# encoder_arch_pretrained.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from torch.utils.checkpoint import checkpoint
import sys, os

from microstructure_ed.config import (
    DEVICE, VIT_MODEL_NAME, VIT_PRETRAINED,
    VIT_DIM, DEFAULT_TARGET_DIM, ATTENTION_POOLER_HEADS,
    ATTENTION_POOLER_NUM_QUERIES,
)


class MultiQueryAttentionPooler(nn.Module):
    """
    Multi-query Perceiver-style pooler for richer spatial summarization.

    Upgrades the single-query pooler by introducing N learned queries that
    each specialize on different aspects of the spatial map (e.g. structure,
    texture, color, high-frequency detail). The queries first cross-attend
    to all ViT tokens, then exchange information via self-attention so that
    they can divide labor explicitly. Finally, the N specialized vectors are
    concatenated and projected back down to a single 1280-D summary that
    feeds the 512-D bottleneck.

    Architectural pieces:
        1. cross_attn  : queries → ViT tokens (information extraction)
        2. self_attn   : queries ↔ queries     (specialization / labor split)
        3. ffn         : per-token refinement
        4. merge       : (N × dim) → dim       (single vector for bottleneck)

    Output shape stays (B, dim) so target_proj remains unchanged.
    """

    def __init__(
        self,
        dim: int = VIT_DIM,
        num_heads: int = ATTENTION_POOLER_HEADS,
        num_queries: int = ATTENTION_POOLER_NUM_QUERIES,
        spatial: bool = False,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.dim = dim
        # Spatial mode: return the Q refined queries as a token sequence
        # (B, Q, dim) WITHOUT collapsing them to one vector. The Compressor
        # then projects each token to d_z and concatenates -> flat (B, Q*d_z),
        # preserving spatial structure inside a flat latent (optimizer-compat).
        self.spatial = spatial

        # N learnable "search queries" — each can specialize.
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)

        # Stage 1: cross-attention from queries to ViT tokens.
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        # Stage 2: self-attention among queries so they can divide labor.
        self.ln_sa = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        # Stage 3: per-query FFN refinement.
        self.ln_ff = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

        # Stage 4: merge N specialized vectors → single 1280-D summary.
        if not spatial:
            self.ln_merge = nn.LayerNorm(num_queries * dim)
            self.merge = nn.Linear(num_queries * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1370, 1280) — All ViT output tokens (CLS + 1369 patches).

        Returns:
            (B, 1280) — Single dense summary vector for the bottleneck.
        """
        B = x.shape[0]
        q = self.queries.expand(B, -1, -1)  # (B, Q, D)

        # 1. Cross-attention: queries actively scan all ViT tokens.
        q_norm = self.ln_q(q)
        kv_norm = self.ln_kv(x)
        attn_out, _ = self.cross_attn(q_norm, kv_norm, kv_norm)
        q = q + attn_out                                          # (B, Q, D)

        # 2. Self-attention: queries communicate to specialize.
        q_sa = self.ln_sa(q)
        sa_out, _ = self.self_attn(q_sa, q_sa, q_sa)
        q = q + sa_out                                            # (B, Q, D)

        # 3. FFN refinement (per-query).
        q = q + self.ffn(self.ln_ff(q))                           # (B, Q, D)

        # 4. Merge: concat queries → linear projection back to D.
        if self.spatial:
            return q                                              # (B, Q, D)
        q_flat = q.reshape(B, self.num_queries * self.dim)        # (B, Q*D)
        return self.merge(self.ln_merge(q_flat))                  # (B, D)


# Backward-compatible alias (in case external code references the old name).
AttentionPooler = MultiQueryAttentionPooler
    

class Compressor(nn.Module):
    """
    Unified Compression Encoder: Image (512x512) -> z_512 (512-D vector).
    Used by ViT-SDXL, ViT-FMDiT, and ViT-DiT pipelines.
    
    Architecture Strategy:
    1. Uses ViT-H/14 (630M params) for maximum feature extraction capacity.
    2. Pads 512x512 to 518x518 to perfectly fit the 14x14 patch size without 
       cropping pixels (preserving 1-pixel grain boundaries).
    3. SMART FREEZING: Trains the *last* N blocks (closest to output), freezes
       the *first* blocks (closest to pixels).
       - Early blocks (0-15): Universal edge/texture features that transfer 
         perfectly from CLIP — no adaptation needed for microstructures.
       - Late blocks (16-31): CLIP semantic compression that discards visual
         detail irrelevant to text matching. These MUST be retrained to preserve
         per-image microstructure identity for reconstruction.
    4. Uses AttentionPooler to actively extract spatial info before compression.
    5. Compresses 1280-D -> 512-D via a Linear layer.
    
    Args:
        use_gradient_checkpointing (bool): If True, recomputes activations during
            backward for trainable blocks to save massive VRAM.
        trainable_blocks (int): Number of LAST blocks to train.
            16 = Freeze 0-15, Train 16-31 (Default).
            0  = Freeze all ViT blocks (only attention_pooler + target_proj train).
            32 = Train all blocks.
    """

    def __init__(self, use_gradient_checkpointing: bool = True, trainable_blocks: int = 16,
                 target_dim: int = DEFAULT_TARGET_DIM, spatial_tokens: int = 0):
        super().__init__()
        # Store the per-instance bottleneck width so external code can introspect.
        self.target_dim = target_dim
        # Spatial-token latent: when spatial_tokens > 0 the bottleneck is laid
        # out as `spatial_tokens` blocks of d_z = target_dim // spatial_tokens
        # each, concatenated into the SAME flat (B, target_dim) vector so all
        # downstream code (VICReg, pooled_proj, optimizers) is unchanged.
        self.spatial_tokens = spatial_tokens
        if spatial_tokens > 0:
            assert target_dim % spatial_tokens == 0, \
                f"target_dim {target_dim} must be divisible by spatial_tokens {spatial_tokens}"
            self._d_z = target_dim // spatial_tokens
        else:
            self._d_z = target_dim

        # Load Pretrained OpenCLIP ViT-H/14
        clip_model, _, _ = open_clip.create_model_and_transforms(
            VIT_MODEL_NAME, pretrained=VIT_PRETRAINED,
        )
        vis = clip_model.visual

        # Extract necessary visual components
        self.conv1 = vis.conv1   # Patch embedding layer
        self.cls_token = nn.Parameter(vis.class_embedding.data.clone().unsqueeze(0).unsqueeze(0))

        # Interpolate Positional Embeddings (16x16 -> 37x37)
        # Standard ViT-H/14 is trained on 224x224 (16x16 grid + 1 CLS = 257 tokens).
        # We are using 518x518 (37x37 grid + 1 CLS = 1370 tokens).
        pretrained_pos = vis.positional_embedding.data.clone()  # (257, 1280)
        self.pos_embed = nn.Parameter(self._interpolate_pos_embed(pretrained_pos))

        self.ln_pre = vis.ln_pre  # Pre-transformer LayerNorm
        self.vit_blocks = vis.transformer.resblocks  # 32 Transformer blocks
        self.ln_post = vis.ln_post  # Post-transformer LayerNorm

        # ── SMART FREEZING STRATEGY ─────────────────────────────────────────
        # 1. ALWAYS freeze spatial grids. Training these causes catastrophic 
        #    instability because they dictate absolute pixel positions.
        self.conv1.requires_grad_(False)
        self.cls_token.requires_grad_(False)
        self.pos_embed.requires_grad_(False)
        self.ln_pre.requires_grad_(False)

        # 2. Freeze early blocks, train late blocks.
        # Block 0 is closest to pixels (universal features — keep frozen).
        # Block 31 is closest to output (CLIP semantics — needs retraining).
        n_blocks = len(self.vit_blocks)  # 32
        freeze_up_to = n_blocks - trainable_blocks  # e.g. 32-16=16 → freeze 0-15
        for i, block in enumerate(self.vit_blocks):
            if i < freeze_up_to:
                block.requires_grad_(False)
            # else: trainable (requires_grad=True by default from pretrained load)
        # ──────────────────────────────────────────────────────────────────

        # ALWAYS train the Attention Pooler and Bottleneck Projection
        self.attention_pooler = MultiQueryAttentionPooler(
            dim=VIT_DIM,
            num_heads=ATTENTION_POOLER_HEADS,
            num_queries=(spatial_tokens if spatial_tokens > 0
                         else ATTENTION_POOLER_NUM_QUERIES),
            spatial=(spatial_tokens > 0),
        )

        # The bottleneck. Upgraded from a single Linear+LN to a small MLP
        # so that the violent 1280→target_dim compression can learn a curved
        # manifold rather than a hyperplane projection. The intermediate hidden
        # width equals VIT_DIM to preserve information capacity before the squeeze.
        # Final LayerNorm stabilizes the bottleneck output.
        # `target_dim` is per-instance (default 512) so multiple variants
        # (e.g. 768-D / 1024-D bottlenecks) can share this class.
        # In spatial mode this Linear stack is applied per-token (Linear acts
        # on the last dim), projecting each of the Q pooled tokens 1280 -> d_z.
        _proj_out_dim = self._d_z
        self.target_proj = nn.Sequential(
            nn.Linear(VIT_DIM, VIT_DIM),
            nn.GELU(),
            nn.Linear(VIT_DIM, _proj_out_dim),
            nn.LayerNorm(_proj_out_dim),
        )

        self.gradient_checkpointing = use_gradient_checkpointing

        # Free the text encoder to save massive amounts of VRAM
        del clip_model

        # CLIP normalization buffers (specific to OpenCLIP, NOT ImageNet)
        self.register_buffer('img_mean', torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer('img_std', torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

    # ── Phased freeze / unfreeze helpers ─────────────────────────────────────

    def freeze_vit_blocks(self):
        """Freeze ALL ViT transformer blocks. Attention pooler + target_proj remain trainable."""
        for block in self.vit_blocks:
            block.requires_grad_(False)
        self.ln_post.requires_grad_(False)

    def unfreeze_vit_blocks(self, last_n: int = 16):
        """Unfreeze the last `last_n` ViT blocks for fine-tuning.
        
        Call this at the start of Phase 2 to enable gradual encoder adaptation.
        The attention pooler and target_proj are always trainable.
        """
        n_blocks = len(self.vit_blocks)
        freeze_up_to = n_blocks - last_n
        for i, block in enumerate(self.vit_blocks):
            if i >= freeze_up_to:
                block.requires_grad_(True)
        self.ln_post.requires_grad_(True)

    def _interpolate_pos_embed(self, pos_embed: torch.Tensor) -> torch.Tensor:
        """
        Bicubically interpolates 2D positional embeddings to match the new image size.
    
        We separate the CLS token (which represents global position 0) from the 
        spatial grid, scale the grid from 16x16 to 37x37, and re-concatenate.
    
        Args:
            pos_embed: (257, 1280) — 1 CLS + 256 patch positions (16x16 grid)
        
        Returns:
            (1370, 1280) — 1 CLS + 1369 patch positions (37x37 grid)
        """
        cls_pos = pos_embed[:1, :]  # (1, 1280)
        patch_pos = pos_embed[1:, :] # (256, 1280)
        old_grid, new_grid = 16, 37   # 224 / 14 = 16, 518 / 14 = 37

        # Reshape to 2D grid format expected by F.interpolate: (B, C, H, W)
        patch_pos = patch_pos.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)

        # Scale the 2D coordinate system smoothly
        patch_pos = F.interpolate(patch_pos.float(), size=(new_grid, new_grid), mode="bicubic", align_corners=False)

        # Flatten back to sequence format
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(new_grid * new_grid, -1)

        return torch.cat([cls_pos, patch_pos], dim=0) # (1370, 1280)
    
    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """
        Prepares raw microstructure images for the ViT.
    
        CRITICAL: We DO NOT resize to 224x224. Resizing applies a smoothing filter
        that destroys 1-pixel grain boundaries. Instead, we pad to 518x518.
    
        Args:
            images: (B, 3, 512, 512) in [-1, 1] range (standard diffusion format)
        
        Returns:
            (B, 3, 518, 518) with CLIP normalization applied.
        """
        # 1. Convert from diffusion range [-1, 1] to standard image range [0, 1]
        x = (images + 1.0) / 2.0

        # 2. THE PADDING TRICK
        # ViT-H/14 uses a patch size of 14. 512 is NOT cleanly divisible by 14.
        # 512 / 14 = 36.57. If we just pass 512, Conv2d slices off the last pixels.
        # We pad to 518 (which is 37 * 14) to preserve edge microstructure data.
        B, C, H, W = x.shape
        pad_h, pad_w = 518 - H, 518 - W
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)

        # 3. Apply CLIP-specific normalization
        return (x - self.img_mean) / self.img_std
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Full encoding pipeline: Image -> Extreme 512-D Bottleneck.
    
        Args:
            images: (B, 3, 512, 512) in [-1, 1] -- raw RGB microstructure images
        
        Returns:
            z_512: (B, 512) -- The strict bottleneck conditioning vector
        """
        B = images.shape[0]

        # Preprocess: Pad and CLIP normalize
        x = self.preprocess(images) # (B, 3, 518, 518)

        # Convert to sequence of tokens via patch embedding
        x = self.conv1(x)    # (B, 1280, 37, 37)
        x = x.reshape(B, x.shape[1], -1).permute(0, 2, 1)  # (B, 1369, 1280)

        # Prepend CLS Token & Add Interpolated Pos Embeds
        cls = self.cls_token.expand(B, -1, -1).to(x.dtype)
        x = torch.cat([cls, x], dim=1)     # (B, 1370, 1280)
        x = x + self.pos_embed.to(x.dtype)

        # Pre-transformer layerNorm
        x = self.ln_pre(x)

        # Pass through all 32 transformer blocks
        # OPTIMIZATION: We only apply gradient checkpointing to blocks that actually 
        # require gradients (the trainable bottom half). Applying it to frozen blocks 
        # would waste compute re-calculating activations we don't need gradients for.
        for i, block in enumerate(self.vit_blocks):
            # Check parameters directly to bypass OpenCLIP's custom __getattr__ bug
            block_needs_grad = any(p.requires_grad for p in block.parameters())
            
            if self.gradient_checkpointing and self.training and block_needs_grad:
                # OpenCLIP forward signature is block(x, attn_mask)
                x = checkpoint(block, x, None, use_reentrant=False)
            else:
                x = block(x)

        # Post-transformer layerNorm on all tokens
        x = self.ln_post(x)

        # Attention pooler: 1370 tokens -> summary.
        #   spatial mode  -> (B, Q, 1280) token sequence (no collapse)
        #   legacy mode   -> (B, 1280) single vector
        pooled = self.attention_pooler(x)

        if self.spatial_tokens > 0:
            # Per-token projection 1280 -> d_z, then concat tokens -> flat z.
            tokens = self.target_proj(pooled)              # (B, Q, d_z)
            z = tokens.reshape(pooled.shape[0], self.target_dim)  # (B, target_dim)
        else:
            z = self.target_proj(pooled)                   # (B, target_dim)

        return z