"""
decoder_vitdit.py  —  ViTDiT Decoder (DiT-XL/2 + DDPM, global-modulation baseline)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from diffusers import AutoencoderKL
from typing import Tuple
import sys, os
from diffusers import DiffusionPipeline
from diffusers.models.transformers import DiTTransformer2DModel
from diffusers import DiTPipeline 


from microstructure_ed.config import (
    DEVICE,
    DIT_HIDDEN_SIZE,      
    DIT_DEPTH,
    DIT_NUM_HEADS,
    DIT_PATCH_SIZE,
    DIT_CLASS_EMB_DIM,
    SD_VAE_MODEL_ID,      
    SD_VAE_CHANNELS,
    SD_VAE_DOWNSAMPLE_FACTOR,
    DDPM_NUM_TIMESTEPS,
    DDPM_BETA_START,
    DDPM_BETA_END,
    VITDIT_NUM_INFERENCE_STEPS,
)

# ============================================================================
# Variant: 512-D bottleneck (canonical ViTDiT). The trainer must instantiate
# the encoder with `target_dim=TARGET_DIM` to keep encoder and decoder in sync.
# ============================================================================
TARGET_DIM = 512

# Convenience alias kept so the rest of the file is unchanged
NUM_INFERENCE_STEPS = VITDIT_NUM_INFERENCE_STEPS


# ── Core DiT Block (adaLN-Zero) ───────────────────────────────────────────────

class DiTBlock(nn.Module):
    """
    Standard DiT block from Peebles & Xie (2022).
    Conditions the denoiser globally via Adaptive Layer Norm — no cross-attention.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = hidden_size // num_heads

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.mlp  = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * mlp_ratio),
            nn.GELU(approximate='tanh'),
            nn.Linear(hidden_size * mlp_ratio, hidden_size),
        )

        # adaLN-Zero: z → 6 parameters (γ1, β1, α1, γ2, β2, α2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 6),
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        mod = self.adaLN_modulation(cond).unsqueeze(1)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = mod.chunk(6, dim=-1)

        h = self.norm1(x) * (1 + gamma1) + beta1
        attn_out, _ = self.attn(h, h, h)
        x = x + alpha1 * attn_out

        h = self.norm2(x) * (1 + gamma2) + beta2
        x = x + alpha2 * self.mlp(h)
        return x

# ── QK-Norm Attention ────────────────────────────────────────────────────────

def scaled_dot_product_attention_qk_norm(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Normalising Q and K to unit vectors bounds attention logits to [-1, 1]."""
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    d_k = q.shape[-1]
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * (d_k ** -0.5)
    attn_weights = F.softmax(attn_weights, dim=-1)
    return torch.matmul(attn_weights, v)


# ── AdaLN-Zero Block for Token Generator ─────────────────────────────────────

class AdaLNZeroBlock(nn.Module):
    """
    Accepts z_512 (z_dim) to modulate token sequences (dim).
    """
    def __init__(self, dim: int, z_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(z_dim, dim * 6),  # Takes 512-D, outputs 6 * 1152
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

        self.ln1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, q: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        B, N, D = q.shape
        mod = self.adaLN_modulation(z).unsqueeze(1)      
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = mod.chunk(6, dim=-1)

        q_mod = self.ln1(q) * (1 + gamma1) + beta1
        qkv_out = self.qkv(q_mod)
        q_in, k_in, v_in = qkv_out.chunk(3, dim=-1)

        def reshape(t):
            return t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out = scaled_dot_product_attention_qk_norm(
            reshape(q_in), reshape(k_in), reshape(v_in)
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
        q = q + alpha1 * self.out_proj(attn_out)

        ff_mod = self.ln2(q) * (1 + gamma2) + beta2
        q = q + alpha2 * self.ff(ff_mod)
        return q


# ── DiT-XL/2 Backbone ────────────────────────────────────────────────────────

class DiTBackbone(nn.Module):
    """Pure Diffusion Transformer (DiT-XL/2 configuration)."""

    def __init__(
        self,
        in_channels: int,
        patch_size:  int,
        hidden_size: int,
        depth:       int,
        num_heads:   int,
    ):
        super().__init__()
        self.patch_size  = patch_size
        self.num_patches = (64 // patch_size) ** 2   # 64×64 latent → 32×32 patches

        self.patchify   = nn.Linear(in_channels * patch_size ** 2, hidden_size)
        self.unpatchify = nn.Linear(hidden_size, in_channels * patch_size ** 2)

        # 32×32 = 1024 patches (pre-interpolation for 256×256 images)
        self.pos_embed  = nn.Parameter(torch.zeros(1, 1024, hidden_size))

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])

        self.final_norm       = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size)
        )
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.final_modulation[-1]:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.patchify(x) + self.pos_embed
        for block in self.blocks:
            x = block(x, cond)
        scale, shift = self.final_modulation(cond).chunk(2, dim=-1)
        x = self.final_norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.unpatchify(x)
        B, N, C = x.shape
        p = self.patch_size
        h = w = int(math.sqrt(N))
        x = x.reshape(B, h, w, p, p, -1)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, -1, h * p, w * p)
        return x

class _CLIPInjector(nn.Module):
    """Drop-in replacement for nn.Embedding to pass continuous vectors."""
    def forward(self, x):
        return x


class _ZeroEmbedder(nn.Module):
    """Zero out class embedding inside blocks (top-level already adds class_emb)."""
    def forward(self, x):
        return torch.zeros_like(x)


# ── 1. ADD: The Spatial Token Generator ────────────────
class LatentToSpatialTokens(nn.Module):
    """
    z_512 (B, 512) → spatial conditioning tokens (B, num_tokens, inner_dim).

    Default depth bumped from 2 → 4 to match the deeper-decoder strategy
    used by the FMDiT adapter. More AdaLN-Zero blocks let the adapter
    decompose the 512→(16×1152) expansion hierarchically rather than in
    just two hops.
    """
    def __init__(self, z_dim=512, num_tokens=16, inner_dim=1152, num_heads=8, depth=4):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_tokens, inner_dim) * 0.02)
        self.blocks = nn.ModuleList([
            AdaLNZeroBlock(inner_dim, z_dim, num_heads) for _ in range(depth)
        ])
        self.proj_out = nn.Linear(inner_dim, inner_dim)
        self.norm_out = nn.LayerNorm(inner_dim)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, z):
        B = z.shape[0]
        q = self.queries.expand(B, -1, -1).clone()
        for block in self.blocks:
            q = block(q, z)
        return self.norm_out(self.proj_out(q))

# ── 2. ADD: QK-Norm Cross Attention Fuser ────────────────────────────────────
class SpatialCrossFuser(nn.Module):
    def __init__(self, dim=1152, num_heads=16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, patches, tokens):
        # patches: (B, 1024, 1152), tokens: (B, 16, 1152)
        B, N, D = patches.shape
        q = F.normalize(self.q_proj(self.norm_q(patches)), dim=-1)
        k = F.normalize(self.k_proj(self.norm_kv(tokens)), dim=-1)
        v = self.v_proj(self.norm_kv(tokens))

        # Reshape for multi-head attention
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, tokens.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, tokens.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        
        out = out.transpose(1, 2).reshape(B, N, D)
        return patches + self.out_proj(out)

# ── ViTDiT Wrapper ────────────────────────────────────────────────────────────

class ViTDiTDecoder(nn.Module):
    """
    ViTDiT: z_512 → 512×512 via DiT-XL/2 + DDPM + Spatial Cross-Attention.
    Uses a PyTorch hook to inject cross-attention, avoiding diffusers version mismatches.
    """

    def __init__(self, dit_checkpoint: str = "facebook/DiT-XL-2-256"):
        super().__init__()

        print("Initialising DiT-XL/2 backbone for 512-D hijack + Spatial Cross-Attention …")
        
        # 1. Initialize backbone with correct 64x64 latent size
        self.backbone = DiTTransformer2DModel(
            in_channels=SD_VAE_CHANNELS,  # 4
            out_channels=8,
            sample_size=64,               # <--- 512 / 8 = 64
            patch_size=DIT_PATCH_SIZE,
            num_layers=DIT_DEPTH,
            attention_head_dim=72,
            num_attention_heads=DIT_NUM_HEADS,
            num_embeds_ada_norm=1000,
            norm_type="ada_norm_zero",
        )
        
        # 2. Load pretrained weights safely using DiTPipeline
        try:
            from diffusers import DiTPipeline  # Specific pipeline for DiT
            pipe = DiTPipeline.from_pretrained(dit_checkpoint, torch_dtype=torch.float32)
            state_dict = pipe.transformer.state_dict()
            
            # strict=False skips position embeddings (different size: 256 vs 1024)
            self.backbone.load_state_dict(state_dict, strict=False)
            print(" Pretrained transformer weights loaded successfully!")
            print("   (Position embeddings will initialize fresh and fine-tune)")
            
        except Exception as e:
            print(f"⚠️  Error loading pretrained: {e}")
            print("Training from scratch...")
            
        # Unfreeze the FULL backbone — pretrained weights on 256×256 (1024 patches)
        # cannot generalise to our 512×512 (4096 patches) with frozen attention.
        # Position embeddings are fresh, attention patterns must adapt, adaLN must
        # learn our continuous conditioning.  Memory budget allows this (~60 GB free).
        self.backbone.requires_grad_(True)
        self.backbone.enable_gradient_checkpointing()
        trainable_count = sum(p.numel() for p in self.backbone.parameters())
        print(f"Backbone FULLY TRAINABLE: {trainable_count:,} params (grad checkpointing ON)")

        # ── 3. Neutralise the top-level class_embedding ──────────────────
        # DiTTransformer2DModel stores an nn.Embedding(1000, hidden) here but
        # never calls it in forward().  Replace with a no-param identity so
        # model introspection doesn't encounter the wrong type.
        self.backbone.class_embedding = _CLIPInjector()

        # ── 4. Per-block class conditioning: identity pass-through ─────
        # Each block's CombinedTimestepLabelEmbeddings calls class_embedder
        # on class_labels. The original LabelEmbedding expects integer IDs;
        # we replace it with an identity so our continuous 1152-D class_emb
        # flows straight through:  conditioning = timestep_proj + class_emb.
        # token_drop (CFG dropout) is also bypassed since we don't use CFG.
        # CRITICAL: top-level class_embedding already adds class_emb to timestep
        # inside each block's CombinedTimestepLabelEmbeddings. If we use identity
        # here, class_emb is added 28x (once per block). Zero it out at block level.
        for block in self.backbone.transformer_blocks:
            block.norm1.emb.class_embedder = _ZeroEmbedder()
            block.norm1.emb.token_drop = lambda labels, force_drop_ids=None: labels

        # ── 5. Trainable Projection ───────────────────────────────────────────
        # Upgraded from a single Linear(512, 1152) to a small MLP so the
        # global class-conditioning vector can learn a nonlinear mapping
        # rather than a hyperplane projection. Mirrors the target_proj
        # upgrade in the encoder. Both layers use xavier init (NOT zero) —
        # ViTDiT's backbone is fully trainable and needs an active conditioning
        # signal from step 0 (unlike FMDiT where the backbone is frozen and
        # zero-init protects the pretrained weights).
        self.class_proj = nn.Sequential(
            nn.Linear(TARGET_DIM, DIT_CLASS_EMB_DIM),
            nn.GELU(),
            nn.Linear(DIT_CLASS_EMB_DIM, DIT_CLASS_EMB_DIM),
        )
        nn.init.xavier_uniform_(self.class_proj[0].weight)
        nn.init.zeros_(self.class_proj[0].bias)
        nn.init.xavier_uniform_(self.class_proj[2].weight)
        nn.init.zeros_(self.class_proj[2].bias)

        # ── 6. Spatial Conditioning Modules ───────────────────────────────────
        self.token_gen = LatentToSpatialTokens(z_dim=TARGET_DIM, inner_dim=DIT_CLASS_EMB_DIM)
        # spatial_fusers created below as nn.ModuleList (one per injection block)

        self._current_z_tokens = None

        # One independent SpatialCrossFuser per injection point so each
        # layer can learn its own cross-attention mapping.
        self._injection_blocks = [0, 7, 14, 21]
        self.spatial_fusers = nn.ModuleList([
            SpatialCrossFuser(dim=DIT_CLASS_EMB_DIM, num_heads=DIT_NUM_HEADS)
            for _ in self._injection_blocks
        ])

        def _make_hook(fuser):
            def cross_attn_pre_hook(module, args):
                hidden_states = args[0]
                if self._current_z_tokens is not None:
                    hidden_states = fuser(hidden_states, self._current_z_tokens)
                return (hidden_states,) + args[1:]
            return cross_attn_pre_hook

        for fuser, blk_idx in zip(self.spatial_fusers, self._injection_blocks):
            self.backbone.transformer_blocks[blk_idx].register_forward_pre_hook(
                _make_hook(fuser)
            )

        # DDPM schedule
        self.register_buffer("betas", torch.linspace(DDPM_BETA_START, DDPM_BETA_END, DDPM_NUM_TIMESTEPS))
        alphas = 1.0 - self.betas
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, x_t, t, z_512):
        # Project z_512 to class embedding dimension
        class_emb = self.class_proj(z_512)
        
        # Generate spatial tokens and store them for the hook to access
        self._current_z_tokens = self.token_gen(z_512)
        
        # Call backbone forward completely normally! 
        # It handles patching, pos_embed, time_proj, unpatchify automatically behind the scenes.
        output = self.backbone(
            hidden_states=x_t, 
            timestep=t,
            class_labels=class_emb, 
            return_dict=False
        )[0]
        
        # Backbone outputs 8 channels, slice to 4 for SD VAE
        return output[:, :x_t.shape[1], :, :]

    # ── inference ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        z_512:     torch.Tensor,
        num_steps: int = NUM_INFERENCE_STEPS,
    ) -> torch.Tensor:
        """DDIM Sampling (ODE-based, stable for large timestep jumps)."""
        B      = z_512.shape[0]
        device = z_512.device
        img    = torch.randn(B, SD_VAE_CHANNELS, 64, 64, device=device)

        # Create evenly spaced timestep schedule (e.g., 1000 -> 50 steps)
        timesteps = torch.linspace(DDPM_NUM_TIMESTEPS - 1, 0, num_steps, dtype=torch.long, device=device)

        for i in range(len(timesteps)):
            t = timesteps[i].unsqueeze(0).expand(B)
            eps_pred = self.forward(img, t, z_512)

            # Get current alpha
            alpha_t = self.alphas_cumprod[t].view(B, 1, 1, 1)

            # Predict x0
            x0_pred = (img - torch.sqrt(1 - alpha_t) * eps_pred) / torch.sqrt(alpha_t)

            # Get next alpha
            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1].unsqueeze(0).expand(B)
                alpha_t_next = self.alphas_cumprod[t_next].view(B, 1, 1, 1)
            else:
                alpha_t_next = torch.ones_like(alpha_t)

            # DDIM update step (no random noise added!)
            img = torch.sqrt(alpha_t_next) * x0_pred + torch.sqrt(1 - alpha_t_next) * eps_pred

        return img

# ── Logit-Normal Time Sampler ─────────────────────────────────────────────────

def _sample_logit_normal_timesteps(B: int, device: torch.device, shift: float = 3.0) -> torch.Tensor:
    """Focus training on intermediate timesteps where structure forms."""
    u = torch.randn(B, device=device)
    t = torch.sigmoid(u + math.log(shift))
    return t.clamp(1e-5, 1 - 1e-5)



# ── DDP helper for buffer access ───────────────────────────────────────────────
def _get_base(model):
    """Unwrap DDP/FSDP for buffer access while keeping forward through wrapper."""
    return getattr(model, 'module', model)

# ── Standard DDPM Loss (Used for Validation) ──────────────────────────────────

def ddpm_loss(
    model: 'ViTDiTDecoder',
    x_0:   torch.Tensor,
    z_512: torch.Tensor,
) -> Tuple[torch.Tensor, dict]:
    """L_simple = || ε - ε_θ(x_t, t) ||²  (Keeps uniform sampling for fast val metric)"""
    B      = x_0.shape[0]
    device = x_0.device
    t      = torch.randint(0, DDPM_NUM_TIMESTEPS, (B,), device=device).long()
    eps    = torch.randn_like(x_0)
    
    alpha_t = _get_base(model).alphas_cumprod[t].view(B, 1, 1, 1)
    x_t     = torch.sqrt(alpha_t) * x_0 + torch.sqrt(1 - alpha_t) * eps
    
    eps_pred = model(x_t, t, z_512)
    loss     = F.mse_loss(eps_pred.float(), eps.float())
    
    return loss, {"loss": loss.item(), "mean_t": t.float().mean().item()}

# ── Composite ViTDiT Loss (Used for Training) ─────────────────────────────────

def composite_ddpm_loss(
    model,
    x_0,
    z_512,
    z_512_aug,
    contrastive_fn,
    lambda_diff=1.0,
    lambda_recon=0.5,
    lambda_contrastive=0.1,
    vicreg_fn=None,
    freq_fn=None,
    vae=None,
    lambda_vicreg=0.0,
    lambda_freq=0.0,
    freq_t_threshold=0.7,
    freq_max_samples=2,
):
    B      = x_0.shape[0]
    device = x_0.device

    # 1. Sample t using Logit-Normal (shift=3.0) instead of uniform
    t_cont = _sample_logit_normal_timesteps(B, device, shift=3.0)
    t_idx  = (t_cont * DDPM_NUM_TIMESTEPS).long()
    
    eps = torch.randn_like(x_0)
    alpha_t           = _get_base(model).alphas_cumprod[t_idx].view(B, 1, 1, 1)
    sqrt_alpha_t      = torch.sqrt(alpha_t)
    sqrt_one_minus_at = torch.sqrt(1.0 - alpha_t)
    
    x_t      = sqrt_alpha_t * x_0 + sqrt_one_minus_at * eps
    eps_pred = model(x_t, t_idx, z_512)

    # L1: Standard epsilon prediction loss
    loss_eps = F.mse_loss(eps_pred.float(), eps.float())

    # L2: x0-prediction reconstruction (weighted)
    x0_pred      = (x_t.float() - sqrt_one_minus_at * eps_pred.float()) / sqrt_alpha_t.clamp(min=1e-4)
    
    # Weight by (1-t)^2 so it focuses on clean images and ignores pure noise
    recon_weight = (1.0 - t_cont.view(B, 1, 1, 1)) ** 2
    loss_recon   = (recon_weight * (x0_pred - x_0.float()) ** 2).mean()

    # L3: Contrastive loss on encoder embeddings
    embeddings_both = torch.cat([z_512, z_512_aug], dim=0)
    loss_contr      = contrastive_fn(embeddings_both.float())

    # ── VICReg on z_512 (all-gather across GPUs for better cov estimate) ────
    loss_vicreg = torch.tensor(0.0, device=device)
    if vicreg_fn is not None and lambda_vicreg > 0:
        z_local = z_512.float()
        if dist.is_initialized() and dist.get_world_size() > 1:
            gathered = [torch.zeros_like(z_local) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, z_local)
            # Replace local shard so grads flow back through this rank
            gathered[dist.get_rank()] = z_local
            z_all = torch.cat(gathered, dim=0)
        else:
            z_all = z_local
        loss_vicreg = vicreg_fn(z_all)

    # ── Frequency loss in pixel space (gated on small t, sample-capped) ──────
    loss_freq = torch.tensor(0.0, device=device)
    if (freq_fn is not None and vae is not None and lambda_freq > 0):
        mask = t_cont < freq_t_threshold
        if mask.any():
            idx_sel = mask.nonzero(as_tuple=False).squeeze(1)[:freq_max_samples]
            sf = vae.scaling_factor
            vae_dtype = next(vae.vae.parameters()).dtype
            with torch.amp.autocast('cuda', enabled=False):
                def _vae_decode(z):
                    return vae.vae.decode(z).sample

                z_in = (x0_pred[idx_sel] / sf).to(vae_dtype)
                img_pred = grad_checkpoint(
                    _vae_decode, z_in, use_reentrant=False
                ).clamp(-1, 1)

                with torch.no_grad():
                    z_tgt = (x_0[idx_sel].float() / sf).to(vae_dtype)
                    img_tgt = vae.vae.decode(z_tgt).sample.clamp(-1, 1)

            loss_freq = freq_fn(img_pred, img_tgt)

    total = (
        lambda_diff        * loss_eps     +
        lambda_recon       * loss_recon   +
        lambda_contrastive * loss_contr   +
        lambda_vicreg      * loss_vicreg  +
        lambda_freq        * loss_freq
    )

    return total, {
        "loss_epsilon":     loss_eps.item(),
        "loss_recon":       loss_recon.item(),
        "loss_contrastive": loss_contr.item(),
        "loss_vicreg":      loss_vicreg.item(),
        "loss_freq":        loss_freq.item(),
        "loss_total":       total.item(),
        "mean_t":           t_cont.mean().item(),
    }


# ── Standard SD-VAE Utility ───────────────────────────────────────────────────

class SDVAE:
    """Standard SD 1.5 VAE (4-channel, frozen)."""

    def __init__(self, model_id: str = SD_VAE_MODEL_ID):
        print(f"Loading frozen SD VAE ({model_id}) …")
        self.vae = AutoencoderKL.from_pretrained(model_id).eval()
        for p in self.vae.parameters():
            p.requires_grad = False
        self.scaling_factor = 0.18215

    def to(self, device):
        self.vae = self.vae.to(device)
        return self

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        latents = self.vae.encode(images).latent_dist.sample()
        return latents * self.scaling_factor

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents / self.scaling_factor
        return self.vae.decode(latents).sample.float()