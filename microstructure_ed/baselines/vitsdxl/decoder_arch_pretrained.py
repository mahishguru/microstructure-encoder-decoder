# decoder_arch_pretrained.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline, DDPMScheduler, EulerDiscreteScheduler, AutoencoderKL
import sys, os
from microstructure_ed.config import (
    SDXL_MODEL_NAME, COND_DIM, N_TOKENS, 
    NUM_INFERENCE_STEPS, LATENT_TO_TOKENS_DEPTH,
)

# ============================================================================
# Variant: 512-D bottleneck (canonical SDXL). The trainer must instantiate
# the encoder with `target_dim=TARGET_DIM` to keep encoder and decoder in sync.
# ============================================================================
TARGET_DIM = 512

# ─────────────────────────────────────────────────────────────────────────────
# Architecture (v4): Single latent z for inverse design
#
#   Compressor (ViT-H/14 @ 512×512 + AttentionPooler) → z: (B, 512)
#
#   z → LatentToTokens (AdaLN-Zero) → encoder_hidden_states (B, 77, 2048) → UNet cross-attn
#   z → pooled_proj                 → text_embeds (B, 1280)               → UNet time-embedding
#   z → ContrastiveLoss (InfoNCE) + VICReg
#
# Both SDXL UNet conditioning channels used, both driven by z:
#   Channel 1 (cross-attention)  : LatentToTokens(z)  → (B, 77, 2048)
#   Channel 2 (time-embedding)   : pooled_proj(z)     → (B, 1280) + time_ids (B, 6)
#
# ControlNet is permanently disabled in this architecture.
# ─────────────────────────────────────────────────────────────────────────────


# ── AdaLN-Zero Block ──────────────────────────────────────────────────────────

class AdaLNZeroBlock(nn.Module):
    """
    DiT-style Adaptive LayerNorm Zero block with gated residuals.

    z → SiLU → Linear → 6 modulation params per dim:
        γ₁, β₁ : scale & shift for pre-attention LayerNorm
        α₁     : gate on attention residual
        γ₂, β₂ : scale & shift for pre-FFN LayerNorm
        α₂     : gate on FFN residual

    Key differences from FiLM:
        - elementwise_affine=False on LayerNorm (z provides ALL affine params)
        - Modulates BOTH attention AND FFN (FiLM only modulated attention)
        - α gates on residuals — zero-initialized so block starts as identity
    """

    def __init__(self, dim: int, z_dim: int, num_heads: int):
        super().__init__()

        # z → 6 modulation parameters (γ₁, β₁, α₁, γ₂, β₂, α₂)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(z_dim, dim * 6),
        )
        # Zero-init → all gates start at 0, block is identity at init
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

        # LayerNorms without learnable affine (z provides scale/shift)
        self.ln1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(dim, elementwise_affine=False)

        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, q: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        q : (B, N, dim) — learned queries
        z : (B, z_dim)  — 512-D latent vector from Compressor

        Returns: updated queries (B, N, dim)
        """
        # z → 6 modulation vectors
        mod = self.adaLN_modulation(z).unsqueeze(1)       # (B, 1, 6*dim)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = mod.chunk(6, dim=-1)

        # Modulated self-attention with gated residual
        q_mod = self.ln1(q) * (1 + gamma1) + beta1        # (B, N, dim)
        attn_out, _ = self.attn(q_mod, q_mod, q_mod)
        q = q + alpha1 * attn_out                          # gated residual

        # Modulated FFN with gated residual
        ff_mod = self.ln2(q) * (1 + gamma2) + beta2        # (B, N, dim)
        q = q + alpha2 * self.ff(ff_mod)                   # gated residual

        return q


# ── LatentToTokens ──────────────────────────────────────────────────────────

class LatentToTokens(nn.Module):
    """
    Maps a single latent z (B, 512) → N conditioning tokens (B, N, 2048).

    Uses learned positional queries + AdaLN-Zero self-attention (DiT-style).
    This is NOT a flat MLP (which would produce degenerate near-identical tokens
    causing attention collapse in the UNet cross-attention layers).

    Each query specialises via:
        1. Unique learned positional identity
        2. AdaLN-Zero modulation from z at every self-attention layer
           (scale, shift, gate on BOTH attention and FFN)
        3. Self-attention allowing queries to differentiate from each other

    Output is zero-initialised (proj_out + norm_out.bias) so that the
    conditioning starts neutral and ramps up during training.
    """

    def __init__(
        self,
        z_dim: int      = TARGET_DIM,             # 512 — Bottleneck dimension
        inner_dim: int  = TARGET_DIM,             # 512 — internal processing dimension
        output_dim: int = COND_DIM,               # 2048 — SDXL cross-attention dimension
        num_tokens: int = N_TOKENS,               # 77 — number of output conditioning tokens
        num_heads: int  = 8,
        depth: int      = LATENT_TO_TOKENS_DEPTH, # 3 — from config
    ):
        super().__init__()
        self.num_tokens = num_tokens

        # Learned positional queries — each one specialises in a different
        # aspect of the microstructure (e.g. grain boundaries, phase regions)
        self.queries = nn.Parameter(
            torch.randn(1, num_tokens, inner_dim) * 0.02
        )

        # AdaLN-Zero self-attention blocks (DiT-style)
        self.blocks = nn.ModuleList([
            AdaLNZeroBlock(inner_dim, z_dim, num_heads)
            for _ in range(depth)
        ])

        # Project to SDXL cross-attention space
        self.proj_out = nn.Linear(inner_dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        # Initialize the output path with SMALL non-zero Xavier so the frozen
        # UNet sees a meaningful conditioning signal from the first step and the
        # gradient-feedback loop can boot up. (Pure zero-init produced a dead
        # adapter — see run_20260426_164407 diagnosis: proj_out norm stayed at
        # 0.46 vs input layers at 13-27 even after 23 epochs.)
        nn.init.xavier_uniform_(self.proj_out.weight, gain=0.10)
        nn.init.zeros_(self.proj_out.bias)
        nn.init.zeros_(self.norm_out.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z       : (B, 512)  — single latent from Compressor (AttentionPooler)
        returns : (B, num_tokens, output_dim) — conditioning for SDXL cross-attention
        """
        B = z.shape[0]
        q = self.queries.expand(B, -1, -1).clone()   # (B, N, inner_dim)

        for block in self.blocks:
            q = block(q, z)                           # (B, N, inner_dim)

        return self.norm_out(self.proj_out(q))        # (B, N, output_dim)


# ── SDXLDecoder ───────────────────────────────────────────────────────────────

class SDXLDecoder(nn.Module):
    """
    SDXL-based decoder conditioned entirely on a single latent z (512-D).

    Two conditioning channels (both driven by z):
      1. Cross-attention (encoder_hidden_states):
             z → LatentToTokens (3 AdaLN-Zero blocks) → (B, 77, 2048)
      2. Time-embedding additive (added_cond_kwargs):
             z → pooled_proj → text_embeds (B, 1280)

    Trainable components:
        LatentToTokens    ~8 M   (77 queries + 3 AdaLN-Zero blocks + proj)
        pooled_proj       ~0.8 M (512 → 1280 → 1280)
        null_token        512

    Frozen:
        SDXL UNet  (2.6B params, bf16)
    """

    def __init__(self, p_drop: float = 0.1):
        super().__init__()
        self.p_drop = p_drop

        # ── Channel 2: Pooled global conditioning ─────────────────────────────
        self.pooled_proj = nn.Sequential(
            nn.Linear(TARGET_DIM, 1280),
            nn.SiLU(),
            nn.Linear(1280, 1280),
        )
        # Initialize the final projection with small non-zero Xavier so SDXL's
        # add_embedding sees real pooled conditioning (norm ~30 expected) from
        # epoch 1. Pure zero-init kept this layer at norm 1.6 forever.
        nn.init.xavier_uniform_(self.pooled_proj[2].weight, gain=0.10)
        nn.init.zeros_(self.pooled_proj[2].bias)

        # Null token for classifier-free guidance dropout (applied to z)
        self.null_token = nn.Parameter(torch.randn(TARGET_DIM) * 0.02)

        # ── Channel 1: LatentToTokens (cross-attention conditioning) ──────────
        self.token_proj = LatentToTokens(
            z_dim=TARGET_DIM,
            inner_dim=TARGET_DIM,
            output_dim=COND_DIM,
            num_tokens=N_TOKENS,
            depth=LATENT_TO_TOKENS_DEPTH,
        )

        # ── Pretrained frozen UNet ─────────────────────────────────────────────
        print("Loading SDXL UNet (pretrained, frozen)...")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            SDXL_MODEL_NAME,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        # Convert fp16 → bf16: bf16 has the same exponent range as float32
        # (max ≈ 3.4e38) so gradient flow through the frozen UNet backward
        # cannot overflow, eliminating NaN gradients from the pixel-loss path.
        self.unet = pipe.unet.to(torch.bfloat16)
        self.unet.requires_grad_(False)
        self.unet.eval()

        # Cache SDXL null (empty-string) text embeddings while text encoders
        # are still loaded in the pipeline.
        self._compute_null_embeddings(pipe)

        # ControlNet permanently disabled in this architecture
        self.controlnet = None

        # Training scheduler: DDPMScheduler — used only for add_noise() and alphas_cumprod.
        self.scheduler = DDPMScheduler.from_pretrained(
            SDXL_MODEL_NAME,
            subfolder="scheduler",
        )
        self.scheduler.set_timesteps(NUM_INFERENCE_STEPS)

        # Inference scheduler: EulerDiscreteScheduler — SDXL's native scheduler.
        self.inference_scheduler = EulerDiscreteScheduler.from_pretrained(
            SDXL_MODEL_NAME,
            subfolder="scheduler",
        )

    @torch.no_grad()
    def _compute_null_embeddings(self, pipe) -> None:
        """
        Pre-compute SDXL null text embeddings (empty-string CLIP) and cache as buffers.

        SDXL null condition = CLIP("") — NOT zeros.
        During CFG training the UNet sees CLIP("") as the unconditional branch; zeros
        are completely out-of-distribution and produce neon banding at epoch 0.
        """
        print("  Computing SDXL null text embeddings (empty-string CLIP)...")
        try:
            tok1, tok2 = pipe.tokenizer, pipe.tokenizer_2
            te1,  te2  = pipe.text_encoder, pipe.text_encoder_2

            ids1 = tok1(
                "", padding="max_length", max_length=tok1.model_max_length,
                truncation=True, return_tensors="pt",
            ).input_ids
            ids2 = tok2(
                "", padding="max_length", max_length=tok2.model_max_length,
                truncation=True, return_tensors="pt",
            ).input_ids

            out1 = te1(ids1, output_hidden_states=True)
            out2 = te2(ids2, output_hidden_states=True)

            h1 = out1.hidden_states[-2].float()   # (1, 77, 768)
            h2 = out2.hidden_states[-2].float()   # (1, 77, 1280)
            null_hidden = torch.cat([h1, h2], dim=-1)  # (1, 77, 2048)

            null_pooled = out2.text_embeds.float()  # (1, 1280)

            self.register_buffer("sdxl_null_hidden", null_hidden, persistent=True)
            self.register_buffer("sdxl_null_pooled", null_pooled,  persistent=True)
            print(f"    null_hidden {null_hidden.shape}  norm={null_hidden.norm():.3f}")
            print(f"    null_pooled {null_pooled.shape}  norm={null_pooled.norm():.3f}")

        except Exception as e:
            print(f"  Warning: null embed computation failed ({e}); falling back to zeros.")
            self.register_buffer("sdxl_null_hidden", torch.zeros(1, 77, 2048), persistent=True)
            self.register_buffer("sdxl_null_pooled", torch.zeros(1, 1280),     persistent=True)

    # ── Conditioning helpers ───────────────────────────────────────────────────

    def get_conditioning(self, z: torch.Tensor) -> torch.Tensor:
        """
        Channel 1: z (B, 512) → encoder_hidden_states (B, 77, 2048).
        """
        return self.token_proj(z)

    def apply_cfg_dropout(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Classifier-free guidance dropout on z.

        Replaces each sample's embedding with the null_token with probability p_drop.

        Returns:
            embedding_cf : (B, 512) — z with some samples replaced by null_token
        """
        B = embedding.shape[0]
        keep = (torch.rand(B, device=embedding.device) >= self.p_drop)  # True = keep

        null_emb = self.null_token.unsqueeze(0).expand(B, -1).to(embedding.dtype)
        embedding_cf = torch.where(keep.view(B, 1).expand_as(embedding), embedding, null_emb)

        return embedding_cf

    def get_added_cond_kwargs(
        self,
        B: int,
        embedding: torch.Tensor,
        h: int = 512,
        w: int = 512,
        device: torch.device = None,
    ) -> dict:
        """
        Channel 2: SDXL additional conditioning kwargs.
        z → pooled_proj → text_embeds + time_ids.
        """
        device = device or next(self.parameters()).device
        pooled_emb = self.pooled_proj(embedding)                       # (B, 1280)
        time_ids = torch.tensor(
            [[h, w, 0, 0, h, w]], device=device, dtype=torch.float32
        ).repeat(B, 1)
        return {"text_embeds": pooled_emb, "time_ids": time_ids}

    # ── Inference ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        embedding: torch.Tensor,
        vae=None,
        num_inference_steps: int = None,
        guidance_scale: float = 1.0,
        seed: int = None,
        return_latent: bool = False,
        use_sdxl_null: bool = False,
    ) -> torch.Tensor:
        """
        Full denoising loop conditioned on z via LatentToTokens + pooled_proj.

        Args:
            embedding      : (B, 512)  — z from Compressor (AttentionPooler output)
            vae            : SDVAE for latent → pixel decoding
            guidance_scale : CFG scale. 1.0 = no CFG
            seed           : RNG seed for reproducibility
            return_latent  : return raw fp16 latent instead of decoded image
            use_sdxl_null  : bypass LatentToTokens and use the cached SDXL empty-string
                             CLIP embeddings as conditioning (epoch 0 baseline)

        Returns:
            image  (B, 3, 512, 512) in [-1, 1], or latent (B, 4, 64, 64)
        """
        if num_inference_steps is None:
            num_inference_steps = NUM_INFERENCE_STEPS

        embedding = embedding.float()

        B = embedding.shape[0]
        device = embedding.device

        if seed is not None:
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(seed)

        if use_sdxl_null:
            # Epoch-0 baseline: use SDXL's native CLIP("") embeddings.
            # CFG is meaningless here (cond == uncond), so force guidance_scale=1.0.
            # n_tokens=77 now matches sdxl_null_hidden (77 tokens) — no shape mismatch.
            cond_tokens = self.sdxl_null_hidden.to(device=device, dtype=torch.float32).expand(B, -1, -1)
            null_pooled = self.sdxl_null_pooled.to(device=device, dtype=torch.float32).expand(B, -1)
            h, w = 512, 512
            time_ids = torch.tensor([[h, w, 0, 0, h, w]], device=device, dtype=torch.float32).repeat(B, 1)
            added_kwargs = {"text_embeds": null_pooled, "time_ids": time_ids}
            guidance_scale = 1.0  # no CFG for null baseline
        else:
            # Channel 1: z → LatentToTokens → cross-attention tokens
            cond_tokens = self.get_conditioning(embedding)               # (B, 77, 2048)
            # Channel 2: z → pooled_proj → time-embedding
            added_kwargs = self.get_added_cond_kwargs(B, embedding, device=device)

        # Expand for CFG: stack unconditional before conditional
        if guidance_scale > 1.0:
            null_emb = self.null_token.float().unsqueeze(0).expand(B, -1)
            uncond_tokens = self.get_conditioning(null_emb)              # (B, 77, 2048)
            cond_tokens = torch.cat([uncond_tokens, cond_tokens])        # (2B, 77, 2048)

            null_added = self.get_added_cond_kwargs(B, null_emb, device=device)
            added_kwargs = {
                k: torch.cat([null_added[k], added_kwargs[k]]) for k in added_kwargs
            }

        # ── Denoising loop ────────────────────────────────────────────────────
        unet_dtype = next(self.unet.parameters()).dtype  # bf16

        self.inference_scheduler.set_timesteps(num_inference_steps)
        noisy_latent = torch.randn(B, 4, 64, 64, device=device, dtype=unet_dtype)
        noisy_latent = noisy_latent * self.inference_scheduler.init_noise_sigma

        for t in self.inference_scheduler.timesteps:
            latent_input = noisy_latent
            if guidance_scale > 1.0:
                latent_input = torch.cat([latent_input] * 2)
            latent_input = self.inference_scheduler.scale_model_input(latent_input, t)

            noise_pred = self.unet(
                latent_input,
                t,
                encoder_hidden_states=cond_tokens.to(unet_dtype),
                added_cond_kwargs={k: v.to(unet_dtype) for k, v in added_kwargs.items()},
            ).sample

            if guidance_scale > 1.0:
                noise_uncond, noise_cond = noise_pred.chunk(2)
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            noisy_latent = self.inference_scheduler.step(
                noise_pred, t, noisy_latent
            ).prev_sample.to(unet_dtype)

        if return_latent:
            return noisy_latent

        if vae is None:
            raise ValueError("vae must be provided when return_latent=False")

        # ── VAE decode ─────────────────────────────────────────────────────────
        decode_latent = (noisy_latent / vae.vae.config.scaling_factor).to(torch.float32)
        original_dtype = next(vae.vae.parameters()).dtype
        if original_dtype != torch.float32:
            vae.vae.to(torch.float32)

        image = vae.vae.decode(decode_latent).sample

        if original_dtype != torch.float32:
            vae.vae.to(original_dtype)

        return image.clamp(-1.0, 1.0)  # (B, 3, 512, 512)
    
# ── SDXL VAE Utility ─────────────────────────────────────────────────────────

class SDVAE:
    """SDXL VAE wrapper (4-channel latent space)."""

    def __init__(self, freeze: bool = True, model_id: str = SDXL_MODEL_NAME):
        print(f"Loading frozen SDXL VAE from {model_id} ...")
        self.vae = AutoencoderKL.from_pretrained(
            model_id, 
            subfolder="vae", 
            torch_dtype=torch.bfloat16, 
            variant="fp16"
        ).eval()
        
        if freeze:
            for p in self.vae.parameters():
                p.requires_grad = False

    def to(self, device):
        self.vae = self.vae.to(device)
        return self

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 3, 512, 512) → (B, 4, 64, 64)"""
        latents = self.vae.encode(images.to(self.vae.dtype)).latent_dist.sample()
        return latents * self.vae.config.scaling_factor