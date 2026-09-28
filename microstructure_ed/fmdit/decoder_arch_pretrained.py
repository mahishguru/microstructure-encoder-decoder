"""
decoder_vitfmdit.py  —  FM-DiT Decoder (SD3.5 Hijack + Flow Matching)
"""

import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from typing import Tuple
from diffusers import SD3Transformer2DModel, FlowMatchEulerDiscreteScheduler, AutoencoderKL
import sys, os


from microstructure_ed.config import (
    ASSETS_DIR,
    DEVICE,
    FMDIT_MODEL_ID,
    FMDIT_JOINT_ATTENTION_DIM,
    FMDIT_POOLED_DIM,
    FMDIT_VAE_CHANNELS,
    FMDIT_VAE_DOWNSAMPLE_FACTOR,
    FMDIT_ADAPTER_INNER_DIM,
    FMDIT_NUM_TOKENS,
    FMDIT_ADAPTER_DEPTH,
    FMDIT_ADAPTER_NUM_HEADS,
    FMDIT_FLOW_SHIFT,
    FM_NUM_TRAIN_TIMESTEPS,
    FMDIT_NUM_INFERENCE_STEPS,
    FMDIT_VAE_SCALING_FACTOR,  
)

# ============================================================================
# Bottleneck width d = 16 x d_z. The paper variants are FM-DiT-512/768/1024/1280.
# Default for this process: FMDIT_TARGET_DIM (512 if unset). Pass
# `target_dim=` to FlowMatchingDiTDecoder to override per instance; the
# encoder (Compressor) must be built with the same value.
# ============================================================================
TARGET_DIM = int(os.environ.get("FMDIT_TARGET_DIM", "512"))

# Spatial-token latent: the flat (B, 512) bottleneck is read as
# SPATIAL_NUM_TOKENS blocks of d_z (16 x 32 = 512), preserving spatial
# structure through the bottleneck with NO encoder->decoder skip.
SPATIAL_LATENT = True
SPATIAL_NUM_TOKENS = FMDIT_NUM_TOKENS  # 16

# LoRA rank on the frozen MMDiT (decoder-capacity lever). 0 disables.
try:
    from microstructure_ed.config import FMDIT_DECODER_LORA_RANK, FMDIT_DECODER_LORA_ALPHA
except Exception:
    FMDIT_DECODER_LORA_RANK, FMDIT_DECODER_LORA_ALPHA = 0, 0

# ── QK-Norm Attention ────────────────────────────────────────────────────────

def scaled_dot_product_attention_qk_norm(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """
    Standard scaled dot-product attention with QK-Norm stability trick.
    Normalising Q and K to unit vectors bounds attention logits to [-1, 1],
    preventing explosion when the early-training z_512 is still unstable.
    """
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    d_k = q.shape[-1]
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * (d_k ** -0.5)
    attn_weights = F.softmax(attn_weights, dim=-1)
    return torch.matmul(attn_weights, v)


# ── AdaLN-Zero Block ─────────────────────────────────────────────────────────

class AdaLNZeroBlock(nn.Module):
    """
    DiT-style Adaptive LayerNorm Zero block (zero-initialised → identity at epoch 0).
    z_512 generates 6 modulation parameters: γ₁, β₁, α₁, γ₂, β₂, α₂.
    """

    def __init__(self, dim: int, z_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(z_dim, dim * 6),
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
        """
        q : (B, N, dim)  — learned positional queries
        z : (B, z_dim)   — 512-D bottleneck
        """
        B, N, D = q.shape
        mod = self.adaLN_modulation(z).unsqueeze(1)      # (B, 1, 6*dim)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = mod.chunk(6, dim=-1)

        # Self-attention with QK-Norm
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

        # FFN
        ff_mod = self.ln2(q) * (1 + gamma2) + beta2
        q = q + alpha2 * self.ff(ff_mod)
        return q


# ── LatentToTokens Adapter ───────────────────────────────────────────────────

class TokenSelfAttnBlock(nn.Module):
    """Plain pre-norm self-attention + FFN block over the spatial tokens.

    Used by the spatial LatentToTokens path so the 16 conditioning tokens can
    exchange information (global coherence) before being lifted to the MMDiT
    joint-attention dim. No z-conditioning: the tokens already carry the latent.
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x


class LatentToTokens(nn.Module):
    """
    z_512 (B, 512)  →  fake text tokens (B, FMDIT_NUM_TOKENS, 4096)

    Uses a Mini-DiT of AdaLN-Zero blocks so the 16 output tokens specialise
    rather than collapsing to degenerate identical copies.
    """

    def __init__(self, target_dim: int = None):
        super().__init__()
        target_dim = TARGET_DIM if target_dim is None else int(target_dim)
        self.spatial = SPATIAL_LATENT
        if self.spatial:
            # Read the flat latent as SPATIAL_NUM_TOKENS x d_z. Each block is one
            # conditioning token: d_z -> inner, + learned pos, self-attend across
            # tokens, then lift to the MMDiT joint-attention dim. proj_out zero-
            # init keeps a neutral start (matches the legacy adapter).
            self.d_z = target_dim // SPATIAL_NUM_TOKENS
            self.token_in = nn.Linear(self.d_z, FMDIT_ADAPTER_INNER_DIM)
            self.pos = nn.Parameter(
                torch.randn(1, SPATIAL_NUM_TOKENS, FMDIT_ADAPTER_INNER_DIM) * 0.02
            )
            self.blocks = nn.ModuleList([
                TokenSelfAttnBlock(FMDIT_ADAPTER_INNER_DIM, FMDIT_ADAPTER_NUM_HEADS)
                for _ in range(FMDIT_ADAPTER_DEPTH)
            ])
            self.proj_out = nn.Linear(FMDIT_ADAPTER_INNER_DIM, FMDIT_JOINT_ATTENTION_DIM)
            self.norm_out = nn.LayerNorm(FMDIT_JOINT_ATTENTION_DIM)
            nn.init.zeros_(self.proj_out.weight)
            nn.init.zeros_(self.proj_out.bias)
            nn.init.zeros_(self.norm_out.bias)
        else:
            self.queries = nn.Parameter(
                torch.randn(1, FMDIT_NUM_TOKENS, FMDIT_ADAPTER_INNER_DIM) * 0.02
            )
            self.blocks = nn.ModuleList([
                AdaLNZeroBlock(FMDIT_ADAPTER_INNER_DIM, target_dim, FMDIT_ADAPTER_NUM_HEADS)
                for _ in range(FMDIT_ADAPTER_DEPTH)
            ])
            self.proj_out = nn.Linear(FMDIT_ADAPTER_INNER_DIM, FMDIT_JOINT_ATTENTION_DIM)
            self.norm_out = nn.LayerNorm(FMDIT_JOINT_ATTENTION_DIM)
            nn.init.zeros_(self.proj_out.weight)
            nn.init.zeros_(self.proj_out.bias)
            nn.init.zeros_(self.norm_out.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, 512)  →  (B, 16, 4096)"""
        if self.spatial:
            B = z.shape[0]
            t = z.view(B, SPATIAL_NUM_TOKENS, self.d_z)   # (B, 16, d_z)
            t = self.token_in(t) + self.pos               # (B, 16, inner)
            for block in self.blocks:
                t = block(t)
            return self.norm_out(self.proj_out(t))        # (B, 16, 4096)
        B = z.shape[0]
        q = self.queries.expand(B, -1, -1).clone()
        for block in self.blocks:
            q = block(q, z)
        return self.norm_out(self.proj_out(q))


# ── Flow Matching DiT Decoder ────────────────────────────────────────────────

class FlowMatchingDiTDecoder(nn.Module):
    """
    FM-DiT Decoder: z_512 → 512×512 image via frozen SD3.5 MMDiT backbone.

    Only the ~15 M-parameter LatentToTokens adapter and pooled_proj are trained.
    """

    def __init__(self, model_id: str = FMDIT_MODEL_ID, token: str = None,
                 target_dim: int = None):
        super().__init__()
        self.target_dim = TARGET_DIM if target_dim is None else int(target_dim)

        self._hf_token = token or os.environ.get("HF_TOKEN")

        self.token_generator = LatentToTokens(self.target_dim)

        # C3 (texture head): tiny aux MLP z -> per-image texture summaries
        # (log TI/maxMRD, basal+prismatic). Phase-A probe showed z is texture
        # blind (kNN R2 ~ 0 vs class one-hot R2 0.6-0.73); this head forces
        # the ENCODER to carry per-image texture information through z. It is
        # never used at inference. Env-gated so existing ckpts load untouched.
        if os.environ.get("FMDIT_TEXHEAD", "0") == "1":
            self.tex_head = nn.Sequential(
                nn.Linear(self.target_dim, 256), nn.GELU(),
                nn.Linear(256, 4),
            )

        self.pooled_proj = nn.Sequential(
            nn.Linear(self.target_dim, FMDIT_POOLED_DIM),
            nn.SiLU(),
            nn.Linear(FMDIT_POOLED_DIM, FMDIT_POOLED_DIM),
        )
        nn.init.zeros_(self.pooled_proj[2].weight)
        nn.init.zeros_(self.pooled_proj[2].bias)

        print("Loading frozen SD 3.5 MMDiT Backbone (2.5 B params) in bfloat16 \u2026")
        self.transformer = SD3Transformer2DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=torch.bfloat16,
            token=self._hf_token,
        )
        self.transformer.requires_grad_(False)
        self.transformer.eval()

        # LoRA on the frozen MMDiT joint-attention projections (capacity lever).
        # v3 (variant-local): env overrides so the 512 arm can raise capacity
        # without touching the SHARED config keys the other variants read.
        #   FMDIT_LORA_RANK_OVERRIDE / FMDIT_LORA_ALPHA_OVERRIDE: rank/alpha.
        #   FMDIT_LORA_FFN=1: also adapt the FFN projections - in DiTs they
        #   carry most of the per-token appearance capacity, which is where
        #   the OOD IPF colour statistics (sharp texture) must be expressed.
        import os as _os
        _lora_rank  = int(_os.environ.get("FMDIT_LORA_RANK_OVERRIDE",
                                          FMDIT_DECODER_LORA_RANK))
        _lora_alpha = int(_os.environ.get("FMDIT_LORA_ALPHA_OVERRIDE",
                                          FMDIT_DECODER_LORA_ALPHA))
        _lora_ffn   = _os.environ.get("FMDIT_LORA_FFN", "0") == "1"
        self.lora_enabled = _lora_rank > 0
        if self.lora_enabled:
            from peft import LoraConfig
            _targets = [
                "to_q", "to_k", "to_v", "to_out.0",
                "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
            ]
            if _lora_ffn:
                _targets += ["ff.net.0.proj", "ff.net.2",
                             "ff_context.net.0.proj", "ff_context.net.2"]
            lora_cfg = LoraConfig(
                r=_lora_rank,
                lora_alpha=_lora_alpha,
                init_lora_weights="gaussian",
                target_modules=_targets,
            )
            self.transformer.add_adapter(lora_cfg)
            _n_lora = sum(p.numel() for p in self.transformer.parameters()
                          if p.requires_grad)
            print(f"[FMDiT] LoRA on MMDiT: rank={_lora_rank} "
                  f"alpha={_lora_alpha} ffn={_lora_ffn} trainable={_n_lora:,}")

        self.scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=FM_NUM_TRAIN_TIMESTEPS,
            shift=FMDIT_FLOW_SHIFT,          
        )

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        x_t:   torch.Tensor,   # (B, 16, 64, 64) noised latent
        t:     torch.Tensor,   # (B,) timestep indices
        z_512: torch.Tensor,   # (B, 512)
        tex_z: torch.Tensor = None,  # C3: run tex_head inside the DDP forward
    ) -> torch.Tensor:
        """Returns predicted velocity v_pred (B, 16, 64, 64)."""
        z_bf16 = z_512.to(self.transformer.dtype)
        prompt_embeds = self.token_generator(z_bf16)    # (B, 16, 4096)
        pooled_embeds = self.pooled_proj(z_bf16)        # (B, 2048)

        model_pred = self.transformer(
            hidden_states=x_t,
            timestep=t,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
        if tex_z is not None:
            # C3: aux texture prediction from the TRUE z. Must run inside
            # forward: DDP reducer hooks fire per-forward; calling the head
            # via model.module outside forward marks its params ready twice.
            return model_pred, self.tex_head(tex_z)
        return model_pred

    # ── inference ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        z_512:     torch.Tensor,
        num_steps: int = FMDIT_NUM_INFERENCE_STEPS,
        noise_temp: float = 1.0,
        shift: float = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """Euler ODE integration: pure noise → denoised latent.

        B4 sampling knobs (defaults = legacy behaviour):
          noise_temp — initial-noise temperature tau; tau < 1 directly
                       counteracts conditional-mean variance shrinkage.
          shift      — override the scheduler flow shift at inference
                       (train value FMDIT_FLOW_SHIFT; lower biases detail).
        """
        B      = z_512.shape[0]
        device = z_512.device
        dtype  = self.transformer.dtype

        latent = torch.randn(B, FMDIT_VAE_CHANNELS, 64, 64, device=device, dtype=dtype) * float(noise_temp)
        if shift is not None and abs(float(shift) - float(self.scheduler.config.shift)) > 1e-9:
            self.scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=self.scheduler.config.num_train_timesteps,
                shift=float(shift),
            )
        self.scheduler.set_timesteps(num_steps, device=device)

        use_cfg = abs(float(guidance_scale) - 1.0) > 1e-9
        z_null = torch.zeros_like(z_512)
        for t_val in self.scheduler.timesteps:
            #latent_input = self.scheduler.scale_model_input(latent, t_val)
            if use_cfg:
                # C1: classifier-free guidance. Requires a ckpt fine-tuned
                # with cond_dropout > 0 (null = zero embedding).
                v_both = self.forward(
                    torch.cat([latent, latent], dim=0),
                    t_val.unsqueeze(0).expand(2 * B),
                    torch.cat([z_512, z_null], dim=0),
                )
                v_cond, v_null = v_both.chunk(2, dim=0)
                v_pred = v_null + guidance_scale * (v_cond - v_null)
            else:
                v_pred = self.forward(
                    latent,
                    t_val.unsqueeze(0).expand(B),
                    z_512,
                )
            latent = self.scheduler.step(v_pred, t_val, latent).prev_sample

        return latent

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Flow Matching Loss ────────────────────────────────────────────────────────

def _sample_logit_normal_timesteps(
    B: int,
    device: torch.device,
    shift: float = FMDIT_FLOW_SHIFT,
) -> torch.Tensor:
    """
    Sample t ∈ [0, 1] from the logit-normal distribution used by SD3/SD3.5.
    shift > 1 concentrates mass around intermediate t, where structure forms.
    """
    u = torch.randn(B, device=device)
    # logit-normal: sigmoid(N(0,1) + log(shift))
    t = torch.sigmoid(u + math.log(shift))
    return t.clamp(1e-5, 1 - 1e-5)


def flow_matching_loss(
    model: FlowMatchingDiTDecoder,
    x_0:   torch.Tensor,   # (B, 16, 64, 64) target latent from SD3.5 VAE
    z_512: torch.Tensor,   # (B, 512)
) -> Tuple[torch.Tensor, dict]:
    """
    Rectified Flow (RF) loss.

    Math:
        path  :  x_t = (1-t) * x_0 + t * x_1     where x_1 ~ N(0,I)
        target:  v   = x_1 - x_0
        loss  :  || v_pred(x_t, t) - v_target ||²

    We implement the RF math directly — no scheduler needed for the loss.
    """
    B      = x_0.shape[0]
    device = x_0.device

    # Sample destination noise
    noise  = torch.randn_like(x_0)                                 # x_1

    t_cont = _sample_logit_normal_timesteps(B, device)             # (B,) ∈ [0,1]
    t_idx  = (t_cont * FM_NUM_TRAIN_TIMESTEPS).long()              # integer timesteps for model

    # Interpolate: x_t = (1-t) * x_0 + t * noise
    t_view     = t_cont.view(B, 1, 1, 1)
    noisy_latents = (1.0 - t_view) * x_0 + t_view * noise

    # True velocity
    v_target = noise - x_0                                         # (B, 16, 64, 64)

    # Predicted velocity
    v_pred = model(noisy_latents, t_idx, z_512)

    loss = F.mse_loss(v_pred.float(), v_target.float())

    return loss, {
        "loss":   loss.item(),
        "mean_t": t_cont.mean().item(),
        "std_t":  t_cont.std().item(),
    }


# ── Composite FM-DiT Loss ─────────────────────────────────────────────────────

def composite_fm_loss(
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
    orient_fn=None,
    lambda_orient=0.0,
    lambda_latstat=0.0,
    orient_weight_power=2.0,
    orient_t_threshold=None,   # v3: tighter t-gate for orientation
                               # (None -> share freq_t_threshold)
    cond_dropout=0.0,          # C1: per-sample conditioning dropout for CFG
    lambda_texhead=0.0,        # C3: aux texture-head loss weight
):
    B      = x_0.shape[0]
    device = x_0.device

    noise         = torch.randn_like(x_0)
    t_cont        = _sample_logit_normal_timesteps(B, device)
    t_idx         = (t_cont * FM_NUM_TRAIN_TIMESTEPS).long()
    t_view        = t_cont.view(B, 1, 1, 1)

    noisy_latents = (1.0 - t_view) * x_0 + t_view * noise
    v_target      = noise - x_0
    # C1: drop conditioning to the null (zero) embedding for a random subset
    # so the model learns p(x0) alongside p(x0|z) -> enables CFG at inference.
    # Applies ONLY to the diffusion conditioning; contrastive/vicreg terms
    # keep the true z.
    z_diff = z_512
    if cond_dropout > 0.0:
        keep = (torch.rand(B, device=device) >= cond_dropout).float().view(B, 1)
        z_diff = z_512 * keep
    use_texhead = (lambda_texhead > 0.0 and
                   getattr(getattr(model, "module", model), "tex_head", None) is not None)
    if use_texhead:
        # tex head reads the TRUE z (not the dropout-masked z_diff)
        v_pred, tex_pred_all = model(noisy_latents, t_idx, z_diff, tex_z=z_512)
    else:
        v_pred = model(noisy_latents, t_idx, z_diff)

    loss_v = F.mse_loss(v_pred.float(), v_target.float())

    x0_pred      = noisy_latents.float() - t_view * v_pred.float()
    recon_weight = (1.0 - t_view) ** 2
    loss_recon   = (recon_weight * (x0_pred - x_0.float()) ** 2).mean()

    # ── B3: always-on latent-statistics loss (anti-shrinkage, no VAE decode) ──
    # Match per-channel spatial mean and the 16x16 channel covariance of
    # x0_pred against x_0 for ALL samples at ALL t, weighted by (1-t). Texture
    # strength lives in these first/second moments; conditional-mean shrinkage
    # shows up as deflated covariance -> this term pushes it back up.
    loss_latstat = torch.tensor(0.0, device=device)
    if lambda_latstat > 0:
        w_ls   = (1.0 - t_cont.float()).view(B, 1)                    # (B,1)
        mu_p   = x0_pred.mean(dim=(2, 3))                             # (B,16)
        mu_t   = x_0.float().mean(dim=(2, 3))
        loss_mu = (w_ls * (mu_p - mu_t) ** 2).mean()
        fp     = x0_pred.flatten(2)                                   # (B,16,N)
        ft     = x_0.float().flatten(2)
        N_pix  = fp.shape[-1]
        cp     = torch.bmm(fp - mu_p.unsqueeze(-1), (fp - mu_p.unsqueeze(-1)).transpose(1, 2)) / N_pix
        ct     = torch.bmm(ft - mu_t.unsqueeze(-1), (ft - mu_t.unsqueeze(-1)).transpose(1, 2)) / N_pix
        loss_cov = (w_ls.view(B, 1, 1) * (cp - ct) ** 2).mean()
        loss_latstat = loss_mu + loss_cov

    # Contrastive on encoder embeddings (skipped entirely when disabled)
    if lambda_contrastive > 0:
        embeddings_both = torch.cat([z_512, z_512_aug], dim=0)
        loss_contr      = contrastive_fn(embeddings_both.float())
    else:
        loss_contr      = torch.tensor(0.0, device=device)

    # ── VICReg on z_512 (all-gather across GPUs for better cov estimate) ─────
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

    # ── Pixel-space losses (frequency + orientation), gated on small t ────────
    # Both need the VAE-decoded x̂₀; only meaningful when t is small (x̂₀ close to
    # x₀). Decode at most `freq_max_samples` pairs ONCE through grad checkpoint
    # (bounds VRAM on the SD3.5 16-channel VAE) and SHARE them between the two
    # losses, so adding orientation supervision costs no extra VAE decode.
    loss_freq   = torch.tensor(0.0, device=device)
    loss_orient = torch.tensor(0.0, device=device)
    orient_info = {}
    loss_texhead = torch.tensor(0.0, device=device)
    need_freq   = (freq_fn   is not None and lambda_freq   > 0)
    need_orient = (orient_fn is not None and lambda_orient > 0)
    if vae is not None and (need_freq or need_orient):
        mask = t_cont < freq_t_threshold
        if mask.any():
            idx_sel = mask.nonzero(as_tuple=False).squeeze(1)[:freq_max_samples]
            vae_dtype = next(vae.vae.parameters()).dtype
            with torch.amp.autocast('cuda', enabled=False):
                def _vae_decode(z):
                    return vae.vae.decode(z).sample

                z_in = vae.denorm_to_raw(x0_pred[idx_sel]).to(vae_dtype)
                img_pred = grad_checkpoint(
                    _vae_decode, z_in, use_reentrant=False
                ).clamp(-1, 1)

                with torch.no_grad():
                    z_tgt = vae.denorm_to_raw(x_0[idx_sel].float()).to(vae_dtype)
                    img_tgt = vae.vae.decode(z_tgt).sample.clamp(-1, 1)

            if need_freq:
                loss_freq = freq_fn(img_pred, img_tgt)
            if need_orient:
                # v3: mid-t x0_pred is E[x0|xt] - inherently blurred toward the
                # class mean. Demanding sharp texture there teaches mean-collapse
                # (the v2 regression). Gate orientation to near-clean predictions
                # via a SEPARATE, tighter threshold, and weight by (1-t)^p.
                o_thr = freq_t_threshold if orient_t_threshold is None else orient_t_threshold
                o_mask = t_cont[idx_sel] < o_thr
                if o_mask.any():
                    o_idx = o_mask.nonzero(as_tuple=False).squeeze(1)
                    orient_w = (1.0 - t_cont[idx_sel][o_idx].float()) ** orient_weight_power
                    loss_orient, orient_info = orient_fn(
                        img_pred[o_idx], img_tgt[o_idx], sample_weight=orient_w)

            # C3: texture-head aux loss. Targets = log TI/maxMRD (basal +
            # prismatic) of the GROUND-TRUTH image (img_tgt is the exact VAE
            # decode of x_0, already computed -> zero extra VAE cost). The
            # head reads z; its gradient flows into the ENCODER, forcing z to
            # carry per-image texture info (Phase-A: z was texture-blind).
            if use_texhead and orient_fn is not None:
                from microstructure_ed.orientation_loss import rgb_to_quat_wxyz as _rgb2q
                with torch.no_grad():
                    q_t = _rgb2q(img_tgt.float())                     # (b,4,H,W)
                    qf = q_t.flatten(2).transpose(1, 2)               # (b,HW,4)
                    n_sub = min(2048, qf.shape[1])
                    ridx = torch.randint(qf.shape[1], (n_sub,), device=qf.device)
                    h_t = orient_fn._pf_hist(qf[:, ridx])             # (b, 2G)
                    G = orient_fn.pf_nodes.shape[0]
                    pb, pp = h_t[:, :G], h_t[:, G:]
                    tex_tgt = torch.stack([
                        (G * (pb ** 2).sum(-1)).clamp_min(1e-4).log(),
                        (G * pb.amax(-1)).clamp_min(1e-4).log(),
                        (G * (pp ** 2).sum(-1)).clamp_min(1e-4).log(),
                        (G * pp.amax(-1)).clamp_min(1e-4).log(),
                    ], dim=-1)                                        # (b,4)
                loss_texhead = F.smooth_l1_loss(
                    tex_pred_all[idx_sel].float(), tex_tgt.float())

    total = (
        lambda_diff        * loss_v       +
        lambda_recon       * loss_recon   +
        lambda_contrastive * loss_contr   +
        lambda_vicreg      * loss_vicreg  +
        lambda_freq        * loss_freq    +
        lambda_orient      * loss_orient  +
        lambda_latstat     * loss_latstat +
        lambda_texhead     * loss_texhead
    )

    info = {
        "loss_velocity":    loss_v.item(),
        "loss_recon":       loss_recon.item(),
        "loss_contrastive": loss_contr.item(),
        "loss_vicreg":      loss_vicreg.item(),
        "loss_freq":        loss_freq.item(),
        "loss_orient":      loss_orient.item(),
        "loss_latstat":     float(loss_latstat.detach()),
        "loss_texhead":     float(loss_texhead.detach()),
        "loss_total":       total.item(),
        "mean_t":           t_cont.mean().item(),
        "std_t":            t_cont.std().item(),
    }
    # V3: orientation sub-terms (odf / misori / pf / scatter) when available.
    for k, v in orient_info.items():
        if k != "loss_orient":
            info[k] = v
    return total, info


# ── SD3.5 VAE Utility ────────────────────────────────────────────────────────

class SD35VAE:
    """16-channel SD3.5 VAE wrapper (fully frozen)."""

    def __init__(self, model_id: str = FMDIT_MODEL_ID,  token: str = None):
        print("Loading frozen SD 3.5 VAE (16-channel) …")
        self.vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.bfloat16, token=token
        ).eval()
        for p in self.vae.parameters():
            p.requires_grad = False
        self.scaling_factor = FMDIT_VAE_SCALING_FACTOR
        # Per-channel latent whitening. SD3.5 latents of IPF maps are far from
        # N(0,1) after *scaling_factor (measured per-ch mean range ~[-1.4,0.9],
        # std ~[0.42,1.15]). Flow matching samples from N(0,1) at t=1; those
        # per-channel DC offsets otherwise reappear as a directional RGB cast
        # and the sub-unit-std channels get an imbalanced noise path.
        _stats_path = os.path.join(ASSETS_DIR, 'latent_stats_fmdit768.json')
        with open(_stats_path) as _f:
            _st = json.load(_f)
        assert abs(_st['scaling_factor'] - self.scaling_factor) < 1e-6, \
            'latent_stats_fmdit768.json scaling_factor mismatch'
        self.latent_mean = torch.tensor(_st['mean'], dtype=torch.float32).view(1, 16, 1, 1)
        self.latent_std  = torch.tensor(_st['std'],  dtype=torch.float32).view(1, 16, 1, 1)
        # Latent whitening DISABLED by default (A/B test 2026-06-16: whitening ON
        # produces a uniform low-saturation pink veil; OFF recovers more vivid grain
        # colour and wins recon PSNR 8/10 + vivid-pixel recovery 82% vs 78%).
        # Set FMDIT_ENABLE_WHITENING=1 to restore the old whitened behaviour.
        if os.environ.get('FMDIT_ENABLE_WHITENING') == '1':
            print('[SD35VAE] latent whitening ON (FMDIT_ENABLE_WHITENING=1)')
        else:
            self.latent_mean = torch.zeros_like(self.latent_mean)
            self.latent_std  = torch.ones_like(self.latent_std)
            print('[SD35VAE] latent whitening OFF (default; set FMDIT_ENABLE_WHITENING=1 to enable)')
    def to(self, device):
        self.vae = self.vae.to(device)
        self.latent_mean = self.latent_mean.to(device)
        self.latent_std  = self.latent_std.to(device)
        return self
    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 3, 512, 512) → (B, 16, 64, 64), whitened to ~N(0,1)"""
        latents = self.vae.encode(images.to(self.vae.dtype)).latent_dist.sample().float() * self.scaling_factor
        return (latents - self.latent_mean) / self.latent_std

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """(B, 16, 64, 64) whitened → (B, 3, 512, 512) in [-1, 1]"""
        self.vae.to(torch.float32)
        latents = self.denorm_to_raw(latents).to(torch.float32)
        images  = self.vae.decode(latents).sample
        self.vae.to(torch.bfloat16)
        return images

    def denorm_to_raw(self, latents: torch.Tensor) -> torch.Tensor:
        """Whitened latents -> raw VAE latents for vae.vae.decode()."""
        return (latents * self.latent_std + self.latent_mean) / self.scaling_factor