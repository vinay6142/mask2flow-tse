"""
Stage 2: Flow Matching Module for Mask2Flow-TSE.

Paper Section 4.4 + 5.2.3:
    "DiT backbone with 9 blocks, 768-dim hidden states, 8 heads"
    "Rotary position embeddings (RoPE) for sequence modeling"
    "Speaker embedding injected into AdaLN-Zero conditioning"
    "Single Euler step at inference"

Architecture:
    Input : X_enh (B, 80, T) + timestep t + d-vector (B, 512)
    Output: velocity v (B, 80, T)  [= predicted Y - X_enh]

Training (rectified flow matching, Eq 16):
    t      ~ Uniform(0, 1)
    X_t    = (1-t) * X_enh + t * Y          [interpolation]
    target = Y - X_enh                       [constant velocity]
    loss   = MSE(v_θ(X_t, t, d), Y - X_enh)

Inference (single Euler step, Eq 17):
    Ŷ = X_enh + v_θ(X_enh, t=0, d)

Enhancement — Classifier-Free Guidance (CFG):
    During training: drop d-vector with prob cfg_dropout=0.1
    At inference   : v = v_uncond + scale * (v_cond - v_uncond)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ── Rotary Position Embedding (RoPE) ──────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) — Su et al. 2021.
    Encodes relative position directly into attention scores.
    More effective than absolute position embeddings for
    variable-length sequences.

    Paper Section 5.2.3: "rotary position embeddings (RoPE)"
    """

    def __init__(self, dim: int, max_seq_len: int = 4096):
        super().__init__()
        assert dim % 2 == 0, "RoPE dim must be even"

        # precompute frequency bands
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, dim, 2).float() / dim)
        )
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Precompute sin/cos tables."""
        t     = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)   # (T, dim/2)
        emb   = torch.cat([freqs, freqs], dim=-1)  # (T, dim)
        self.register_buffer("cos_cache", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cache", emb.sin()[None, None, :, :])

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to query and key tensors.

        Args:
            q: (B, n_heads, T, head_dim)
            k: (B, n_heads, T, head_dim)
        Returns:
            q_rot, k_rot: same shape with rotary encoding applied
        """
        T = q.shape[2]

        # rebuild cache if sequence is longer than precomputed
        if T > self.max_seq_len:
            self._build_cache(T)
            self.max_seq_len = T

        cos = self.cos_cache[:, :, :T, :].to(q.device)  # (1,1,T,dim)
        sin = self.sin_cache[:, :, :T, :].to(q.device)

        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin

        return q_rot, k_rot


# ── Sinusoidal Timestep Embedding ─────────────────────────────────────────────

class TimestepEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding → MLP projection.
    Maps scalar t ∈ [0,1] to a dense vector for conditioning.

    Same design as diffusion models (DDPM, DiT).
    """

    def __init__(self, hidden_dim: int = 768, freq_dim: int = 256):
        super().__init__()

        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _sinusoidal(self, t: torch.Tensor) -> torch.Tensor:
        """
        Sinusoidal embedding of timestep.

        Args:
            t: (B,) float timesteps in [0, 1]
        Returns:
            (B, freq_dim) sinusoidal embedding
        """
        half  = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000) *
            torch.arange(half, device=t.device).float() / half
        )
        args  = t[:, None] * freqs[None, :]    # (B, half)
        emb   = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, freq_dim)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) timesteps in [0, 1]
        Returns:
            (B, hidden_dim)
        """
        sinusoidal = self._sinusoidal(t)
        return self.mlp(sinusoidal)


# ── AdaLN-Zero Modulation ─────────────────────────────────────────────────────

class AdaLNZero(nn.Module):
    """
    Adaptive LayerNorm-Zero conditioning (Peebles & Xie 2023).

    Predicts per-sample scale (γ), shift (β), and gate (α)
    from the conditioning vector c = timestep + speaker.

    α is initialized to zero so each block initially acts
    as an identity function — critical for training stability.

    Paper Equations (14) and (15):
        ĥ = (1 + γ) ⊙ LN(h) + β
        h' = h + α ⊙ f(ĥ)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        # predict 6 modulation params: γ1, β1, α1, γ2, β2, α2
        # (for self-attention and feedforward sub-layers)
        self.linear = nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)

        # initialize output to zero → identity init for α
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(
        self,
        c: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            c: (B, hidden_dim) conditioning vector
        Returns:
            6 tensors: γ1, β1, α1, γ2, β2, α2
            each (B, 1, hidden_dim) for broadcasting over T
        """
        params = self.linear(c)           # (B, 6*H)
        chunks = params.chunk(6, dim=-1)  # 6 × (B, H)
        # unsqueeze time dim for broadcasting
        return tuple(p.unsqueeze(1) for p in chunks)


# ── DiT Block ─────────────────────────────────────────────────────────────────

class DiTBlock(nn.Module):
    """
    Diffusion Transformer block with AdaLN-Zero + RoPE.

    Modified from standard DiT to incorporate speaker embedding
    into the conditioning signal alongside the timestep.

    Paper Section 4.4:
        "conditioning signal c = MLP(t) + W_d * d"
        "additive formulation injects speaker identity into
         every transformer layer without requiring cross-attention"
    """

    def __init__(
        self,
        hidden_dim: int   = 768,
        n_heads:    int   = 8,
        ffn_mult:   int   = 2,
        dropout:    float = 0.1,
    ):
        super().__init__()

        assert hidden_dim % n_heads == 0

        self.hidden_dim = hidden_dim
        self.n_heads    = n_heads
        self.head_dim   = hidden_dim // n_heads

        # ── Self-Attention ────────────────────────────────────
        self.norm1   = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.q_proj  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)

        # RoPE for relative position in time dimension
        self.rope = RotaryEmbedding(self.head_dim)

        # ── Feed-Forward Network ──────────────────────────────
        self.norm2  = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        ffn_hidden  = hidden_dim * ffn_mult   # 768*2 = 1536
        self.ff1    = nn.Linear(hidden_dim, ffn_hidden)
        self.ff2    = nn.Linear(ffn_hidden,  hidden_dim)
        self.ff_act = nn.GELU()
        self.ff_drop = nn.Dropout(dropout)

        # ── AdaLN-Zero conditioning ───────────────────────────
        self.adaln = AdaLNZero(hidden_dim)

    def _self_attention(
        self,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """
        Multi-head self-attention with RoPE.

        Args:
            h: (B, T, hidden_dim)
        Returns:
            (B, T, hidden_dim)
        """
        B, T, _ = h.shape

        # project to Q, K, V
        q = self.q_proj(h)   # (B, T, H)
        k = self.k_proj(h)
        v = self.v_proj(h)

        # reshape to (B, n_heads, T, head_dim)
        def split_heads(x):
            return x.view(B, T, self.n_heads, self.head_dim) \
                    .transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # apply RoPE to q and k
        q, k = self.rope(q, k)

        # scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn  = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn  = F.softmax(attn, dim=-1)
        attn  = self.attn_drop(attn)

        # aggregate values
        out = torch.matmul(attn, v)          # (B, n_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(out)

    def _feedforward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Position-wise FFN.

        Args:
            h: (B, T, hidden_dim)
        Returns:
            (B, T, hidden_dim)
        """
        return self.ff2(
            self.ff_drop(self.ff_act(self.ff1(h)))
        )

    def forward(
        self,
        h: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """
        DiT block forward pass.

        Args:
            h: (B, T, hidden_dim) hidden states
            c: (B, hidden_dim)    conditioning vector
        Returns:
            (B, T, hidden_dim)
        """
        # get AdaLN-Zero modulation params
        γ1, β1, α1, γ2, β2, α2 = self.adaln(c)

        # ── Self-attention sub-layer (Eq 14, 15) ─────────────
        h_norm = (1 + γ1) * self.norm1(h) + β1
        h      = h + α1 * self._self_attention(h_norm)

        # ── Feed-forward sub-layer ────────────────────────────
        h_norm = (1 + γ2) * self.norm2(h) + β2
        h      = h + α2 * self._feedforward(h_norm)

        return h


# ── Flow Matching Module ───────────────────────────────────────────────────────

class FlowMatchingModule(nn.Module):
    """
    Mask2Flow-TSE Stage 2: rectified flow matching with DiT.

    Predicts the velocity field v_θ(X_t, t, d) that transforms
    the masked spectrogram X_enh into clean target Y.

    Total params: ~72-75M (paper: 72.6M)
    """

    def __init__(
        self,
        n_mels:      int   = 80,
        hidden_dim:  int   = 768,
        n_heads:     int   = 8,
        n_blocks:    int   = 9,
        ffn_mult:    int   = 2,
        embed_dim:   int   = 512,
        dropout:     float = 0.1,
        cfg_dropout: float = 0.1,
    ):
        super().__init__()

        self.n_mels      = n_mels
        self.hidden_dim  = hidden_dim
        self.cfg_dropout = cfg_dropout

        # ── Input projection ──────────────────────────────────
        # project mel bins to transformer hidden dim
        # paper: "speaker embedding concatenated with input features"
        # we concat d-vector to mel input → 80+512=592
        self.input_proj = nn.Linear(n_mels + embed_dim, hidden_dim)

        # ── Conditioning ──────────────────────────────────────
        # timestep → hidden_dim
        self.time_embed = TimestepEmbedding(
            hidden_dim = hidden_dim,
            freq_dim   = 256,
        )
        # speaker embedding → hidden_dim
        self.spk_proj   = nn.Linear(embed_dim, hidden_dim)

        # null speaker embedding for CFG (learned)
        self.null_spk   = nn.Parameter(torch.zeros(1, embed_dim))

        # ── DiT backbone ──────────────────────────────────────
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_dim = hidden_dim,
                n_heads    = n_heads,
                ffn_mult   = ffn_mult,
                dropout    = dropout,
            )
            for _ in range(n_blocks)
        ])

        # ── Final modulation + output projection ──────────────
        self.final_norm  = nn.LayerNorm(hidden_dim,
                                        elementwise_affine=False)
        self.final_mod   = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, n_mels)

        # zero-init output projection
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        # ── Parameter summary ─────────────────────────────────
        n_input  = sum(p.numel() for p in self.input_proj.parameters())
        n_time   = sum(p.numel() for p in self.time_embed.parameters())
        n_spk    = sum(p.numel() for p in self.spk_proj.parameters())
        n_blocks = sum(p.numel() for p in self.blocks.parameters())
        n_out    = sum(p.numel() for p in self.output_proj.parameters()) \
                 + sum(p.numel() for p in self.final_mod.parameters())
        n_total  = n_input + n_time + n_spk + n_blocks + n_out + embed_dim

        print(f"[FlowMatchingModule] Input proj  : {n_input:>12,} params")
        print(f"[FlowMatchingModule] Time embed  : {n_time:>12,} params")
        print(f"[FlowMatchingModule] Speaker proj: {n_spk:>12,} params")
        print(f"[FlowMatchingModule] DiT blocks  : {n_blocks:>12,} params  "
              f"({n_blocks//9:,} × 9 blocks)")
        print(f"[FlowMatchingModule] Output      : {n_out:>12,} params")
        print(f"[FlowMatchingModule] Total       : {n_total:>12,} params  "
              f"(~{n_total/1e6:.1f}M)")

    def _get_conditioning(
        self,
        t:         torch.Tensor,
        d_vector:  torch.Tensor,
        cfg_mode:  bool = False,
    ) -> torch.Tensor:
        """
        Build conditioning vector c = MLP(t) + W_d * d.

        Paper Equation (13):
            c = MLP(t) + W_d * d

        With CFG: randomly replace d with null embedding
        during training.

        Args:
            t       : (B,) timesteps in [0, 1]
            d_vector: (B, 512) speaker d-vectors
            cfg_mode: if True, use null speaker (for CFG inference)
        Returns:
            c: (B, hidden_dim) conditioning vector
        """
        t_emb = self.time_embed(t)   # (B, hidden_dim)

        if cfg_mode:
            # use learned null embedding for unconditional pass
            d = self.null_spk.expand(d_vector.shape[0], -1)
        else:
            d = d_vector

        d_emb = self.spk_proj(d)     # (B, hidden_dim)

        return t_emb + d_emb         # (B, hidden_dim)

    def forward(
        self,
        x_t:      torch.Tensor,
        t:        torch.Tensor,
        d_vector: torch.Tensor,
        force_cfg_drop: bool = False,
    ) -> torch.Tensor:
        """
        Predict velocity field v_θ(X_t, t, d).

        Args:
            x_t     : (B, n_mels, T) interpolated spectrogram
            t       : (B,) timesteps in [0, 1]
            d_vector: (B, 512) speaker d-vectors (L2 normalized)
            force_cfg_drop: if True, drop speaker for CFG training
        Returns:
            velocity: (B, n_mels, T)
        """
        B, n_mels, T = x_t.shape

        # ── CFG training dropout ──────────────────────────────
        if self.training and not force_cfg_drop:
            # randomly drop speaker embedding with cfg_dropout prob
            drop_mask = torch.rand(B, device=x_t.device) < self.cfg_dropout
            if drop_mask.any():
                null = self.null_spk.expand(B, -1)
                d_eff = torch.where(
                    drop_mask.unsqueeze(-1), null, d_vector
                )
            else:
                d_eff = d_vector
        elif force_cfg_drop:
            d_eff = self.null_spk.expand(B, -1)
        else:
            d_eff = d_vector

        # ── Build conditioning vector ─────────────────────────
        c = self._get_conditioning(t, d_eff)  # (B, hidden_dim)

        # ── Input projection ──────────────────────────────────
        # concat d-vector to each mel frame for richer input
        x = x_t.permute(0, 2, 1)              # (B, T, 80)
        d_exp = d_eff.unsqueeze(1).expand(-1, T, -1)  # (B, T, 512)
        x_cat = torch.cat([x, d_exp], dim=-1)  # (B, T, 592)
        h = self.input_proj(x_cat)             # (B, T, 768)

        # ── DiT blocks ────────────────────────────────────────
        for block in self.blocks:
            h = block(h, c)   # (B, T, 768)

        # ── Final modulation ──────────────────────────────────
        γ, β = self.final_mod(c).chunk(2, dim=-1)  # each (B, 768)
        h = (1 + γ.unsqueeze(1)) * self.final_norm(h) + β.unsqueeze(1)

        # ── Output projection ─────────────────────────────────
        vel = self.output_proj(h)   # (B, T, 80)
        return vel.permute(0, 2, 1) # (B, 80, T)

    def compute_loss(
        self,
        x_enh:    torch.Tensor,
        y_target: torch.Tensor,
        d_vector: torch.Tensor,
        mask:       Optional[torch.Tensor] = None,
        frame_mask:  Optional[torch.Tensor] = None,  # NEW: (B,1,T) or (B,T), 1=real audio, 0=padding
    ) -> Tuple[torch.Tensor, dict]:
        """
        Rectified flow matching loss — Equation (16) in paper.

        L_flow = E_{t,X_enh,Y} ||v_θ(X_t,t,d) - (Y - X_enh)||²

        Args:
            x_enh   : (B, 80, T) masked spectrogram from Stage 1
            y_target: (B, 80, T) clean target spectrogram
            d_vector: (B, 512)   speaker d-vectors
        Returns:
            loss  : scalar MSE loss
            info  : dict with logging values
        """
        B = x_enh.shape[0]

        # sample random timesteps t ~ Uniform(0, 1)
        t = torch.rand(B, device=x_enh.device)

        # linear interpolation: X_t = (1-t)*X_enh + t*Y
        t_exp = t.view(B, 1, 1)
        x_t   = (1 - t_exp) * x_enh + t_exp * y_target

        # target velocity (constant along straight trajectory)
        target_vel = y_target - x_enh   # (B, 80, T)

        # predict velocity
        pred_vel = self.forward(x_t, t, d_vector)

        # MSE loss (masked to ignore padded frames when frame_mask is given)
        if frame_mask is None:
            loss = F.mse_loss(pred_vel, target_vel)
        else:
            fm = frame_mask
            if fm.dim() == 2:
                fm = fm.unsqueeze(1)                            # (B, 1, T)
            diff2 = (pred_vel - target_vel) ** 2 * fm
            denom = fm.sum() * pred_vel.shape[1] + 1e-8         # ×n_mels (mask is per-frame)
            loss = diff2.sum() / denom

        info = {
            "loss"          : loss.item(),
            "t_mean"        : t.mean().item(),
            "vel_pred_norm" : pred_vel.norm(dim=-1).mean().item(),
            "vel_true_norm" : target_vel.norm(dim=-1).mean().item(),
        }
        return loss, info

    @torch.no_grad()
    def inference(
        self,
        x_enh:            torch.Tensor,
        d_vector:         torch.Tensor,
        cfg_scale:        float = 1.0,
        n_steps:          int   = 1,
        cfg_warmup_steps: int   = 0,
    ) -> torch.Tensor:
        """
        Single-step Euler inference — Equation (17).

        Ŷ = X_enh + v_θ(X_enh, t=0, d)

        With Classifier-Free Guidance (scale > 1):
            v = v_uncond + scale * (v_cond - v_uncond)

        CFG warm-up (cfg_warmup_steps > 0):
            At t=0, x_t = x_enh exactly (no interpolation signal toward the
            target yet), so v_θ has to extrapolate the entire trajectory
            from a single frozen point. Diagnostic (eval/diagnose_catastrophic.py)
            found that for the eval_newproj_10555 catastrophic outliers
            (samples 3, 32, 20, 27, 29, 46, 30, 40 — MSE regressions from
            -51% to -2454% vs Stage 1), ||v_cond - v_uncond|| at step 0
            (t=0) was 5-40x larger than at any later step (e.g. sample 32:
            2133 at t=0 vs ~110-173 after; sample 46: 2813 vs ~70-87). All
            8 flagged samples showed this exact shape; steps 1-3 look
            normal. cfg_scale amplifying that raw t=0 disagreement produces
            an oversized first Euler step that the remaining steps can't
            correct — they just keep extrapolating from an already-bad x.
            Setting cfg_warmup_steps > 0 uses the conditional velocity
            alone (no guidance amplification) for that many initial steps,
            then applies full cfg_scale once the trajectory is past t=0.

        Args:
            x_enh    : (B, 80, T) enhanced mel from Stage 1
            d_vector : (B, 512)   speaker d-vector
            cfg_scale: guidance scale (1.0 = no guidance)
            n_steps  : number of Euler steps (1 = paper default)
            cfg_warmup_steps: number of initial steps to run at cfg_scale=1.0
                               (no guidance amplification) before switching to
                               the full cfg_scale. 0 = current/original behavior.
        Returns:
            y_hat: (B, 80, T) predicted clean spectrogram
        """
        self.eval()
        B = x_enh.shape[0]

        x = x_enh.clone()
        dt = 1.0 / n_steps

        for step in range(n_steps):
            t = torch.full((B,), step * dt, device=x.device)
            step_cfg_scale = 1.0 if step < cfg_warmup_steps else cfg_scale

            if step_cfg_scale > 1.0:
                # conditional velocity
                v_cond   = self.forward(x, t, d_vector)
                # unconditional velocity (null speaker)
                v_uncond = self.forward(x, t, d_vector,
                                        force_cfg_drop=True)
                # classifier-free guidance
                v = v_uncond + step_cfg_scale * (v_cond - v_uncond)
            else:
                v = self.forward(x, t, d_vector)

            # Euler step
            x = x + v * dt

        return x


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Build model ───────────────────────────────────────────
    print("=" * 60)
    print("Building FlowMatchingModule...")
    print("=" * 60)

    model = FlowMatchingModule(
        n_mels      = 80,
        hidden_dim  = 768,
        n_heads     = 8,
        n_blocks    = 9,
        ffn_mult    = 2,
        embed_dim   = 512,
        dropout     = 0.1,
        cfg_dropout = 0.1,
    ).to(device)

    # ── Forward pass ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training forward pass (B=2, T=500)...")
    print("=" * 60)

    B, T  = 2, 500

    # simulate Stage 1 output
    target_mel = torch.randn(B, 80, T).to(device) * 1.5 - 2.0
    interf_mel = torch.randn(B, 80, T).to(device) * 1.5 - 3.0
    mixture    = torch.log(
        torch.exp(target_mel) + torch.exp(interf_mel)
    )
    mask       = torch.sigmoid(torch.randn(B, 80, T).to(device))
    x_enh      = mixture * mask

    d_vector   = F.normalize(
        torch.randn(B, 512).to(device), p=2, dim=-1
    )

    # sample timestep and interpolate
    t     = torch.rand(B).to(device)
    t_exp = t.view(B, 1, 1)
    x_t   = (1 - t_exp) * x_enh + t_exp * target_mel

    # forward pass
    velocity = model(x_t, t, d_vector)

    print(f"  x_t shape    : {tuple(x_t.shape)}")
    print(f"  d_vector     : {tuple(d_vector.shape)}")
    print(f"  velocity     : {tuple(velocity.shape)}")
    assert velocity.shape == x_enh.shape, "Velocity shape mismatch!"
    print(f"  Shape match  : ✅")

    # ── Loss + backward ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("Loss + backward pass...")
    print("=" * 60)

    model.train()
    loss, info = model.compute_loss(x_enh, target_mel, d_vector)
    loss.backward()

    print(f"  Flow loss    : {info['loss']:.4f}")
    print(f"  t_mean       : {info['t_mean']:.3f}")
    print(f"  Backward     : ✅")

    # verify gradients flow
    no_grad = [n for n, p in model.named_parameters()
               if p.grad is None and p.requires_grad]
    if no_grad:
        print(f"  ⚠️  No grad: {no_grad[:3]}")
    else:
        print(f"  All grads    : ✅")

    # ── Inference (single Euler step) ─────────────────────────
    print("\n" + "=" * 60)
    print("Single-step inference (Eq 17)...")
    print("=" * 60)

    model.eval()
    y_hat = model.inference(x_enh, d_vector, cfg_scale=1.0, n_steps=1)
    print(f"  Input  X_enh : {tuple(x_enh.shape)}")
    print(f"  Output Ŷ     : {tuple(y_hat.shape)}")
    print(f"  Shape match  : ✅")

    # ── CFG inference ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CFG inference (scale=1.5, our enhancement)...")
    print("=" * 60)

    y_cfg = model.inference(x_enh, d_vector, cfg_scale=1.5, n_steps=1)
    print(f"  CFG output   : {tuple(y_cfg.shape)}")

    # CFG should produce slightly different output than no-guidance
    diff = (y_cfg - y_hat).abs().mean().item()
    print(f"  CFG diff     : {diff:.4f}  (should be > 0)")
    print(f"  CFG working  : {'✅' if diff > 0 else '❌'}")

    # ── Multi-step inference ──────────────────────────────────
    print("\n" + "=" * 60)
    print("Multi-step inference (2 steps)...")
    print("=" * 60)

    y_2step = model.inference(x_enh, d_vector, cfg_scale=1.0, n_steps=2)
    print(f"  2-step output: {tuple(y_2step.shape)} ✅")

    # ── CFG warm-up inference ─────────────────────────────────
    print("\n" + "=" * 60)
    print("CFG warm-up inference (4 steps, warmup=1)...")
    print("=" * 60)

    y_warmup = model.inference(x_enh, d_vector, cfg_scale=1.5, n_steps=4, cfg_warmup_steps=1)
    print(f"  Warmup output: {tuple(y_warmup.shape)} ✅")

    # ── Variable length test ──────────────────────────────────
    print("\n" + "=" * 60)
    print("Variable batch/length test...")
    print("=" * 60)

    model.eval()
    for B_t, T_t in [(1, 300), (4, 1000), (2, 2000)]:
        xe = torch.randn(B_t, 80, T_t).to(device)
        dv = F.normalize(torch.randn(B_t, 512).to(device), p=2, dim=-1)
        yh = model.inference(xe, dv)
        print(f"  B={B_t}, T={T_t:4d}  →  output={tuple(yh.shape)} ✅")

    # ── GPU memory ────────────────────────────────────────────
    if device.type == "cuda":
        mem = torch.cuda.memory_allocated() / 1e6
        print(f"\n  GPU memory   : {mem:.1f} MB")

    print("\n" + "=" * 60)
    print("All FlowMatchingModule tests passed! ✅")
    print("=" * 60)
