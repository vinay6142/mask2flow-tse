"""
Stage 1: Masking Module for Mask2Flow-TSE.

Paper Section 4.3 + 5.2.2:
    Input : log-mel (B, 80, T) + speaker d-vector (B, 512)
    Output: enhanced mel (B, 80, T) = mixture ⊙ soft_mask

Architecture:
    1. ℓ2-normalize input mel
    2. Conv2d × 4  →  8-channel feature maps  (B, 8, 80, T)
    3. Flatten     →  (B, T, 640)
    4. Concat d    →  (B, T, 1152)  [d_vector injected at layer 1]
    5. BiLSTM-1    →  (B, T, 832)
    6. Concat d    →  (B, T, 1344)  [d_vector injected at layer 2]
    7. BiLSTM-2    →  (B, T, 832)
    8. Linear + Sigmoid  →  mask (B, 80, T) ∈ [0, 1]
    9. X_enh = X ⊙ mask

Loss: MSE(X_enh, Y_clean)   — Equation (11) in paper

Key property:
    mask ∈ [0, 1]  →  X_enh ≤ X everywhere
    →  pure deletion (D=100%, I=0%)
    →  flow matching handles insertion in Stage 2
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ── Building blocks ───────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """
    4-layer 2D convolutional block.

    Processes the mel spectrogram as a 2D image (freq × time).
    All layers use kernel=3, padding=1 to preserve spatial dimensions.

    Paper: "processed by convolutional layers...
           producing 8-channel feature maps"
    """

    def __init__(self, channels: list = [1, 4, 6, 8, 8]):
        super().__init__()

        assert len(channels) == 5, "Need exactly 4 conv layers (5 channel values)"

        layers = []
        for i in range(len(channels) - 1):
            in_ch  = channels[i]
            out_ch = channels[i + 1]
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3,
                          padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.1, inplace=True),
            ]

        self.conv        = nn.Sequential(*layers)
        self.out_channels = channels[-1]   # 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, n_mels, T)
        Returns:
            (B, 8, n_mels, T)
        """
        return self.conv(x)


class BiLSTMLayer(nn.Module):
    """
    Single BiLSTM layer with LayerNorm and dropout.

    Paper: "stacked bidirectional LSTM layers with residual connections.
    LayerNorm and residual connections are applied after each layer."

    Note: residual connection is only applied when input and output
    dimensions match (not the case here due to d-vector concatenation,
    so we use LayerNorm only).
    """

    def __init__(
        self,
        input_size:  int,
        hidden_size: int   = 416,
        dropout:     float = 0.1,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size    = input_size,
            hidden_size   = hidden_size,
            num_layers    = 1,
            bidirectional = True,
            batch_first   = True,
        )

        self.norm    = nn.LayerNorm(hidden_size * 2)   # 832
        self.dropout = nn.Dropout(dropout)

        self.output_size = hidden_size * 2   # 832

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_size)
        Returns:
            out: (B, T, 832)
        """
        out, _ = self.lstm(x)
        out    = self.dropout(out)
        out    = self.norm(out)
        return out


# ── Main module ───────────────────────────────────────────────────────────────

class MaskingModule(nn.Module):
    """
    Mask2Flow-TSE Stage 1: soft masking network.

    Estimates M ∈ [0,1]^(80×T) and applies it to suppress interference.
    Output X_enh = X ⊙ M is the coarse-enhanced spectrogram fed to Stage 2.

    Parameter count: ~9-13M (paper: 12.7M)
    """

    def __init__(
        self,
        n_mels:        int   = 80,
        embed_dim:     int   = 512,
        conv_channels: list  = [1, 4, 6, 8, 8],
        lstm_hidden:   int   = 416,
        lstm_dropout:  float = 0.1,
    ):
        super().__init__()

        self.n_mels    = n_mels
        self.embed_dim = embed_dim

        # ── Conv block ────────────────────────────────────────
        self.conv_block  = ConvBlock(conv_channels)
        conv_out_ch      = conv_channels[-1]       # 8
        self.conv_flat   = conv_out_ch * n_mels    # 8 × 80 = 640

        # ── BiLSTM Layer 1 ────────────────────────────────────
        # Input: flattened conv (640) + speaker embedding (512)
        self.lstm1 = BiLSTMLayer(
            input_size  = self.conv_flat + embed_dim,    # 1152
            hidden_size = lstm_hidden,                   # 416
            dropout     = lstm_dropout,
        )

        # ── BiLSTM Layer 2 ────────────────────────────────────
        # Input: lstm1 output (832) + speaker embedding (512)
        # Paper: "speaker embedding concatenated at each LSTM layer"
        self.lstm2 = BiLSTMLayer(
            input_size  = self.lstm1.output_size + embed_dim,  # 1344
            hidden_size = lstm_hidden,                          # 416
            dropout     = lstm_dropout,
        )

        # ── Output mask ───────────────────────────────────────
        # 832 → 80 mel bins, sigmoid squashes to [0, 1]
        self.output_proj = nn.Linear(self.lstm2.output_size, n_mels)

        # ── Parameter summary ─────────────────────────────────
        n_conv  = sum(p.numel() for p in self.conv_block.parameters())
        n_lstm1 = sum(p.numel() for p in self.lstm1.parameters())
        n_lstm2 = sum(p.numel() for p in self.lstm2.parameters())
        n_proj  = sum(p.numel() for p in self.output_proj.parameters())
        n_total = n_conv + n_lstm1 + n_lstm2 + n_proj

        print(f"[MaskingModule] Conv block  : {n_conv:>10,} params")
        print(f"[MaskingModule] BiLSTM 1    : {n_lstm1:>10,} params  "
              f"(in={self.conv_flat+embed_dim}, out={self.lstm1.output_size})")
        print(f"[MaskingModule] BiLSTM 2    : {n_lstm2:>10,} params  "
              f"(in={self.lstm1.output_size+embed_dim}, out={self.lstm2.output_size})")
        print(f"[MaskingModule] Output proj : {n_proj:>10,} params")
        print(f"[MaskingModule] Total       : {n_total:>10,} params  "
              f"(~{n_total/1e6:.1f}M)")

    def forward(
        self,
        x_mel:    torch.Tensor,
        d_vector: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate soft mask and produce enhanced spectrogram.

        Args:
            x_mel   : (B, n_mels=80, T) log-mel of mixture
            d_vector: (B, embed_dim=512) speaker d-vector (L2 normalized)

        Returns:
            x_enhanced: (B, 80, T) X ⊙ M — coarse-enhanced mel
            mask       : (B, 80, T) soft mask values in [0, 1]
        """
        B, n_mels, T = x_mel.shape

        # ── ℓ2-normalize input ────────────────────────────────
        x_norm = F.normalize(x_mel, p=2, dim=(-2, -1))  # (B, 80, T)

        # ── Conv2d block ──────────────────────────────────────
        x_in   = x_norm.unsqueeze(1)          # (B, 1, 80, T)
        x_conv = self.conv_block(x_in)        # (B, 8, 80, T)

        # flatten freq-channel dims for LSTM
        x_flat = x_conv.permute(0, 3, 1, 2)  # (B, T, 8, 80)
        x_flat = x_flat.reshape(B, T, -1)    # (B, T, 640)

        # ── Speaker embedding expansion ───────────────────────
        # broadcast d-vector across all time steps
        d_exp  = d_vector.unsqueeze(1).expand(-1, T, -1)  # (B, T, 512)

        # ── BiLSTM Layer 1 ────────────────────────────────────
        x1_in  = torch.cat([x_flat, d_exp], dim=-1)  # (B, T, 1152)
        x1_out = self.lstm1(x1_in)                   # (B, T, 832)

        # ── BiLSTM Layer 2 ────────────────────────────────────
        # concatenate d-vector again at layer 2 input
        x2_in  = torch.cat([x1_out, d_exp], dim=-1)  # (B, T, 1344)
        x2_out = self.lstm2(x2_in)                   # (B, T, 832)

        # ── Soft mask prediction ──────────────────────────────
        mask_seq = self.output_proj(x2_out)           # (B, T, 80)
        mask     = torch.sigmoid(mask_seq)            # (B, T, 80) ∈ [0,1]
        mask     = mask.permute(0, 2, 1)              # (B, 80, T)

        # ── Apply mask ────────────────────────────────────────
        # apply to original (non-normalized) mel
        x_enhanced = x_mel * mask                    # (B, 80, T)

        return x_enhanced, mask

    def compute_loss(
        self,
        x_enhanced: torch.Tensor,
        y_target:   torch.Tensor,
        mask:       Optional[torch.Tensor] = None,
        frame_mask:  Optional[torch.Tensor] = None,  # NEW: (B,1,T) or (B,T), 1=real audio, 0=padding
    ) -> torch.Tensor:
        """
        Masking stage loss — Equation (11) in paper.

        L_mask = ||X_enh - Y||²

        Optionally weight by speech activity to penalize
        over-suppression of active speech bins more heavily.

        Args:
            x_enhanced: (B, 80, T) masked spectrogram
            y_target   : (B, 80, T) clean target spectrogram
            mask       : (B, 80, T) predicted mask (optional, for logging)
        Returns:
            loss: scalar
        """
        if frame_mask is None:
            return F.mse_loss(x_enhanced, y_target)
        if frame_mask.dim() == 2:
            frame_mask = frame_mask.unsqueeze(1)          # (B,1,T)
        diff2 = (x_enhanced - y_target) ** 2 * frame_mask
        denom = frame_mask.sum() * x_enhanced.shape[1] + 1e-8   # ×n_mels since mask is per-frame
        return diff2.sum() / denom

    def compute_di_proportion(
        self,
        x_input:  torch.Tensor,
        x_output: torch.Tensor,
    ) -> Tuple[float, float]:
        """
        Compute Delete/Insert proportion for this module.

        For a trained masking module, should be D≈100%, I≈0%.

        Args:
            x_input : (B, 80, T) mixture mel
            x_output: (B, 80, T) enhanced mel
        Returns:
            (D_percent, I_percent)
        """
        with torch.no_grad():
            delta = x_output - x_input
            D = (-delta[delta < 0]).sum().item()
            I = ( delta[delta > 0]).sum().item()
            total = D + I + 1e-8
        return D / total * 100, I / total * 100


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Build model ───────────────────────────────────────────
    print("=" * 55)
    print("Building MaskingModule...")
    print("=" * 55)

    model = MaskingModule(
        n_mels        = 80,
        embed_dim     = 512,
        conv_channels = [1, 4, 6, 8, 8],
        lstm_hidden   = 416,
        lstm_dropout  = 0.1,
    ).to(device)

    # ── Forward pass ──────────────────────────────────────────
    print("\n" + "=" * 55)
    print("Forward pass test (B=4, T=1000)...")
    print("=" * 55)

    B, T  = 4, 1000

    # simulate realistic mixture: mixture >= target (log-sum-exp property)
    target_mel = torch.randn(B, 80, T).to(device) * 1.5 - 2.0
    interf_mel = torch.randn(B, 80, T).to(device) * 1.5 - 3.0
    x_mel      = torch.log(torch.exp(target_mel) + torch.exp(interf_mel))

    d_vector = F.normalize(torch.randn(B, 512).to(device), p=2, dim=-1)

    x_enh, mask = model(x_mel, d_vector)

    print(f"  Input mel   : {tuple(x_mel.shape)}")
    print(f"  d-vector    : {tuple(d_vector.shape)}")
    print(f"  Mask shape  : {tuple(mask.shape)}")
    print(f"  Mask range  : [{mask.min():.4f}, {mask.max():.4f}]")
    print(f"  Enhanced    : {tuple(x_enh.shape)}")

    # check mask is truly in [0, 1]
    assert mask.min() >= 0.0 and mask.max() <= 1.0, "Mask out of [0,1]!"
    print(f"  Mask ∈ [0,1]: ✅")

    # ── Loss + backward ───────────────────────────────────────
    loss = model.compute_loss(x_enh, target_mel)
    loss.backward()

    print(f"\n  MSE loss    : {loss.item():.4f}")
    print(f"  Backward    : ✅")

    # check gradients flowing
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"  WARNING: no grad for {name}")

    # ── D/I proportion ────────────────────────────────────────
    D_pct, I_pct = model.compute_di_proportion(x_mel, x_enh)
    print(f"\n  D/I proportion:")
    print(f"    Delete: {D_pct:.1f}%")
    print(f"    Insert: {I_pct:.1f}%")
    print(f"    (Paper: D=100%, I=0% for trained model)")

    # ── Memory ───────────────────────────────────────────────
    if device.type == "cuda":
        mem_mb = torch.cuda.memory_allocated() / 1e6
        print(f"\n  GPU memory  : {mem_mb:.1f} MB")

    # ── Variable batch/length test ───────────────────────────
    print("\n" + "=" * 55)
    print("Variable batch/length test...")
    print("=" * 55)

    for B_test, T_test in [(1, 500), (2, 2000), (8, 300)]:
        xm = torch.randn(B_test, 80, T_test).to(device)
        dv = F.normalize(torch.randn(B_test, 512).to(device), p=2, dim=-1)
        xe, mk = model(xm, dv)
        print(f"  B={B_test:2d}, T={T_test:4d}  →  "
              f"enhanced={tuple(xe.shape)}, mask={tuple(mk.shape)} ✅")

    print("\nAll MaskingModule tests passed! ✅")