"""
Prints complete model architecture summary
with parameter counts matching the paper (~85M total).
"""

import torch
import torch.nn as nn
from omegaconf import OmegaConf


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def print_architecture_summary(cfg):

    print("\n" + "=" * 60)
    print("  Mask2Flow-TSE — Architecture Summary")
    print("=" * 60)

    # ── Stage 1: Masking ──────────────────────────────────────
    print("\n[Stage 1] Masking Module")
    print("-" * 40)

    # Paper says: "4 Conv2d layers producing 8-channel feature maps"
    # Input: (B, 1, 80, T) → Output: (B, 8, 80, T)
    conv_block = nn.Sequential(
        nn.Conv2d(1, 4,  3, padding=1), nn.BatchNorm2d(4),  nn.LeakyReLU(),
        nn.Conv2d(4, 6,  3, padding=1), nn.BatchNorm2d(6),  nn.LeakyReLU(),
        nn.Conv2d(6, 8,  3, padding=1), nn.BatchNorm2d(8),  nn.LeakyReLU(),
        nn.Conv2d(8, 8,  3, padding=1), nn.BatchNorm2d(8),  nn.LeakyReLU(),
    )

    # After flattening: 8 channels * 80 mel bins = 640
    # Concat with 512-dim speaker embedding → 1152
    # Paper: BiLSTM with 416 hidden units, 2 layers
    lstm_input_size = 8 * cfg.mel.n_mels + cfg.speaker_encoder.embed_dim
    lstm = nn.LSTM(
        input_size    = lstm_input_size,    # 8*80 + 512 = 1152
        hidden_size   = cfg.masking.lstm_hidden,   # 416
        num_layers    = cfg.masking.lstm_layers,   # 2
        bidirectional = True,
        batch_first   = True,
    )

    # BiLSTM output: 416 * 2 = 832 → project to 80 mel bins
    output_proj = nn.Linear(
        cfg.masking.lstm_hidden * 2,   # 832
        cfg.mel.n_mels                 # 80
    )

    conv_params  = count_params(conv_block)
    lstm_params  = count_params(lstm)
    proj_params  = count_params(output_proj)
    mask_total   = conv_params + lstm_params + proj_params

    print(f"  Input shape  :  (B, 1, {cfg.mel.n_mels}, T)  → log-mel")
    print(f"  Conv Block   : {conv_params:>12,} params  "
          f"(1→4→6→8 channels, 8×{cfg.mel.n_mels} output)")
    print(f"  LSTM input   :  {lstm_input_size} "
          f"(8×{cfg.mel.n_mels} + {cfg.speaker_encoder.embed_dim} speaker)")
    print(f"  BiLSTM       : {lstm_params:>12,} params  "
          f"(hidden={cfg.masking.lstm_hidden}, layers={cfg.masking.lstm_layers})")
    print(f"  Output Proj  : {proj_params:>12,} params  "
          f"(832 → {cfg.mel.n_mels})")
    print(f"  {'─'*38}")
    print(f"  Stage 1 Total: {mask_total:>12,} params  "
          f"(~{mask_total/1e6:.1f}M)")
    print(f"  Paper reports:     12,700,000 params  (~12.7M)")

    # ── Stage 2: Flow Matching ────────────────────────────────
    print("\n[Stage 2] Flow Matching Module (DiT)")
    print("-" * 40)

    H = cfg.flow.hidden_dim   # 768

    # project mel bins to hidden dim
    input_proj = nn.Linear(cfg.mel.n_mels, H)   # 80 → 768

    # DiT block components
    # AdaLN-Zero: conditioning → scale/shift for 2 sub-layers
    # Input is conditioning vector c (size H), output is 6*H
    single_dit = nn.ModuleDict({
        "norm1"  : nn.LayerNorm(H),
        "attn_qkv": nn.Linear(H, 3 * H),
        "attn_out": nn.Linear(H, H),
        "norm2"  : nn.LayerNorm(H),
        "ff1"    : nn.Linear(H, H * cfg.flow.ffn_mult),  # 768→3072
        "ff2"    : nn.Linear(H * cfg.flow.ffn_mult, H),  # 3072→768
        "adaln"  : nn.Linear(H, 6 * H),
    })
    dit_block_params = count_params(single_dit)

    # timestep embedding: sinusoidal → MLP
    time_embed = nn.Sequential(
        nn.Linear(256, H), nn.SiLU(), nn.Linear(H, H)
    )

    # speaker projection: d-vector → conditioning
    spk_proj = nn.Linear(cfg.speaker_encoder.embed_dim, H)

    # final output projection: hidden → mel bins
    output_proj_flow = nn.Linear(H, cfg.mel.n_mels)

    flow_total = (
        count_params(input_proj)
        + dit_block_params * cfg.flow.n_blocks
        + count_params(time_embed)
        + count_params(spk_proj)
        + count_params(output_proj_flow)
    )

    print(f"  Hidden dim   :  {H}")
    print(f"  Input Proj   : {count_params(input_proj):>12,} params  "
          f"({cfg.mel.n_mels} → {H})")
    print(f"  DiT Block    : {dit_block_params:>12,} params "
          f"× {cfg.flow.n_blocks} blocks")
    print(f"  Time Embed   : {count_params(time_embed):>12,} params")
    print(f"  Speaker Proj : {count_params(spk_proj):>12,} params  "
          f"({cfg.speaker_encoder.embed_dim} → {H})")
    print(f"  Output Proj  : {count_params(output_proj_flow):>12,} params  "
          f"({H} → {cfg.mel.n_mels})")
    print(f"  {'─'*38}")
    print(f"  Stage 2 Total: {flow_total:>12,} params  "
          f"(~{flow_total/1e6:.1f}M)")
    print(f"  Paper reports:     72,600,000 params  (~72.6M)")

    # ── Speaker Encoder ───────────────────────────────────────
    print("\n[Speaker Encoder] WavLM-base-plus-sv (frozen)")
    print("-" * 40)
    print(f"  Params       :       94,000,000  (~94M, FROZEN)")
    print(f"  Output dim   :              512  (d-vector)")
    print(f"  Note         :  NOT counted in trainable params")

    # ── Total ─────────────────────────────────────────────────
    total = mask_total + flow_total
    diff  = abs(total / 1e6 - 85.0)
    match = "✅ Close to paper" if diff < 20 else "⚠️  Check config"

    print("\n" + "=" * 60)
    print(f"  Stage 1 (Masking)  : {mask_total/1e6:>8.1f}M params")
    print(f"  Stage 2 (Flow)     : {flow_total/1e6:>8.1f}M params")
    print(f"  {'─'*38}")
    print(f"  TOTAL (trainable)  : {total/1e6:>8.1f}M params")
    print(f"  Paper reports      :     85.0M params")
    print(f"  Status             :  {match}")
    print("=" * 60)

    # ── Baseline comparison ───────────────────────────────────
    print("\n[Comparison with Baselines]")
    print("-" * 40)
    baselines = {
        "ConVoiFilter"  : 49.9,
        "Mask2Flow-TSE" : total / 1e6,
        "TSELM"         : 195.4,
        "Metis-TSE"     : 1425.0,
    }
    max_params = max(baselines.values())
    for name, params in baselines.items():
        bar    = "█" * int(params / max_params * 30)
        marker = "  ← Ours" if name == "Mask2Flow-TSE" else ""
        print(f"  {name:16s}: {params:7.1f}M  {bar}{marker}")

    print(f"\n  ✅ Mask2Flow-TSE is {1425/total*1e6:.0f}× smaller than Metis-TSE")
    print(f"  ✅ Single inference step vs 50+ for other generative methods\n")


if __name__ == "__main__":
    cfg = OmegaConf.load("configs/default.yaml")
    print_architecture_summary(cfg)