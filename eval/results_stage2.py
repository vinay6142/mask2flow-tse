"""
Mask2Flow-TSE — Combined Stage 1 + Stage 2 Results Generator (REAL DATA)

Loads BOTH the frozen masking checkpoint and the trained flow-matching
checkpoint, runs the full two-stage pipeline (mixture -> mask -> flow)
on real held-out LibriSpeech audio (test-clean), and produces comparison
plots + a terminal summary across all three signals: mixture, Stage 1
output, Stage 2 (final) output, target.

By default this uses REAL data (test-clean split) — this is what should
be used for any quality/results claims, since Stage 2 was trained on
real speech statistics and is not meaningful to evaluate on synthetic
random-noise data.

A --fake flag is kept ONLY for a fast mechanical sanity check (do shapes
match, does it crash, do checkpoints load) — its output must NOT be used
as a quality result; the script prints a loud warning if you use it.

Run (real data, default):
  python3 eval/results_stage2.py \
      --mask_ckpt checkpoints/masking/mask_final.pt \
      --flow_ckpt checkpoints/flow/flow_best.pt

Run (fast mechanical check only, NOT a quality result):
  python3 eval/results_stage2.py \
      --mask_ckpt checkpoints/masking/mask_final.pt \
      --flow_ckpt checkpoints/flow/flow_best.pt \
      --fake
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.mel import make_frame_mask, masked_mse
from data.augment import trim_trailing_silence
from inference.vocoder import load_hifigan_generator, mel_to_audio_hifigan

from models.speaker_encoder import SpeakerEncoder
from models.masking         import MaskingModule
from models.flow            import FlowMatchingModule

os.makedirs("outputs/results", exist_ok=True)

BG   = "#0D1117"
BLUE = "#2196F3"
GRN  = "#4CAF50"
AMB  = "#FF9800"
RED  = "#F44336"
PUR  = "#AB47BC"
GREY = "#78909C"


def load_masking(ckpt_path, cfg, device):
    model = MaskingModule(
        n_mels        = cfg.mel.n_mels,
        embed_dim     = cfg.speaker_encoder.embed_dim,
        conv_channels = cfg.masking.conv_channels,
        lstm_hidden   = cfg.masking.lstm_hidden,
        lstm_dropout  = cfg.masking.lstm_dropout,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)

    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])

    if "ema" in ckpt:
        ema_shadow = ckpt["ema"]["shadow"]
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in ema_shadow:
                    param.data.copy_(ema_shadow[name])
        step = ckpt.get("step", "?")
        print(f"[Results] Stage 1: loaded EMA params + raw buffers from step {step}")
    else:
        print(f"[Results] Stage 1: loaded raw weights only")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_flow(ckpt_path, cfg, device):
    model = FlowMatchingModule(
        n_mels      = cfg.mel.n_mels,
        hidden_dim  = cfg.flow.hidden_dim,
        n_heads     = cfg.flow.n_heads,
        n_blocks    = cfg.flow.n_blocks,
        ffn_mult    = cfg.flow.ffn_mult,
        embed_dim   = cfg.speaker_encoder.embed_dim,
        dropout     = cfg.flow.dropout,
        cfg_dropout = cfg.flow.cfg_dropout,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)

    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])

    if "ema" in ckpt:
        ema_shadow = ckpt["ema"]["shadow"]
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in ema_shadow:
                    param.data.copy_(ema_shadow[name])
        step     = ckpt.get("step", "?")
        val_loss = ckpt.get("val_loss", float("inf"))
        print(f"[Results] Stage 2: loaded EMA params + raw buffers from step {step}"
              f" (val_loss={val_loss:.4f})")
    else:
        print(f"[Results] Stage 2: loaded raw weights only")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def apply_safety_net(stage1_out, stage2_out, threshold_db=5.0):
    """
    Per-frame energy-collapse guard.

    IMPORTANT: stage1_out/stage2_out are LOG-mel values (can be negative,
    roughly -15 to +8). Do NOT compare via squared magnitude — squaring a
    very negative log value produces a LARGE positive number, making true
    silence look like high energy. Compare the mean log-mel LEVEL
    directly: if Stage 2's level has dropped more than `threshold_db`
    below Stage 1's at the same frame, fall back to Stage 1 there.

    stage1_out, stage2_out: (B, n_mels, T) log-mel tensors.
    """
    s1_level = stage1_out.mean(dim=1, keepdim=True)
    s2_level = stage2_out.mean(dim=1, keepdim=True)
    collapsed = (s1_level - s2_level) > threshold_db
    collapsed = collapsed.expand_as(stage2_out)
    n_collapsed_frames = collapsed.any(dim=1).sum().item()
    if n_collapsed_frames > 0:
        print(f"[SafetyNet] Fell back to Stage 1 for {n_collapsed_frames} "
              f"frame-positions across the batch (threshold_db={threshold_db})")
    return torch.where(collapsed, stage1_out, stage2_out)


def get_real_batch(cfg, device, B=4, seed=42):
    from data.librispeech import LibriSpeechTSEDataset

    print(f"[Results] Loading real batch from LibriSpeech test-clean (seed={seed})...")
    val_dataset = LibriSpeechTSEDataset(
        librispeech_root = cfg.data.librispeech_path,
        splits           = ["test-clean"],
        cfg              = cfg,
        is_train         = False,
        seed             = seed,
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        val_dataset, batch_size=B, shuffle=True,
        num_workers=0, generator=generator,
    )
    batch = next(iter(loader))

    mixture      = batch["mixture_mel"].to(device)
    target       = batch["target_mel"].to(device)
    ref_wav      = batch["reference_wav"].to(device)
    condition    = batch["condition"]
    snr_db       = batch["snr_db"].tolist()
    valid_frames = batch["target_valid_frames"].to(device)   # NEW: (B,) real-frame counts
    print(f"[Diag] mixture checksum : {mixture.sum().item():.6f}")
    print(f"[Diag] target checksum  : {target.sum().item():.6f}")
    print(f"[Diag] ref_wav checksum : {ref_wav.sum().item():.6f}  "
          f"(mean={ref_wav.mean().item():.8f}, std={ref_wav.std().item():.8f})")
    frac_padded = 1.0 - (valid_frames.float() / target.shape[-1]).mean().item()
    print(f"[Diag] mean padding fraction in this batch: {frac_padded:.3f}")

    return mixture, target, ref_wav, condition, snr_db, valid_frames


def make_fake_batch(cfg, device, B=4, seed=42):
    torch.manual_seed(seed)
    n_mels   = cfg.mel.n_mels
    n_frames = cfg.audio.segment_length * cfg.audio.sample_rate // cfg.mel.hop_length

    target_mel     = torch.randn(B, n_mels, n_frames) * 1.5 - 2.0
    interferer_mel = torch.randn(B, n_mels, n_frames) * 1.5 - 3.5
    mixture_mel    = torch.log(torch.exp(target_mel) + torch.exp(interferer_mel))
    reference_wav  = torch.randn(B, 3 * cfg.audio.sample_rate) * 0.1

    condition = ["fake_noise"] * B
    snr_db    = [float("nan")] * B
    # NEW: fake/synthetic samples have no real padding — mark every frame valid
    valid_frames = torch.full((B,), n_frames, dtype=torch.long)
    return (mixture_mel.to(device), target_mel.to(device), reference_wav.to(device),
            condition, snr_db, valid_frames.to(device))


def compute_di(x_in, x_out):
    delta = x_out - x_in
    D = (-delta[delta < 0]).sum().item()
    I = ( delta[delta > 0]).sum().item()
    t = D + I + 1e-8
    return D / t * 100, I / t * 100


def plot_pipeline_spectrograms(mixture, stage1_out, stage2_out, target, tag, idx=0):
    print("  Generating full-pipeline spectrogram comparison...")

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), facecolor=BG)
    fig.suptitle(
        f"Mask2Flow-TSE — Full Pipeline ({tag}): Mixture -> Stage 1 -> Stage 2 -> Target",
        fontsize=13, color="white", fontweight="bold", y=0.995,
    )

    panels = [
        (mixture[idx].cpu().numpy(),    "(a) Mixture Input  [interference present]",              RED),
        (stage1_out[idx].cpu().numpy(), "(b) After Stage 1 Masking  [interference suppressed]",   AMB),
        (stage2_out[idx].cpu().numpy(), "(c) After Stage 2 Flow Matching  [refined toward clean]",PUR),
        (target[idx].cpu().numpy(),     "(d) Clean Target  [ground truth]",                        GRN),
    ]

    vmin = min(p[0].min() for p in panels)
    vmax = max(p[0].max() for p in panels)

    for ax, (mel, title, color) in zip(axes, panels):
        im = ax.imshow(mel, aspect="auto", origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(title, color=color, fontsize=11, pad=4, fontweight="bold")
        ax.set_ylabel("Mel Bin", color="white", fontsize=9)
        ax.tick_params(colors="white", labelsize=8)
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_edgecolor(color)
            sp.set_linewidth(1.5)

    axes[-1].set_xlabel("Time Frame", color="white", fontsize=9)
    cbar = plt.colorbar(im, ax=axes)
    cbar.set_label("Log Energy", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")

    fig.subplots_adjust(top=0.94)
    path = f"outputs/results/spectrogram_comparison_full_pipeline_{tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"    -> {path}")


def plot_pipeline_quality(mixture, stage1_out, stage2_out, target, tag, valid_frames=None):
    print("  Generating three-way MSE / improvement comparison...")

    B = mixture.shape[0]
    labels = [f"Sample {i+1}" for i in range(B)]

    # NEW: per-sample frame mask so padded (silence) frames on short
    # utterances don't get counted as trivially-easy "improvement" —
    # test-clean utterances are ~7.4s on average vs this project's fixed
    # 10s segment_length, so this materially affects every number below.
    if valid_frames is None:
        frame_mask_full = None
    else:
        frame_mask_full = make_frame_mask(valid_frames, target.shape[-1], target.device)  # (B, T)

    mix_mse, s1_mse, s2_mse = [], [], []
    s1_imp, s2_imp_vs_mix, s2_imp_vs_s1 = [], [], []

    for b in range(B):
        fm_b = None if frame_mask_full is None else frame_mask_full[[b]]
        bm = masked_mse(mixture[[b]],    target[[b]], fm_b).item()
        sm = masked_mse(stage1_out[[b]], target[[b]], fm_b).item()
        fm = masked_mse(stage2_out[[b]], target[[b]], fm_b).item()
        mix_mse.append(bm); s1_mse.append(sm); s2_mse.append(fm)
        s1_imp.append((bm - sm) / bm * 100)
        s2_imp_vs_mix.append((bm - fm) / bm * 100)
        s2_imp_vs_s1.append((sm - fm) / max(sm, 1e-8) * 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
    fig.suptitle(f"Full-Pipeline Reconstruction Quality ({tag})", fontsize=13, color="white", fontweight="bold")

    x = np.arange(B); w = 0.26
    ax1.set_facecolor("#0D1B2E")
    ax1.bar(x - w, mix_mse, w, label="Mixture MSE", color=RED, alpha=0.85)
    ax1.bar(x,     s1_mse,  w, label="Stage 1 MSE",  color=AMB, alpha=0.85)
    ax1.bar(x + w, s2_mse,  w, label="Stage 2 MSE",  color=PUR, alpha=0.85)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, color="white")
    ax1.set_ylabel("MSE vs Clean Target", color="white")
    ax1.set_title("MSE by Pipeline Stage", color="white", fontsize=11)
    ax1.tick_params(colors="white")
    ax1.legend(facecolor="#0D1B2E", labelcolor="white", fontsize=9)
    ax1.grid(axis="y", alpha=0.15, color="white")
    for sp in ax1.spines.values(): sp.set_edgecolor("#2A3F5F")

    ax2.set_facecolor("#0D1B2E")
    ax2.bar(x - w/2, s2_imp_vs_mix, w, label="Stage 2 vs Mixture", color=PUR, alpha=0.85)
    ax2.bar(x + w/2, s2_imp_vs_s1,  w, label="Stage 2 vs Stage 1", color=BLUE, alpha=0.85)
    ax2.axhline(0, color="white", linewidth=0.8, alpha=0.5)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, color="white")
    ax2.set_ylabel("% MSE Improvement", color="white")
    ax2.set_title("Stage 2's Added Value", color="white", fontsize=11)
    ax2.tick_params(colors="white")
    ax2.legend(facecolor="#0D1B2E", labelcolor="white", fontsize=9)
    ax2.grid(axis="y", alpha=0.15, color="white")
    for sp in ax2.spines.values(): sp.set_edgecolor("#2A3F5F")

    plt.tight_layout()
    path = f"outputs/results/pipeline_reconstruction_quality_{tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"    -> {path}")

    return mix_mse, s1_mse, s2_mse, s1_imp, s2_imp_vs_mix, s2_imp_vs_s1


def print_summary(mix_mse, s1_mse, s2_mse, s1_imp, s2_imp_vs_mix, s2_imp_vs_s1,
                   mask_ckpt, flow_ckpt, tag, condition, snr_db):
    B = len(mix_mse)
    print("\n" + "=" * 96)
    print(f"  MASK2FLOW-TSE — FULL PIPELINE RESULTS SUMMARY  [{tag.upper()}]")
    print("=" * 96)
    print(f"\n  Stage 1 checkpoint: {mask_ckpt}")
    print(f"  Stage 2 checkpoint: {flow_ckpt}")

    print(f"\n  {'Sample':<10} {'Cond':<9} {'SNR(dB)':>8} {'Mixture MSE':>12} {'Stage1 MSE':>11} {'Stage2 MSE':>11}"
          f" {'S1 Imp%':>9} {'S2 vs Mix%':>11} {'S2 vs S1%':>10}")
    print(f"  {'-'*94}")
    for b in range(B):
        snr_str = "NaN" if snr_db[b] != snr_db[b] else f"{snr_db[b]:.2f}"
        print(f"  Sample {b+1:<4} {condition[b]:<9} {snr_str:>8} {mix_mse[b]:>12.4f} {s1_mse[b]:>11.4f} {s2_mse[b]:>11.4f}"
              f" {s1_imp[b]:>8.1f}% {s2_imp_vs_mix[b]:>10.1f}% {s2_imp_vs_s1[b]:>9.1f}%")

    avg = lambda l: sum(l) / len(l)
    med = lambda l: float(np.median(l))
    print(f"  {'-'*94}")
    print(f"  {'Mean (all)':<20} {avg(mix_mse):>12.4f} {avg(s1_mse):>11.4f} {avg(s2_mse):>11.4f}"
          f" {avg(s1_imp):>8.1f}% {avg(s2_imp_vs_mix):>10.1f}% {avg(s2_imp_vs_s1):>9.1f}%")
    print(f"  {'Median (all)':<20} {med(mix_mse):>12.4f} {med(s1_mse):>11.4f} {med(s2_mse):>11.4f}"
          f" {med(s1_imp):>8.1f}% {med(s2_imp_vs_mix):>10.1f}% {med(s2_imp_vs_s1):>9.1f}%")

    print(f"\n  Breakdown by mixing condition:")
    conditions_present = sorted(set(condition))
    for c in conditions_present:
        idxs = [i for i in range(B) if condition[i] == c]
        n = len(idxs)
        c_s2_vs_s1 = [s2_imp_vs_s1[i] for i in idxs]
        c_s2_vs_mix = [s2_imp_vs_mix[i] for i in idxs]
        c_mix_mse = [mix_mse[i] for i in idxs]
        print(f"    {c:<10} n={n:<3}  mean mixture_mse={avg(c_mix_mse):>8.2f}"
              f"   mean S2 vs S1={avg(c_s2_vs_s1):>9.1f}%  median S2 vs S1={med(c_s2_vs_s1):>8.1f}%"
              f"   mean S2 vs Mix={avg(c_s2_vs_mix):>9.1f}%")

    print(f"\n  Key Findings (median, robust to outliers):")
    print(f"    Full pipeline (Stage 1+2) median MSE improvement vs mixture: {med(s2_imp_vs_mix):.1f}%")
    print(f"    Stage 2 median improvement on top of Stage 1 alone: {med(s2_imp_vs_s1):.1f}%")
    print(f"\n  Plots saved to: outputs/results/")
    print("=" * 86)


def compute_speaker_similarity_eval(
    mixture, stage2_out, target, ref_wav, valid_frames,
    encoder, vocoder, cfg, device, max_samples=None, verbose=True,
):
    """
    Waveform-domain speaker-similarity metric across an eval batch —
    complements the mel-domain MSE numbers in plot_pipeline_quality with
    a second, independent quality signal (does extraction actually move
    the output closer to the reference speaker's voice, embedding-wise).

    Uses valid_frames (exact real-frame counts already tracked per sample)
    to trim mixture/target/stage2_out mels to their real-speech region
    BEFORE vocoding — more precise than an energy-threshold heuristic,
    since we know the exact boundary here. reference_wav still uses
    trim_trailing_silence() since its valid length isn't tracked the
    same way.

    Returns:
        sims_mix, sims_tgt, sims_ext: lists of cosine similarities
        (reference vs mixture / vs ground-truth target / vs Stage 2 output)
    """
    B = mixture.shape[0]
    n = B if max_samples is None else min(B, max_samples)
    sr = cfg.audio.sample_rate

    sims_mix, sims_tgt, sims_ext = [], [], []

    for b in range(n):
        vf = max(int(valid_frames[b].item()), 1)

        mix_wav = mel_to_audio_hifigan(mixture[b, :, :vf], vocoder)
        tgt_wav = mel_to_audio_hifigan(target[b, :, :vf], vocoder)
        ext_wav = mel_to_audio_hifigan(stage2_out[b, :, :vf], vocoder)

        mix_wav_t = torch.from_numpy(mix_wav).float().to(device)
        tgt_wav_t = torch.from_numpy(tgt_wav).float().to(device)
        ext_wav_t = torch.from_numpy(ext_wav).float().to(device)
        ref_wav_b = trim_trailing_silence(ref_wav[b], sr)

        with torch.no_grad():
            ref_emb = encoder(ref_wav_b.unsqueeze(0))
            mix_emb = encoder(mix_wav_t.unsqueeze(0))
            tgt_emb = encoder(tgt_wav_t.unsqueeze(0))
            ext_emb = encoder(ext_wav_t.unsqueeze(0))

        sims_mix.append(F.cosine_similarity(ref_emb, mix_emb, dim=-1).item())
        sims_tgt.append(F.cosine_similarity(ref_emb, tgt_emb, dim=-1).item())
        sims_ext.append(F.cosine_similarity(ref_emb, ext_emb, dim=-1).item())

        if verbose and (b + 1) % 10 == 0:
            print(f"    ...speaker-similarity: {b + 1}/{n} samples done")

    return sims_mix, sims_tgt, sims_ext


def print_speaker_similarity_summary(sims_mix, sims_tgt, sims_ext, print_per_sample=True):
    import statistics as _stats

    n = len(sims_ext)
    diffs = [e - m for e, m in zip(sims_ext, sims_mix)]  # raw cosine delta, extraction vs mixture
    n_worse = sum(1 for d in diffs if d < 0)

    if print_per_sample:
        print("\n" + "-" * 86)
        print("  Per-sample speaker similarity (cross-reference Sample # against the MSE table above)")
        print("-" * 86)
        print(f"  {'Sample':<10} {'ref-vs-mix':>12} {'ref-vs-tgt':>12} {'ref-vs-ext':>12} {'delta':>10}  flag")
        for i, (m, t, e, d) in enumerate(zip(sims_mix, sims_tgt, sims_ext, diffs), start=1):
            flag = "  <-- WORSE" if d < 0 else ""
            print(f"  Sample {i:<3} {m:>12.4f} {t:>12.4f} {e:>12.4f} {d:>+10.4f}{flag}")
        print("-" * 86)

    print("\n" + "=" * 86)
    print("  SPEAKER-SIMILARITY (waveform-domain, WavLM cosine) — SECONDARY QUALITY SIGNAL")
    print("=" * 86)
    print(f"  n = {n} samples (mixture/target/Stage2 output all trimmed to real speech before vocoding)")
    print(f"\n  Median cosine sim   ref-vs-mixture: {_stats.median(sims_mix):.4f}"
          f"   ref-vs-target: {_stats.median(sims_tgt):.4f}"
          f"   ref-vs-extracted: {_stats.median(sims_ext):.4f}")
    print(f"  Mean   cosine sim   ref-vs-mixture: {_stats.mean(sims_mix):.4f}"
          f"   ref-vs-target: {_stats.mean(sims_tgt):.4f}"
          f"   ref-vs-extracted: {_stats.mean(sims_ext):.4f}")
    print(f"\n  Median (extracted - mixture) similarity delta: {_stats.median(diffs):+.4f}")
    print(f"  Samples where extraction REDUCED similarity vs mixture: {n_worse}/{n} ({100*n_worse/n:.1f}%)")
    print("=" * 86)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="configs/default.yaml")
    parser.add_argument("--mask_ckpt", required=True)
    parser.add_argument("--flow_ckpt", required=True)
    parser.add_argument("--fake",      action="store_true",
                         help="MECHANICAL SANITY CHECK ONLY (random noise data) — "
                              "do NOT use this output as a quality result.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--plot_sample", type=int, default=0,
                         help="Which sample index in the batch to render in the "
                              "spectrogram figure (e.g. point this at a known "
                              "failing sample index from the summary table).")
    parser.add_argument("--cfg_scale", type=float, default=None,
                         help="Override inference.cfg_scale from config")
    parser.add_argument("--n_steps",   type=int,   default=None,
                         help="Override inference.flow_steps from config")
    parser.add_argument("--cfg_warmup_steps", type=int, default=0,
                         help="Run the first N Euler steps at cfg_scale=1.0 "
                              "(no guidance amplification) before switching "
                              "to the full --cfg_scale. Diagnostic finding: "
                              "at t=0, ||v_cond - v_uncond|| spikes 5-40x "
                              "above later-step levels for the catastrophic "
                              "MSE outliers (samples 3, 32, 20, 27, 29, 46, "
                              "30, 40 in eval_newproj_10555); cfg_scale "
                              "amplifies that spike into an oversized first "
                              "step the remaining steps can't correct. "
                              "Try --cfg_warmup_steps 1 as the fix.")
    parser.add_argument("--safety_net", action="store_true",
                         help="Apply the per-frame energy-collapse fallback: "
                              "wherever Stage 2's log-mel level drops far "
                              "below Stage 1's at the same frame, use Stage 1's "
                              "output there instead. Mitigates the outlier "
                              "failure mode without touching the trained model.")
    parser.add_argument("--safety_net_threshold_db", type=float, default=5.0,
                         help="Collapse threshold in log-mel units (default 5.0): "
                              "flag frames where Stage 2's mean log-mel level is "
                              "more than this many units below Stage 1's.")
    parser.add_argument("--projection_ckpt", default=None,
                         help="Path to a trained SpeakerEncoder projection checkpoint "
                              "(see training/train_speaker_encoder.py). If omitted, "
                              "falls back to the untrained random projection — "
                              "the known bug from project notes.")
    parser.add_argument("--vocoder_ckpt", default="checkpoints_vocoder/vocoder_best.pt")
    parser.add_argument("--vocoder_config", default="configs/vocoder.yaml")
    parser.add_argument("--skip_speaker_sim", action="store_true",
                         help="Skip the waveform-domain speaker-similarity metric "
                              "(vocodes + re-encodes every sample, so this is the "
                              "slowest part of the script, especially on CPU).")
    parser.add_argument("--speaker_sim_samples", type=int, default=None,
                         help="Cap the number of samples used for the speaker-"
                              "similarity metric (default: all samples in the batch).")
    args = parser.parse_args()

    cfg    = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        print(f"[Warn] Could not enable fully deterministic algorithms: {e}")

    cfg_scale = args.cfg_scale if args.cfg_scale is not None else cfg.inference.cfg_scale
    n_steps   = args.n_steps   if args.n_steps   is not None else cfg.inference.flow_steps

    print(f"\n[Results] Device: {device}")

    if args.fake:
        print("\n" + "!" * 78)
        print("  WARNING: --fake uses synthetic random-noise data.")
        print("  This is a MECHANICAL SANITY CHECK ONLY (shapes/crashes/loading).")
        print("  Stage 2 was trained on real speech statistics — these numbers")
        print("  are NOT a valid quality measurement. Do not use in results/report.")
        print("!" * 78 + "\n")

    encoder = SpeakerEncoder(
        model_name = cfg.speaker_encoder.model_name,
        embed_dim  = cfg.speaker_encoder.embed_dim,
        freeze     = True,
        projection_ckpt = args.projection_ckpt,
    ).to(device)
    encoder.eval()

    mask_model = load_masking(args.mask_ckpt, cfg, device)
    flow_model = load_flow(args.flow_ckpt, cfg, device)

    if args.fake:
        mixture, target, ref_wav, condition, snr_db, valid_frames = make_fake_batch(cfg, device, B=args.batch_size, seed=args.seed)
        tag = "fake_sanity_check"
    else:
        mixture, target, ref_wav, condition, snr_db, valid_frames = get_real_batch(cfg, device, B=args.batch_size, seed=args.seed)
        tag = "real_testclean"

    with torch.no_grad():
        d_vec = encoder(ref_wav)
        stage1_out, mask = mask_model(mixture, d_vec)
        stage2_out = flow_model.inference(
            stage1_out, d_vec, cfg_scale=cfg_scale, n_steps=n_steps,
            cfg_warmup_steps=args.cfg_warmup_steps,
        )

        if args.safety_net:
            stage2_out = apply_safety_net(
                stage1_out, stage2_out, threshold_db=args.safety_net_threshold_db
            )

    print(f"\n  Running full pipeline on {mixture.shape[0]} samples ({tag})...")
    print(f"  Mixture shape   : {tuple(mixture.shape)}")
    print(f"  Stage 1 output  : {tuple(stage1_out.shape)}")
    print(f"  Stage 2 output  : {tuple(stage2_out.shape)}")
    print(f"  Inference steps : {n_steps}  (cfg_scale={cfg_scale}, cfg_warmup_steps={args.cfg_warmup_steps})")

    print(f"\n  Generating plots...")
    plot_idx = min(args.plot_sample, mixture.shape[0] - 1)
    if plot_idx != 0:
        print(f"  Plotting sample index {plot_idx} (0-based) in spectrogram figure")
    plot_pipeline_spectrograms(mixture, stage1_out, stage2_out, target, f"{tag}_s{plot_idx}", plot_idx)
    mix_mse, s1_mse, s2_mse, s1_imp, s2_imp_vs_mix, s2_imp_vs_s1 = \
        plot_pipeline_quality(mixture, stage1_out, stage2_out, target, tag, valid_frames=valid_frames)

    print_summary(mix_mse, s1_mse, s2_mse, s1_imp, s2_imp_vs_mix, s2_imp_vs_s1,
                  args.mask_ckpt, args.flow_ckpt, tag, condition, snr_db)

    if not args.skip_speaker_sim:
        print(f"\n  Computing waveform-domain speaker-similarity metric "
              f"(vocoding + re-encoding each sample)...")
        vc_cfg   = OmegaConf.load(args.vocoder_config)
        vocoder  = load_hifigan_generator(args.vocoder_ckpt, vc_cfg, device)
        sims_mix, sims_tgt, sims_ext = compute_speaker_similarity_eval(
            mixture, stage2_out, target, ref_wav, valid_frames,
            encoder, vocoder, cfg, device, max_samples=args.speaker_sim_samples,
        )
        print_speaker_similarity_summary(sims_mix, sims_tgt, sims_ext)
