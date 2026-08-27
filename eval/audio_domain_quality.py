"""
Mask2Flow-TSE — Audio-domain quality check: MSE and SI-SDR after vocoding.

Everything reported so far (median 75.8%/71.9% improvement, the catastrophic
outlier investigation in eval/diagnose_catastrophic.py and
eval/KNOWN_LIMITATIONS.md) is measured in the MEL-SPECTROGRAM domain
(log-mel MSE) — the space the models are trained in, not what a listener
actually hears. This script vocodes mixture, Stage 1 output, Stage 2 output,
and target through the trained HiFi-GAN and measures quality on the actual
reconstructed AUDIO WAVEFORM, across the same three pipeline stages.

Two metrics, for different reasons:
  - Waveform MSE: matches the metric already used throughout this project,
    for direct comparability. But it's a weak metric for audio on its own —
    a small sample-shift or amplitude-scale difference between two
    PERCEPTUALLY IDENTICAL signals can produce a large MSE. Don't over-read
    small MSE differences here.
  - SI-SDR (scale-invariant signal-to-distortion ratio, Le Roux et al.
    2019, "SDR - half-baked or well done?"): the standard metric in the
    source-separation/speech-enhancement literature. Robust to a constant
    amplitude-scale mismatch between estimate and target, which vocoded
    output can have (Stage 1/2 restore log-mel LEVEL, not a
    correctly-scaled waveform amplitude). Treat this as the primary
    audio-domain number; waveform MSE as secondary/for comparability only.

Both vocoded outputs for a given sample are naturally time-aligned:
mel_to_audio_hifigan() always returns exactly T * hop_length samples for a
(n_mels, T) input, and all four signals (mixture/stage1/stage2/target) here
are trimmed to the same valid_frames length T before vocoding — no DTW/
cross-correlation alignment needed.

Also cross-references each sample against the known mel-domain
MSE-catastrophic outliers (samples 3, 20, 27, 29, 30, 32, 40, 46 — see
eval/KNOWN_LIMITATIONS.md) to check whether that failure survives vocoding
into an audible/measurable audio-domain failure, or whether HiFi-GAN
partially smooths it out.

Run:
  python3 eval/audio_domain_quality.py \
      --mask_ckpt checkpoints_v2/masking/mask_best.pt \
      --flow_ckpt checkpoints_v2/flow/flow_best.pt \
      --projection_ckpt checkpoints_speaker_encoder/projection_latest.pt \
      --batch_size 60 --seed 42 --n_steps 4
"""
import os
import sys
import argparse
import statistics as stats

import torch
import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.vocoder import load_hifigan_generator, mel_to_audio_hifigan
from models.speaker_encoder import SpeakerEncoder

from eval.results_stage2 import load_masking, load_flow, get_real_batch

os.makedirs("outputs/results", exist_ok=True)

# Known mel-domain catastrophic outliers, from eval_newproj_10555 / KNOWN_LIMITATIONS.md
# (0-based indices for samples 3, 20, 27, 29, 30, 32, 40, 46)
KNOWN_CATASTROPHIC = {2, 19, 26, 28, 29, 31, 39, 45}


def si_sdr(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Scale-invariant SDR (Le Roux et al. 2019), in dB. Higher is better.
    estimate, target: 1D waveforms, same length.
    """
    target = target - target.mean()
    estimate = estimate - estimate.mean()
    target_energy = torch.sum(target ** 2) + eps
    optimal_scale = torch.sum(target * estimate) / target_energy
    projection = optimal_scale * target
    noise = estimate - projection
    ratio = (torch.sum(projection ** 2) + eps) / (torch.sum(noise ** 2) + eps)
    return (10 * torch.log10(ratio)).item()


def waveform_mse(estimate: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean((estimate - target) ** 2).item()


@torch.no_grad()
def compute_audio_domain_quality(mixture, stage1_out, stage2_out, target,
                                  valid_frames, vocoder, device, max_samples=None):
    B = mixture.shape[0]
    n = B if max_samples is None else min(B, max_samples)

    rows = []
    for b in range(n):
        vf = max(int(valid_frames[b].item()), 1)

        mix_wav = torch.from_numpy(mel_to_audio_hifigan(mixture[b, :, :vf], vocoder)).float().to(device)
        s1_wav  = torch.from_numpy(mel_to_audio_hifigan(stage1_out[b, :, :vf], vocoder)).float().to(device)
        s2_wav  = torch.from_numpy(mel_to_audio_hifigan(stage2_out[b, :, :vf], vocoder)).float().to(device)
        tgt_wav = torch.from_numpy(mel_to_audio_hifigan(target[b, :, :vf], vocoder)).float().to(device)

        mix_mse = waveform_mse(mix_wav, tgt_wav)
        s1_mse  = waveform_mse(s1_wav, tgt_wav)
        s2_mse  = waveform_mse(s2_wav, tgt_wav)

        mix_sisdr = si_sdr(mix_wav, tgt_wav)
        s1_sisdr  = si_sdr(s1_wav, tgt_wav)
        s2_sisdr  = si_sdr(s2_wav, tgt_wav)

        rows.append({
            "idx": b,
            "mix_mse": mix_mse, "s1_mse": s1_mse, "s2_mse": s2_mse,
            "s1_imp_pct": (mix_mse - s1_mse) / max(mix_mse, 1e-12) * 100,
            "s2_vs_mix_pct": (mix_mse - s2_mse) / max(mix_mse, 1e-12) * 100,
            "s2_vs_s1_pct": (s1_mse - s2_mse) / max(s1_mse, 1e-12) * 100,
            "mix_sisdr": mix_sisdr, "s1_sisdr": s1_sisdr, "s2_sisdr": s2_sisdr,
            "s1_sisdr_gain": s1_sisdr - mix_sisdr,
            "s2_sisdr_gain_vs_mix": s2_sisdr - mix_sisdr,
            "s2_sisdr_gain_vs_s1": s2_sisdr - s1_sisdr,
        })

        if (b + 1) % 10 == 0:
            print(f"    ...audio-domain quality: {b + 1}/{n} samples done")

    return rows


def print_summary(rows):
    B = len(rows)
    print("\n" + "=" * 118)
    print("  MASK2FLOW-TSE — AUDIO-DOMAIN QUALITY (post-vocoder waveform, not mel-domain)")
    print("=" * 118)
    print(f"\n  {'Sample':<9} {'Mix MSE':>10} {'S1 MSE':>10} {'S2 MSE':>10} {'S2vsS1%':>9}"
          f"   {'Mix SI-SDR':>11} {'S1 SI-SDR':>10} {'S2 SI-SDR':>10} {'S2vsS1 dB':>10}  flag")
    print(f"  {'-'*116}")
    for r in rows:
        flag = "  <-- known mel-catastrophic" if r["idx"] in KNOWN_CATASTROPHIC else ""
        print(f"  Sample {r['idx']+1:<3} {r['mix_mse']:>10.5f} {r['s1_mse']:>10.5f} {r['s2_mse']:>10.5f}"
              f" {r['s2_vs_s1_pct']:>8.1f}%   {r['mix_sisdr']:>11.2f} {r['s1_sisdr']:>10.2f}"
              f" {r['s2_sisdr']:>10.2f} {r['s2_sisdr_gain_vs_s1']:>+10.2f}{flag}")

    mix_mse_all = [r["mix_mse"] for r in rows]
    s1_mse_all  = [r["s1_mse"] for r in rows]
    s2_mse_all  = [r["s2_mse"] for r in rows]
    s2_vs_s1_all = [r["s2_vs_s1_pct"] for r in rows]
    s2_vs_mix_all = [r["s2_vs_mix_pct"] for r in rows]

    mix_sisdr_all = [r["mix_sisdr"] for r in rows]
    s1_sisdr_all  = [r["s1_sisdr"] for r in rows]
    s2_sisdr_all  = [r["s2_sisdr"] for r in rows]
    s2_gain_s1_all = [r["s2_sisdr_gain_vs_s1"] for r in rows]
    s2_gain_mix_all = [r["s2_sisdr_gain_vs_mix"] for r in rows]

    print(f"  {'-'*116}")
    print(f"  {'Mean':<9} {stats.mean(mix_mse_all):>10.5f} {stats.mean(s1_mse_all):>10.5f}"
          f" {stats.mean(s2_mse_all):>10.5f} {stats.mean(s2_vs_s1_all):>8.1f}%   "
          f"{stats.mean(mix_sisdr_all):>11.2f} {stats.mean(s1_sisdr_all):>10.2f}"
          f" {stats.mean(s2_sisdr_all):>10.2f} {stats.mean(s2_gain_s1_all):>+10.2f}")
    print(f"  {'Median':<9} {stats.median(mix_mse_all):>10.5f} {stats.median(s1_mse_all):>10.5f}"
          f" {stats.median(s2_mse_all):>10.5f} {stats.median(s2_vs_s1_all):>8.1f}%   "
          f"{stats.median(mix_sisdr_all):>11.2f} {stats.median(s1_sisdr_all):>10.2f}"
          f" {stats.median(s2_sisdr_all):>10.2f} {stats.median(s2_gain_s1_all):>+10.2f}")

    print(f"\n  Key findings (audio domain, post-vocoder):")
    print(f"    Median waveform-MSE improvement, Stage 2 vs mixture : {stats.median(s2_vs_mix_all):.1f}%")
    print(f"    Median waveform-MSE improvement, Stage 2 vs Stage 1 : {stats.median(s2_vs_s1_all):.1f}%")
    print(f"    Median SI-SDR gain, Stage 2 vs mixture   : {stats.median(s2_gain_mix_all):+.2f} dB")
    print(f"    Median SI-SDR gain, Stage 2 vs Stage 1   : {stats.median(s2_gain_s1_all):+.2f} dB")
    print(f"    Mean   SI-SDR gain, Stage 2 vs Stage 1   : {stats.mean(s2_gain_s1_all):+.2f} dB"
          f"  (mean vs median gap here is the outlier signature, same as mel-domain)")

    known_present = [r for r in rows if r["idx"] in KNOWN_CATASTROPHIC]
    if known_present:
        print(f"\n  Known mel-domain catastrophic outliers, audio-domain outcome:")
        for r in known_present:
            sisdr_verdict = "still bad" if r["s2_sisdr_gain_vs_s1"] < -1.0 else \
                            ("smoothed out" if r["s2_sisdr_gain_vs_s1"] > 0 else "mild")
            print(f"    Sample {r['idx']+1:<3} mel-MSE was catastrophic -> audio SI-SDR gain vs S1 = "
                  f"{r['s2_sisdr_gain_vs_s1']:+.2f} dB ({sisdr_verdict}), "
                  f"waveform-MSE S2vsS1 = {r['s2_vs_s1_pct']:+.1f}%")

    n_sisdr_worse = sum(1 for r in rows if r["s2_sisdr_gain_vs_s1"] < 0)
    print(f"\n  Samples where Stage 2 REDUCED SI-SDR vs Stage 1 alone: "
          f"{n_sisdr_worse}/{B} ({100*n_sisdr_worse/B:.1f}%)")
    print("=" * 118)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default_v2.yaml")
    parser.add_argument("--mask_ckpt", required=True)
    parser.add_argument("--flow_ckpt", required=True)
    parser.add_argument("--projection_ckpt", default="checkpoints_speaker_encoder/projection_latest.pt")
    parser.add_argument("--vocoder_ckpt", default="checkpoints_vocoder/vocoder_best.pt")
    parser.add_argument("--vocoder_config", default="configs/vocoder.yaml")
    parser.add_argument("--batch_size", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg_scale", type=float, default=None)
    parser.add_argument("--n_steps", type=int, default=None)
    parser.add_argument("--cfg_warmup_steps", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    cfg_scale = args.cfg_scale if args.cfg_scale is not None else cfg.inference.cfg_scale
    n_steps = args.n_steps if args.n_steps is not None else cfg.inference.flow_steps

    print(f"[AudioDomain] Device: {device}  cfg_scale={cfg_scale}  n_steps={n_steps}"
          f"  cfg_warmup_steps={args.cfg_warmup_steps}")

    encoder = SpeakerEncoder(
        model_name=cfg.speaker_encoder.model_name,
        embed_dim=cfg.speaker_encoder.embed_dim,
        freeze=True,
        projection_ckpt=args.projection_ckpt,
    ).to(device)
    encoder.eval()

    mask_model = load_masking(args.mask_ckpt, cfg, device)
    flow_model = load_flow(args.flow_ckpt, cfg, device)

    vc_cfg = OmegaConf.load(args.vocoder_config)
    vocoder = load_hifigan_generator(args.vocoder_ckpt, vc_cfg, device)

    mixture, target, ref_wav, condition, snr_db, valid_frames = get_real_batch(
        cfg, device, B=args.batch_size, seed=args.seed
    )

    with torch.no_grad():
        d_vec = encoder(ref_wav)
        stage1_out, _ = mask_model(mixture, d_vec)
        stage2_out = flow_model.inference(
            stage1_out, d_vec, cfg_scale=cfg_scale, n_steps=n_steps,
            cfg_warmup_steps=args.cfg_warmup_steps,
        )

    print(f"\n  Vocoding {mixture.shape[0]} samples x 4 signals (mixture/Stage1/Stage2/target) "
          f"and computing audio-domain metrics...")
    rows = compute_audio_domain_quality(
        mixture, stage1_out, stage2_out, target, valid_frames, vocoder, device,
        max_samples=args.max_samples,
    )
    print_summary(rows)
