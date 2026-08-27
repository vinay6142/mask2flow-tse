"""
Mask2Flow-TSE — Full test-clean evaluation (mel-domain + optional audio-domain),
resumable, incremental-checkpointed.

Everything reported so far (75.8%/71.9% mel-MSE, +0.95 dB SI-SDR) came from a
SINGLE batch of 60 samples — too thin to be a citable thesis headline number.
This script iterates the WHOLE LibriSpeechTSEDataset(seed=...) deterministically
(same seed => same target/interferer/SNR draw per index every run, matching the
convention already used by eval/results_stage2.py's get_real_batch), computes
per-sample mel-domain metrics for every sample, and OPTIONALLY the more
expensive audio-domain metrics (vocoding + SI-SDR) too.

Design for a long CPU run:
  - Per-sample records are appended to a JSONL file as each batch completes
    (--output), not held in memory until the end — a killed/timed-out SLURM
    job still leaves usable partial results.
  - --resume skips any batch whose samples are already fully present in the
    output file, so a second sbatch submission continues rather than restarts.
  - --skip_audio_domain runs the cheap mel-only pass (no vocoding) across the
    FULL dataset quickly; audio-domain (vocoding + SI-SDR) is the expensive
    part on CPU, so run it separately on a --max_samples subsample if a full
    2620-sample audio-domain pass isn't practical time-wise. Recommended
    strategy: full mel-domain pass (fast, all ~2620 samples) + audio-domain
    pass on a few hundred samples (still far more robust than n=60).
  - --summarize_only reads an existing --output file and just prints stats,
    without touching the GPU/CPU model at all — use this to check progress
    on a still-running or already-finished job, or to re-print stats after
    tweaking the catastrophic-threshold flags.

Run (mel-domain only, full dataset, fast):
  python3 eval/full_eval.py \
      --mask_ckpt checkpoints_v2/masking/mask_best.pt \
      --flow_ckpt checkpoints_v2/flow/flow_best.pt \
      --projection_ckpt checkpoints_speaker_encoder/projection_latest.pt \
      --seed 42 --n_steps 4 --skip_audio_domain \
      --output outputs/results/full_eval_mel.jsonl

Run (mel + audio-domain, subsample, slower):
  python3 eval/full_eval.py \
      --mask_ckpt checkpoints_v2/masking/mask_best.pt \
      --flow_ckpt checkpoints_v2/flow/flow_best.pt \
      --projection_ckpt checkpoints_speaker_encoder/projection_latest.pt \
      --seed 42 --n_steps 4 --max_samples 400 \
      --output outputs/results/full_eval_audio_400.jsonl

Resume an interrupted run:
  ... same command, add --resume

Just print stats on what's done so far / already finished:
  python3 eval/full_eval.py --output outputs/results/full_eval_mel.jsonl --summarize_only
"""
import os
import sys
import json
import time
import argparse
import statistics as stats

import torch
from torch.utils.data import DataLoader, Subset
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.mel import make_frame_mask, masked_mse
from data.librispeech import LibriSpeechTSEDataset
from inference.vocoder import load_hifigan_generator, mel_to_audio_hifigan
from models.speaker_encoder import SpeakerEncoder

from eval.results_stage2 import load_masking, load_flow
from eval.audio_domain_quality import si_sdr, waveform_mse

os.makedirs("outputs/results", exist_ok=True)

SNR_BUCKETS = [(1, 3), (3, 5), (5, 7), (7, 10)]
CATASTROPHIC_MEL_PCT = -30.0   # same threshold used in diagnose_catastrophic.py


def build_loader(cfg, seed, batch_size, max_samples):
    dataset = LibriSpeechTSEDataset(
        librispeech_root=cfg.data.librispeech_path,
        splits=["test-clean"],
        cfg=cfg,
        is_train=False,
        seed=seed,
    )
    total = len(dataset)
    if max_samples is not None and max_samples < total:
        dataset = Subset(dataset, range(max_samples))
        print(f"[FullEval] Using first {max_samples}/{total} deterministic indices "
              f"(dataset __getitem__ is seed+idx keyed, not positional, so this "
              f"is as representative as any other subset of the same size).")
    else:
        print(f"[FullEval] Using the full dataset: {total} samples.")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader, len(dataset)


def load_done_indices(output_path):
    done = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done.add(rec["idx"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


@torch.no_grad()
def process_batch(batch, global_idx0, mask_model, flow_model, encoder, vocoder,
                   cfg_scale, n_steps, cfg_warmup_steps, device, compute_audio):
    mixture      = batch["mixture_mel"].to(device)
    target       = batch["target_mel"].to(device)
    ref_wav      = batch["reference_wav"].to(device)
    condition    = batch["condition"]
    snr_db       = batch["snr_db"].tolist()
    valid_frames = batch["target_valid_frames"].to(device)

    d_vec = encoder(ref_wav)
    stage1_out, _ = mask_model(mixture, d_vec)
    stage2_out = flow_model.inference(
        stage1_out, d_vec, cfg_scale=cfg_scale, n_steps=n_steps,
        cfg_warmup_steps=cfg_warmup_steps,
    )

    frame_mask = make_frame_mask(valid_frames, target.shape[-1], target.device)
    B = mixture.shape[0]
    records = []

    for b in range(B):
        fm_b = frame_mask[[b]]
        mix_mse = masked_mse(mixture[[b]], target[[b]], fm_b).item()
        s1_mse  = masked_mse(stage1_out[[b]], target[[b]], fm_b).item()
        s2_mse  = masked_mse(stage2_out[[b]], target[[b]], fm_b).item()

        rec = {
            "idx": global_idx0 + b,
            "condition": condition[b],
            "snr_db": snr_db[b],
            "mel_mix_mse": mix_mse, "mel_s1_mse": s1_mse, "mel_s2_mse": s2_mse,
            "mel_s1_imp_pct": (mix_mse - s1_mse) / max(mix_mse, 1e-12) * 100,
            "mel_s2_vs_mix_pct": (mix_mse - s2_mse) / max(mix_mse, 1e-12) * 100,
            "mel_s2_vs_s1_pct": (s1_mse - s2_mse) / max(s1_mse, 1e-12) * 100,
        }

        if compute_audio:
            vf = max(int(valid_frames[b].item()), 1)
            mix_wav = torch.from_numpy(mel_to_audio_hifigan(mixture[b, :, :vf], vocoder)).float().to(device)
            s1_wav  = torch.from_numpy(mel_to_audio_hifigan(stage1_out[b, :, :vf], vocoder)).float().to(device)
            s2_wav  = torch.from_numpy(mel_to_audio_hifigan(stage2_out[b, :, :vf], vocoder)).float().to(device)
            tgt_wav = torch.from_numpy(mel_to_audio_hifigan(target[b, :, :vf], vocoder)).float().to(device)

            mix_sisdr = si_sdr(mix_wav, tgt_wav)
            s1_sisdr  = si_sdr(s1_wav, tgt_wav)
            s2_sisdr  = si_sdr(s2_wav, tgt_wav)

            rec.update({
                "wav_mix_mse": waveform_mse(mix_wav, tgt_wav),
                "wav_s1_mse": waveform_mse(s1_wav, tgt_wav),
                "wav_s2_mse": waveform_mse(s2_wav, tgt_wav),
                "sisdr_mix": mix_sisdr, "sisdr_s1": s1_sisdr, "sisdr_s2": s2_sisdr,
                "sisdr_gain_vs_mix": s2_sisdr - mix_sisdr,
                "sisdr_gain_vs_s1": s2_sisdr - s1_sisdr,
            })

        records.append(rec)

    return records


def print_summary(records, catastrophic_pct=CATASTROPHIC_MEL_PCT):
    n = len(records)
    if n == 0:
        print("[FullEval] No records to summarize.")
        return

    has_audio = "sisdr_s2" in records[0]

    print("\n" + "=" * 100)
    print(f"  MASK2FLOW-TSE — FULL-SET EVALUATION SUMMARY  (n = {n} samples)")
    print("=" * 100)

    mel_s2_vs_mix = [r["mel_s2_vs_mix_pct"] for r in records]
    mel_s2_vs_s1  = [r["mel_s2_vs_s1_pct"] for r in records]
    n_catastrophic = sum(1 for v in mel_s2_vs_s1 if v < catastrophic_pct)

    print(f"\n  [Mel-domain]  (n={n})")
    print(f"    Median S2 vs Mixture improvement : {stats.median(mel_s2_vs_mix):.1f}%")
    print(f"    Median S2 vs Stage1 improvement  : {stats.median(mel_s2_vs_s1):.1f}%")
    print(f"    Mean   S2 vs Mixture improvement : {stats.mean(mel_s2_vs_mix):.1f}%")
    print(f"    Mean   S2 vs Stage1 improvement  : {stats.mean(mel_s2_vs_s1):.1f}%"
          f"  (mean << median gap = heavy-tailed outliers, expected)")
    print(f"    Stdev  S2 vs Stage1 improvement  : {stats.pstdev(mel_s2_vs_s1):.1f} pp")
    print(f"    Catastrophic samples (S2 vs S1 < {catastrophic_pct:.0f}%): "
          f"{n_catastrophic}/{n} ({100*n_catastrophic/n:.1f}%)")

    if has_audio:
        n_audio = sum(1 for r in records if "sisdr_s2" in r)
        gain_s1 = [r["sisdr_gain_vs_s1"] for r in records if "sisdr_gain_vs_s1" in r]
        gain_mix = [r["sisdr_gain_vs_mix"] for r in records if "sisdr_gain_vs_mix" in r]
        wav_s2_vs_s1 = [
            (r["wav_s1_mse"] - r["wav_s2_mse"]) / max(r["wav_s1_mse"], 1e-12) * 100
            for r in records if "wav_s2_mse" in r
        ]
        n_sisdr_worse = sum(1 for g in gain_s1 if g < 0)
        print(f"\n  [Audio-domain, post-vocoder]  (n={n_audio})")
        print(f"    Median SI-SDR gain vs mixture : {stats.median(gain_mix):+.2f} dB")
        print(f"    Median SI-SDR gain vs Stage 1 : {stats.median(gain_s1):+.2f} dB")
        print(f"    Mean   SI-SDR gain vs Stage 1 : {stats.mean(gain_s1):+.2f} dB")
        print(f"    Stdev  SI-SDR gain vs Stage 1 : {stats.pstdev(gain_s1):.2f} dB")
        print(f"    Median waveform-MSE improvement vs Stage 1 : {stats.median(wav_s2_vs_s1):.1f}%")
        print(f"    Samples where Stage 2 reduced SI-SDR vs Stage 1: "
              f"{n_sisdr_worse}/{n_audio} ({100*n_sisdr_worse/n_audio:.1f}%)")

    print(f"\n  [Breakdown by SNR bucket]  (mel-domain S2-vs-S1)")
    for lo, hi in SNR_BUCKETS:
        bucket_recs = [r for r in records if lo <= r["snr_db"] < hi or
                       (hi == SNR_BUCKETS[-1][1] and lo <= r["snr_db"] <= hi)]
        if not bucket_recs:
            continue
        vals = [r["mel_s2_vs_s1_pct"] for r in bucket_recs]
        n_cat = sum(1 for v in vals if v < catastrophic_pct)
        print(f"    {lo:>2}-{hi:<2}dB  n={len(bucket_recs):<5} "
              f"median S2vsS1={stats.median(vals):>7.1f}%  "
              f"catastrophic={n_cat}/{len(bucket_recs)} ({100*n_cat/len(bucket_recs):.1f}%)")

    worst = sorted(records, key=lambda r: r["mel_s2_vs_s1_pct"])[:10]
    print(f"\n  [Worst 10 by mel S2-vs-S1%]")
    for r in worst:
        extra = f"  SI-SDR gain vs S1={r['sisdr_gain_vs_s1']:+.2f}dB" if "sisdr_gain_vs_s1" in r else ""
        print(f"    idx={r['idx']:<5} SNR={r['snr_db']:>5.2f}dB  "
              f"mel S2vsS1={r['mel_s2_vs_s1_pct']:>9.1f}%{extra}")

    print("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default_v2.yaml")
    parser.add_argument("--mask_ckpt", default=None)
    parser.add_argument("--flow_ckpt", default=None)
    parser.add_argument("--projection_ckpt", default="checkpoints_speaker_encoder/projection_latest.pt")
    parser.add_argument("--vocoder_ckpt", default="checkpoints_vocoder/vocoder_best.pt")
    parser.add_argument("--vocoder_config", default="configs/vocoder.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--max_samples", type=int, default=None,
                         help="Cap total samples evaluated (default: full dataset, ~2620).")
    parser.add_argument("--cfg_scale", type=float, default=None)
    parser.add_argument("--n_steps", type=int, default=None)
    parser.add_argument("--cfg_warmup_steps", type=int, default=0)
    parser.add_argument("--skip_audio_domain", action="store_true",
                         help="Skip vocoding + SI-SDR (the expensive part on CPU). "
                              "Recommended for a full ~2620-sample pass.")
    parser.add_argument("--output", default="outputs/results/full_eval.jsonl")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize_only", action="store_true",
                         help="Skip everything model-related; just read --output and print stats.")
    parser.add_argument("--catastrophic_pct", type=float, default=CATASTROPHIC_MEL_PCT)
    args = parser.parse_args()

    if args.summarize_only:
        records = []
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        print(f"[FullEval] Loaded {len(records)} records from {args.output}")
        print_summary(records, catastrophic_pct=args.catastrophic_pct)
        sys.exit(0)

    assert args.mask_ckpt and args.flow_ckpt, "--mask_ckpt and --flow_ckpt required unless --summarize_only"

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    cfg_scale = args.cfg_scale if args.cfg_scale is not None else cfg.inference.cfg_scale
    n_steps = args.n_steps if args.n_steps is not None else cfg.inference.flow_steps

    print(f"[FullEval] Device: {device}  cfg_scale={cfg_scale}  n_steps={n_steps}"
          f"  audio_domain={'off' if args.skip_audio_domain else 'on'}")

    encoder = SpeakerEncoder(
        model_name=cfg.speaker_encoder.model_name,
        embed_dim=cfg.speaker_encoder.embed_dim,
        freeze=True,
        projection_ckpt=args.projection_ckpt,
    ).to(device)
    encoder.eval()

    mask_model = load_masking(args.mask_ckpt, cfg, device)
    flow_model = load_flow(args.flow_ckpt, cfg, device)

    vocoder = None
    if not args.skip_audio_domain:
        vc_cfg = OmegaConf.load(args.vocoder_config)
        vocoder = load_hifigan_generator(args.vocoder_ckpt, vc_cfg, device)

    loader, total_n = build_loader(cfg, args.seed, args.batch_size, args.max_samples)

    done_indices = set()
    out_mode = "a"
    if args.resume:
        done_indices = load_done_indices(args.output)
        print(f"[FullEval] Resuming: {len(done_indices)} samples already recorded in {args.output}")
    elif os.path.exists(args.output):
        print(f"[FullEval] {args.output} already exists and --resume not set — "
              f"appending anyway (pass --resume to skip already-done samples, "
              f"or delete the file first for a clean run).")

    all_records = []
    if done_indices:
        all_records = [json.loads(l) for l in open(args.output, "r", encoding="utf-8") if l.strip()]

    t_start = time.time()
    n_processed_this_run = 0

    with open(args.output, out_mode, encoding="utf-8") as f_out:
        for batch_i, batch in enumerate(loader):
            batch_size_actual = batch["mixture_mel"].shape[0]
            global_idx0 = batch_i * args.batch_size
            batch_indices = set(range(global_idx0, global_idx0 + batch_size_actual))

            if batch_indices.issubset(done_indices):
                continue  # whole batch already done, skip re-running the model

            records = process_batch(
                batch, global_idx0, mask_model, flow_model, encoder, vocoder,
                cfg_scale, n_steps, args.cfg_warmup_steps, device,
                compute_audio=not args.skip_audio_domain,
            )

            for rec in records:
                if rec["idx"] in done_indices:
                    continue
                f_out.write(json.dumps(rec) + "\n")
                all_records.append(rec)
                n_processed_this_run += 1
            f_out.flush()

            done_so_far = len(done_indices) + n_processed_this_run
            if batch_i % 5 == 0 or done_so_far >= total_n:
                elapsed = time.time() - t_start
                rate = n_processed_this_run / max(elapsed, 1e-6)
                remaining = total_n - done_so_far
                eta_min = (remaining / rate / 60) if rate > 0 else float("nan")
                print(f"[FullEval] {done_so_far}/{total_n} samples done "
                      f"({elapsed/60:.1f} min elapsed this run, "
                      f"~{eta_min:.0f} min remaining at current rate)")

    print(f"\n[FullEval] Done. {len(all_records)} total records in {args.output}")
    print_summary(all_records, catastrophic_pct=args.catastrophic_pct)
