"""
Mask2Flow-TSE -- Libri2Mix cross-corpus evaluation.

Runs the trained two-stage pipeline (masking + flow matching) on the
Libri2Mix benchmark (JorisCos/LibriMix, 2-speaker "mix_clean" condition)
instead of this project's own synthetic on-the-fly LibriSpeech mixing --
a second, independently-built corpus for the same headline metrics
(mel-MSE improvement, catastrophic-outlier rate, SI-SDR gain), so the
existing test-clean numbers in eval/KNOWN_LIMITATIONS.md aren't the only
evidence for the thesis.

How this differs from eval/full_eval.py:
  - Libri2Mix mixtures have REAL variable length (not this project's
    fixed cfg.audio.segment_length training window). This reuses the
    SAME chunked overlap-add inference path added to inference/infer.py
    (_chunk_starts / _overlap_add) instead of a single fixed-size pass --
    the real-world chunking code, exercised end-to-end for the first
    time on genuinely variable-length input from an external corpus,
    rather than a resynthesized one.
  - Libri2Mix's generated metadata doesn't include a separate speaker-
    enrollment clip (its source_i wav files ARE the exact utterances
    mixed in). For each mixture this looks up a DIFFERENT utterance by
    the same speaker from the original LibriSpeech test-clean directory
    to use as the reference/enrollment clip, matching the
    reference != target-utterance convention used everywhere else in
    this project (data/librispeech.py, inference/infer.py).
  - By default evaluates BOTH directions per mixture (source_1 as
    target, then source_2 as target) -- doubling n for free from the
    same generated mixture set.

Needs two metadata CSVs per test mixture:
  - data/raw/LibriMix/metadata/Libri2Mix/libri2mix_test-clean.csv
    (shipped with the LibriMix repo -- ORIGINAL LibriSpeech-relative
    source paths, used here only to find each target's speaker ID and
    which exact utterance to exclude when picking an enrollment clip)
  - <librimix_dir>/metadata/mixture_test_mix_clean.csv
    (written by LibriMix's own generation script -- absolute paths to
    the actually-generated mixture/s1/s2 wav files, already resampled/
    gain-matched/length-fit; this is what actually gets loaded)
    NOTE: LibriMix writes metadata/ as a SIBLING of the split dirs
    (test/, dev/, train-100/, ...), not nested inside them -- i.e.
    Libri2Mix/wav16k/<mode>/metadata/, not .../wav16k/<mode>/test/metadata/.
    --librimix_dir must point at the wav16k/<mode> level (the parent of
    test/), even though this script never reads wav files by walking
    that directory itself (their paths come straight out of the CSV).

Run:
  python3 eval/eval_libri2mix.py \
      --mask_ckpt checkpoints_v2/masking/mask_best.pt \
      --flow_ckpt checkpoints_v2/flow/flow_best.pt \
      --projection_ckpt checkpoints_speaker_encoder/projection_latest.pt \
      --librimix_dir data/raw/LibriMix_storage/Libri2Mix/wav16k/min \
      --librimix_input_csv data/raw/LibriMix/metadata/Libri2Mix/libri2mix_test-clean.csv \
      --librispeech_dir data/raw/LibriSpeech \
      --output outputs/results/eval_libri2mix_min.jsonl

Resume an interrupted run:
  ... same command, add --resume

Just print stats on what is done so far / already finished:
  python3 eval/eval_libri2mix.py --output outputs/results/eval_libri2mix_min.jsonl --summarize_only
"""
import os
import sys
import csv
import json
import glob
import time
import argparse
import statistics as stats

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.mel import MelSpectrogramExtractor, pad_or_trim, segment_waveform, masked_mse
from data.augment import load_audio, trim_trailing_silence
from inference.infer import _chunk_starts, _overlap_add
from inference.vocoder import load_hifigan_generator, mel_to_audio_hifigan
from models.speaker_encoder import SpeakerEncoder

from eval.results_stage2 import load_masking, load_flow
from eval.audio_domain_quality import si_sdr, waveform_mse

os.makedirs("outputs/results", exist_ok=True)

CATASTROPHIC_MEL_PCT = -30.0   # same threshold used everywhere else in eval/


# ── metadata plumbing ─────────────────────────────────────────────────────

def load_orig_source_paths(input_csv):
    """
    mixture_ID -> {1: 'test-clean/spk/chap/file.flac', 2: '...'} using the
    ORIGINAL LibriSpeech-relative paths from the shipped input metadata
    (NOT the generated output metadata, whose source_i_path columns point
    at Libri2Mix's own resampled/gain-adjusted derivative wav files).
    """
    lookup = {}
    with open(input_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lookup[row["mixture_ID"]] = {
                1: row["source_1_path"],
                2: row["source_2_path"],
            }
    return lookup


def load_generated_mixtures(output_csv):
    with open(output_csv, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_metrics_snr(metrics_csv):
    """Optional: mixture_ID -> {1: source_1_SNR, 2: source_2_SNR}, if present."""
    if not os.path.exists(metrics_csv):
        return {}
    lookup = {}
    with open(metrics_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lookup[row["mixture_ID"]] = {
                1: float(row["source_1_SNR"]),
                2: float(row["source_2_SNR"]),
            }
    return lookup


_speaker_utt_cache = {}


def find_enrollment_path(librispeech_dir, orig_rel_path):
    """
    A DIFFERENT utterance by the same speaker as orig_rel_path (which is
    the exact utterance mixed into this Libri2Mix sample), for use as the
    speaker-encoder's enrollment/reference clip. Returns None if the
    speaker has only this one utterance in the split (shouldn't happen
    for test-clean, which has dozens of utterances per speaker).
    """
    parts = orig_rel_path.replace("\\", "/").split("/")
    split, speaker_id = parts[0], parts[-3]
    key = (split, speaker_id)
    if key not in _speaker_utt_cache:
        pattern = os.path.join(librispeech_dir, split, speaker_id, "**", "*.flac")
        _speaker_utt_cache[key] = sorted(glob.glob(pattern, recursive=True))
    candidates = _speaker_utt_cache[key]
    exclude_abs = os.path.normpath(os.path.join(librispeech_dir, orig_rel_path))
    others = [c for c in candidates if os.path.normpath(c) != exclude_abs]
    return others[0] if others else None


def build_tasks(generated_rows, orig_lookup, both_directions):
    """Yield (sample_id, row, target_idx) -- one or two per generated mixture."""
    for row in generated_rows:
        mid = row["mixture_ID"]
        if mid not in orig_lookup:
            continue  # generation covered a mixture not in this input csv -- skip
        yield (f"{mid}__t1", row, 1)
        if both_directions:
            yield (f"{mid}__t2", row, 2)


# ── inference (reuses inference/infer.py's chunking exactly) ──────────────

@torch.no_grad()
def run_chunked_pipeline(mixture_wav, ref_wav, mel_extractor, mask_model, flow_model,
                          encoder, chunk_frames, overlap_frames, cfg_scale, n_steps, device):
    """
    Same per-chunk masking + flow-matching + overlap-add reassembly as
    inference/infer.py's extract_target_speaker(), factored out here so
    both stage1 and stage2 full-length mels are returned (infer.py only
    keeps stage2, since it's building a file to save, not comparing
    against a stage1 baseline).
    """
    mixture_mel_full = mel_extractor(mixture_wav.to(device))
    total_frames = mixture_mel_full.shape[-1]
    starts = _chunk_starts(total_frames, chunk_frames, overlap_frames)

    d_vec = encoder(ref_wav.unsqueeze(0).to(device))

    stage1_chunks, stage2_chunks = [], []
    for start in starts:
        end = start + chunk_frames
        chunk = mixture_mel_full[:, start:end]
        if chunk.shape[-1] < chunk_frames:
            chunk = pad_or_trim(chunk, chunk_frames, pad_value=-11.5)
        chunk = chunk.unsqueeze(0)

        s1_out, _ = mask_model(chunk, d_vec)
        s2_out = flow_model.inference(s1_out, d_vec, cfg_scale=cfg_scale, n_steps=n_steps)
        stage1_chunks.append(s1_out[0])
        stage2_chunks.append(s2_out[0])

    if len(starts) == 1:
        stage1_final = stage1_chunks[0][:, :total_frames]
        stage2_final = stage2_chunks[0][:, :total_frames]
    else:
        stage1_final = _overlap_add(stage1_chunks, starts, chunk_frames, total_frames, overlap_frames, device)
        stage2_final = _overlap_add(stage2_chunks, starts, chunk_frames, total_frames, overlap_frames, device)

    return mixture_mel_full, stage1_final, stage2_final, d_vec, len(starts)


def process_one(sample_id, row, target_idx, orig_lookup, librispeech_dir, sr, ref_seg_len,
                 mel_extractor, mask_model, flow_model, encoder, vocoder,
                 chunk_frames, overlap_frames, cfg_scale, n_steps, device, compute_audio,
                 snr_lookup):
    mid = row["mixture_ID"]
    orig_paths = orig_lookup[mid]
    target_orig_rel = orig_paths[target_idx]

    enroll_path = find_enrollment_path(librispeech_dir, target_orig_rel)
    if enroll_path is None:
        return None  # speaker has no other utterance in this split -- skip, not an error

    mixture_wav = load_audio(row["mixture_path"], sr)
    target_wav = load_audio(row[f"source_{target_idx}_path"], sr)
    ref_wav = segment_waveform(load_audio(enroll_path, sr), sr, ref_seg_len, random_start=False)
    # Trim trailing silence off the reference clip -- segment_waveform zero-pads
    # any enrollment utterance shorter than ref_seg_len (3.0s), and that padding
    # dilutes the mean-pooled WavLM embedding used for speaker conditioning.
    # Same fix already applied in eval_multi_speaker.py / results_stage2.py; see
    # trim_trailing_silence()'s own docstring in data/augment.py.
    ref_wav = trim_trailing_silence(ref_wav, sr)

    mixture_mel, s1_mel, s2_mel, d_vec, n_chunks = run_chunked_pipeline(
        mixture_wav, ref_wav, mel_extractor, mask_model, flow_model, encoder,
        chunk_frames, overlap_frames, cfg_scale, n_steps, device,
    )
    target_mel = mel_extractor(target_wav.to(device))

    # Mixture/target should already be sample-exact same length (LibriMix's
    # own fit_lengths() reshapes every source + the mixture together at
    # generation time) -- trim defensively to the shorter in case of a
    # stray off-by-one STFT frame.
    T = min(mixture_mel.shape[-1], target_mel.shape[-1], s1_mel.shape[-1], s2_mel.shape[-1])
    mixture_mel, target_mel, s1_mel, s2_mel = (
        mixture_mel[:, :T], target_mel[:, :T], s1_mel[:, :T], s2_mel[:, :T]
    )

    mix_mse = masked_mse(mixture_mel.unsqueeze(0), target_mel.unsqueeze(0)).item()
    s1_mse = masked_mse(s1_mel.unsqueeze(0), target_mel.unsqueeze(0)).item()
    s2_mse = masked_mse(s2_mel.unsqueeze(0), target_mel.unsqueeze(0)).item()

    rec = {
        "sample_id": sample_id,
        "mixture_id": mid,
        "target_idx": target_idx,
        "n_chunks": n_chunks,
        "length_sec": T * mel_extractor.hop_length / sr,
        "mel_mix_mse": mix_mse, "mel_s1_mse": s1_mse, "mel_s2_mse": s2_mse,
        "mel_s1_imp_pct": (mix_mse - s1_mse) / max(mix_mse, 1e-12) * 100,
        "mel_s2_vs_mix_pct": (mix_mse - s2_mse) / max(mix_mse, 1e-12) * 100,
        "mel_s2_vs_s1_pct": (s1_mse - s2_mse) / max(s1_mse, 1e-12) * 100,
    }
    if mid in snr_lookup:
        rec["snr_db"] = snr_lookup[mid][target_idx]

    if compute_audio:
        mix_wav = torch.from_numpy(mel_to_audio_hifigan(mixture_mel, vocoder)).float().to(device)
        s1_wav = torch.from_numpy(mel_to_audio_hifigan(s1_mel, vocoder)).float().to(device)
        s2_wav = torch.from_numpy(mel_to_audio_hifigan(s2_mel, vocoder)).float().to(device)
        tgt_wav = torch.from_numpy(mel_to_audio_hifigan(target_mel, vocoder)).float().to(device)

        mix_sisdr, s1_sisdr, s2_sisdr = si_sdr(mix_wav, tgt_wav), si_sdr(s1_wav, tgt_wav), si_sdr(s2_wav, tgt_wav)
        rec.update({
            "wav_mix_mse": waveform_mse(mix_wav, tgt_wav),
            "wav_s1_mse": waveform_mse(s1_wav, tgt_wav),
            "wav_s2_mse": waveform_mse(s2_wav, tgt_wav),
            "sisdr_mix": mix_sisdr, "sisdr_s1": s1_sisdr, "sisdr_s2": s2_sisdr,
            "sisdr_gain_vs_mix": s2_sisdr - mix_sisdr,
            "sisdr_gain_vs_s1": s2_sisdr - s1_sisdr,
        })

        emb = encoder(s2_wav.unsqueeze(0))[0]
        rec["verify_s2_genuine_sim"] = F.cosine_similarity(emb, d_vec[0], dim=0).item()

    return rec


# ── resumable JSONL driver + summary (same pattern as eval/full_eval.py) ──

def load_done_ids(output_path):
    done = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["sample_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def print_summary(records, catastrophic_pct=CATASTROPHIC_MEL_PCT):
    n = len(records)
    if n == 0:
        print("[EvalLibri2Mix] No records to summarize.")
        return

    has_audio = "sisdr_s2" in records[0]
    n_mix = len(set(r["mixture_id"] for r in records))

    print("\n" + "=" * 100)
    print(f"  MASK2FLOW-TSE -- LIBRI2MIX CROSS-CORPUS EVALUATION  (n = {n} samples, {n_mix} mixtures)")
    print("=" * 100)

    mel_s2_vs_mix = [r["mel_s2_vs_mix_pct"] for r in records]
    mel_s2_vs_s1 = [r["mel_s2_vs_s1_pct"] for r in records]
    n_catastrophic = sum(1 for v in mel_s2_vs_s1 if v < catastrophic_pct)

    print(f"\n  [Mel-domain]  (n={n})")
    print(f"    Median S2 vs Mixture improvement : {stats.median(mel_s2_vs_mix):.1f}%")
    print(f"    Median S2 vs Stage1 improvement  : {stats.median(mel_s2_vs_s1):.1f}%")
    print(f"    Mean   S2 vs Stage1 improvement  : {stats.mean(mel_s2_vs_s1):.1f}%")
    print(f"    Catastrophic samples (S2 vs S1 < {catastrophic_pct:.0f}%): "
          f"{n_catastrophic}/{n} ({100 * n_catastrophic / n:.1f}%)")

    if has_audio:
        n_audio = sum(1 for r in records if "sisdr_s2" in r)
        gain_s1 = [r["sisdr_gain_vs_s1"] for r in records if "sisdr_gain_vs_s1" in r]
        gain_mix = [r["sisdr_gain_vs_mix"] for r in records if "sisdr_gain_vs_mix" in r]
        n_sisdr_worse = sum(1 for g in gain_s1 if g < 0)
        print(f"\n  [Audio-domain, post-vocoder]  (n={n_audio})")
        print(f"    Median SI-SDR gain vs mixture : {stats.median(gain_mix):+.2f} dB")
        print(f"    Median SI-SDR gain vs Stage 1 : {stats.median(gain_s1):+.2f} dB")
        print(f"    Samples where Stage 2 reduced SI-SDR vs Stage 1: "
              f"{n_sisdr_worse}/{n_audio} ({100 * n_sisdr_worse / n_audio:.1f}%)")

    n_multi_chunk = sum(1 for r in records if r["n_chunks"] > 1)
    print(f"\n  [Length] median {stats.median(r['length_sec'] for r in records):.1f}s, "
          f"{n_multi_chunk}/{n} samples needed multi-chunk overlap-add "
          f"(> {records[0]['n_chunks'] and 'model training window'})")

    worst = sorted(records, key=lambda r: r["mel_s2_vs_s1_pct"])[:10]
    print(f"\n  [Worst 10 by mel S2-vs-S1%]")
    for r in worst:
        extra = f"  SI-SDR gain vs S1={r['sisdr_gain_vs_s1']:+.2f}dB" if "sisdr_gain_vs_s1" in r else ""
        print(f"    {r['sample_id']:<40} mel S2vsS1={r['mel_s2_vs_s1_pct']:>9.1f}%{extra}")

    print("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default_v2.yaml")
    parser.add_argument("--mask_ckpt", default=None)
    parser.add_argument("--flow_ckpt", default=None)
    parser.add_argument("--projection_ckpt", default="checkpoints_speaker_encoder/projection_latest.pt")
    parser.add_argument("--vocoder_ckpt", default="checkpoints_vocoder/vocoder_best.pt")
    parser.add_argument("--vocoder_config", default="configs/vocoder.yaml")
    parser.add_argument("--librimix_dir", default=None,
                         help="e.g. data/raw/LibriMix_storage/Libri2Mix/wav16k/min -- the "
                              "wav16k/<mode> level, NOT .../wav16k/<mode>/test (LibriMix "
                              "writes metadata/ as a sibling of test/, not nested inside "
                              "it). Must contain metadata/mixture_test_mix_clean.csv.")
    parser.add_argument("--librimix_input_csv",
                         default="data/raw/LibriMix/metadata/Libri2Mix/libri2mix_test-clean.csv")
    parser.add_argument("--librispeech_dir", default="data/raw/LibriSpeech")
    parser.add_argument("--both_directions", action="store_true", default=True)
    parser.add_argument("--single_direction", dest="both_directions", action="store_false",
                         help="Only evaluate source_1 as target (default: both source_1 and "
                              "source_2, doubling n from the same mixture set).")
    parser.add_argument("--overlap_sec", type=float, default=2.0)
    parser.add_argument("--cfg_scale", type=float, default=None)
    parser.add_argument("--n_steps", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--skip_audio_domain", action="store_true",
                         help="Skip vocoding + SI-SDR (the expensive part on CPU).")
    parser.add_argument("--output", default="outputs/results/eval_libri2mix.jsonl")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize_only", action="store_true")
    parser.add_argument("--catastrophic_pct", type=float, default=CATASTROPHIC_MEL_PCT)
    args = parser.parse_args()

    if args.summarize_only:
        records = []
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        print(f"[EvalLibri2Mix] Loaded {len(records)} records from {args.output}")
        print_summary(records, catastrophic_pct=args.catastrophic_pct)
        sys.exit(0)

    assert args.mask_ckpt and args.flow_ckpt and args.librimix_dir, \
        "--mask_ckpt, --flow_ckpt, --librimix_dir required unless --summarize_only"

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sr = cfg.audio.sample_rate

    cfg_scale = args.cfg_scale if args.cfg_scale is not None else cfg.inference.cfg_scale
    n_steps = args.n_steps if args.n_steps is not None else cfg.inference.flow_steps
    ref_seg_len = getattr(cfg.audio, "reference_length", 3.0)

    print(f"[EvalLibri2Mix] Device: {device}  cfg_scale={cfg_scale}  n_steps={n_steps}  "
          f"both_directions={args.both_directions}  audio_domain={'off' if args.skip_audio_domain else 'on'}")

    mel_extractor = MelSpectrogramExtractor(
        sample_rate=sr, n_mels=cfg.mel.n_mels, n_fft=cfg.mel.n_fft,
        hop_length=cfg.mel.hop_length, win_length=cfg.mel.win_length,
        f_min=cfg.mel.f_min, f_max=cfg.mel.f_max, log_offset=cfg.mel.log_offset,
    ).to(device)
    dummy = torch.zeros(int(cfg.audio.segment_length * sr), device=device)
    chunk_frames = mel_extractor(dummy).shape[-1]
    overlap_frames = min(int(args.overlap_sec * sr / cfg.mel.hop_length), chunk_frames // 2)

    encoder = SpeakerEncoder(
        model_name=cfg.speaker_encoder.model_name, embed_dim=cfg.speaker_encoder.embed_dim,
        freeze=True, projection_ckpt=args.projection_ckpt,
    ).to(device)
    encoder.eval()
    mask_model = load_masking(args.mask_ckpt, cfg, device)
    flow_model = load_flow(args.flow_ckpt, cfg, device)

    vocoder = None
    if not args.skip_audio_domain:
        vc_cfg = OmegaConf.load(args.vocoder_config)
        vocoder = load_hifigan_generator(args.vocoder_ckpt, vc_cfg, device)

    orig_lookup = load_orig_source_paths(args.librimix_input_csv)
    generated_rows = load_generated_mixtures(os.path.join(args.librimix_dir, "metadata", "mixture_test_mix_clean.csv"))
    snr_lookup = load_metrics_snr(os.path.join(args.librimix_dir, "metadata", "metrics_test_mix_clean.csv"))
    tasks = list(build_tasks(generated_rows, orig_lookup, args.both_directions))
    if args.max_samples is not None:
        tasks = tasks[:args.max_samples]
    print(f"[EvalLibri2Mix] {len(generated_rows)} generated mixtures -> {len(tasks)} eval samples "
          f"({'both directions' if args.both_directions else 'source_1 only'})")

    done_ids = load_done_ids(args.output) if args.resume else set()
    if args.resume:
        print(f"[EvalLibri2Mix] Resuming: {len(done_ids)} samples already recorded in {args.output}")
    elif os.path.exists(args.output):
        print(f"[EvalLibri2Mix] {args.output} already exists and --resume not set -- "
              f"appending anyway (pass --resume to skip already-done samples).")

    all_records = [json.loads(l) for l in open(args.output, "r", encoding="utf-8") if l.strip()] if done_ids else []
    t_start = time.time()
    n_processed = 0

    with open(args.output, "a", encoding="utf-8") as f_out:
        for i, (sample_id, row, target_idx) in enumerate(tasks):
            if sample_id in done_ids:
                continue
            rec = process_one(
                sample_id, row, target_idx, orig_lookup, args.librispeech_dir, sr, ref_seg_len,
                mel_extractor, mask_model, flow_model, encoder, vocoder,
                chunk_frames, overlap_frames, cfg_scale, n_steps, device,
                compute_audio=not args.skip_audio_domain, snr_lookup=snr_lookup,
            )
            if rec is None:
                continue
            f_out.write(json.dumps(rec) + "\n")
            f_out.flush()
            all_records.append(rec)
            n_processed += 1

            if i % 50 == 0 or i == len(tasks) - 1:
                elapsed = time.time() - t_start
                rate = n_processed / max(elapsed, 1e-6)
                remaining = len(tasks) - i - 1
                eta_min = (remaining / rate / 60) if rate > 0 else float("nan")
                print(f"[EvalLibri2Mix] {i + 1}/{len(tasks)} tasks done "
                      f"({elapsed/60:.1f} min elapsed, ~{eta_min:.0f} min remaining)")

    print(f"\n[EvalLibri2Mix] Done. {len(all_records)} total records in {args.output}")
    print_summary(all_records, catastrophic_pct=args.catastrophic_pct)
