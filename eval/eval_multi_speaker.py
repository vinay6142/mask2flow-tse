"""
Mask2Flow-TSE -- multi-speaker (>2 total speakers) generalization test.

The model is only ever TRAINED on 2-total-speaker mixtures (1 target + exactly
1 interferer -- see data/augment.py's MixtureCreator / data/librispeech.py).
This script asks the out-of-training-distribution question directly: what
happens if a THIRD (or more) simultaneous talker is added to the mixture?
Same methodology and metrics as this project's other in-domain eval scripts
(eval/results_stage2.py, eval/full_eval.py) -- masked mel-MSE, catastrophic-
outlier rate, audio-domain SI-SDR after vocoding -- plus a speaker-verification
similarity check (extracted-vs-reference cosine sim, and mixture-vs-reference
as the do-nothing baseline), so a 2-speaker run and a 3-speaker run from this
same script are directly comparable to each other AND to the existing n=2620
test-clean headline (77.7% median S2vsS1, 3.5% catastrophic, +1.58dB SI-SDR).

Uses REAL LibriSpeech test-clean audio (same corpus as the existing in-domain
headline), synthetically mixed via the NEW data.augment.mix_multi_at_snr()
(a pure eval-time generalization of mix_at_snr() to N interferers -- the
training-time MixtureCreator is untouched). --n_interferers controls how many
simultaneous talkers besides the target: 1 reproduces the existing trained
condition (2 total speakers) as an in-script baseline; 2 is the 3-speaker
test; higher values are supported too, same tool.

Run (recommended: run BOTH conditions for a paired comparison):
  python3 eval/eval_multi_speaker.py --n_interferers 1 \
      --mask_ckpt checkpoints_v2/masking/mask_best.pt \
      --flow_ckpt checkpoints_v2/flow/flow_best.pt \
      --projection_ckpt checkpoints_speaker_encoder/projection_latest.pt \
      --output outputs/results/eval_multispeaker_2total.jsonl

  python3 eval/eval_multi_speaker.py --n_interferers 2 \
      --mask_ckpt checkpoints_v2/masking/mask_best.pt \
      --flow_ckpt checkpoints_v2/flow/flow_best.pt \
      --projection_ckpt checkpoints_speaker_encoder/projection_latest.pt \
      --output outputs/results/eval_multispeaker_3total.jsonl

Resume an interrupted run: same command, add --resume
Just print stats on what is done so far:
  python3 eval/eval_multi_speaker.py --output <file> --summarize_only
"""
import os
import sys
import json
import time
import random
import argparse
import statistics as stats

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

from data.mel import MelSpectrogramExtractor, segment_waveform, make_frame_mask, masked_mse
from data.augment import load_audio, mix_multi_at_snr, trim_trailing_silence
from inference.vocoder import load_hifigan_generator, mel_to_audio_hifigan
from models.speaker_encoder import SpeakerEncoder

from eval.results_stage2 import load_masking, load_flow
from eval.audio_domain_quality import si_sdr, waveform_mse
from eval.verify_eer import genuine_impostor_scores, compute_eer_auc

os.makedirs("outputs/results", exist_ok=True)

CATASTROPHIC_MEL_PCT = -30.0   # same threshold used everywhere else in eval/


# -- speaker index (mirrors data/librispeech.py's _build_index) ------------

def build_speaker_index(librispeech_dir, split, min_utterances=2):
    root = Path(librispeech_dir) / split
    speaker_utterances = {}
    for flac_file in root.rglob("*.flac"):
        speaker_id = flac_file.parts[-3]
        speaker_utterances.setdefault(speaker_id, []).append(str(flac_file))
    speakers = [s for s, utts in speaker_utterances.items() if len(utts) >= min_utterances]
    print(f"[EvalMultiSpeaker] {len(speakers)} speakers (>= {min_utterances} utterances) "
          f"in {split}, {sum(len(v) for v in speaker_utterances.values())} utterances total")
    return speaker_utterances, speakers


# -- one sample: pick target + N interferers, mix, run the pipeline --------

@torch.no_grad()
def process_one(sample_idx, base_seed, speaker_utterances, speakers, n_interferers,
                 cfg, mel_extractor, mask_model, flow_model, encoder, vocoder,
                 cfg_scale, n_steps, snr_min, snr_max, device, compute_audio):
    seed = base_seed + sample_idx
    random.seed(seed)
    torch.manual_seed(seed)

    sr          = cfg.audio.sample_rate
    seg_len     = cfg.audio.segment_length
    ref_seg_len = getattr(cfg.audio, "reference_length", 3.0)
    segment_samples = int(seg_len * sr)

    target_speaker = random.choice(speakers)
    utterances     = speaker_utterances[target_speaker]
    utt_idx        = random.sample(range(len(utterances)), 2)
    target_path    = utterances[utt_idx[0]]
    ref_path       = utterances[utt_idx[1]]

    other_speakers = [s for s in speakers if s != target_speaker]
    interferer_speakers = random.sample(other_speakers, n_interferers)
    interferer_paths = [random.choice(speaker_utterances[s]) for s in interferer_speakers]

    target_wav = load_audio(target_path, sr)
    ref_wav    = load_audio(ref_path, sr)
    interferer_wavs_raw = [load_audio(p, sr) for p in interferer_paths]

    # valid (non-padded) sample count, BEFORE segment_waveform pads/crops --
    # same convention as data/librispeech.py's __getitem__.
    target_valid_samples = min(target_wav.shape[0], segment_samples)

    target_wav = segment_waveform(target_wav, sr, seg_len, random_start=True)
    ref_wav    = segment_waveform(ref_wav, sr, ref_seg_len, random_start=True)
    # Trim trailing silence off the reference clip -- segment_waveform zero-pads
    # any reference utterance shorter than ref_seg_len (3.0s), and that padding
    # dilutes the mean-pooled WavLM embedding (same padding-dilution mechanism
    # already fixed elsewhere in this project -- see trim_trailing_silence()'s
    # use in inference/speaker_similarity.py / eval/results_stage2.py, which
    # this script had been missing until now).
    ref_wav    = trim_trailing_silence(ref_wav, sr)
    interferer_wavs = [segment_waveform(w, sr, seg_len, random_start=True) for w in interferer_wavs_raw]

    snr_db_list = [random.uniform(snr_min, snr_max) for _ in range(n_interferers)]
    mixture_wav, target_matched, _ = mix_multi_at_snr(target_wav, interferer_wavs, snr_db_list)

    mixture_mel   = mel_extractor(mixture_wav.to(device))
    target_mel    = mel_extractor(target_matched.to(device))
    # NOTE: no reference_mel here -- the speaker encoder consumes ref_wav (raw
    # waveform) directly, not a mel spectrogram, so computing one would be dead
    # code (an earlier version of this function did, harmlessly but pointlessly).

    target_valid_frames = min(target_valid_samples // mel_extractor.hop_length, target_mel.shape[-1])
    frame_mask = make_frame_mask(torch.tensor([target_valid_frames]), target_mel.shape[-1], device=device)

    d_vec = encoder(ref_wav.unsqueeze(0).to(device))
    stage1_out, _ = mask_model(mixture_mel.unsqueeze(0), d_vec)
    stage2_out = flow_model.inference(stage1_out, d_vec, cfg_scale=cfg_scale, n_steps=n_steps)

    mix_b = mixture_mel.unsqueeze(0)
    tgt_b = target_mel.unsqueeze(0)
    mix_mse = masked_mse(mix_b, tgt_b, frame_mask).item()
    s1_mse  = masked_mse(stage1_out, tgt_b, frame_mask).item()
    s2_mse  = masked_mse(stage2_out, tgt_b, frame_mask).item()

    rec = {
        "sample_id": f"sample_{sample_idx:05d}",
        "seed": seed,
        "n_interferers": n_interferers,
        "n_total_speakers": n_interferers + 1,
        "target_speaker": target_speaker,
        "interferer_speakers": interferer_speakers,
        "snr_db_list": snr_db_list,
        "mel_mix_mse": mix_mse, "mel_s1_mse": s1_mse, "mel_s2_mse": s2_mse,
        "mel_s1_imp_pct": (mix_mse - s1_mse) / max(mix_mse, 1e-12) * 100,
        "mel_s2_vs_mix_pct": (mix_mse - s2_mse) / max(mix_mse, 1e-12) * 100,
        "mel_s2_vs_s1_pct": (s1_mse - s2_mse) / max(s1_mse, 1e-12) * 100,
    }

    if compute_audio:
        vf = max(target_valid_frames, 1)
        mix_wav = torch.from_numpy(mel_to_audio_hifigan(mixture_mel[:, :vf], vocoder)).float().to(device)
        s1_wav  = torch.from_numpy(mel_to_audio_hifigan(stage1_out[0, :, :vf], vocoder)).float().to(device)
        s2_wav  = torch.from_numpy(mel_to_audio_hifigan(stage2_out[0, :, :vf], vocoder)).float().to(device)
        tgt_wav = torch.from_numpy(mel_to_audio_hifigan(target_mel[:, :vf], vocoder)).float().to(device)

        mix_sisdr, s1_sisdr, s2_sisdr = si_sdr(mix_wav, tgt_wav), si_sdr(s1_wav, tgt_wav), si_sdr(s2_wav, tgt_wav)
        rec.update({
            "wav_mix_mse": waveform_mse(mix_wav, tgt_wav),
            "wav_s2_mse": waveform_mse(s2_wav, tgt_wav),
            "sisdr_mix": mix_sisdr, "sisdr_s1": s1_sisdr, "sisdr_s2": s2_sisdr,
            "sisdr_gain_vs_mix": s2_sisdr - mix_sisdr,
            "sisdr_gain_vs_s1": s2_sisdr - s1_sisdr,
        })

        # speaker-identity check: does extraction pull the output CLOSER to the
        # reference speaker's embedding than doing nothing (the raw mixture)?
        emb_s2  = encoder(s2_wav.unsqueeze(0))[0]
        emb_mix = encoder(mix_wav.unsqueeze(0))[0]
        emb_tgt = encoder(tgt_wav.unsqueeze(0))[0]
        rec["verify_s2_sim"]  = F.cosine_similarity(emb_s2, d_vec[0], dim=0).item()
        rec["verify_mix_sim"] = F.cosine_similarity(emb_mix, d_vec[0], dim=0).item()

        # Raw embeddings, saved so a corpus-wide genuine/impostor EER (the
        # SAME real "accuracy" metric as eval/verify_eer.py's 86.2% headline
        # number, not just the per-sample cosine sim above) can be computed
        # in print_summary() across the WHOLE evaluated set -- a single
        # sample's cosine sim to its own reference says nothing about
        # separability from OTHER speakers, which is what "accuracy" means.
        rec["ref_embedding"] = d_vec[0].detach().cpu().tolist()
        rec["mix_embedding"] = emb_mix.detach().cpu().tolist()
        rec["s2_embedding"]  = emb_s2.detach().cpu().tolist()
        rec["tgt_embedding"] = emb_tgt.detach().cpu().tolist()

    return rec


def compute_corpus_eer(records):
    """
    Corpus-wide genuine/impostor EER/accuracy, reusing eval/verify_eer.py's
    exact methodology (genuine_impostor_scores + compute_eer_auc) so these
    numbers are directly comparable to the existing 2-speaker headline
    (86.2% accuracy / 13.8% EER / 0.9353 AUC, n=400). Three probes against
    the SAME reference matrix: mixture (do-nothing baseline), Stage 2
    extraction, and the ground-truth target (oracle ceiling -- how well the
    encoder itself can do on clean, unmixed audio, an upper bound the
    extraction system can't be expected to beat).
    Returns None if fewer than 2 distinct speakers are present (can't form
    impostor trials) or no records carry embeddings (--skip_audio_domain run).
    """
    if not records or "ref_embedding" not in records[0]:
        return None

    speakers = [r["target_speaker"] for r in records]
    if len(set(speakers)) < 2:
        return None

    ref_emb = torch.tensor([r["ref_embedding"] for r in records])
    mix_emb = torch.tensor([r["mix_embedding"] for r in records])
    s2_emb  = torch.tensor([r["s2_embedding"] for r in records])
    tgt_emb = torch.tensor([r["tgt_embedding"] for r in records])

    results = {}
    for name, probe_emb in [("mixture", mix_emb), ("stage2", s2_emb), ("target", tgt_emb)]:
        sim = probe_emb @ ref_emb.T
        genuine, impostor = genuine_impostor_scores(sim, speakers)
        eer, _, auc = compute_eer_auc(genuine, impostor)
        results[name] = {"accuracy": (1.0 - eer) * 100, "eer": eer * 100, "auc": auc}
    return results


# -- resumable JSONL driver + summary (same pattern as eval/eval_libri2mix.py) --

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
        print("[EvalMultiSpeaker] No records to summarize.")
        return

    has_audio = "sisdr_s2" in records[0]
    n_total_speakers = records[0].get("n_total_speakers", "?")

    print("\n" + "=" * 100)
    print(f"  MASK2FLOW-TSE -- MULTI-SPEAKER GENERALIZATION TEST  "
          f"(n = {n} samples, {n_total_speakers} total speakers per mixture)")
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

        s2_sim = [r["verify_s2_sim"] for r in records if "verify_s2_sim" in r]
        mix_sim = [r["verify_mix_sim"] for r in records if "verify_mix_sim" in r]
        n_id_worse = sum(1 for r in records if "verify_s2_sim" in r and r["verify_s2_sim"] < r["verify_mix_sim"])
        print(f"\n  [Speaker identity, cosine sim to reference]  (n={len(s2_sim)})")
        print(f"    Median sim, mixture (do-nothing baseline) : {stats.median(mix_sim):.4f}")
        print(f"    Median sim, Stage 2 (extracted)            : {stats.median(s2_sim):.4f}")
        print(f"    Samples where extraction moved FARTHER from target identity than doing nothing: "
              f"{n_id_worse}/{len(s2_sim)} ({100 * n_id_worse / len(s2_sim):.1f}%)")

        eer = compute_corpus_eer(records)
        if eer is not None:
            print(f"\n  [Speaker-verification ACCURACY -- same genuine/impostor EER methodology as "
                  f"eval/verify_eer.py's 86.2% 2-speaker headline; THIS is the number comparable "
                  f"to a target like \"75-80% accuracy\", not the raw cosine sim above]")
            print(f"    {'Probe':<32} {'Accuracy (1-EER)':>18} {'EER':>8} {'AUC':>8}")
            for name, label in [("mixture", "Mixture (do-nothing)"),
                                 ("stage2", "Stage 2 extraction"),
                                 ("target", "Ground-truth target (ceiling)")]:
                r = eer[name]
                print(f"    {label:<32} {r['accuracy']:>17.1f}% {r['eer']:>7.1f}% {r['auc']:>8.4f}")
        else:
            print(f"\n  [Speaker-verification accuracy: skipped -- need >=2 distinct target speakers "
                  f"in this run to form impostor trials]")

    worst = sorted(records, key=lambda r: r["mel_s2_vs_s1_pct"])[:10]
    print(f"\n  [Worst 10 by mel S2-vs-S1%]")
    for r in worst:
        extra = f"  SI-SDR gain vs S1={r['sisdr_gain_vs_s1']:+.2f}dB" if "sisdr_gain_vs_s1" in r else ""
        print(f"    {r['sample_id']:<16} mel S2vsS1={r['mel_s2_vs_s1_pct']:>9.1f}%{extra}")

    print("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default_v2.yaml")
    parser.add_argument("--mask_ckpt", default=None)
    parser.add_argument("--flow_ckpt", default=None)
    parser.add_argument("--projection_ckpt", default="checkpoints_speaker_encoder/projection_latest.pt")
    parser.add_argument("--vocoder_ckpt", default="checkpoints_vocoder/vocoder_best.pt")
    parser.add_argument("--vocoder_config", default="configs/vocoder.yaml")
    parser.add_argument("--librispeech_dir", default="data/raw/LibriSpeech")
    parser.add_argument("--split", default="test-clean")
    parser.add_argument("--n_interferers", type=int, default=2,
                         help="Simultaneous interferers besides the target. 1 reproduces the "
                              "trained 2-total-speaker condition; 2 is the 3-speaker test.")
    parser.add_argument("--n_samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--snr_min", type=float, default=None, help="Default: cfg.data.snr_min")
    parser.add_argument("--snr_max", type=float, default=None, help="Default: cfg.data.snr_max")
    parser.add_argument("--cfg_scale", type=float, default=None)
    parser.add_argument("--n_steps", type=int, default=None)
    parser.add_argument("--skip_audio_domain", action="store_true",
                         help="Skip vocoding + SI-SDR + speaker-identity check (the expensive part on CPU).")
    parser.add_argument("--output", default="outputs/results/eval_multi_speaker.jsonl")
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
        print(f"[EvalMultiSpeaker] Loaded {len(records)} records from {args.output}")
        print_summary(records, catastrophic_pct=args.catastrophic_pct)
        sys.exit(0)

    assert args.mask_ckpt and args.flow_ckpt, "--mask_ckpt and --flow_ckpt required unless --summarize_only"
    assert args.n_interferers >= 1, "--n_interferers must be >= 1"

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg_scale = args.cfg_scale if args.cfg_scale is not None else cfg.inference.cfg_scale
    n_steps   = args.n_steps if args.n_steps is not None else cfg.inference.flow_steps
    snr_min   = args.snr_min if args.snr_min is not None else cfg.data.snr_min
    snr_max   = args.snr_max if args.snr_max is not None else cfg.data.snr_max

    print(f"[EvalMultiSpeaker] Device: {device}  n_interferers={args.n_interferers} "
          f"({args.n_interferers + 1} total speakers)  cfg_scale={cfg_scale}  n_steps={n_steps}  "
          f"snr_range=[{snr_min}, {snr_max}]  audio_domain={'off' if args.skip_audio_domain else 'on'}")

    mel_extractor = MelSpectrogramExtractor(
        sample_rate=cfg.audio.sample_rate, n_mels=cfg.mel.n_mels, n_fft=cfg.mel.n_fft,
        hop_length=cfg.mel.hop_length, win_length=cfg.mel.win_length,
        f_min=cfg.mel.f_min, f_max=cfg.mel.f_max, log_offset=cfg.mel.log_offset,
    ).to(device)

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

    speaker_utterances, speakers = build_speaker_index(args.librispeech_dir, args.split)
    assert len(speakers) > args.n_interferers, \
        f"Need more than {args.n_interferers} eligible speakers in {args.split}, found {len(speakers)}"

    done_ids = load_done_ids(args.output) if args.resume else set()
    if args.resume:
        print(f"[EvalMultiSpeaker] Resuming: {len(done_ids)} samples already recorded in {args.output}")
    elif os.path.exists(args.output):
        print(f"[EvalMultiSpeaker] {args.output} already exists and --resume not set -- "
              f"appending anyway (pass --resume to skip already-done samples).")

    all_records = [json.loads(l) for l in open(args.output, "r", encoding="utf-8") if l.strip()] if done_ids else []
    t_start = time.time()
    n_processed = 0

    with open(args.output, "a", encoding="utf-8") as f_out:
        for i in range(args.n_samples):
            sample_id = f"sample_{i:05d}"
            if sample_id in done_ids:
                continue
            rec = process_one(
                i, args.seed, speaker_utterances, speakers, args.n_interferers,
                cfg, mel_extractor, mask_model, flow_model, encoder, vocoder,
                cfg_scale, n_steps, snr_min, snr_max, device,
                compute_audio=not args.skip_audio_domain,
            )
            f_out.write(json.dumps(rec) + "\n")
            f_out.flush()
            all_records.append(rec)
            n_processed += 1

            if i % 5 == 0 or i == args.n_samples - 1:
                # Print (and explicitly flush) every 5 samples, not 50 -- the
                # first real run of this script turned out to be ~140x slower
                # per sample than the closely comparable eval_libri2mix.py
                # (root cause not yet understood), and stdout was fully
                # buffered so a 2-hour SLURM run showed NOTHING in the log
                # even though 97 samples were actually completed and flushed
                # to the output JSONL the whole time. Frequent, explicitly-
                # flushed progress prints are cheap insurance against that
                # happening again unexplained.
                elapsed = time.time() - t_start
                rate = n_processed / max(elapsed, 1e-6)
                remaining = args.n_samples - i - 1
                eta_min = (remaining / rate / 60) if rate > 0 else float("nan")
                print(f"[EvalMultiSpeaker] {i + 1}/{args.n_samples} samples done "
                      f"({elapsed/60:.1f} min elapsed, {elapsed/max(n_processed,1):.1f}s/sample, "
                      f"~{eta_min:.0f} min remaining)", flush=True)

    print(f"\n[EvalMultiSpeaker] Done. {len(all_records)} total records in {args.output}")
    print_summary(all_records, catastrophic_pct=args.catastrophic_pct)
