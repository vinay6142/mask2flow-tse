"""Diagnose the onset damping reported by ear on 2026-09-14.

Listening report: 4-speaker extractions sound "suppressed ... damped
especially at the start"; 3-speaker sample_03 likewise. Frame-level
measurement against target_clean confirmed it and localized it:

    first speech burst (starts at frame 0) : -4.08 dB median, worst -16
    interior speech onsets (after silence) : +0.29 dB
    sustained speech                       : +0.01 dB

So it is a POSITION-0 effect, not an acoustic one -- interior onsets, which
are acoustically the same event, are reproduced perfectly. Stage 1 does not
show it (~0 dB); Stage 2 does.

Hypothesis: Stage 2's DiT attends with RoPE (relative positions), so frame 0
is the only frame with no left context, and Stage 2 GENERATES the mel from
noise -- where Stage 1 merely scales the mixture and so inherits a plausible
level even when its mask is wrong. Giving Stage 2 a silent lead-in should
move the weak region into frames that are thrown away before vocoding.

Test: the SAME samples as the listening export (same seeds), Stage 2 run
with P frames of silence prepended to the Stage-1 mel and dropped again
afterwards. P=0 is run TWICE with different flow-noise seeds, so the
noise-induced spread is measurable and the padding effect can be judged
against it rather than against zero.

This is a diagnostic only -- it writes no checkpoint and changes no config.
"""
import os
import sys
import json
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from omegaconf import OmegaConf

from data.mel import MelSpectrogramExtractor, segment_waveform, make_frame_mask, masked_mse
from data.augment import load_audio, mix_multi_at_snr, trim_trailing_silence
from inference.vocoder import load_hifigan_generator, mel_to_audio_hifigan
from models.speaker_encoder import SpeakerEncoder

from eval.results_stage2 import load_masking, load_flow
from eval.audio_domain_quality import si_sdr
from eval.eval_multi_speaker import build_speaker_index

PAD_VALUE = -11.5   # data/mel.py pad_or_trim's silence value


def build_sample(sample_idx, base_seed, speaker_utterances, speakers, n_interferers,
                 cfg, mel_extractor, encoder, snr_min, snr_max, device):
    """Rebuild one listening-export sample EXACTLY (same seed, same RNG order)."""
    seed = base_seed + sample_idx
    random.seed(seed)
    torch.manual_seed(seed)

    sr = cfg.audio.sample_rate
    seg_len = cfg.audio.segment_length
    ref_seg_len = getattr(cfg.audio, "reference_length", 3.0)
    segment_samples = int(seg_len * sr)

    target_speaker = random.choice(speakers)
    utterances = speaker_utterances[target_speaker]
    utt_idx = random.sample(range(len(utterances)), 2)
    target_path, ref_path = utterances[utt_idx[0]], utterances[utt_idx[1]]

    other_speakers = [s for s in speakers if s != target_speaker]
    interferer_speakers = random.sample(other_speakers, n_interferers)
    interferer_paths = [random.choice(speaker_utterances[s]) for s in interferer_speakers]

    target_wav = load_audio(target_path, sr)
    ref_wav = load_audio(ref_path, sr)
    interferer_wavs_raw = [load_audio(p, sr) for p in interferer_paths]

    target_valid_samples = min(target_wav.shape[0], segment_samples)

    target_wav = segment_waveform(target_wav, sr, seg_len, random_start=True)
    ref_wav = segment_waveform(ref_wav, sr, ref_seg_len, random_start=True)
    ref_wav = trim_trailing_silence(ref_wav, sr)
    interferer_wavs = [segment_waveform(w, sr, seg_len, random_start=True)
                       for w in interferer_wavs_raw]

    snr_db_list = [random.uniform(snr_min, snr_max) for _ in range(n_interferers)]
    mixture_wav, target_matched, _ = mix_multi_at_snr(target_wav, interferer_wavs, snr_db_list)

    mixture_mel = mel_extractor(mixture_wav.to(device))
    target_mel = mel_extractor(target_matched.to(device))
    target_valid_frames = min(target_valid_samples // mel_extractor.hop_length,
                              target_mel.shape[-1])
    frame_mask = make_frame_mask(torch.tensor([target_valid_frames]),
                                 target_mel.shape[-1], device=device)
    d_vec = encoder(ref_wav.unsqueeze(0).to(device))
    return mixture_mel, target_mel, d_vec, frame_mask, target_valid_frames


def run_stage2(stage1_out, d_vec, flow_model, pad_frames, noise_seed, cfg_scale, n_steps,
               cfg_warmup_steps=0):
    """Stage 2 with `pad_frames` of silence prepended, then removed again.

    `noise_seed` is vestigial: run 11174 was designed to use it as a
    noise-variance control, and the two zero-padding columns came back
    bit-identical because FlowMatchingModule.inference starts from
    `x = x_enh.clone()` and Euler-integrates -- it never samples noise. The
    seeding is kept only so the call site stays explicit about determinism.

    The prepended frames are pure padding: they are dropped before anything
    downstream sees the result, so the only thing they can change is how much
    left context frame 0 of the REAL audio has.
    """
    torch.manual_seed(noise_seed)
    if pad_frames > 0:
        pad = torch.full((stage1_out.shape[0], stage1_out.shape[1], pad_frames),
                         PAD_VALUE, device=stage1_out.device, dtype=stage1_out.dtype)
        inp = torch.cat([pad, stage1_out], dim=-1)
    else:
        inp = stage1_out
    out = flow_model.inference(inp, d_vec, cfg_scale=cfg_scale, n_steps=n_steps,
                               cfg_warmup_steps=cfg_warmup_steps)
    return out[:, :, pad_frames:] if pad_frames > 0 else out


def make_splice(base_mel, padded_mel, splice_frames, xfade_frames):
    """Padded output for the first `splice_frames`, baseline afterwards.

    Motivation (run 11174): silent lead-in roughly halves the onset damping,
    but costs 0.24dB SI-SDR and 2.4% mel MSE across the whole utterance --
    the gain is concentrated in the first ~0.5s while the cost is spread over
    all 10s, so global padding fails the pre-registered bar. Splicing keeps
    the tail BIT-IDENTICAL to current behaviour, so the utterance-averaged
    metrics can only move through the region that actually improved.

    The blend is linear in log-mel, i.e. geometric in magnitude -- fine over
    a short crossfade, and it is what the vocoder is fed either way.
    """
    out = base_mel.clone()
    K = min(splice_frames, out.shape[-1])
    out[..., :K] = padded_mel[..., :K]
    x = min(xfade_frames, out.shape[-1] - K)
    if x > 0:
        w = torch.linspace(1.0, 0.0, x, device=out.device, dtype=out.dtype)
        out[..., K:K + x] = w * padded_mel[..., K:K + x] + (1 - w) * base_mel[..., K:K + x]
    return out


def frame_rms(x, frame=320):
    n = len(x) // frame
    return np.sqrt((x[:n * frame].reshape(n, frame) ** 2).mean(axis=1) + 1e-12)


def region_db(est_wav, tgt_wav, sr, lo=0.0, hi=0.5):
    """Median level of `est` vs `tgt`, in dB, over the target's ACTIVE frames
    in [lo, hi) seconds -- the quantity the listener described as damping.

    lo=0, hi=0.5 gives the onset region; lo=1.0, hi=inf gives the tail, which
    is what shows whether a change bought its onset gain at the expense of
    everything else.
    """
    e, t = frame_rms(est_wav), frame_rms(tgt_wav)
    n = min(len(e), len(t))
    e, t = e[:n], t[:n]
    act = t > t.max() * 10 ** (-40 / 20)
    idx = np.arange(n) * 320.0 / sr
    m = act & (idx >= lo) & (idx < hi)
    if m.sum() < 3:
        return float("nan")
    return float(np.median(20 * np.log10((e[m] + 1e-12) / (t[m] + 1e-12))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default_v2.yaml")
    p.add_argument("--mask_ckpt", default="checkpoints_v2/masking/mask_best.pt")
    p.add_argument("--flow_ckpt", default="checkpoints_v2/flow/flow_best.pt")
    p.add_argument("--projection_ckpt", default="checkpoints_speaker_encoder/projection_latest.pt")
    p.add_argument("--vocoder_ckpt", default="checkpoints_vocoder/vocoder_best.pt")
    p.add_argument("--vocoder_config", default="configs/vocoder.yaml")
    p.add_argument("--librispeech_dir", default="data/raw/LibriSpeech")
    p.add_argument("--split", default="test-clean")
    # NOTE: seed 123 and the cfg.data SNR range are export_listening_samples.py's
    # OWN defaults. Run 11174 used seed 42 at [-5,5] and therefore scored a
    # different, harder draw than the samples that were actually listened to --
    # matching them here is the point, so per-sample findings line up with the
    # by-ear report.
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--n_samples", type=int, default=8)
    p.add_argument("--n_interferers", default="1,2,3")
    p.add_argument("--pads", default="0",
                   help="silence frames to prepend (mel frames; hop 160 => 100/s)")
    p.add_argument("--cfg_variants", default="1.5:0,2.0:0,2.5:0,2.5:1,2.5:2",
                   help="comma-separated cfg_scale:cfg_warmup_steps pairs to compare. "
                        "2.5:0 is the deployed setting; 1.5:0 is what the 2026-09-04 "
                        "export (no onset gate) used; warmup>0 runs that many initial "
                        "Euler steps at cfg_scale=1.0, the built-in t=0 mitigation that "
                        "no inference path currently passes.")
    p.add_argument("--splice_frames", type=int, default=100,
                   help="use the padded output for this many leading frames (100 = 1.0s), "
                        "baseline afterwards; 0 disables the splice variant")
    p.add_argument("--xfade_frames", type=int, default=20,
                   help="crossfade length between padded and baseline at the splice point")
    p.add_argument("--snr_min", type=float, default=None, help="default: cfg.data.snr_min")
    p.add_argument("--snr_max", type=float, default=None, help="default: cfg.data.snr_max")
    p.add_argument("--cfg_scale", type=float, default=None)
    p.add_argument("--n_steps", type=int, default=None)
    p.add_argument("--output", default="outputs/results/diag_onset_padding.jsonl")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(args.config)
    cfg_scale = args.cfg_scale if args.cfg_scale is not None else cfg.inference.cfg_scale
    # NOTE: the config key is `flow_steps`, not `n_steps` -- same spelling the
    # other eval scripts use (full_eval.py, export_listening_samples.py).
    n_steps = args.n_steps if args.n_steps is not None else cfg.inference.flow_steps
    pads = [int(x) for x in args.pads.split(",")]
    n_int_list = [int(x) for x in args.n_interferers.split(",")]
    sr = cfg.audio.sample_rate
    snr_min = args.snr_min if args.snr_min is not None else cfg.data.snr_min
    snr_max = args.snr_max if args.snr_max is not None else cfg.data.snr_max
    cfg_variants = []
    for tok in args.cfg_variants.split(","):
        sc, _, wu = tok.partition(":")
        cfg_variants.append((float(sc), int(wu or 0)))
    print(f"cfg variants (scale:warmup) = {cfg_variants}")
    print(f"seed={args.seed}  snr=[{snr_min},{snr_max}]dB  "
          f"splice_frames={args.splice_frames}  xfade={args.xfade_frames}")

    print(f"device={device}  cfg_scale={cfg_scale}  n_steps={n_steps}  pads={pads}")
    # NOTE: both of these take explicit kwargs, NOT the cfg object -- copied
    # verbatim from eval/export_listening_samples.py so this diagnostic builds
    # exactly the same front-end as the export it is explaining.
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
    vc_cfg = OmegaConf.load(args.vocoder_config)
    vocoder = load_hifigan_generator(args.vocoder_ckpt, vc_cfg, device)

    speaker_utterances, speakers = build_speaker_index(args.librispeech_dir, args.split)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    records = []

    with torch.no_grad():
        for n_int in n_int_list:
            n_total = n_int + 1
            print(f"\n=== {n_total} total speakers ===")
            for i in range(args.n_samples):
                mixture_mel, target_mel, d_vec, frame_mask, vf = build_sample(
                    i, args.seed, speaker_utterances, speakers, n_int,
                    cfg, mel_extractor, encoder, snr_min, snr_max, device)
                vf = max(vf, 1)
                stage1_out, _ = mask_model(mixture_mel.unsqueeze(0), d_vec)
                tgt_wav = mel_to_audio_hifigan(target_mel[:, :vf], vocoder)
                tgt_t = torch.from_numpy(tgt_wav).float().to(device)

                # Stage 2 is deterministic (inference starts from x_enh, no
                # noise is drawn), so each variant is computed once and any
                # difference between variants is exact, not sampling spread.
                mels = {}
                for pad in pads:
                    for sc, wu in cfg_variants:
                        name = f"cfg{sc:g}w{wu}" + (f"p{pad}" if pad else "")
                        mels[name] = run_stage2(stage1_out, d_vec, flow_model,
                                                pad, 0, sc, n_steps, wu)
                if args.splice_frames > 0:
                    for sc, wu in cfg_variants:
                        base = f"cfg{sc:g}w{wu}"
                        for pad in pads:
                            if pad == 0 or f"{base}p{pad}" not in mels:
                                continue
                            mels[f"{base}spl{pad}"] = make_splice(
                                mels[base], mels[f"{base}p{pad}"],
                                args.splice_frames, args.xfade_frames)

                row = {"n_total_speakers": n_total, "sample": i}
                for key, s2 in mels.items():
                    wav = mel_to_audio_hifigan(s2[0, :, :vf], vocoder)
                    row[f"{key}_onset_db"] = region_db(wav, tgt_wav, sr, 0.0, 0.5)
                    row[f"{key}_tail_db"] = region_db(wav, tgt_wav, sr, 1.0, 1e9)
                    # NOTE: si_sdr already returns a Python float (it .item()s
                    # internally) -- do NOT call .item() on it again.
                    row[f"{key}_sisdr"] = si_sdr(
                        torch.from_numpy(wav).float().to(device), tgt_t)
                    row[f"{key}_melmse"] = masked_mse(
                        s2, target_mel.unsqueeze(0), frame_mask).item()
                records.append(row)
                cells = "  ".join(
                    f"{k.replace('_onset_db',''):>9s}:{row[k]:6.2f}dB"
                    for k in row if k.endswith("_onset_db"))
                print(f"  sample_{i:02d}  {cells}")

    with open(args.output, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print("\n" + "=" * 86)
    print("SUMMARY -- median over samples. Stage 2 is DETERMINISTIC (inference starts")
    print("from x_enh and Euler-steps; no noise is drawn), so every difference below is")
    print("exact -- there is no run-to-run spread to discount it against.")
    print("  onset = level vs ground truth over the first 0.5s of speech (0 = correct,")
    print("          negative = damped: the defect reported by ear)")
    print("  tail  = the same measure after 1.0s. A splice variant MUST match pad0")
    print("          here to the decimal -- it is baseline by construction past the")
    print("          splice point, so any difference means the splice is wrong.")
    print("=" * 86)
    for metric, label, fmt in (("onset_db", "onset damping dB (higher/0 is better)", "12.2f"),
                               ("tail_db", "tail damping dB (sanity check)", "12.2f"),
                               ("sisdr", "SI-SDR dB, whole utterance (higher better)", "12.4f"),
                               ("melmse", "mel MSE, whole utterance (lower better)", "12.4f")):
        keys = [k for k in records[0] if k.endswith("_" + metric)]
        print(f"\n{label}")
        print(f"{'condition':>12s} " + "".join(
            f"{k[:-len(metric) - 1]:>12s}" for k in keys))
        for n_total in sorted({r["n_total_speakers"] for r in records}):
            sub = [r for r in records if r["n_total_speakers"] == n_total]
            print(f"{str(n_total) + ' speakers':>12s} " + "".join(
                f"{np.nanmedian([r[k] for r in sub]):{fmt}}" for k in keys))
        print(f"{'ALL':>12s} " + "".join(
            f"{np.nanmedian([r[k] for r in records]):{fmt}}" for k in keys))
    # The defect is a GATE, not gradual damping: in the affected samples the
    # output sits on a flat ~-60dB floor for the first ~0.5s and then switches
    # on. So the count of gated samples tracks what is audible far better than
    # the median does -- one sample at -35dB is heard, a median shift of 1dB
    # is not. Reference points, measured on the exported samples with this
    # same threshold: 2026-09-04 export (old ckpts, cfg 1.5) = 0/12 gated;
    # 2026-09-14 export (current ckpts, cfg 2.5) = 5/12.
    keys = [k for k in records[0] if k.endswith("_onset_db")]
    print("\ngated samples (onset < -10dB) -- the count that tracks audibility")
    print(f"{'condition':>12s} " + "".join(f"{k[:-9]:>12s}" for k in keys))
    def gated(rs, k):
        # samples too short to score come back nan -- exclude them from BOTH
        # numerator and denominator rather than silently counting them as fine
        valid = [r[k] for r in rs if not np.isnan(r[k])]
        return f"{sum(1 for v in valid if v < -10):>7d}/{len(valid):<4d}"

    for n_total in sorted({r["n_total_speakers"] for r in records}):
        sub = [r for r in records if r["n_total_speakers"] == n_total]
        print(f"{str(n_total) + ' speakers':>12s} " + "".join(gated(sub, k) for k in keys))
    print(f"{'ALL':>12s} " + "".join(gated(records, k) for k in keys))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
