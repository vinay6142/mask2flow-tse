"""
Mask2Flow-TSE -- export real, listenable .wav files for manual/human
comparison of target-speaker extraction, across 2/3/4(+)-speaker mixtures.

Every automated metric in this project (mel-MSE, SI-SDR, WavLM cosine
similarity, corpus-wide EER) is a PROXY for "did we actually extract the
target speaker's voice, and does it sound right" -- this script produces
the actual audio so that question can be answered by ear, not just
inferred from a number. Built directly in response to two listening
observations the automated metrics alone hadn't caught:

  (1) the vocoder sometimes sounds "metallic" even when the words/content
      are correct. This script also exports a GROUND-TRUTH-mel-vocoded
      file (target_clean.wav) through the SAME vocoder, so listening can
      isolate whether the metallic quality comes from the vocoder itself
      (would show up in target_clean.wav too, since that's real audio's
      own mel run back through the vocoder) or specifically from Stage 2's
      imperfect mel prediction feeding an otherwise-fine vocoder (would
      show up in extracted.wav only).
  (2) low speaker-similarity scores despite a subjectively-correct-sounding
      voice, traced to trailing silence in the reference clip diluting the
      mean-pooled WavLM embedding. This script (and, as of this session,
      eval/eval_multi_speaker.py itself) now applies trim_trailing_silence()
      to the reference clip -- the same fix already used in
      inference/speaker_similarity.py / eval/results_stage2.py, but which
      had been missing from the multi-speaker eval path until now.

For each of --n_interferers (default 1,2,3 -- i.e. 2/3/4 total speakers),
saves --n_samples_per_condition (default 4) sets of:
  mixture.wav       -- the raw mixture (what a microphone would pick up)
  reference.wav     -- the enrollment clip used to condition extraction
                       (trailing silence trimmed)
  target_clean.wav  -- ground-truth clean target, vocoded from its OWN
                       real mel (not Stage 2's prediction) -- the oracle/
                       ceiling, and the vocoder-artifact diagnostic above
  stage1_only.wav   -- masking-only output (before flow matching)
  extracted.wav     -- the actual system output (Stage 2) -- what the
                       project's "did we extract the target speaker" claim
                       is about
plus an info.json per sample with the speaker IDs, SNRs, and every
relevant metric (mel/SI-SDR/cosine-similarity) for that specific sample,
so what you hear can be directly cross-referenced against the numbers.

Uses a DIFFERENT default seed (123) than every metrics eval in this
project (42), so this is a fresh spot-check draw, not literally the exact
same samples already scored elsewhere.

Run:
  python3 eval/export_listening_samples.py \
      --mask_ckpt checkpoints_v2/masking/mask_best.pt \
      --flow_ckpt checkpoints_v2/flow/flow_best.pt \
      --projection_ckpt checkpoints_speaker_encoder/projection_latest.pt \
      --n_interferers 1,2,3 --n_samples_per_condition 4 \
      --output_dir outputs/listening_samples
"""
import os
import sys
import json
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import torch
import torch.nn.functional as F
import soundfile as sf
from omegaconf import OmegaConf

from data.mel import MelSpectrogramExtractor, segment_waveform, make_frame_mask, masked_mse
from data.augment import load_audio, mix_multi_at_snr, trim_trailing_silence
from inference.vocoder import load_hifigan_generator, mel_to_audio_hifigan
from models.speaker_encoder import SpeakerEncoder

from eval.results_stage2 import load_masking, load_flow
from eval.audio_domain_quality import si_sdr
from eval.eval_multi_speaker import build_speaker_index

README_TEXT = """Mask2Flow-TSE -- listening samples for manual verification
============================================================================

WHY THIS EXISTS
----------------
The project's automated metrics (mel-MSE %, SI-SDR, WavLM cosine similarity,
corpus-wide accuracy/EER) are all proxies for "did we extract the right
speaker's voice, and does it sound right." This directory has the actual
audio so you can judge that yourself, for 2/3/4-total-speaker mixtures.

FOLDER LAYOUT
-------------
  2speakers/sample_00/, sample_01/, ...
  3speakers/sample_00/, sample_01/, ...
  4speakers/sample_00/, sample_01/, ...

Each sample_NN/ folder has 5 files + info.json:
  mixture.wav       what a microphone would pick up (target + all interferers)
  reference.wav     the enrollment clip the system was told to extract
                     ("find THIS voice in the mixture") -- trailing silence
                     already trimmed off
  extracted.wav     *** the actual system output -- listen to this one for
                     "did it get the right voice" ***
  target_clean.wav  ground-truth clean target, vocoded from ITS OWN real
                     mel (not predicted) -- the ceiling / oracle
  stage1_only.wav   masking-only output, before flow matching -- useful to
                     hear how much Stage 2 changes things
  info.json         speaker IDs, SNRs, and every metric for this exact
                     sample (mel %, SI-SDR, cosine similarity)

SUGGESTED LISTENING ORDER
--------------------------
1. reference.wav       -- learn what the target voice sounds like
2. mixture.wav          -- hear how buried it is in the mixture
3. extracted.wav        -- did the system pull the right voice out?
4. target_clean.wav     -- compare against the true target
5. stage1_only.wav      -- optional, hear the masking-only intermediate

TWO SPECIFIC THINGS TO LISTEN FOR
-----------------------------------
(A) "Metallic" vocoder quality. Compare extracted.wav against
    target_clean.wav. Both go through the SAME vocoder.
      - If target_clean.wav ALSO sounds metallic: that's a vocoder quality
        issue, not an extraction issue -- the vocoder itself struggles even
        with a real, undistorted mel spectrogram.
      - If target_clean.wav sounds clean/natural but extracted.wav sounds
        metallic: that's specifically Stage 2's mel prediction being
        imperfect (a distribution mismatch from what the vocoder was
        trained on), not a vocoder training problem.

(B) Low speaker-similarity despite a correct-sounding voice. info.json's
    speaker_sim_ref_vs_extracted is the cosine similarity between the
    reference embedding and the extracted output's embedding (same metric
    family as the project's accuracy/EER numbers). If the voice sounds
    right to you but this number seems low, check
    speaker_sim_ref_vs_target_clean first -- that's the SAME reference
    compared against the true clean target (same speaker, different
    utterance), and should be the highest number in the file. If even
    THAT number looks lower than expected, the reference clip itself may
    still be short/atypical for this speaker (trailing silence is now
    trimmed, but a reference clip that's just inherently a-typical for a
    speaker's voice is a separate, real limitation of any embedding-based
    metric, not a bug).

METRICS DEFINITIONS (info.json)
---------------------------------
  mel_s2_vs_s1_pct               mel-domain %-improvement, Stage 2 vs Stage 1
  sisdr_*_vs_target              SI-SDR (dB) of that signal against the true
                                  clean target -- higher is better
  speaker_sim_ref_vs_target_clean  reference vs. true target (sanity ceiling)
  speaker_sim_ref_vs_mixture       reference vs. raw mixture (do-nothing baseline)
  speaker_sim_ref_vs_stage1        reference vs. masking-only output
  speaker_sim_ref_vs_extracted     reference vs. the actual system output
                                    (THIS is the number reported throughout
                                    the project as "accuracy"/"similarity")
"""


@torch.no_grad()
def generate_one(sample_idx, base_seed, speaker_utterances, speakers, n_interferers,
                  cfg, mel_extractor, mask_model, flow_model, encoder, vocoder,
                  cfg_scale, n_steps, snr_min, snr_max, device):
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
    target_path = utterances[utt_idx[0]]
    ref_path = utterances[utt_idx[1]]

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
    interferer_wavs = [segment_waveform(w, sr, seg_len, random_start=True) for w in interferer_wavs_raw]

    snr_db_list = [random.uniform(snr_min, snr_max) for _ in range(n_interferers)]
    mixture_wav, target_matched, _ = mix_multi_at_snr(target_wav, interferer_wavs, snr_db_list)

    mixture_mel = mel_extractor(mixture_wav.to(device))
    target_mel = mel_extractor(target_matched.to(device))

    target_valid_frames = min(target_valid_samples // mel_extractor.hop_length, target_mel.shape[-1])
    frame_mask = make_frame_mask(torch.tensor([target_valid_frames]), target_mel.shape[-1], device=device)

    d_vec = encoder(ref_wav.unsqueeze(0).to(device))
    stage1_out, _ = mask_model(mixture_mel.unsqueeze(0), d_vec)
    stage2_out = flow_model.inference(stage1_out, d_vec, cfg_scale=cfg_scale, n_steps=n_steps)

    mix_b = mixture_mel.unsqueeze(0)
    tgt_b = target_mel.unsqueeze(0)
    mix_mse = masked_mse(mix_b, tgt_b, frame_mask).item()
    s1_mse = masked_mse(stage1_out, tgt_b, frame_mask).item()
    s2_mse = masked_mse(stage2_out, tgt_b, frame_mask).item()

    vf = max(target_valid_frames, 1)
    mix_wav_out = mel_to_audio_hifigan(mixture_mel[:, :vf], vocoder)
    s1_wav_out = mel_to_audio_hifigan(stage1_out[0, :, :vf], vocoder)
    s2_wav_out = mel_to_audio_hifigan(stage2_out[0, :, :vf], vocoder)
    tgt_wav_out = mel_to_audio_hifigan(target_mel[:, :vf], vocoder)

    mix_t = torch.from_numpy(mix_wav_out).float().to(device)
    s1_t = torch.from_numpy(s1_wav_out).float().to(device)
    s2_t = torch.from_numpy(s2_wav_out).float().to(device)
    tgt_t = torch.from_numpy(tgt_wav_out).float().to(device)

    sisdr_mix = si_sdr(mix_t, tgt_t)
    sisdr_s1 = si_sdr(s1_t, tgt_t)
    sisdr_s2 = si_sdr(s2_t, tgt_t)

    emb_ref = d_vec[0]
    emb_mix = encoder(mix_t.unsqueeze(0))[0]
    emb_s1 = encoder(s1_t.unsqueeze(0))[0]
    emb_s2 = encoder(s2_t.unsqueeze(0))[0]
    emb_tgt = encoder(tgt_t.unsqueeze(0))[0]

    def sim(a, b):
        return F.cosine_similarity(a, b, dim=0).item()

    info = {
        "sample_idx": sample_idx,
        "seed": seed,
        "n_interferers": n_interferers,
        "n_total_speakers": n_interferers + 1,
        "target_speaker": target_speaker,
        "interferer_speakers": interferer_speakers,
        "snr_db_list": [round(s, 2) for s in snr_db_list],
        "mel_mix_mse": mix_mse, "mel_s1_mse": s1_mse, "mel_s2_mse": s2_mse,
        "mel_s2_vs_s1_pct": (s1_mse - s2_mse) / max(s1_mse, 1e-12) * 100,
        "sisdr_mixture_vs_target": sisdr_mix,
        "sisdr_stage1_vs_target": sisdr_s1,
        "sisdr_extracted_vs_target": sisdr_s2,
        "speaker_sim_ref_vs_target_clean": sim(emb_ref, emb_tgt),
        "speaker_sim_ref_vs_mixture": sim(emb_ref, emb_mix),
        "speaker_sim_ref_vs_stage1": sim(emb_ref, emb_s1),
        "speaker_sim_ref_vs_extracted": sim(emb_ref, emb_s2),
    }

    audio = {
        "mixture.wav": mixture_wav.numpy(),
        "reference.wav": ref_wav.numpy(),
        "target_clean.wav": tgt_wav_out,
        "stage1_only.wav": s1_wav_out,
        "extracted.wav": s2_wav_out,
    }
    return audio, info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default_v2.yaml")
    parser.add_argument("--mask_ckpt", default="checkpoints_v2/masking/mask_best.pt")
    parser.add_argument("--flow_ckpt", default="checkpoints_v2/flow/flow_best.pt")
    parser.add_argument("--projection_ckpt", default="checkpoints_speaker_encoder/projection_latest.pt")
    parser.add_argument("--vocoder_ckpt", default="checkpoints_vocoder/vocoder_best.pt")
    parser.add_argument("--vocoder_config", default="configs/vocoder.yaml")
    parser.add_argument("--librispeech_dir", default="data/raw/LibriSpeech")
    parser.add_argument("--split", default="test-clean")
    parser.add_argument("--n_interferers", default="1,2,3",
                         help="Comma-separated interferer counts, e.g. 1,2,3 = 2/3/4 total speakers.")
    parser.add_argument("--n_samples_per_condition", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123,
                         help="Different default from every metrics eval in this project (42), "
                              "so this is a fresh spot-check draw, not the exact same samples "
                              "already scored elsewhere.")
    parser.add_argument("--snr_min", type=float, default=None, help="Default: cfg.data.snr_min")
    parser.add_argument("--snr_max", type=float, default=None, help="Default: cfg.data.snr_max")
    parser.add_argument("--cfg_scale", type=float, default=None)
    parser.add_argument("--n_steps", type=int, default=None)
    parser.add_argument("--output_dir", default="outputs/listening_samples")
    args = parser.parse_args()

    n_interferers_list = [int(x) for x in args.n_interferers.split(",")]

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_scale = args.cfg_scale if args.cfg_scale is not None else cfg.inference.cfg_scale
    n_steps = args.n_steps if args.n_steps is not None else cfg.inference.flow_steps
    snr_min = args.snr_min if args.snr_min is not None else cfg.data.snr_min
    snr_max = args.snr_max if args.snr_max is not None else cfg.data.snr_max

    print(f"[ExportSamples] Device: {device}  cfg_scale={cfg_scale}  n_steps={n_steps}  "
          f"conditions(n_interferers)={n_interferers_list}  "
          f"n_per_condition={args.n_samples_per_condition}")

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

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    sr = cfg.audio.sample_rate
    manifest = []

    for n_interferers in n_interferers_list:
        n_total = n_interferers + 1
        cond_dir = out_root / f"{n_total}speakers"
        cond_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {n_total} total speakers ({args.n_samples_per_condition} samples) -> {cond_dir} ===")

        for i in range(args.n_samples_per_condition):
            audio, info = generate_one(
                i, args.seed, speaker_utterances, speakers, n_interferers,
                cfg, mel_extractor, mask_model, flow_model, encoder, vocoder,
                cfg_scale, n_steps, snr_min, snr_max, device,
            )
            sample_dir = cond_dir / f"sample_{i:02d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            for fname, wav_array in audio.items():
                sf.write(str(sample_dir / fname), wav_array, sr)
            with open(sample_dir / "info.json", "w", encoding="utf-8") as f:
                json.dump(info, f, indent=2)
            manifest.append({"sample_dir": str(sample_dir), **info})
            print(f"  sample_{i:02d}: target={info['target_speaker']} "
                  f"interferers={info['interferer_speakers']} "
                  f"ref_vs_extracted_sim={info['speaker_sim_ref_vs_extracted']:.3f} "
                  f"(baseline ref_vs_mixture={info['speaker_sim_ref_vs_mixture']:.3f}, "
                  f"ceiling ref_vs_target={info['speaker_sim_ref_vs_target_clean']:.3f}) "
                  f"-> {sample_dir}")

    with open(out_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    readme_path = out_root / "README.txt"
    readme_path.write_text(README_TEXT, encoding="utf-8")

    print(f"\n[ExportSamples] Done. {len(manifest)} samples written under {out_root}/")
    print(f"[ExportSamples] Read {readme_path} first -- explains what each file is, listening "
          f"order, and how to diagnose the metallic-vocoder / low-similarity questions.")
