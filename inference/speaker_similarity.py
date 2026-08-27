"""
Speaker-similarity metric — the "accuracy" number for real clips with no
ground-truth clean target. Computes cosine similarity between the
reference speaker's embedding and the extracted output's embedding,
using the SAME frozen WavLM encoder as the rest of the pipeline (so this
number is self-consistent with everything the model itself was trained
and conditioned on).

Usage:
  python3 inference/speaker_similarity.py \
      --reference outputs/test_reference.wav \
      --extracted outputs/08_extracted_hifigan.wav
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from data.augment import load_audio, trim_trailing_silence
from models.speaker_encoder import SpeakerEncoder


def speaker_similarity(reference_path, extracted_path, cfg, device, trim_silence=True, verbose=True):
    encoder = SpeakerEncoder(
        model_name=cfg.speaker_encoder.model_name,
        embed_dim=cfg.speaker_encoder.embed_dim,
        freeze=True,
        projection_ckpt=args.projection_ckpt,
    ).to(device).eval()

    sr = cfg.audio.sample_rate
    ref_wav = load_audio(reference_path, sr).to(device)
    ext_wav = load_audio(extracted_path, sr).to(device)

    if trim_silence:
        ref_dur_before = ref_wav.shape[-1] / sr
        ext_dur_before = ext_wav.shape[-1] / sr
        ref_wav = trim_trailing_silence(ref_wav, sr)
        ext_wav = trim_trailing_silence(ext_wav, sr)
        if verbose:
            print(f"[Trim] reference: {ref_dur_before:.2f}s -> {ref_wav.shape[-1]/sr:.2f}s")
            print(f"[Trim] extracted: {ext_dur_before:.2f}s -> {ext_wav.shape[-1]/sr:.2f}s")

    with torch.no_grad():
        ref_emb = encoder(ref_wav.unsqueeze(0))
        ext_emb = encoder(ext_wav.unsqueeze(0))

    return F.cosine_similarity(ref_emb, ext_emb, dim=-1).item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--extracted", required=True)
    parser.add_argument("--config", default="configs/default_v2.yaml")
    parser.add_argument("--projection_ckpt", default="checkpoints_speaker_encoder/projection_latest.pt")
    parser.add_argument("--no_trim", action="store_true",
                         help="Disable trailing-silence trimming (old behavior, for comparison)")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sim = speaker_similarity(args.reference, args.extracted, cfg, device, trim_silence=not args.no_trim)
    print(f"\nSpeaker similarity (cosine): {sim:.4f}")
    print("Note: thresholds below are a rough illustrative scale, not a")
    print("calibrated standard — run the calibration commands in the reply")
    print("to see what 'good' looks like for THIS encoder/setup specifically.")
    if sim > 0.85:
        print("-> Very high: extracted voice strongly matches the reference speaker")
    elif sim > 0.70:
        print("-> Good match: likely the correct speaker, some residual artifacts")
    elif sim > 0.50:
        print("-> Weak match: partial speaker leakage or heavy degradation")
    else:
        print("-> Poor match: likely wrong speaker or severe corruption")