"""
Real-world inference: mixture.wav + reference.wav -> extracted_speaker.wav

No synthetic mixing, no ground truth — takes a user-supplied mixture
recording directly. Replicates the exact preprocessing chain used in
data/librispeech.py's __getitem__ (mel extraction with NO mixture
normalization, reference via raw waveform for the speaker encoder),
so results stay consistent with everything already validated in eval/.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import soundfile as sf
from omegaconf import OmegaConf

from data.augment import load_audio
from data.mel import MelSpectrogramExtractor, pad_or_trim, segment_waveform
from eval.results_stage2 import load_masking, load_flow
from models.speaker_encoder import SpeakerEncoder
from inference.vocoder import mel_to_audio_griffinlim, load_hifigan_generator, mel_to_audio_hifigan


def extract_target_speaker(
    mixture_path, reference_path, cfg, device,
    mask_ckpt="checkpoints_v2/masking/mask_best.pt",
    flow_ckpt="checkpoints_v2/flow/flow_best.pt",
    out_path="outputs/extracted.wav",
):
    sr = cfg.audio.sample_rate

    mel_extractor = MelSpectrogramExtractor(
        sample_rate=sr, n_mels=cfg.mel.n_mels, n_fft=cfg.mel.n_fft,
        hop_length=cfg.mel.hop_length, win_length=cfg.mel.win_length,
        f_min=cfg.mel.f_min, f_max=cfg.mel.f_max, log_offset=cfg.mel.log_offset,
    ).to(device)

    encoder    = SpeakerEncoder(cfg.speaker_encoder.model_name, cfg.speaker_encoder.embed_dim, freeze=True).to(device).eval()
    mask_model = load_masking(mask_ckpt, cfg, device)
    flow_model = load_flow(flow_ckpt, cfg, device)

    # --- load real audio, no synthetic mixing (user already supplied a real mixture) ---
    mixture_wav   = load_audio(mixture_path, sr).to(device)
    reference_wav = load_audio(reference_path, sr).to(device)

    # Reference: fixed-length, DETERMINISTIC crop (random_start=False —
    # training/eval use True on purpose for augmentation variety; a
    # single real inference call must be reproducible, not random).
    ref_seg_len   = getattr(cfg.audio, "reference_length", 3.0)
    reference_wav = segment_waveform(reference_wav, sr, ref_seg_len, random_start=False)

    # --- mel extraction: mixture_mel stays UNNORMALIZED, matching training exactly ---
    mixture_mel = mel_extractor(mixture_wav)  # (n_mels, T_real) — T_real varies by clip length

    # Match the model's trained frame count. Compute it empirically from
    # the extractor itself (avoids off-by-one guessing vs. hardcoding).
    dummy = torch.zeros(int(cfg.audio.segment_length * sr), device=device)
    target_frames = mel_extractor(dummy).shape[-1]
    mixture_mel = pad_or_trim(mixture_mel, target_frames, pad_value=-11.5)  # silence, matches data/mel.py's convention

    mixture_mel = mixture_mel.unsqueeze(0)  # (1, n_mels, T)

    with torch.no_grad():
        d_vec = encoder(reference_wav.unsqueeze(0))
        stage1_out, _ = mask_model(mixture_mel, d_vec)
        stage2_out = flow_model.inference(
            stage1_out, d_vec,
            cfg_scale=cfg.inference.cfg_scale,
            n_steps=cfg.inference.flow_steps,
        )

    if args.vocoder == "hifigan":
        vc_cfg = OmegaConf.load(args.vocoder_config)
        generator = load_hifigan_generator(args.vocoder_ckpt, vc_cfg, device)
        waveform = mel_to_audio_hifigan(stage2_out[0], generator)
    else:
        waveform = mel_to_audio_griffinlim(stage2_out[0], cfg)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(out_path, waveform, sr)
    print(f"Saved extracted speech -> {out_path}")
    return stage2_out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixture",   required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--config",    default="configs/default_v2.yaml")
    parser.add_argument("--mask_ckpt", default="checkpoints_v2/masking/mask_best.pt")
    parser.add_argument("--flow_ckpt", default="checkpoints_v2/flow/flow_best.pt")
    parser.add_argument("--out",       default="outputs/extracted.wav")
    parser.add_argument("--vocoder", choices=["griffinlim", "hifigan"], default="hifigan")
    parser.add_argument("--vocoder_ckpt", default="checkpoints_vocoder/vocoder_best.pt")
    parser.add_argument("--vocoder_config", default="configs/vocoder.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extract_target_speaker(args.mixture, args.reference, cfg, device,
                            args.mask_ckpt, args.flow_ckpt, args.out)
