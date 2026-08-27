"""
Griffin-Lim vocoder — mel spectrogram -> waveform, no training required.

This is a placeholder for Week 1-2: it lets the full pipeline run
end-to-end and produce audible output today. Quality is rougher than a
trained neural vocoder (HiFi-GAN) — swap this out once one is trained
or found matching this project's exact mel config (see notes: hop=160,
win=400, 16kHz — NOT a standard TTS vocoder config, most pretrained
HiFi-GAN checkpoints will NOT match and will sound garbled).
"""
import torch
import librosa
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vocoder_train"))
from hifigan import Generator


def mel_to_audio_griffinlim(log_mel, cfg, n_iter=60):
    """
    log_mel: (n_mels, T) tensor, log-scale as produced by the model
              (matches this project's mel = log(power_mel + log_offset))
    cfg: the loaded OmegaConf config (needs cfg.mel.*, cfg.audio.sample_rate)
    Returns: 1D numpy waveform
    """
    log_mel_np = log_mel.detach().cpu().numpy()

    # Invert the log step: log_mel = log(power_mel + log_offset)
    power_mel = np.exp(log_mel_np) - cfg.mel.log_offset
    power_mel = np.clip(power_mel, a_min=0.0, a_max=None)  # guard tiny negative noise

    waveform = librosa.feature.inverse.mel_to_audio(
        power_mel,
        sr=cfg.audio.sample_rate,
        n_fft=cfg.mel.n_fft,
        hop_length=cfg.mel.hop_length,
        win_length=cfg.mel.win_length,
        power=cfg.mel.power,
        fmin=cfg.mel.f_min,
        fmax=cfg.mel.f_max,
        n_iter=n_iter,
    )
    return waveform


def load_hifigan_generator(ckpt_path, vc_cfg, device):
    """
    Loads the trained HiFi-GAN generator for inference.

    vc_cfg is the vocoder config (configs/vocoder.yaml) — NOT the main
    default_v2.yaml, since only vocoder.yaml has the `vocoder:` block with
    the generator's architecture hyperparameters (upsample_rates etc).
    Only the generator is loaded — the discriminators (MPD/MSD) exist
    purely for computing the training-time adversarial loss and play no
    role in inference; the checkpoint's "mpd"/"msd" keys are ignored here.
    """

    vc = vc_cfg.vocoder
    generator = Generator(
        n_mels=vc_cfg.mel.n_mels,
        upsample_rates=tuple(vc.upsample_rates),
        upsample_kernel_sizes=tuple(vc.upsample_kernel_sizes),
        upsample_initial_channel=vc.upsample_initial_channel,
        resblock_kernel_sizes=tuple(vc.resblock_kernel_sizes),
        resblock_dilation_sizes=tuple(tuple(d) for d in vc.resblock_dilation_sizes),
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    generator.load_state_dict(ckpt["generator"])
    generator.eval()
    for p in generator.parameters():
        p.requires_grad = False

    step = ckpt.get("step", "?")
    val_loss = ckpt.get("val_loss", "?")
    print(f"[Vocoder] Loaded HiFi-GAN generator from step {step} (val_loss={val_loss})")
    return generator


def mel_to_audio_hifigan(log_mel, generator):
    """
    log_mel: (n_mels, T) tensor, log-scale, as produced by Stage 2 — same
             representation the vocoder was trained to invert, no manual
             de-log/de-power step needed here (unlike Griffin-Lim, the
             network was trained directly on this exact representation).
    Returns: 1D numpy waveform, length T * hop_length (160)
    """
    with torch.no_grad():
        mel_batched = log_mel.unsqueeze(0).to(next(generator.parameters()).device)
        wav = generator(mel_batched)  # (1, 1, T*160)
    return wav.squeeze().cpu().numpy()
