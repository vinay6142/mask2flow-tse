"""
On-the-fly data augmentation for Mask2Flow-TSE.
Creates three types of mixtures during training:
  1. Clean       — target only (no interference)
  2. Additive    — target + interfering speaker
  3. Reverb      — additive mixture convolved with RIR
"""

import os
import sys
import random
import torch
import torchaudio
import soundfile as sf
import numpy as np
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_audio(
    path: str,
    target_sr: int = 16000,
) -> torch.Tensor:
    """
    Load audio file and resample if needed.

    Uses soundfile directly instead of torchaudio.load(), since the
    installed torchaudio version requires the optional torchcodec
    package (which itself requires ffmpeg) for .load() — neither is
    available on this cluster. soundfile reads flac/wav natively via
    libsndfile with no such dependency.

    Returns:
        waveform: (T,) mono float32 in [-1, 1]
    """
    data, sr = sf.read(path, dtype="float32", always_2d=True)  # (T, C)
    waveform = torch.from_numpy(data).T                        # (C, T)

    # convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # resample
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform  = resampler(waveform)

    return waveform.squeeze(0)   # (T,)


def trim_trailing_silence(
    wav: torch.Tensor,
    sr: int,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    rel_threshold_db: float = -40.0,
    trailing_buffer_ms: float = 100.0,
) -> torch.Tensor:
    """
    Trim trailing near-silence from a 1D waveform using a threshold RELATIVE
    to the clip's own peak energy, not an absolute zero check.

    This project's fixed-length (segment_length=10s) padding isn't always
    true digital silence: raw waveforms and mel-domain zero-padding produce
    exact zeros, but a vocoder (e.g. HiFi-GAN) fed a near-silent mel input
    can output a low-level noise floor (~0.0001 RMS) instead of true zero.
    A relative-dB threshold catches both cases, so this same function works
    whether the input is raw audio, a mixture, a ground-truth target, or a
    vocoder's output.

    Args:
        wav               : (T,) or (1, T) waveform
        sr                : sample rate
        frame_ms/hop_ms   : analysis frame/hop size for the energy scan
        rel_threshold_db  : frames this many dB below the clip's peak RMS
                             are considered "silent"
        trailing_buffer_ms: extra audio kept after the last active frame,
                             so natural decay/reverb tails aren't clipped
    Returns:
        trimmed waveform (T',) — unchanged if no clear silent tail is found
    """
    if wav.dim() == 2:
        wav = wav.squeeze(0)

    frame_len = int(sr * frame_ms / 1000)
    hop_len   = int(sr * hop_ms / 1000)
    n_samples = wav.shape[0]

    if n_samples < frame_len:
        return wav  # too short to analyze meaningfully, leave as-is

    n_frames = 1 + (n_samples - frame_len) // hop_len
    if n_frames <= 1:
        return wav

    frame_rms = torch.empty(n_frames, device=wav.device)
    for i in range(n_frames):
        start = i * hop_len
        seg = wav[start:start + frame_len]
        frame_rms[i] = torch.sqrt(torch.mean(seg ** 2) + 1e-12)

    peak = frame_rms.max().clamp(min=1e-8)
    rms_db = 20.0 * torch.log10(frame_rms / peak + 1e-8)
    active = rms_db > rel_threshold_db

    if not active.any():
        # degenerate case (e.g. uniform noise floor with no clear peak) —
        # don't trim anything, safer than accidentally trimming everything
        return wav

    last_active_frame = active.nonzero().max().item()
    end_sample = last_active_frame * hop_len + frame_len
    end_sample = min(n_samples, end_sample + int(sr * trailing_buffer_ms / 1000))

    return wav[:end_sample]


def mix_at_snr(
    target: torch.Tensor,
    interferer: torch.Tensor,
    snr_db: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mix target and interferer at a given SNR (dB).
    
    Args:
        target    : (T,) target speaker waveform
        interferer: (T,) interfering speaker waveform
        snr_db    : desired signal-to-noise ratio in dB
    Returns:
        mixture  : (T,) mixed waveform
        interferer_scaled: (T,) scaled interferer
    """
    # match lengths
    min_len    = min(len(target), len(interferer))
    target     = target[:min_len]
    interferer = interferer[:min_len]

    # compute RMS energies
    target_rms     = target.pow(2).mean().sqrt().clamp(min=1e-8)
    interferer_rms = interferer.pow(2).mean().sqrt().clamp(min=1e-8)

    # scale interferer to achieve desired SNR
    desired_interferer_rms = target_rms / (10 ** (snr_db / 20))
    scale                  = desired_interferer_rms / interferer_rms
    interferer_scaled      = interferer * scale

    mixture = target + interferer_scaled

    # normalize mixture to prevent clipping
    max_val = mixture.abs().max().clamp(min=1e-8)
    if max_val > 1.0:
        mixture           = mixture / max_val
        target            = target / max_val
        interferer_scaled = interferer_scaled / max_val

    return mixture, interferer_scaled


def apply_rir(
    waveform: torch.Tensor,
    rir: torch.Tensor,
) -> torch.Tensor:
    """
    Convolve waveform with a room impulse response (RIR) using
    FFT-based convolution — critical for speed with long kernels.
    The previous direct-form torch.nn.functional.conv1d was O(T*K)
    and took ~500ms per call at T=160,000, K=4,000, causing severe
    per-step slowdowns whenever a batch sampled the "reverb" condition.

    Args:
        waveform: (T,)
        rir     : (T_rir,) room impulse response
    Returns:
        reverberant: (T,) same length as input
    """
    # normalize RIR
    rir = rir / (rir.abs().max().clamp(min=1e-8))

    waveform_len = len(waveform)
    conv_len     = waveform_len + len(rir) - 1

    # next power of 2 for efficient FFT
    n_fft = 1
    while n_fft < conv_len:
        n_fft *= 2

    W = torch.fft.rfft(waveform, n=n_fft)
    R = torch.fft.rfft(rir,      n=n_fft)
    reverberant = torch.fft.irfft(W * R, n=n_fft)[:waveform_len]

    # normalize
    max_val = reverberant.abs().max().clamp(min=1e-8)
    if max_val > 1.0:
        reverberant = reverberant / max_val

    return reverberant

class RIRLoader:
    """
    Loads and caches room impulse responses from a directory.
    Falls back to a synthetic RIR if no RIRs are available.
    """

    def __init__(self, rir_dir: Optional[str] = None, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.rir_paths: List[str] = []

        if rir_dir and Path(rir_dir).exists():
            self.rir_paths = [
                str(p) for p in Path(rir_dir).rglob("*.wav")
            ]
            print(f"[RIRLoader] Found {len(self.rir_paths)} RIRs in {rir_dir}")
        else:
            print("[RIRLoader] No RIR directory found — will use synthetic RIRs")

    def sample(self) -> torch.Tensor:
        """Return a random RIR tensor."""
        if self.rir_paths:
            path = random.choice(self.rir_paths)
            return load_audio(path, self.sample_rate)
        else:
            return self._synthetic_rir()

    def _synthetic_rir(self, length: int = 4000) -> torch.Tensor:
        """
        Simple synthetic exponentially decaying RIR.
        Used as fallback when no real RIRs are available.
        """
        t      = torch.arange(length, dtype=torch.float32)
        decay  = torch.exp(-t / (self.sample_rate * 0.1))   # 100ms decay
        noise  = torch.randn(length)
        rir    = noise * decay
        return rir / rir.abs().max().clamp(min=1e-8)


class MixtureCreator:
    """
    Creates speech mixtures for training Mask2Flow-TSE.
    
    Three conditions (probabilities configurable):
      - Clean   : target only, no interferer
      - Additive: target + interferer at random SNR
      - Reverb  : additive mixture + room reverberation
    """

    def __init__(
        self,
        rir_dir:        Optional[str] = None,
        sample_rate:    int   = 16000,
        snr_min:        float = 1.0,
        snr_max:        float = 10.0,
        clean_prob:     float = 0.33,
        additive_prob:  float = 0.33,
        reverb_prob:    float = 0.34,
    ):
        self.sample_rate   = sample_rate
        self.snr_min       = snr_min
        self.snr_max       = snr_max
        self.clean_prob    = clean_prob
        self.additive_prob = additive_prob
        self.reverb_prob   = reverb_prob

        self.rir_loader = RIRLoader(rir_dir, sample_rate)

        assert abs(clean_prob + additive_prob + reverb_prob - 1.0) < 1e-5, \
            "Condition probabilities must sum to 1.0"

    def create_mixture(
        self,
        target: torch.Tensor,
        interferer: Optional[torch.Tensor] = None,
        condition: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, str, Optional[float]]:
        """
        Create a mixture from target and optional interferer.
        
        Args:
            target    : (T,) clean target speaker waveform
            interferer: (T,) interfering speaker waveform (None for clean)
            condition : force a specific condition, or None to sample randomly
        Returns:
            mixture  : (T,) mixed/augmented waveform
            target   : (T,) clean target (possibly trimmed to match)
            condition: which condition was applied
        """
        # sample condition if not forced
        if condition is None:
            r = random.random()
            if r < self.clean_prob:
                condition = "clean"
            elif r < self.clean_prob + self.additive_prob:
                condition = "additive"
            else:
                condition = "reverb"

        # fall back to clean if no interferer provided
        if interferer is None and condition != "clean":
            condition = "clean"

        if condition == "clean":
            return target.clone(), target.clone(), condition, None

        # sample SNR
        snr_db = random.uniform(self.snr_min, self.snr_max)

        if condition == "additive":
            mixture, _ = mix_at_snr(target, interferer, snr_db)
            # trim target to mixture length
            target = target[:len(mixture)]
            return mixture, target, condition, snr_db

        elif condition == "reverb":
            # mix first, then reverberate both mixture and target
            mixture, _ = mix_at_snr(target, interferer, snr_db)
            rir        = self.rir_loader.sample()
            mixture    = apply_rir(mixture, rir)
            target     = target[:len(mixture)]
            return mixture, target, condition, snr_db

        else:
            raise ValueError(f"Unknown condition: {condition}")

# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing augmentation pipeline...")

    creator = MixtureCreator(snr_min=1.0, snr_max=10.0)

    # fake waveforms
    target     = torch.randn(16000 * 5)   # 5 seconds
    interferer = torch.randn(16000 * 5)

    for condition in ["clean", "additive", "reverb"]:
        mix, tgt, cond = creator.create_mixture(
            target, interferer, condition=condition
        )
        print(f"  [{cond:10s}] mixture: {mix.shape}  target: {tgt.shape}  "
              f"max: {mix.abs().max():.3f}")

    print("All augmentation tests passed!")