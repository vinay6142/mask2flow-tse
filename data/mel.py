"""
Mel spectrogram utilities for Mask2Flow-TSE.
All mel parameters match Whisper's expected input format
so that our output can be directly fed into Whisper ASR.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import numpy as np
from typing import Optional, Tuple


class MelSpectrogramExtractor(nn.Module):
    """
    Converts raw waveform to log-mel spectrogram.
    Parameters match Whisper's preprocessing exactly.
    
    Input : (B, T) or (T,)  waveform, float32, range [-1, 1]
    Output: (B, n_mels, frames) or (n_mels, frames)  log-mel
    """

    def __init__(
        self,
        sample_rate: int   = 16000,
        n_mels: int        = 80,
        n_fft: int         = 1024,
        hop_length: int    = 160,
        win_length: int    = 400,
        f_min: float       = 0.0,
        f_max: float       = 8000.0,
        log_offset: float  = 1e-8,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_mels      = n_mels
        self.hop_length  = hop_length
        self.log_offset  = log_offset

        self.mel_transform = T.MelSpectrogram(
            sample_rate = sample_rate,
            n_fft       = n_fft,
            win_length  = win_length,
            hop_length  = hop_length,
            f_min       = f_min,
            f_max       = f_max,
            n_mels      = n_mels,
            power       = 2.0,          # power spectrogram
            normalized  = False,
            center      = True,
            pad_mode    = "reflect",
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, T) or (T,)
        Returns:
            log_mel: (B, n_mels, frames) or (n_mels, frames)
        """
        squeeze = waveform.dim() == 1
        if squeeze:
            waveform = waveform.unsqueeze(0)         # (1, T)

        mel = self.mel_transform(waveform)           # (B, n_mels, frames)
        log_mel = torch.log(mel + self.log_offset)   # log compression
        
        if squeeze:
            log_mel = log_mel.squeeze(0)             # (n_mels, frames)

        return log_mel

    def frames_to_samples(self, n_frames: int) -> int:
        """Convert number of mel frames to audio samples."""
        return n_frames * self.hop_length

    def samples_to_frames(self, n_samples: int) -> int:
        """Convert number of audio samples to mel frames."""
        return n_samples // self.hop_length


class MelNormalizer:
    """
    L2-normalizes mel spectrograms per the masking module input,
    as described in Section 5.2.2 of the paper.
    
    Also provides global stats normalization as an alternative.
    """

    def __init__(self, mode: str = "l2"):
        """
        Args:
            mode: 'l2'    — per-sample L2 normalization (paper's method)
                  'global' — normalize using dataset mean/std
        """
        assert mode in ("l2", "global"), f"Unknown mode: {mode}"
        self.mode = mode

        # populated only when mode='global'
        self.mean: Optional[float] = None
        self.std:  Optional[float] = None

    def normalize(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (B, n_mels, T) or (n_mels, T)
        Returns:
            normalized mel of same shape
        """
        if self.mode == "l2":
            return self._l2_normalize(mel)
        else:
            return self._global_normalize(mel)

    def _l2_normalize(self, mel: torch.Tensor) -> torch.Tensor:
        """Per-sample L2 normalization across all elements."""
        norm = mel.norm(p=2, dim=(-2, -1), keepdim=True).clamp(min=1e-8)
        return mel / norm

    def _global_normalize(self, mel: torch.Tensor) -> torch.Tensor:
        """Normalize using precomputed dataset statistics."""
        assert self.mean is not None, "Call compute_stats() first."
        return (mel - self.mean) / (self.std + 1e-8)

    def compute_stats(
        self,
        mel_list: list,
        max_samples: int = 10000,
    ) -> Tuple[float, float]:
        """
        Compute dataset-level mean and std from a list of mel tensors.
        
        Args:
            mel_list  : list of (n_mels, T) tensors
            max_samples: max number of frames to use
        Returns:
            (mean, std)
        """
        all_values = []
        for mel in mel_list[:max_samples]:
            all_values.append(mel.flatten())
        
        all_values = torch.cat(all_values)
        self.mean  = all_values.mean().item()
        self.std   = all_values.std().item()

        print(f"[MelNormalizer] mean={self.mean:.4f}  std={self.std:.4f}")
        return self.mean, self.std


def pad_or_trim(
    mel: torch.Tensor,
    target_frames: int,
    pad_value: float = -11.5,  # log(1e-5), a sensible silence value
) -> torch.Tensor:
    """
    Pad or trim a mel spectrogram to exactly target_frames.
    
    Args:
        mel          : (..., n_mels, T)
        target_frames: desired number of time frames
        pad_value    : value used for padding (silence)
    Returns:
        mel of shape (..., n_mels, target_frames)
    """
    current_frames = mel.shape[-1]

    if current_frames == target_frames:
        return mel
    elif current_frames > target_frames:
        # trim
        return mel[..., :target_frames]
    else:
        # pad
        pad_size = target_frames - current_frames
        pad = torch.full(
            (*mel.shape[:-1], pad_size),
            fill_value = pad_value,
            dtype      = mel.dtype,
            device     = mel.device,
        )
        return torch.cat([mel, pad], dim=-1)


def segment_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    segment_length: float,
    random_start: bool = True,
) -> torch.Tensor:
    """
    Extract a fixed-length segment from a waveform.
    Used during training to get consistent-length inputs.
    
    Args:
        waveform      : (T,) or (1, T)
        sample_rate   : audio sample rate
        segment_length: desired segment length in seconds
        random_start  : if True, pick a random start point
    Returns:
        segment: (segment_samples,)
    """
    if waveform.dim() == 2:
        waveform = waveform.squeeze(0)

    segment_samples = int(segment_length * sample_rate)
    total_samples   = waveform.shape[0]

    if total_samples >= segment_samples:
        if random_start:
            max_start = total_samples - segment_samples
            start     = torch.randint(0, max_start + 1, (1,)).item()
        else:
            start = 0
        return waveform[start : start + segment_samples]
    else:
        # pad if shorter than desired length
        pad = torch.zeros(segment_samples - total_samples, dtype=waveform.dtype)
        return torch.cat([waveform, pad])


def make_frame_mask(
    valid_frames: torch.Tensor,
    total_frames: int,
    device=None,
) -> torch.Tensor:
    """
    Build a binary (B, T) mask marking real (non-padded) mel frames.

    segment_waveform/pad_or_trim always append padding at the END of a
    short utterance (never a random offset), so a single valid-length
    count per sample is sufficient to reconstruct which frames are real.

    Args:
        valid_frames: (B,) long tensor — number of real (non-padded)
                      frames for each sample in the batch
        total_frames: T, the padded/trimmed frame count all samples share
        device      : device to build the mask on (defaults to
                      valid_frames.device)
    Returns:
        mask: (B, T) float tensor, 1.0 = real frame, 0.0 = padding
    """
    if device is None:
        device = valid_frames.device
    idx = torch.arange(total_frames, device=device).unsqueeze(0)   # (1, T)
    vf  = valid_frames.to(device).unsqueeze(1)                     # (B, 1)
    return (idx < vf).float()                                      # (B, T)


def masked_mse(
    pred:       torch.Tensor,
    target:     torch.Tensor,
    frame_mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    MSE loss that ignores padded frames.

    Args:
        pred, target: (B, n_mels, T)
        frame_mask  : (B, T) or (B, 1, T), 1=real frame, 0=padding.
                      If None, falls back to plain F.mse_loss (unmasked)
                      so existing callers keep working unchanged.
    Returns:
        scalar loss
    """
    if frame_mask is None:
        return F.mse_loss(pred, target)
    if frame_mask.dim() == 2:
        frame_mask = frame_mask.unsqueeze(1)                # (B, 1, T)
    diff2 = (pred - target) ** 2 * frame_mask
    denom = frame_mask.sum() * pred.shape[1] + 1e-8          # ×n_mels (mask is per-frame)
    return diff2.sum() / denom


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing MelSpectrogramExtractor...")
    
    extractor  = MelSpectrogramExtractor()
    normalizer = MelNormalizer(mode="l2")

    # fake waveform: batch of 2, 3 seconds each
    waveform = torch.randn(2, 16000 * 3)

    mel     = extractor(waveform)
    mel_norm = normalizer.normalize(mel)

    print(f"  Waveform shape : {waveform.shape}")
    print(f"  Mel shape      : {mel.shape}")
    print(f"  Mel norm shape : {mel_norm.shape}")
    print(f"  Mel min/max    : {mel.min():.2f} / {mel.max():.2f}")

    # test single waveform (no batch dim)
    single   = torch.randn(16000 * 5)
    mel_s    = extractor(single)
    print(f"  Single mel     : {mel_s.shape}")

    # test pad/trim
    padded   = pad_or_trim(mel, target_frames=500)
    trimmed  = pad_or_trim(mel, target_frames=200)
    print(f"  Padded shape   : {padded.shape}")
    print(f"  Trimmed shape  : {trimmed.shape}")

    print("All tests passed!")