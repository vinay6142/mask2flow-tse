"""
Generates fake data for testing the pipeline
without needing real LibriSpeech data.
Saves ~30GB of storage during development.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import random
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig

from data.mel import MelSpectrogramExtractor, MelNormalizer


class FakeTSEDataset(Dataset):
    """
    Generates random mel spectrograms that
    mimic the shape of real TSE training data.
    Use this to test the full pipeline without
    downloading any real data.
    """

    def __init__(self, cfg: DictConfig, size: int = 1000):
        """
        Args:
            cfg : full omegaconf config
            size: number of fake samples to generate
        """
        self.cfg  = cfg
        self.size = size

        self.mel = MelSpectrogramExtractor(
            sample_rate = cfg.audio.sample_rate,
            n_mels      = cfg.mel.n_mels,
            n_fft       = cfg.mel.n_fft,
            hop_length  = cfg.mel.hop_length,
            win_length  = cfg.mel.win_length,
        )

        self.normalizer = MelNormalizer(mode="l2")

        # fixed frame count for segment_length seconds
        self.n_frames = (
            cfg.audio.segment_length
            * cfg.audio.sample_rate
            // cfg.mel.hop_length
        )

        print(f"[FakeDataset] {size} samples, "
              f"mel shape: ({cfg.mel.n_mels}, {self.n_frames})")

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        n_mels    = self.cfg.mel.n_mels
        n_frames  = self.n_frames

        # simulate realistic mel ranges
        # real log-mel values are roughly in [-10, 2]
        mixture_mel   = torch.randn(n_mels, n_frames) * 2 - 5
        target_mel    = torch.randn(n_mels, n_frames) * 2 - 5
        reference_mel = torch.randn(n_mels, n_frames // 4) * 2 - 5

        # raw waveform for speaker encoder (WavLM needs ~3 seconds)
        ref_samples   = 3 * self.cfg.audio.sample_rate
        reference_wav = torch.randn(ref_samples) * 0.1

        # NOTE: mixture_mel stays at raw log-mel scale, matching target_mel.
        # The model's forward() already applies its own internal F.normalize()
        # for the conv pathway; normalizing it here too collapsed it toward ~0,
        # making X_enh = X * mask unable to ever reach the target's scale.
        reference_mel = self.normalizer.normalize(reference_mel)

        return {
            "mixture_mel"  : mixture_mel,
            "target_mel"   : target_mel,
            "reference_mel": reference_mel,
            "reference_wav": reference_wav,
        }


def build_fake_dataloaders(
    cfg: DictConfig,
    train_size: int = 1000,
    val_size:   int = 100,
) -> tuple:
    """
    Build fake train and val dataloaders.
    Drop-in replacement for build_dataloaders()
    during development.
    """
    train_dataset = FakeTSEDataset(cfg, size=train_size)
    val_dataset   = FakeTSEDataset(cfg, size=val_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size  = cfg.train_masking.batch_size,
        shuffle     = True,
        num_workers = 0,     # 0 for fake data (no I/O bottleneck)
        pin_memory  = False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size  = cfg.train_masking.batch_size,
        shuffle     = False,
        num_workers = 0,
    )

    return train_loader, val_loader


# ── Quick sanity check ───────────────────────────────────────
if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg     = OmegaConf.load("configs/default.yaml")
    loader, _ = build_fake_dataloaders(cfg, train_size=50)

    batch = next(iter(loader))
    print("Fake batch shapes:")
    for k, v in batch.items():
        print(f"  {k:20s}: {v.shape}")

    print("\nFake data test passed!")
    print("No real data needed until training on Kaggle.")