"""
VocoderDataset — builds (mel, waveform) training pairs directly from
LibriSpeech clean audio, using this project's own MelSpectrogramExtractor
so the vocoder is trained on exactly the same mel representation Stage 2
actually outputs at inference time.

No separate pairing/preprocessing step or cached dataset needed — pairs
are built on the fly per __getitem__, same philosophy as this project's
existing LibriSpeechTSEDataset.
"""
import random
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import Dataset

from data.augment import load_audio
from data.mel import MelSpectrogramExtractor, segment_waveform


class VocoderDataset(Dataset):
    def __init__(
        self,
        librispeech_root: str,
        splits: List[str],
        cfg,
        segment_size: int = 8192,   # waveform samples per training crop
        is_train: bool = True,
        seed: int = None,
    ):
        self.cfg = cfg
        self.sr = cfg.audio.sample_rate
        self.segment_size = segment_size
        self.is_train = is_train
        self.seed = seed

        self.mel = MelSpectrogramExtractor(
            sample_rate = cfg.audio.sample_rate,
            n_mels      = cfg.mel.n_mels,
            n_fft       = cfg.mel.n_fft,
            hop_length  = cfg.mel.hop_length,
            win_length  = cfg.mel.win_length,
            f_min       = cfg.mel.f_min,
            f_max       = cfg.mel.f_max,
            log_offset  = cfg.mel.log_offset,
        )

        self.utterances = self._build_index(librispeech_root, splits)
        print(f"[VocoderDataset] {len(self.utterances)} utterances "
              f"({'train' if is_train else 'val'})")

    def _build_index(self, root: str, splits: List[str]) -> List[str]:
        paths = []
        root_path = Path(root).resolve()
        for split in splits:
            split_path = root_path / split
            paths.extend(str(p) for p in split_path.rglob("*.flac"))
        if len(paths) == 0:
            raise FileNotFoundError(
                f"Found 0 .flac files under {root_path} for splits {splits}. "
                f"Check that you're running from the project root (not a "
                f"subdirectory) and that librispeech_path in the config is "
                f"correct relative to your current working directory."
            )
        return paths

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.seed is not None:
            random.seed(self.seed + idx)
            torch.manual_seed(self.seed + idx)

        path = self.utterances[idx]
        wav = load_audio(path, self.sr)  # (T,) mono float32

        seg_seconds = self.segment_size / self.sr
        wav = segment_waveform(wav, self.sr, seg_seconds, random_start=self.is_train)

        # Guarantee EXACT segment_size after crop/pad (segment_waveform
        # rounds to whole seconds internally in some edge cases — enforce
        # the exact sample count HiFi-GAN needs for its fixed-length crops).
        if wav.shape[-1] > self.segment_size:
            wav = wav[..., :self.segment_size]
        elif wav.shape[-1] < self.segment_size:
            wav = torch.nn.functional.pad(wav, (0, self.segment_size - wav.shape[-1]))

        mel = self.mel(wav)  # (n_mels, T) — computed from the SAME crop, guaranteed aligned

        return {"mel": mel, "waveform": wav}


def build_vocoder_dataloaders(cfg, segment_size=8192, batch_size=16, num_workers=4):
    from torch.utils.data import DataLoader

    train_ds = VocoderDataset(
        librispeech_root=cfg.data.librispeech_path,
        splits=cfg.data.train_splits,
        cfg=cfg, segment_size=segment_size, is_train=True,
    )
    val_ds = VocoderDataset(
        librispeech_root=cfg.data.librispeech_path,
        splits=["test-clean"],
        cfg=cfg, segment_size=segment_size, is_train=False, seed=42,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, drop_last=False)
    return train_loader, val_loader