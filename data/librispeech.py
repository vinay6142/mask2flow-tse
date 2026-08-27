"""
LibriSpeech dataset loader for Mask2Flow-TSE.
Builds on-the-fly speech mixtures during training
following the VoiceFilter-Lite data protocol.
"""

import os
import sys
import random
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from omegaconf import DictConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.mel import MelSpectrogramExtractor, MelNormalizer, segment_waveform
from data.augment import MixtureCreator, load_audio


class LibriSpeechTSEDataset(Dataset):
    """
    Dataset for Target Speaker Extraction training.
    
    For each sample:
      - Picks a random target speaker
      - Picks one utterance as the mixture component
      - Picks another utterance from same speaker as reference
      - Picks a random interfering speaker
      - Creates a mixture using MixtureCreator
    
    Returns mel spectrograms ready for Mask2Flow-TSE.
    """

    def __init__(
        self,
        librispeech_root: str,
        splits: List[str],
        cfg: DictConfig,
        is_train: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Args:
            librispeech_root: path to LibriSpeech directory
            splits          : list of splits e.g. ['train-clean-100']
            cfg             : full config (omegaconf)
            is_train        : if True, apply augmentation; else clean only
            seed            : optional random seed for reproducibility
        """
        self.cfg      = cfg
        self.is_train = is_train
        # If set, every __getitem__ call becomes fully deterministic —
        # same idx always yields the same speaker/utterance/interferer/
        # mixing choice, regardless of DataLoader worker/shuffle state.
        # Left as None during training so normal per-epoch randomness
        # is preserved; only set this for reproducible evaluation.
        self.seed     = seed

        # mel extractor
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

        self.normalizer = MelNormalizer(mode="l2")

        # mixture creator
        self.mixer = MixtureCreator(
            rir_dir       = cfg.data.rir_path if is_train else None,
            sample_rate   = cfg.audio.sample_rate,
            snr_min       = cfg.data.snr_min,
            snr_max       = cfg.data.snr_max,
            clean_prob    = cfg.data.conditions.clean_prob    if is_train else 0.0,
            additive_prob = cfg.data.conditions.additive_prob if is_train else 1.0,
            reverb_prob   = cfg.data.conditions.reverb_prob   if is_train else 0.0,
        )

        # build speaker -> utterance index
        self.speaker_utterances = self._build_index(librispeech_root, splits)
        self.speakers           = list(self.speaker_utterances.keys())

        # filter out speakers with fewer than 2 utterances
        # (we need one for mixture, one for reference)
        self.speakers = [
            s for s in self.speakers
            if len(self.speaker_utterances[s]) >= 2
        ]

        print(f"[Dataset] {len(self.speakers)} speakers, "
              f"{sum(len(v) for v in self.speaker_utterances.values())} utterances "
              f"({'train' if is_train else 'val'})")

    def _build_index(
        self,
        root: str,
        splits: List[str],
    ) -> Dict[str, List[str]]:
        """
        Scan LibriSpeech directory and build
        {speaker_id: [path1, path2, ...]} mapping.
        """
        speaker_utterances: Dict[str, List[str]] = {}
        root_path = Path(root)

        for split in splits:
            split_path = root_path / split
            if not split_path.exists():
                print(f"[Dataset] Warning: {split_path} not found, skipping.")
                continue

            # LibriSpeech structure: split/speaker/chapter/file.flac
            for flac_file in split_path.rglob("*.flac"):
                speaker_id = flac_file.parts[-3]   # speaker directory name
                if speaker_id not in speaker_utterances:
                    speaker_utterances[speaker_id] = []
                speaker_utterances[speaker_id].append(str(flac_file))

        return speaker_utterances

    def __len__(self) -> int:
        # We generate samples on-the-fly, so this is somewhat arbitrary.
        # Return total utterance count as a reasonable epoch length.
        return sum(len(v) for v in self.speaker_utterances.values())

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dict with:
            mixture_mel  : (n_mels, T) noisy input to model
            target_mel   : (n_mels, T) clean target for supervision
            reference_mel: (n_mels, T_ref) reference utterance mel
            reference_wav: (T_ref,) reference utterance waveform
                           (for speaker encoder which needs raw audio)
        """
        if self.seed is not None:
            # Deterministic per-index seed, covering BOTH random-number
            # sources used in this method: Python's random module
            # (speaker/utterance/interferer/SNR/condition selection) AND
            # PyTorch's RNG (segment_waveform's random crop window uses
            # torch.randint, not Python's random — without seeding both,
            # the same "seed" can still yield a different audio crop of
            # the same file, as observed empirically).
            random.seed(self.seed + idx)
            torch.manual_seed(self.seed + idx)

        # pick random target speaker
        target_speaker = random.choice(self.speakers)
        utterances     = self.speaker_utterances[target_speaker]

        # pick two different utterances from same speaker
        utt_idx     = random.sample(range(len(utterances)), 2)
        target_path = utterances[utt_idx[0]]
        ref_path    = utterances[utt_idx[1]]

        # pick a different speaker as interferer
        interferer_speaker = random.choice(
            [s for s in self.speakers if s != target_speaker]
        )
        interferer_path = random.choice(
            self.speaker_utterances[interferer_speaker]
        )

        # load audio
        sr          = self.cfg.audio.sample_rate
        seg_len     = self.cfg.audio.segment_length
        segment_samples = int(seg_len * sr)

        target_wav     = load_audio(target_path,     sr)
        interferer_wav = load_audio(interferer_path, sr)
        reference_wav  = load_audio(ref_path,        sr)

        # NEW: track how much of target_wav is real audio vs. will-be-padding,
        # before segment_waveform pads/crops it. Padding is always appended at
        # the end (see segment_waveform), so a single valid-length count suffices.
        target_valid_samples = min(target_wav.shape[0], segment_samples)

        # segment to fixed length
        # NOTE: reference_wav must also be segmented to a fixed length —
        # without this, utterances of different durations across a batch
        # can't be stacked by the DataLoader's default_collate, causing
        # "Trying to resize storage that is not resizable". Using 3.0s
        # to match the convention in data/fake_data.py (sufficient length
        # for WavLM speaker verification). No reference_length key exists
        # in config, so this falls back to the 3.0s default.
        ref_seg_len     = getattr(self.cfg.audio, "reference_length", 3.0)
        target_wav      = segment_waveform(target_wav,     sr, seg_len,     random_start=True)
        interferer_wav  = segment_waveform(interferer_wav, sr, seg_len,     random_start=True)
        reference_wav   = segment_waveform(reference_wav,  sr, ref_seg_len, random_start=True)

        # create mixture
        mixture_wav, target_wav, condition, snr_db = self.mixer.create_mixture(
            target_wav, interferer_wav
        )

        # compute mel spectrograms
        mixture_mel   = self.mel(mixture_wav)    # (n_mels, T)
        target_mel    = self.mel(target_wav)     # (n_mels, T)
        reference_mel = self.mel(reference_wav)  # (n_mels, T_ref)

        # NEW: convert the valid-sample count to a valid-frame count for the
        # mel domain, capped at target_mel's actual length. This is what
        # training/eval use to mask out padding from the loss/MSE — without
        # it, ~5-10s of silence-padded frames on short utterances (test-clean
        # averages 7.4s vs this project's 10s fixed segment_length) get
        # counted as "trivially easy" in every loss/eval number.
        target_valid_frames = min(
            target_valid_samples // self.mel.hop_length,
            target_mel.shape[-1],
        )

        # NOTE: mixture_mel is intentionally left at raw log-mel scale here,
        # matching target_mel. The model's MaskingModule.forward() already
        # applies its own internal F.normalize() on x_mel for the conv
        # pathway; normalizing mixture_mel here as well collapses it toward
        # ~0, making it numerically impossible for X * mask to ever approach
        # target_mel's scale. reference_mel is normalized since it's a
        # separate embedding input, not used in the mask-multiply step.
        reference_mel = self.normalizer.normalize(reference_mel)

        return {
            "mixture_mel"  : mixture_mel,
            "target_mel"   : target_mel,
            "reference_mel": reference_mel,
            "reference_wav": reference_wav,   # raw wav for WavLM speaker encoder
            "target_valid_frames": torch.tensor(target_valid_frames, dtype=torch.long),
                                               # NEW: real (non-padded) frame count
                                               # in target_mel/mixture_mel's T axis
            "condition"    : condition,       # "clean" | "additive" | "reverb" — which
                                               # mixing condition was applied to this sample
            "snr_db"       : snr_db if snr_db is not None else float("nan"),
                                               # realized additive/reverb SNR; NaN for "clean"
        }


def build_dataloaders(
    cfg: DictConfig,
    num_workers: int = 8,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders.
    
    Args:
        cfg        : full omegaconf config
        num_workers: number of dataloader workers
    Returns:
        train_loader, val_loader
    """
    train_dataset = LibriSpeechTSEDataset(
        librispeech_root = cfg.data.librispeech_path,
        splits           = ["train-clean-100"],
        cfg              = cfg,
        is_train         = True,
    )

    val_dataset = LibriSpeechTSEDataset(
        librispeech_root = cfg.data.librispeech_path,
        splits           = ["test-clean"],
        cfg              = cfg,
        is_train         = False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = cfg.train_masking.batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = True,
        persistent_workers  = True,
        prefetch_factor     = 4,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size  = cfg.train_masking.batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = False,
        persistent_workers  = True,
        prefetch_factor     = 4,
    )

    return train_loader, val_loader


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("configs/default.yaml")

    print("Testing dataset (requires LibriSpeech)...")

    dataset = LibriSpeechTSEDataset(
        librispeech_root = cfg.data.librispeech_path,
        splits           = ["train-clean-100"],
        cfg              = cfg,
        is_train         = True,
    )

    sample = dataset[0]
    for k, v in sample.items():
        print(f"  {k:20s}: {v.shape}")

    print("Dataset test passed!")