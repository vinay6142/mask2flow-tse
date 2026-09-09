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
from data.augment import MixtureCreator, load_audio, mix_multi_at_snr


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
        interferer_count_probs: Optional[List[float]] = None,
    ):
        """
        Args:
            librispeech_root: path to LibriSpeech directory
            splits          : list of splits e.g. ['train-clean-100']
            cfg             : full config (omegaconf)
            is_train        : if True, apply augmentation; else clean only
            seed            : optional random seed for reproducibility
            interferer_count_probs: NEW, opt-in, default None. A probability
                distribution over how many SIMULTANEOUS interferers a training
                example gets: interferer_count_probs[i] = P(exactly i+1
                interferers), so index 0 = the original 2-total-speaker case,
                index 1 = 3-total-speaker, index 2 = 4-total-speaker, etc.
                Must sum to 1.0. Left as None (default) for every existing
                caller (train_flow.py, train_mask.py, eval scripts, ...) --
                that reproduces the ORIGINAL single-interferer __getitem__
                path EXACTLY, unchanged. Added for
                training/finetune_flow_hard_multispeaker.py, which closes the
                speaker-count-coverage gap characterized in
                docs/results_and_limitations.md Sec 5.5.3 (Stage 2 loses
                78.3%->53.7% of its improvement-over-Stage-1 as speaker count
                rises 2->4; Stage 1 itself is unaffected). When set,
                n_interferers>=2 examples use the eval-proven
                mix_multi_at_snr() (additive-only, no clean/reverb branch --
                see __getitem__) instead of self.mixer.
        """
        self.cfg      = cfg
        self.is_train = is_train
        if interferer_count_probs is not None:
            assert abs(sum(interferer_count_probs) - 1.0) < 1e-5, \
                "interferer_count_probs must sum to 1.0"
        self.interferer_count_probs = interferer_count_probs
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

        # NEW: how many simultaneous interferers this example gets. Default
        # (interferer_count_probs=None) always draws exactly 1 -- IDENTICAL
        # to the original behavior below, so every existing caller that
        # doesn't pass the new kwarg is completely unaffected.
        if self.interferer_count_probs is not None:
            n_interferers = random.choices(
                range(1, len(self.interferer_count_probs) + 1),
                weights=self.interferer_count_probs,
            )[0]
        else:
            n_interferers = 1

        other_speakers = [s for s in self.speakers if s != target_speaker]
        interferer_speakers = random.sample(other_speakers, n_interferers)
        interferer_paths = [
            random.choice(self.speaker_utterances[s]) for s in interferer_speakers
        ]

        # load audio
        sr          = self.cfg.audio.sample_rate
        seg_len     = self.cfg.audio.segment_length
        segment_samples = int(seg_len * sr)

        target_wav      = load_audio(target_path, sr)
        reference_wav   = load_audio(ref_path,    sr)
        interferer_wavs = [load_audio(p, sr) for p in interferer_paths]

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
        target_wav      = segment_waveform(target_wav,    sr, seg_len,     random_start=True)
        reference_wav   = segment_waveform(reference_wav, sr, ref_seg_len, random_start=True)
        interferer_wavs = [
            segment_waveform(w, sr, seg_len, random_start=True) for w in interferer_wavs
        ]

        # create mixture
        if n_interferers == 1:
            # UNCHANGED path -- exactly the original single-interferer mixer,
            # including its clean/additive/reverb condition sampling.
            mixture_wav, target_wav, condition, snr_db = self.mixer.create_mixture(
                target_wav, interferer_wavs[0]
            )
        else:
            # NEW: 2+ simultaneous interferers. Additive-only -- no clean/
            # reverb branch here (see interferer_count_probs docstring in
            # __init__) -- each interferer independently drawn at its own SNR
            # in the SAME trained [snr_min, snr_max] range, via the
            # eval-proven mix_multi_at_snr() generalization of mix_at_snr()
            # (see data/augment.py, and its first use in eval/eval_multi_speaker.py).
            snr_db_list = [
                random.uniform(self.cfg.data.snr_min, self.cfg.data.snr_max)
                for _ in range(n_interferers)
            ]
            mixture_wav, target_wav, _ = mix_multi_at_snr(
                target_wav, interferer_wavs, snr_db_list
            )
            condition = f"additive_{n_interferers + 1}spk"
            snr_db    = snr_db_list[0]   # representative value; full per-interferer
                                          # list isn't tracked in this dict's schema

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
            "target_speaker": target_speaker, # NEW: speaker ID string. Needed so a
                                               # verification-accuracy eval (genuine vs.
                                               # in-batch impostor trials) can exclude
                                               # same-speaker collisions from the impostor
                                               # pool -- test-clean only has ~40 speakers,
                                               # so a batch_size=20 draw has real odds of
                                               # the same speaker appearing twice.
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


def build_dataloaders_multispeaker(
    cfg: DictConfig,
    interferer_count_probs: List[float],
    num_workers: int = 8,
) -> Tuple[DataLoader, DataLoader]:
    """
    Like build_dataloaders() above, but the TRAIN split samples a variable
    number of simultaneous interferers per example (see
    LibriSpeechTSEDataset's interferer_count_probs docstring) instead of
    always exactly 1. Added for training/finetune_flow_hard_multispeaker.py.

    VALIDATION deliberately stays on the ORIGINAL fixed 2-speaker
    distribution (interferer_count_probs omitted for val_dataset) -- matches
    the discipline already established in finetune_flow_hard_t0.py: the
    val_loss proxy needs to stay comparable to every other training run's
    val curve, not shift under a different task mix. Track the REAL target
    metric (eval/eval_multi_speaker.py's accuracy table across 2/3/4
    speakers) periodically instead of trusting this proxy -- same guidance
    as the hard-t0 script gives for its own val_loss.

    Args:
        cfg                    : full omegaconf config
        interferer_count_probs : passed through to the TRAIN dataset only
        num_workers            : number of dataloader workers
    Returns:
        train_loader, val_loader
    """
    train_dataset = LibriSpeechTSEDataset(
        librispeech_root = cfg.data.librispeech_path,
        splits           = ["train-clean-100"],
        cfg              = cfg,
        is_train         = True,
        interferer_count_probs = interferer_count_probs,
    )

    val_dataset = LibriSpeechTSEDataset(
        librispeech_root = cfg.data.librispeech_path,
        splits           = ["test-clean"],
        cfg              = cfg,
        is_train         = False,
        # interferer_count_probs intentionally omitted -- stays 2-speaker-only
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = cfg.train_flow.batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = True,
        persistent_workers  = True,
        prefetch_factor     = 4,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size  = cfg.train_flow.batch_size,
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