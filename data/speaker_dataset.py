"""
SpeakerClassificationDataset — builds (waveform, speaker_label) pairs from
LibriSpeech for training SpeakerEncoder's projection layer via a speaker
classification objective (standard d-vector pretraining approach: force the
embedding to be linearly speaker-separable, then discard the classifier
head and keep the embedding for downstream use).

This is a ONE-TIME, standalone pretraining stage for the projection only —
WavLM stays frozen throughout, exactly as in the main TSE pipeline. It does
not touch Stage 1 or Stage 2.
"""
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from torch.utils.data import Dataset

from data.augment import load_audio
from data.mel import segment_waveform


class SpeakerClassificationDataset(Dataset):
    def __init__(
        self,
        librispeech_root: str,
        splits: List[str],
        sample_rate: int = 16000,
        segment_length: float = 3.0,
        is_train: bool = True,
        min_utterances_per_speaker: int = 2,
    ):
        self.sr = sample_rate
        self.segment_length = segment_length
        self.is_train = is_train

        speaker_utterances = self._build_index(librispeech_root, splits)
        self.speaker_ids = sorted([
            s for s, utts in speaker_utterances.items()
            if len(utts) >= min_utterances_per_speaker
        ])
        self.label_map = {spk: i for i, spk in enumerate(self.speaker_ids)}
        self.num_speakers = len(self.speaker_ids)

        # flatten to a list of (path, label) — every utterance seen each epoch,
        # unlike the TSE dataset's random per-index sampling
        self.samples: List[Tuple[str, int]] = []
        for spk in self.speaker_ids:
            label = self.label_map[spk]
            for path in speaker_utterances[spk]:
                self.samples.append((path, label))

        print(f"[SpeakerClassificationDataset] {self.num_speakers} speakers, "
              f"{len(self.samples)} utterances ({'train' if is_train else 'val'})")

    def _build_index(self, root: str, splits: List[str]) -> Dict[str, List[str]]:
        speaker_utterances: Dict[str, List[str]] = {}
        root_path = Path(root)
        for split in splits:
            split_path = root_path / split
            if not split_path.exists():
                continue
            for speaker_dir in split_path.iterdir():
                if not speaker_dir.is_dir():
                    continue
                spk_id = speaker_dir.name
                paths = [str(p) for p in speaker_dir.rglob("*.flac")]
                if paths:
                    speaker_utterances.setdefault(spk_id, []).extend(paths)
        if not speaker_utterances:
            raise FileNotFoundError(
                f"Found 0 speakers under {root_path} for splits {splits}. "
                f"Check you're running from the project root and the path is correct."
            )
        return speaker_utterances

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path, label = self.samples[idx]
        wav = load_audio(path, self.sr)
        wav = segment_waveform(wav, self.sr, self.segment_length, random_start=self.is_train)
        return {"waveform": wav, "label": label}