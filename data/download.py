"""
Data download script for Mask2Flow-TSE.
Downloads LibriSpeech and optionally WHAM! noise + RIRs.

With 200GB storage, recommended downloads:
  LibriSpeech train-clean-100  :  6.3 GB
  LibriSpeech train-clean-360  : 23.1 GB
  LibriSpeech train-other-500  : 30.3 GB
  LibriSpeech dev/test sets    :  1.5 GB
  WHAM! noise                  : 52.0 GB
  OpenSLR RIRs                 :  1.0 GB
  ─────────────────────────────────────
  Total                        :~115 GB
  Remaining for checkpoints    : ~85 GB

Run from project root:
    python data/download.py --splits all
    python data/download.py --splits minimal   (train-clean-100 only)
    python data/download.py --wham             (also download WHAM! noise)
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchaudio
from tqdm import tqdm

# ── paths ─────────────────────────────────────────────────────
ROOT          = Path("data/raw")
LIBRISPEECH   = ROOT / "LibriSpeech"
WHAM_DIR      = ROOT / "wham_noise"
RIR_DIR       = ROOT / "RIRS"

# ── LibriSpeech split sizes ───────────────────────────────────
SPLITS = {
    "train-clean-100": "6.3 GB",
    "train-clean-360": "23.1 GB",
    "train-other-500": "30.3 GB",
    "dev-clean"      : "337 MB",
    "dev-other"      : "314 MB",
    "test-clean"     : "346 MB",
    "test-other"     : "328 MB",
}

MINIMAL_SPLITS = [
    "train-clean-100",
    "dev-clean",
    "test-clean",
    "test-other",
]

ALL_SPLITS = list(SPLITS.keys())


def download_librispeech(splits: list):
    """
    Download LibriSpeech splits using torchaudio.
    torchaudio handles download, extraction, and caching.
    """
    ROOT.mkdir(parents=True, exist_ok=True)

    total_size = sum(
        SPLITS[s] for s in splits if s in SPLITS
    )

    print(f"\n{'─'*50}")
    print(f"  Downloading LibriSpeech")
    print(f"  Splits  : {splits}")
    print(f"  Save to : {LIBRISPEECH.absolute()}")
    print(f"{'─'*50}")

    for split in splits:
        size = SPLITS.get(split, "unknown")
        print(f"\n  [{split}] ({size})")

        # check if already downloaded
        split_path = LIBRISPEECH / split
        if split_path.exists() and any(split_path.rglob("*.flac")):
            n_files = len(list(split_path.rglob("*.flac")))
            print(f"    Already downloaded ({n_files} .flac files) ✅")
            continue

        try:
            # torchaudio automatically downloads and extracts
            print(f"    Downloading... (this may take a while)")
            dataset = torchaudio.datasets.LIBRISPEECH(
                root     = str(ROOT),
                url      = split,
                download = True,
            )
            n_files = len(dataset)
            print(f"    Done ✅  ({n_files} utterances)")

        except Exception as e:
            print(f"    ERROR: {e}")
            print(f"    Try manual download from:")
            print(f"    https://www.openslr.org/12/")


def download_wham():
    """
    Download WHAM! noise dataset (~52 GB).
    Used for Libri2Mix noisy condition evaluation.
    """
    print(f"\n{'─'*50}")
    print(f"  Downloading WHAM! noise")
    print(f"  Size    : ~52 GB")
    print(f"  Save to : {WHAM_DIR.absolute()}")
    print(f"{'─'*50}")

    if WHAM_DIR.exists() and any(WHAM_DIR.rglob("*.wav")):
        n_files = len(list(WHAM_DIR.rglob("*.wav")))
        print(f"  Already downloaded ({n_files} files) ✅")
        return

    WHAM_URL  = "http://wham.whisper.ai/wham_noise.zip"
    WHAM_ZIP  = ROOT / "wham_noise.zip"

    print(f"  Downloading from: {WHAM_URL}")
    print(f"  This is a 52GB file — may take 30+ minutes")

    try:
        # use wget for large file downloads (shows progress)
        result = subprocess.run(
            ["wget", "-c", "-P", str(ROOT), WHAM_URL],
            check=True
        )
        print(f"  Extracting...")
        subprocess.run(
            ["unzip", str(WHAM_ZIP), "-d", str(ROOT)],
            check=True
        )
        WHAM_ZIP.unlink()   # remove zip after extraction
        print(f"  WHAM! noise ready ✅")

    except FileNotFoundError:
        print(f"  wget not found. Install with: sudo apt install wget")
        print(f"  Or download manually from: http://wham.whisper.ai/")
    except subprocess.CalledProcessError as e:
        print(f"  Download failed: {e}")


def download_rirs():
    """
    Download OpenSLR Room Impulse Responses (~1 GB).
    Used for speech reverb augmentation condition.
    """
    print(f"\n{'─'*50}")
    print(f"  Downloading OpenSLR RIRs")
    print(f"  Size    : ~1 GB")
    print(f"  Save to : {RIR_DIR.absolute()}")
    print(f"{'─'*50}")

    if RIR_DIR.exists() and any(RIR_DIR.rglob("*.wav")):
        n_files = len(list(RIR_DIR.rglob("*.wav")))
        print(f"  Already downloaded ({n_files} RIR files) ✅")
        return

    RIR_URL = (
        "https://www.openslr.org/resources/28/rirs_noises.zip"
    )
    RIR_ZIP = ROOT / "rirs_noises.zip"

    print(f"  Downloading from: {RIR_URL}")

    try:
        subprocess.run(
            ["wget", "-c", "-P", str(ROOT), RIR_URL],
            check=True
        )
        print(f"  Extracting...")
        RIR_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["unzip", str(RIR_ZIP), "-d", str(RIR_DIR)],
            check=True
        )
        RIR_ZIP.unlink()
        print(f"  RIRs ready ✅")

    except FileNotFoundError:
        print(f"  wget not found. Install with: sudo apt install wget")
    except subprocess.CalledProcessError as e:
        print(f"  Download failed: {e}")


def verify_downloads():
    """
    Verify all downloaded data looks correct.
    Checks file counts and loads a sample file.
    """
    print(f"\n{'─'*50}")
    print(f"  Verifying downloads...")
    print(f"{'─'*50}")

    # check LibriSpeech
    if LIBRISPEECH.exists():
        splits_found = [
            d.name for d in LIBRISPEECH.iterdir()
            if d.is_dir()
        ]
        for split in splits_found:
            split_path = LIBRISPEECH / split
            n_flac = len(list(split_path.rglob("*.flac")))
            print(f"  LibriSpeech/{split}: {n_flac} files")

            # load one sample to verify it works
            if n_flac > 0:
                sample = next(split_path.rglob("*.flac"))
                wav, sr = torchaudio.load(str(sample))
                print(f"    Sample: {sample.name} "
                      f"({wav.shape[1]/sr:.1f}s, {sr}Hz) ✅")
    else:
        print(f"  LibriSpeech: not found ❌")

    # check WHAM!
    if WHAM_DIR.exists():
        n_wav = len(list(WHAM_DIR.rglob("*.wav")))
        print(f"  WHAM! noise: {n_wav} files")
    else:
        print(f"  WHAM! noise: not downloaded (optional)")

    # check RIRs
    if RIR_DIR.exists():
        n_rir = len(list(RIR_DIR.rglob("*.wav")))
        print(f"  RIRs       : {n_rir} files")
    else:
        print(f"  RIRs       : not downloaded (optional)")

    # disk usage
    try:
        result = subprocess.run(
            ["du", "-sh", str(ROOT)],
            capture_output=True, text=True
        )
        print(f"\n  Total data size: {result.stdout.split()[0]}")
    except Exception:
        pass


def print_disk_space():
    """Print current disk usage."""
    try:
        result = subprocess.run(
            ["df", "-h", "."],
            capture_output=True, text=True
        )
        lines = result.stdout.strip().split("\n")
        print(f"\n  Disk space:")
        for line in lines:
            print(f"    {line}")
    except Exception:
        pass


# ── CLI ───────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download data for Mask2Flow-TSE"
    )
    parser.add_argument(
        "--splits",
        choices=["minimal", "recommended", "all"],
        default="recommended",
        help=(
            "minimal=train-clean-100+test (~7GB), "
            "recommended=+train-clean-360 (~30GB), "
            "all=everything (~60GB)"
        )
    )
    parser.add_argument(
        "--wham", action="store_true",
        help="Also download WHAM! noise (~52GB)"
    )
    parser.add_argument(
        "--rirs", action="store_true",
        help="Also download OpenSLR RIRs (~1GB)"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Only verify existing downloads"
    )
    args = parser.parse_args()

    print("=" * 50)
    print("  Mask2Flow-TSE — Data Downloader")
    print("=" * 50)
    print_disk_space()

    if args.verify:
        verify_downloads()
        sys.exit(0)

    # determine splits to download
    if args.splits == "minimal":
        splits = MINIMAL_SPLITS
        print(f"\n  Mode: minimal (~7 GB)")
    elif args.splits == "recommended":
        splits = [s for s in ALL_SPLITS
                  if s not in ["train-other-500"]]
        print(f"\n  Mode: recommended (~30 GB)")
    else:  # all
        splits = ALL_SPLITS
        print(f"\n  Mode: all (~60 GB)")

    # download
    download_librispeech(splits)

    if args.rirs:
        download_rirs()

    if args.wham:
        download_wham()

    verify_downloads()

    print(f"\n{'='*50}")
    print(f"  Download complete!")
    print(f"  Next: python check_setup.py")
    print(f"{'='*50}")