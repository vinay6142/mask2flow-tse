#!/bin/bash
#SBATCH --partition=gpupart_p100
#SBATCH --time=04:00:00
#SBATCH --job-name=generate_libri2mix_test
#SBATCH --output=scripts/logs/generate_libri2mix_test_%j.out

# Mask2Flow-TSE -- generate the Libri2Mix 2-speaker "mix_clean" test set,
# scoped down from LibriMix's stock generate_librimix.sh in three ways
# (each saves real time/storage and none of it is needed for this project):
#   1. n_src=2 only         (skip Libri3Mix entirely)
#   2. freq=16k only        (matches this project's sample rate; skip 8k)
#   3. test split only      (metadata dir points at a pruned copy containing
#                            only libri2mix_test-clean.csv -- see
#                            data/raw/LibriMix/metadata/Libri2Mix_test_only/,
#                            already created; dev/train-100/train-360 metadata
#                            is left out entirely, so those mixtures are
#                            never generated or downloaded)
#   4. mode=min only         (the more commonly-cited condition; add "max"
#                            to --modes below too if you also want that)
#   5. types=mix_clean only  (2-speaker mixture, no added noise -- matches
#                            this project's own clean-mixture evaluation
#                            convention; mix_both/mix_single/noise files are
#                            never written)
#
# Even with types=mix_clean, the underlying script still reads a real WHAM
# noise file per mixture row (unused in the mix, but required for the read
# to succeed) -- run scripts/download_wham.sh FIRST and let it finish.
# augment_train_noise.py is skipped entirely here: it only augments the
# WHAM tr/ (train) split, and test-clean's metadata only references tt/
# (test) noise, which needs no augmentation.
#
# Output lands in data/raw/LibriMix_storage/Libri2Mix/wav16k/min/test/
# {s1,s2,mix_clean}/*.wav -- about 1.4GB for 3000 mixtures.
#
# Requires: WHAM noise already downloaded+extracted (scripts/download_wham.sh),
# and LibriSpeech test-clean already present at data/raw/LibriSpeech/test-clean
# (already true for this project).
#
# Submit:  sbatch scripts/generate_libri2mix_test.sh

set -eo pipefail

echo "--- conda activation ---"
source /home/mtech1/25CS60R85/miniconda3/etc/profile.d/conda.sh
conda activate mask2flow
echo "which python3: $(which python3)"

LIBRIMIX_REPO="data/raw/LibriMix"
LIBRISPEECH_DIR="data/raw/LibriSpeech"
WHAM_DIR="data/raw/wham_noise"
METADATA_DIR="$LIBRIMIX_REPO/metadata/Libri2Mix_test_only"
OUT_DIR="data/raw/LibriMix_storage"

if [ ! -d "$WHAM_DIR/tt" ]; then
    echo "ERROR: $WHAM_DIR/tt not found -- run scripts/download_wham.sh first and let it finish." >&2
    exit 1
fi
if [ ! -f "$METADATA_DIR/libri2mix_test-clean.csv" ]; then
    echo "ERROR: $METADATA_DIR/libri2mix_test-clean.csv not found." >&2
    exit 1
fi

echo "--- checking LibriMix's own python deps (soundfile, pandas, pysndfx, tqdm) ---"
python3 -c "import soundfile, pandas, pysndfx, tqdm" || \
    pip install -r "$LIBRIMIX_REPO/requirements.txt"

echo "=== REAL JOB: create_librimix_from_metadata.py (n_src=2, 16k, min, mix_clean, test only) ==="
python3 "$LIBRIMIX_REPO/scripts/create_librimix_from_metadata.py" \
    --librispeech_dir "$LIBRISPEECH_DIR" \
    --wham_dir "$WHAM_DIR" \
    --metadata_dir "$METADATA_DIR" \
    --librimix_outdir "$OUT_DIR" \
    --n_src 2 \
    --freqs 16k \
    --modes min \
    --types mix_clean

echo "=== Done. Output tree: ==="
find "$OUT_DIR/Libri2Mix/wav16k/min/test" -maxdepth 2 | sort
echo ""
echo "File counts:"
for d in s1 s2 mix_clean; do
    n=$(find "$OUT_DIR/Libri2Mix/wav16k/min/test/$d" -name '*.wav' 2>/dev/null | wc -l)
    echo "  $d: $n files"
done
