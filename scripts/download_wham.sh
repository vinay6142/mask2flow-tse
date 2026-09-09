#!/bin/bash
#SBATCH --partition=gpupart_p100
#SBATCH --time=03:00:00
#SBATCH --job-name=download_wham
#SBATCH --output=scripts/logs/download_wham_%j.out

# Mask2Flow-TSE -- download WHAM! noise for the Libri2Mix cross-corpus eval
# (see eval/eval_libri2mix.py). No GPU needed -- this is a pure download/
# unzip job. Left on gpupart_p100 only because it is the one partition
# already confirmed to work on this cluster (see run_finetune_flow_hardt0_gpu.sh);
# if `sinfo` shows a genuine CPU-only partition, switch PARTITION below to
# that instead so this doesn't occupy a scarce GPU node for a plain download.
#
# ~52GB compressed, 30-60+ min depending on link speed. wget -c makes this
# safe to just resubmit if it times out or gets interrupted partway --
# it resumes the partial download rather than restarting.
#
# Submit:  sbatch scripts/download_wham.sh

set -eo pipefail

STORAGE_DIR="data/raw"
WHAM_DIR="$STORAGE_DIR/wham_noise"
# URL from the actual JorisCos/LibriMix repo's generate_librimix.sh as of
# this clone (2026-08) -- NOT the same URL as data/download.py's download_wham()
# function, which points at http://wham.whisper.ai/wham_noise.zip. That
# whisper.ai URL is the one bundled with this project since June; the one
# below is what LibriMix's own maintained script currently uses. If this
# one 404s by the time you run it, check data/raw/LibriMix/generate_librimix.sh
# for whatever URL is current then.
WHAM_URL="https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip"

mkdir -p "$STORAGE_DIR"

if [ -d "$WHAM_DIR" ] && [ -d "$WHAM_DIR/tt" ]; then
    echo "=== $WHAM_DIR already has a tt/ subdir -- looks already downloaded+extracted. Skipping. ==="
    find "$WHAM_DIR" -name '*.wav' | wc -l
    exit 0
fi

echo "=== Downloading WHAM! noise from $WHAM_URL into $STORAGE_DIR ==="
wget -c --tries=0 --read-timeout=20 "$WHAM_URL" -P "$STORAGE_DIR"

echo "=== Extracting ==="
unzip -qn "$STORAGE_DIR/wham_noise.zip" -d "$STORAGE_DIR"

echo "=== Done. File counts: ==="
for sub in tr cv tt; do
    n=$(find "$WHAM_DIR/$sub" -name '*.wav' 2>/dev/null | wc -l)
    echo "  $WHAM_DIR/$sub: $n files"
done
echo "NOTE: wham_noise.zip ($STORAGE_DIR/wham_noise.zip) is left in place in case "
echo "the extraction needs re-running -- unzip -n above skips files already on "
echo "disk, so resubmitting this script is always safe/idempotent if a run gets "
echo "cut short (e.g. a compute-node munge/auth error killing the job mid-unzip, "
echo "as happened in job 10790 -- see mask2flow-tse-next-steps memory). Only "
echo "delete the zip once tt/ is confirmed present and non-empty above."
