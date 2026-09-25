#!/bin/bash
#SBATCH --partition=gpupart_p100
#SBATCH --time=12:00:00
#SBATCH --job-name=dl_librispeech
#SBATCH --output=scripts/logs/dl_librispeech_%j.out

# Mask2Flow-TSE -- download the LibriSpeech training splits this project was
# always configured for but never actually had, plus real RIRs.
#
# WHY (found 2026-09-17): configs/default_v2.yaml lists
#   train-clean-100 + train-clean-360 + train-other-500
# as data.train_splits, with the comment "with 200GB use all three" -- but only
# train-clean-100 is on disk. Every checkpoint in this project (Stage 1, Stage 2,
# vocoder, speaker projection) was therefore trained on ~100h instead of ~960h.
# Separately, data/raw/RIRS does not exist, so data/augment.py's RIRLoader has
# been silently falling back to SYNTHETIC RIRs for the 34% of training samples
# that draw the reverb condition (cfg.data.conditions.reverb_prob).
#
# No GPU needed -- pure download/extract. Left on gpupart_p100 only because it
# is the partition confirmed to work on this cluster (same reasoning as
# scripts/download_wham.sh); switch if `sinfo` shows a real CPU-only partition.
#
# IDEMPOTENT: data/download.py skips any split whose directory already exists,
# so resubmitting after a timeout re-checks rather than re-downloads. CAVEAT:
# it skips on directory EXISTENCE, not completeness -- if a run is killed during
# extraction, delete that split's directory before resubmitting or it will be
# treated as done (the exact trap that bit the Libri2Mix generation in job 10817).
#
# Submit:  sbatch scripts/download_librispeech_full.sh

set -eo pipefail

LS_DIR="data/raw/LibriSpeech"

echo "=== Disk before ==="
df -h . || true
echo ""
echo "=== Quota (if the cluster reports one -- df above is the SHARED volume, ==="
echo "===  which can look empty while a per-user quota is already full)     ==="
quota -s 2>/dev/null || echo "[no quota command / no quota reported]"
echo ""

# Size the requirement from what is actually MISSING, not from the full job.
# A fixed NEED_GB=110 (sized for both large splits) blocked jobs 11221/11222
# at 38GB and 93GB free even though only train-clean-360 remained, which needs
# about 46GB peak. Each split costs its compressed size again transiently while
# torchaudio extracts it, hence the x2, plus a small margin.
SPLIT_NAMES="train-clean-100 train-clean-360 train-other-500 dev-clean dev-other test-clean test-other"
split_gb() {
    case "$1" in
        train-clean-100) echo 6.3 ;;
        train-clean-360) echo 23.1 ;;
        train-other-500) echo 30.3 ;;
        dev-clean)       echo 0.33 ;;
        dev-other)       echo 0.31 ;;
        test-clean)      echo 0.34 ;;
        test-other)      echo 0.33 ;;
        *)               echo 0 ;;
    esac
}

MISSING=""
MISSING_GB=0
for s in $SPLIT_NAMES; do
    if [ ! -d "$LS_DIR/$s" ]; then
        MISSING="$MISSING $s"
        MISSING_GB=$(awk -v a="$MISSING_GB" -v b="$(split_gb "$s")" 'BEGIN{print a+b}')
    fi
done
# x2 for extraction transient, +5GB margin (RIRs are ~1GB), rounded up.
NEED_GB=$(awk -v g="$MISSING_GB" 'BEGIN{printf "%d", int(g*2)+6}')

if [ -z "$MISSING" ]; then
    echo "=== All LibriSpeech splits already present; only RIRs may be fetched. ==="
else
    echo "=== Missing splits:$MISSING  (~${MISSING_GB}GB compressed) ==="
fi
echo "=== Need >=${NEED_GB}GB free (compressed x2 for extraction, + margin) ==="
echo ""

AVAIL_GB=$(df -BG --output=avail . 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "$AVAIL_GB" ] && [ "$AVAIL_GB" -lt "$NEED_GB" ]; then
    echo "ERROR: only ${AVAIL_GB}GB available, want >=${NEED_GB}GB."
    echo "Reclaimable right now, in order of safety:"
    echo "  - data/raw/wham_noise (36G) -- its ONLY purpose was generating"
    echo "    Libri2Mix, which is already built in data/raw/LibriMix_storage."
    echo "    Only needed again if you regenerate a different Libri2Mix split."
    echo "  - superseded checkpoints_v2/**/*_step*.pt snapshots (~80G) -- keep"
    echo "    flow_best.pt, mask_best.pt and the *_backup.pt files."
    exit 1
fi

echo "=== Splits currently on disk ==="
ls -1 "$LS_DIR" 2>/dev/null | grep -E '^(train|dev|test)' || echo "(none)"
echo ""

source /home/mtech1/25CS60R85/miniconda3/etc/profile.d/conda.sh
conda activate mask2flow
echo "which python3: $(which python3)"
echo ""

# --splits all  = train-clean-100 + train-clean-360 + train-other-500
#                 (100 is already present and will be skipped)
# --rirs        = OpenSLR RIRS_NOISES, replacing the synthetic fallback
echo "=== Downloading LibriSpeech (all train splits) + RIRs ==="
python3 -u data/download.py --splits all --rirs

echo ""
echo "=== Disk after ==="
df -h . || true
echo ""
echo "=== Verify ==="
python3 -u data/download.py --verify

echo ""
echo "=== RIR layout check -- READ THIS BEFORE TRAINING WITH REVERB ==="
# openslr resource 28 is RIRS_NOISES, which is NOT all impulse responses: it
# ships simulated_rirs/ and real_rirs_isotropic_noises/ (genuine RIRs) ALONGSIDE
# pointsource_noises/ (ordinary noise recordings). data/augment.py's RIRLoader
# takes `rglob("*.wav")` over whatever rir_path points at, so aiming it at the
# top level would convolve speech with noise clips as if they were impulse
# responses -- silently wrong reverb on 34% of training samples.
for sub in RIRS_NOISES/simulated_rirs RIRS_NOISES/real_rirs_isotropic_noises RIRS_NOISES/pointsource_noises; do
    n=$(find "data/raw/RIRS/$sub" -name '*.wav' 2>/dev/null | wc -l)
    echo "  data/raw/RIRS/$sub : $n wav"
done
echo "  TOTAL under data/raw/RIRS: $(find data/raw/RIRS -name '*.wav' 2>/dev/null | wc -l) wav"
echo ""
echo "  If pointsource_noises is non-empty, set cfg.data.rir_path to the RIR-only"
echo "  subtree (data/raw/RIRS/RIRS_NOISES/simulated_rirs) rather than data/raw/RIRS,"
echo "  or the noise files get used as impulse responses."

echo ""
echo "NEXT: nothing retrains automatically. The new data only takes effect in a"
echo "run that builds its dataloaders fresh -- data/librispeech.py reads"
echo "cfg.data.train_splits, so a Stage-1 retrain picks all three up with no"
echo "code change. Confirm the split list in that job's log before trusting it."
