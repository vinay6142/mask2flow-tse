#!/bin/bash
#SBATCH --partition=gpupart_p100
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --job-name=eval_candidate
#SBATCH --output=scripts/logs/eval_candidate_%j.out

# Mask2Flow-TSE -- generic candidate-checkpoint evaluator. Runs the full
# promotion-decision battery against ANY candidate -- a Stage 2 checkpoint,
# or a Stage 1 one via the optional third argument -- so each new experiment
# doesn't need its own near-duplicate eval script.
#
# Usage:
#   sbatch scripts/run_eval_candidate_gpu.sh <flow_ckpt_path> <tag> [mask_ckpt_path] [cfg_scale]
#
# e.g. validating a new guidance setting on the CURRENT checkpoints, without
# editing configs/default_v2.yaml (so a failed validation leaves nothing changed):
#   sbatch scripts/run_eval_candidate_gpu.sh checkpoints_v2/flow/flow_best.pt cfg2.5 \
#       checkpoints_v2/masking/mask_best.pt 2.5
#
# e.g. a Stage-2 candidate (Stage 1 defaults to mask_best.pt):
#   sbatch scripts/run_eval_candidate_gpu.sh \
#       checkpoints_v2/flow_finetune_lowsnr25/flow_ft_lowsnr25_final.pt lowsnr25
# e.g. a Stage-1 candidate, paired with the current Stage 2:
#   sbatch scripts/run_eval_candidate_gpu.sh checkpoints_v2/flow/flow_best.pt mask_log_gain \
#       checkpoints_v2/mask_finetune_lowsnr_log_gain/mask_ft_log_gain_final.pt
# The Stage-1 mask formulation is read from the checkpoint itself
# (eval/results_stage2.py load_masking), so no flag is needed for it.
#
# <tag> is appended to every output filename, so results never collide with
# each other or with the canonical files docs/results_and_limitations.md
# cites. flow_best.pt is NEVER touched -- this is assessment only.
#
# outputs/results/eval_lowsnr_indomain_baseline.jsonl holds the PRE-PROMOTION
# system (old Stage 1 + flow_best.pt) at [-10,1)dB. Stage 1 was promoted on
# 2026-09-12, so that file is no longer the current baseline -- the current
# system's numbers are job 11081's, listed below.

set -eo pipefail
# NOTE: deliberately NOT using `set -u` -- conda activation references unset vars.

CKPT="$1"
TAG="$2"
MASK_CKPT="${3:-checkpoints_v2/masking/mask_best.pt}"
CFG_SCALE="${4:-}"     # optional; empty = each script's config default (inference.cfg_scale)
CFG_ARG=""
# NOTE: written as a full if, not `[ -n "$X" ] && VAR=...` -- that form returns
# non-zero when the variable is empty, which is the common path here, and this
# script runs under `set -e`.
if [ -n "$CFG_SCALE" ]; then
    CFG_ARG="--cfg_scale $CFG_SCALE"
fi

if [ -z "$CKPT" ] || [ -z "$TAG" ]; then
    echo "ERROR: usage: sbatch $0 <flow_ckpt_path> <tag> [mask_ckpt_path] [cfg_scale]"
    exit 1
fi
for f in "$CKPT" "$MASK_CKPT"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: checkpoint not found: $f"
        exit 1
    fi
done

echo "=== DIAGNOSTIC ==="
nvidia-smi || echo "[Diag] nvidia-smi failed -- no GPU visible."
source /home/mtech1/25CS60R85/miniconda3/etc/profile.d/conda.sh
conda activate mask2flow
echo "which python3: $(which python3)"
echo "=== END DIAGNOSTIC ==="
echo ""
echo "=== Stage 2   : $CKPT"
echo "=== Stage 1   : $MASK_CKPT"
echo "=== Tag       : $TAG"
echo "=== cfg_scale : ${CFG_SCALE:-(config default, currently 2.5)}"
echo ""

BASE_ARGS="--mask_ckpt $MASK_CKPT \
    --flow_ckpt $CKPT \
    --projection_ckpt checkpoints_speaker_encoder/projection_latest.pt $CFG_ARG"

echo "======================================================================"
echo " REFERENCE NUMBERS -- current system = mask_best.pt + flow_best.pt"
echo " (Stage 1 promoted 2026-09-12, Stage 2 2026-09-13, cfg_scale 2.5 adopted"
echo "  2026-09-13 -- these are job 11112's numbers)"
echo "----------------------------------------------------------------------"
echo "  low-SNR in-domain [-10,1)dB, n=300:"
echo "    accuracy                    80.3%   (do-nothing mixture 64.0%)"
echo "    SI-SDR gain vs mixture      +6.51dB   S2 vs S1 +1.72dB"
echo "  must hold:"
echo "    corpus-wide accuracy (n=400)   86.2%"
echo "    in-domain n=2620 catastrophic  10.2% mel (severe SI-SDR was ~2.7% at cfg1.5)"
echo "    in-domain n=2620 SI-SDR vs mixture +1.75dB (vs S1 +1.30dB)"
echo "    2/3/4-speaker accuracy      85.7 / 80.0 / 77.0"
echo "    Libri2Mix ALL SI-SDR vs mix +3.39dB, in-dist catastrophic 7.7%"
echo "  HISTORICAL, do not read as a live ceiling: job 11075's oracles reached"
echo "    73.0% (best mask in the paper's formulation) and 89.9% (true energy"
echo "    deletion) at low SNR -- but paired with the OLD Stage 2. The current"
echo "    system already exceeds the 73.0% figure, so those bound that PAIR."
echo "  (pre-low-SNR-work system: 61.3% low-SNR acc / -15.86dB, 86.5% corpus-"
echo "   wide, 3.7% in-domain catastrophic, 87.3/82.1/77.0 speakers)"
echo "======================================================================"
echo ""
echo "=== [1/5] CONTROLLED in-domain low-SNR, n=300 at [-10,1)dB (~5min) ==="
echo "    The cleanest read on whether the target regime improved -- same corpus,"
echo "    mixer, speakers and seed as the baseline, only the checkpoint differs."
python3 -u eval/eval_multi_speaker.py \
    --n_interferers 1 --n_samples 300 \
    --snr_min -10 --snr_max 1 \
    $BASE_ARGS \
    --output outputs/results/eval_lowsnr_indomain_${TAG}.jsonl \
    --resume

echo ""
echo "=== [2/5] Libri2Mix cross-corpus, n=6000 (~56min) ==="
echo "    Yields BOTH the sub-1dB slice (target) and the >=1dB slice (must hold)."
python3 -u eval/eval_libri2mix.py \
    $BASE_ARGS \
    --librimix_dir data/raw/LibriMix_storage/Libri2Mix/wav16k/min \
    --librimix_input_csv data/raw/LibriMix/metadata/Libri2Mix_test_only/libri2mix_test-clean.csv \
    --librispeech_dir data/raw/LibriSpeech \
    --output outputs/results/eval_libri2mix_min_${TAG}.jsonl \
    --resume

echo ""
echo "=== [3/5] in-domain n=2620 -- primary headline (~45min) ==="
python3 -u eval/full_eval.py \
    $BASE_ARGS \
    --output outputs/results/full_eval_audio_gpu_${TAG}.jsonl \
    --resume

echo ""
echo "=== [4/5] corpus-wide speaker-verification accuracy, n=400 (~4min) ==="
python3 -u eval/verify_eer.py \
    $BASE_ARGS \
    --seed 42 --n_steps 4 --max_samples 400

echo ""
echo "=== [5/5] speaker-count conditions, n=300 each ==="
for N in 1 2 3; do
    TOTAL=$((N + 1))
    echo ""
    echo "--- n_interferers=$N ($TOTAL total speakers) ---"
    python3 -u eval/eval_multi_speaker.py \
        --n_interferers $N --n_samples 300 \
        $BASE_ARGS \
        --output outputs/results/eval_multispeaker_${TOTAL}total_${TAG}.jsonl \
        --resume
done

echo ""
echo "======================================================================"
echo "  Done: $TAG"
echo "  For the Libri2Mix SNR split (in-dist vs out-of-dist), the summary"
echo "  above reports only the raw n=6000 aggregate:"
echo "    python3 eval/snr_split_summary.py outputs/results/eval_libri2mix_min_${TAG}.jsonl"
echo "======================================================================"
