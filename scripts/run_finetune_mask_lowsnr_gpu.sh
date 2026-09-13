#!/bin/bash
#SBATCH --partition=gpupart_p100
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --job-name=finetune_mask_lowsnr
#SBATCH --output=scripts/logs/finetune_mask_lowsnr_%j.out

# Mask2Flow-TSE -- Stage-1 low-SNR fine-tune, one arm of the mask-formulation
# A/B (docs/methodology_and_project_history.md entries 20-21). Submit BOTH:
#
#   sbatch scripts/run_finetune_mask_lowsnr_gpu.sh multiplicative   # paper's X*M
#   sbatch scripts/run_finetune_mask_lowsnr_gpu.sh log_gain         # X+log(M)
#
# Everything except the formulation is identical between the two: warm start
# from mask_best.pt, 50% [-10,1) dB curriculum with the 2/3/4-speaker mix,
# 25k steps, lr 1e-4, seed 42. ~6h each at Stage 1's measured ~0.84 s/step,
# longer if both arms share one node's CPUs for data loading; --time leaves
# room. Auto-resumes from this arm's latest.pt. flow_best.pt and mask_best.pt
# are never modified.

set -eo pipefail
# NOTE: deliberately NOT using `set -u` -- conda activation references unset vars.

MODE="$1"
case "$MODE" in
    multiplicative|log_gain) ;;
    *) echo "ERROR: usage: sbatch $0 <multiplicative|log_gain>"; exit 1 ;;
esac

echo "=== DIAGNOSTIC ==="
nvidia-smi || echo "[Diag] nvidia-smi failed -- no GPU visible."
source /home/mtech1/25CS60R85/miniconda3/etc/profile.d/conda.sh
conda activate mask2flow
echo "which python3: $(which python3)"
echo "=== END DIAGNOSTIC ==="
echo "=== arm: mask_mode=$MODE ==="

LATEST="checkpoints_v2/mask_finetune_lowsnr_${MODE}/mask_ft_${MODE}_latest.pt"
RESUME_ARG=""
if [ -f "$LATEST" ]; then
    echo "=== resuming from $LATEST ==="
    RESUME_ARG="--resume $LATEST"
fi

# --low_snr_range MUST use '=': space-separated, argparse reads '-10,1' as an
# option name (this exact mistake killed job 10990).
python3 -u training/finetune_mask_lowsnr.py \
    --config configs/default_v2.yaml \
    --mask_ckpt checkpoints_v2/masking/mask_best.pt \
    --mask_mode "$MODE" \
    --low_snr_prob 0.5 \
    --low_snr_range=-10,1 \
    --interferer_count_probs 0.5,0.3,0.2 \
    --max_steps 25000 \
    $RESUME_ARG
