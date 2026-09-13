#!/bin/bash
#SBATCH --partition=gpupart_p100
#SBATCH --gres=gpu:1
#SBATCH --time=47:59:00
#SBATCH --job-name=finetune_flow_lowsnr
#SBATCH --output=scripts/logs/finetune_flow_lowsnr_%j.out

# Mask2Flow-TSE -- Stage-2 fine-tune closing the SNR-coverage gap, the last
# of the three characterized limitations still open (see
# docs/results_and_limitations.md Sec 5.5.2 and Sec 4 timeline entry 14).
#
# Curriculum: 50% of train examples drawn from [-10, 1) dB -- a regime the
# model has NEVER been trained on, where Stage 2 currently makes 94.8% of
# sub--5dB samples worse than Stage 1 alone -- and 50% from the trained
# [1, 10] dB range so the already-validated in-distribution numbers are
# preserved. The speaker-count curriculum stays ON at the same time so the
# 3-/4-speaker gains from the previous fine-tune aren't handed back.
#
# ~25k steps, expect roughly the same ~13h the multispeaker run took on a
# P100. --time is this partition's confirmed max; auto-resumes from
# flow_ft_lowsnr_latest.pt if this gets cut off, so resubmitting is safe.
#
# checkpoints_v2/flow/flow_best.pt is NOT touched -- this writes to
# checkpoints_v2/flow_finetune_lowsnr/ and a promotion decision comes later,
# only after the same full re-validation discipline every prior promotion
# followed (Libri2Mix both SNR slices, in-domain n=2620, verify_eer n=400,
# and 2/3/4-speaker).
#
# Submit:  sbatch scripts/run_finetune_flow_lowsnr_gpu.sh

set -eo pipefail
# NOTE: deliberately NOT using `set -u` -- conda's activation shell functions
# reference variables that can be unset, and `-u` kills the job at
# `conda activate` on many conda versions.

echo "=== DIAGNOSTIC ==="
nvidia-smi || echo "[Diag] nvidia-smi failed or not found -- no GPU visible to this job."
source /home/mtech1/25CS60R85/miniconda3/etc/profile.d/conda.sh
conda activate mask2flow
echo "which python3: $(which python3)"
python3 -c "import torch; print('torch.cuda.is_available():', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
echo "=== END DIAGNOSTIC ==="
echo ""

RESUME_ARG=""
LATEST="checkpoints_v2/flow_finetune_lowsnr/flow_ft_lowsnr_latest.pt"
if [ -f "$LATEST" ]; then
    echo "=== Found $LATEST -- resuming this fine-tune from it ==="
    RESUME_ARG="--resume $LATEST"
else
    echo "=== No existing checkpoint -- starting fresh from --flow_ckpt ==="
fi
echo ""

# python3 -u: unbuffered, so per-step/val prints reach the log immediately
# instead of sitting in a block buffer for hours on a non-tty (learned from
# the multispeaker run, whose log stayed silent until it finished).
python3 -u training/finetune_flow_hard_lowsnr.py \
    --config configs/default_v2.yaml \
    --mask_ckpt checkpoints_v2/masking/mask_best.pt \
    --flow_ckpt checkpoints_v2/flow/flow_best.pt \
    --low_snr_prob 0.5 \
    --low_snr_range=-10,1 \
    --interferer_count_probs 0.5,0.3,0.2 \
    --max_steps 25000 \
    $RESUME_ARG

echo ""
echo "=== Fine-tune done. Next: evaluate BEFORE promoting anything ==="
echo "  1. eval/eval_libri2mix.py against checkpoints_v2/flow_finetune_lowsnr/flow_ft_lowsnr_final.pt"
echo "     -- the target metric: does the sub-1dB slice improve?"
echo "  2. Re-validate the untouched headlines (in-domain n=2620, verify_eer n=400,"
echo "     2/3/4-speaker) -- same discipline as every prior promotion."
