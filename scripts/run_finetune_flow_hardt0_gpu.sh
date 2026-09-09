#!/bin/bash
#SBATCH --partition=gpupart_p100
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=finetune_flow_hardt0
#SBATCH --output=scripts/logs/finetune_flow_hardt0_%j.out

# Mask2Flow-TSE -- Stage 2 fine-tune, hard t~0 timestep reweighting.
# See training/finetune_flow_hard_t0.py for the full design rationale.
#
# Auto-resume: if checkpoints_v2/flow_finetune_hardt0/flow_ft_latest.pt
# already exists (e.g. this job got killed by a time limit), resubmitting
# this SAME script picks up from there automatically -- no manual --resume
# flag needed. A fresh run always starts from checkpoints_v2/flow/flow_best.pt
# (the current best Stage 2 checkpoint), never overwriting it.
#
# 24h is a starting guess for 25k steps on a P100 -- check your cluster's
# max walltime for this partition (sinfo -p gpupart_p100) and adjust; if
# it is shorter than 24h, that is fine, just resubmit this script to
# continue from flow_ft_latest.pt until max_steps is reached.
#
# Submit:  sbatch scripts/run_finetune_flow_hardt0_gpu.sh
# Monitor progress against the REAL target metric (not the val_loss proxy
# printed during training) by running eval/verify_eer.py against any saved
# checkpoint, e.g.:
#   python3 eval/verify_eer.py --mask_ckpt checkpoints_v2/masking/mask_best.pt \
#       --flow_ckpt checkpoints_v2/flow_finetune_hardt0/flow_ft_best.pt \
#       --seed 42 --n_steps 4 --max_samples 400

set -eo pipefail
# NOTE: deliberately NOT using `set -u` -- conda's activation shell
# functions reference variables that can be unset, and `-u` silently
# kills the job right at `conda activate` on many conda versions.

echo "=== DIAGNOSTIC ==="
echo "--- nvidia-smi ---"
nvidia-smi || echo "[Diag] nvidia-smi failed or not found -- no GPU visible to this job."

echo "--- conda activation ---"
source /home/mtech1/25CS60R85/miniconda3/etc/profile.d/conda.sh
conda activate mask2flow
echo "which python3: $(which python3)"
echo "conda env: ${CONDA_DEFAULT_ENV:-<not set>}"

echo "--- torch/CUDA check ---"
python3 -c "
import torch
print('torch version      :', torch.__version__)
print('torch.cuda avail    :', torch.cuda.is_available())
print('cuda device count   :', torch.cuda.device_count())
if torch.cuda.is_available():
    print('cuda device name    :', torch.cuda.get_device_name(0))
"
echo "=== END DIAGNOSTIC ==="
echo ""

RESUME_CKPT="checkpoints_v2/flow_finetune_hardt0/flow_ft_latest.pt"
RESUME_ARG=""
if [ -f "$RESUME_CKPT" ]; then
    echo "=== Resuming from $RESUME_CKPT ==="
    RESUME_ARG="--resume $RESUME_CKPT"
else
    echo "=== Fresh start from checkpoints_v2/flow/flow_best.pt ==="
fi

echo "=== REAL JOB: finetune_flow_hard_t0.py ==="
python3 training/finetune_flow_hard_t0.py \
    --config configs/default_v2.yaml \
    --mask_ckpt checkpoints_v2/masking/mask_best.pt \
    --flow_ckpt checkpoints_v2/flow/flow_best.pt \
    --max_steps 25000 \
    $RESUME_ARG
