#!/bin/bash
#SBATCH --partition=gpupart_p100
#SBATCH --gres=gpu:1
#SBATCH --time=47:59:00
#SBATCH --job-name=finetune_flow_multispeaker
#SBATCH --output=scripts/logs/finetune_flow_multispeaker_%j.out

# NOTE (added after job 10875): python3 -u below is required -- the FIRST
# submission of this script omitted it, and Python's default full block
# buffering on a non-tty stdout meant every "[Step N] val_loss=..."/
# "checkpoint saved" print sat unflushed for hours despite validation and
# checkpointing genuinely happening on schedule (confirmed independently by
# checking checkpoints_v2/flow_finetune_multispeaker/ on disk -- new
# flow_ft_ms_best_step*.pt files appeared right on the val_every=1000
# cadence). Not a training bug, just a monitoring blind spot -- -u fixes it
# for any future submission of this script.
#
# Mask2Flow-TSE -- Stage 2 fine-tune, hard multi-speaker curriculum.
# See training/finetune_flow_hard_multispeaker.py for the full design
# rationale (short version: Stage 2 loses 78.3%->53.7% of its improvement-
# over-Stage-1 as speaker count rises 2->4 total speakers; Stage 1 itself is
# unaffected; a cfg_scale/n_steps sweep ruled out an inference-time fix; this
# closes the gap by training-time exposure to 3/4-speaker mixtures instead).
#
# Auto-resume: if checkpoints_v2/flow_finetune_multispeaker/flow_ft_ms_latest.pt
# already exists (e.g. this job got killed by the time limit), resubmitting
# this SAME script picks up from there automatically -- no manual --resume
# flag needed. A fresh run always starts from --flow_ckpt below (currently
# checkpoints_v2/flow/flow_best.pt, the current best Stage 2 checkpoint),
# never overwriting it -- checkpoints from THIS fine-tune land in a separate
# checkpoints_v2/flow_finetune_multispeaker/ directory.
#
# 47:59:00 is this partition's confirmed GPU max (see
# mask2flow-tse-multi-speaker-test memory). ~25k steps is a STARTING POINT
# budget, similar to the successful hard-t0 fine-tune (~11-12h on a P100 for
# that run) -- if this doesn't finish in one submission, just resubmit; the
# auto-resume above picks it up. Do NOT assume 25k steps is enough on its
# own merit -- check the real target metric (see below) at intermediate
# checkpoints and extend or stop based on that, not on step count alone.
#
# Submit:  sbatch scripts/run_finetune_flow_multispeaker_gpu.sh
# Monitor progress against the REAL target metric (not the val_loss proxy
# printed during training, which stays 2-speaker-only by design -- see
# build_dataloaders_multispeaker's docstring) by running
# eval/eval_multi_speaker.py against any saved checkpoint at n_interferers=1,
# 2, AND 3 -- both that 2-speaker accuracy hasn't regressed below ~86% AND
# that 4-speaker accuracy has improved from the 72.6% pre-fine-tune baseline
# toward the 75-80% target, e.g.:
#   python3 eval/eval_multi_speaker.py --n_interferers 3 --n_samples 300 \
#       --mask_ckpt checkpoints_v2/masking/mask_best.pt \
#       --flow_ckpt checkpoints_v2/flow_finetune_multispeaker/flow_ft_ms_best.pt \
#       --projection_ckpt checkpoints_speaker_encoder/projection_latest.pt \
#       --output outputs/results/eval_multispeaker_4total_ft.jsonl

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

RESUME_CKPT="checkpoints_v2/flow_finetune_multispeaker/flow_ft_ms_latest.pt"
RESUME_ARG=""
if [ -f "$RESUME_CKPT" ]; then
    echo "=== Resuming from $RESUME_CKPT ==="
    RESUME_ARG="--resume $RESUME_CKPT"
else
    echo "=== Fresh start from checkpoints_v2/flow/flow_best.pt ==="
fi

echo "=== REAL JOB: finetune_flow_hard_multispeaker.py ==="
python3 -u training/finetune_flow_hard_multispeaker.py \
    --config configs/default_v2.yaml \
    --mask_ckpt checkpoints_v2/masking/mask_best.pt \
    --flow_ckpt checkpoints_v2/flow/flow_best.pt \
    --interferer_count_probs 0.5,0.3,0.2 \
    --max_steps 25000 \
    $RESUME_ARG
