#!/bin/bash
#SBATCH --partition=gpupart_p100
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --job-name=full_eval_audio_gpu
#SBATCH --output=scripts/logs/full_eval_audio_gpu_%j.out

# Mask2Flow-TSE — full-set audio-domain eval (mel + vocoding + SI-SDR), on GPU.
#
# WHY THIS SCRIPT EXISTS: every previous run this project — INCLUDING earlier
# sbatch jobs that already requested --gres=gpu:1 (e.g. eval_newproj_10555)
# — printed "Device: cpu", not cuda. That means the earlier CPU fallback was
# NOT simply "ran outside sbatch" — GPU was requested and still unused. The
# likely cause is the job's shell not properly activating the `mask2flow`
# conda env (a --wrap one-liner doesn't source .bashrc/conda.sh the way an
# interactive login shell does, so `python3` can silently resolve to a
# different interpreter without a CUDA-enabled torch build). This script
# explicitly sources conda and activates the env, then runs a short
# diagnostic block BEFORE the real job so the log tells us definitively
# whether the GPU is visible in this job environment and why, instead of
# guessing again.
#
# HOW TO READ THE OUTPUT: check scripts/logs/full_eval_audio_gpu_<jobid>.out
# for a "=== DIAGNOSTIC ===" block near the top:
#   - "nvidia-smi" should list a GPU (P100) — if this is empty/errors, the
#     SLURM allocation itself isn't handing over a GPU to this job/partition.
#   - "torch.cuda.is_available()" should print True — if nvidia-smi shows a
#     GPU but this prints False, it's a torch/CUDA-build mismatch inside the
#     mask2flow conda env specifically.
#   - If both pass, "[FullEval] Device: cuda" should appear once the real
#     job's Python process starts below the diagnostic block.
#
# Submit:  sbatch scripts/run_full_eval_audio_gpu.sh
# Resume (if it times out before finishing all 2620 samples): resubmit this
#   same script unchanged — full_eval.py's --resume flag below picks up from
#   wherever outputs/results/full_eval_audio_gpu.jsonl left off.

set -eo pipefail
# NOTE: deliberately NOT using `set -u` here — conda's activation shell
# functions reference variables that can be unset, and `-u` silently kills
# the job right at `conda activate` on many conda versions. This bit people
# often enough that it's worth avoiding on purpose rather than rediscovering
# it from a confusing failure.

echo "=== DIAGNOSTIC ==="
echo "--- nvidia-smi ---"
nvidia-smi || echo "[Diag] nvidia-smi failed or not found — no GPU visible to this job."

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

echo "=== REAL JOB: full_eval.py (audio-domain, full test-clean, n=2620) ==="
python3 eval/full_eval.py \
    --mask_ckpt checkpoints_v2/masking/mask_best.pt \
    --flow_ckpt checkpoints_v2/flow/flow_best.pt \
    --projection_ckpt checkpoints_speaker_encoder/projection_latest.pt \
    --seed 42 --n_steps 4 \
    --batch_size 60 \
    --output outputs/results/full_eval_audio_gpu.jsonl \
    --resume
