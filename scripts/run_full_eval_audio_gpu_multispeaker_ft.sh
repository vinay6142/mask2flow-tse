#!/bin/bash
#SBATCH --partition=gpupart_p100
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --job-name=full_eval_audio_gpu_ms_ft
#SBATCH --output=scripts/logs/full_eval_audio_gpu_ms_ft_%j.out

# Mask2Flow-TSE -- full-set audio-domain eval (mel + vocoding + SI-SDR), on GPU,
# against the hard-multispeaker fine-tuned Stage 2 checkpoint (job 10875,
# training/finetune_flow_hard_multispeaker.py, completed 2026-09-03 --
# see mask2flow-tse-multi-speaker-test memory).
#
# This is the FINAL pre-promotion check: the multi-speaker-specific numbers
# already look good (job 10887: 2-speaker accuracy held at 86.7% vs the
# 86.3% pre-fine-tune baseline; 4-speaker accuracy climbed 72.6% -> 76.7%,
# clearing the 75-80% target) but that only tested n=300 synthetic
# multi-speaker mixtures. This re-runs the project's actual PRIMARY,
# highest-rigor headline (full n=2620 test-clean, the same eval as every
# other checkpoint promotion in this project) against flow_ft_ms_final.pt,
# to confirm nothing outside the multi-speaker scope regressed before
# promoting it over checkpoints_v2/flow/flow_best.pt.
#
# Same script/diagnostic pattern as run_full_eval_audio_gpu_hardt0.sh
# (confirmed GPU-working) -- only --flow_ckpt and --output changed, so this
# run is directly comparable to outputs/results/full_eval_audio_gpu_hardt0.jsonl
# (the current flow_best.pt / pre-multispeaker-fine-tune baseline: 77.7%
# median mel S2vsS1, 3.5% catastrophic, +1.58dB SI-SDR gain vs S1).
#
# Submit:  sbatch scripts/run_full_eval_audio_gpu_multispeaker_ft.sh
# Resume (if it times out before finishing all 2620 samples): resubmit this
#   same script unchanged -- full_eval.py's --resume flag picks up from
#   wherever the output JSONL left off.
# Compare after it finishes:
#   python3 eval/full_eval.py --output outputs/results/full_eval_audio_gpu_hardt0.jsonl --summarize_only
#   python3 eval/full_eval.py --output outputs/results/full_eval_audio_gpu_multispeaker_ft.jsonl --summarize_only
#   Look specifically at the catastrophic-rate line and median SI-SDR gain in both --
#   the multispeaker-ft numbers should be close to (not necessarily identical to,
#   since training saw the SAME 2-speaker distribution only 50% of the time now)
#   the hardt0 baseline. A material regression here would argue against promoting
#   even though the multi-speaker numbers look good.

set -eo pipefail
# NOTE: deliberately NOT using `set -u` here -- conda's activation shell
# functions reference variables that can be unset, and `-u` silently kills
# the job right at `conda activate` on many conda versions.

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

echo "=== REAL JOB: full_eval.py (audio-domain, full test-clean, n=2620, hard-multispeaker checkpoint) ==="
python3 -u eval/full_eval.py \
    --mask_ckpt checkpoints_v2/masking/mask_best.pt \
    --flow_ckpt checkpoints_v2/flow_finetune_multispeaker/flow_ft_ms_final.pt \
    --projection_ckpt checkpoints_speaker_encoder/projection_latest.pt \
    --seed 42 --n_steps 4 \
    --batch_size 60 \
    --output outputs/results/full_eval_audio_gpu_multispeaker_ft.jsonl \
    --resume
