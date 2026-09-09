#!/bin/bash
#SBATCH --partition=gpupart_p100
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --job-name=full_eval_audio_gpu_hardt0
#SBATCH --output=scripts/logs/full_eval_audio_gpu_hardt0_%j.out

# Mask2Flow-TSE — full-set audio-domain eval (mel + vocoding + SI-SDR), on GPU,
# against the hard-t0 fine-tuned Stage 2 checkpoint (see run_finetune_flow_hardt0_gpu.sh,
# job 10606, and eval/verify_eer.py's corpus-wide EER result for the backstory).
#
# Same script/diagnostic pattern as run_full_eval_audio_gpu.sh (confirmed GPU-working
# via job 10606) — only --flow_ckpt and --output changed, so this run is directly
# comparable to outputs/results/full_eval_audio_gpu.jsonl (the pre-finetune baseline).
#
# Submit:  sbatch scripts/run_full_eval_audio_gpu_hardt0.sh
# Resume (if it times out before finishing all 2620 samples): resubmit this
#   same script unchanged — full_eval.py's --resume flag picks up from
#   wherever outputs/results/full_eval_audio_gpu_hardt0.jsonl left off.
# Compare after it finishes:
#   python3 eval/full_eval.py --output outputs/results/full_eval_audio_gpu.jsonl --summarize_only
#   python3 eval/full_eval.py --output outputs/results/full_eval_audio_gpu_hardt0.jsonl --summarize_only
#   Look specifically at the catastrophic-rate line and median SI-SDR gain in both.

set -eo pipefail
# NOTE: deliberately NOT using `set -u` here — conda's activation shell
# functions reference variables that can be unset, and `-u` silently kills
# the job right at `conda activate` on many conda versions.

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

echo "=== REAL JOB: full_eval.py (audio-domain, full test-clean, n=2620, hard-t0 checkpoint) ==="
python3 eval/full_eval.py \
    --mask_ckpt checkpoints_v2/masking/mask_best.pt \
    --flow_ckpt checkpoints_v2/flow_finetune_hardt0/flow_ft_final.pt \
    --projection_ckpt checkpoints_speaker_encoder/projection_latest.pt \
    --seed 42 --n_steps 4 \
    --batch_size 60 \
    --output outputs/results/full_eval_audio_gpu_hardt0.jsonl \
    --resume
