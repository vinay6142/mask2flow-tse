#!/bin/bash
#SBATCH --partition=gpupart_p100
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --job-name=full_eval_audio_gpu_hardt0_16k
#SBATCH --output=scripts/logs/full_eval_audio_gpu_hardt0_step16000_%j.out

# Mask2Flow-TSE — full-set audio-domain eval (mel + vocoding + SI-SDR), on GPU,
# against flow_ft_best.pt (step 16000) -- the symmetric counterpart to
# run_full_eval_audio_gpu_hardt0.sh (job 10661, which covered flow_ft_final.pt /
# step 25000). Purpose: verify_eer.py showed step 16000 marginally ahead on EER
# (13.2% vs 13.8%, n=400 -- likely within noise), but only step 25000 has had the
# full n=2620 catastrophic-rate/SI-SDR sweep run against it. This job closes that
# gap so the checkpoint-promotion decision rests on the same metric for both.
#
# Submit:  sbatch scripts/run_full_eval_audio_gpu_hardt0_step16000.sh
# Resume (if it times out before finishing all 2620 samples): resubmit this
#   same script unchanged -- full_eval.py's --resume flag picks up from
#   wherever outputs/results/full_eval_audio_gpu_hardt0_step16000.jsonl left off.
# Compare after it finishes (all three should be summarized side by side):
#   python3 eval/full_eval.py --output outputs/results/full_eval_audio_gpu.jsonl --summarize_only                 # flow_best.pt (pre-finetune)
#   python3 eval/full_eval.py --output outputs/results/full_eval_audio_gpu_hardt0.jsonl --summarize_only          # flow_ft_final.pt (step 25000)
#   python3 eval/full_eval.py --output outputs/results/full_eval_audio_gpu_hardt0_step16000.jsonl --summarize_only # flow_ft_best.pt (step 16000)
#   Look specifically at the catastrophic-rate line and median SI-SDR gain in all three.

set -eo pipefail
# NOTE: deliberately NOT using `set -u` here -- conda's activation shell
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

echo "=== REAL JOB: full_eval.py (audio-domain, full test-clean, n=2620, flow_ft_best.pt/step16000) ==="
python3 eval/full_eval.py \
    --mask_ckpt checkpoints_v2/masking/mask_best.pt \
    --flow_ckpt checkpoints_v2/flow_finetune_hardt0/flow_ft_best.pt \
    --projection_ckpt checkpoints_speaker_encoder/projection_latest.pt \
    --seed 42 --n_steps 4 \
    --batch_size 60 \
    --output outputs/results/full_eval_audio_gpu_hardt0_step16000.jsonl \
    --resume
