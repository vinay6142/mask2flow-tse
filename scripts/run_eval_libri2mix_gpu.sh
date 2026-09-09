#!/bin/bash
#SBATCH --partition=gpupart_p100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=eval_libri2mix
#SBATCH --output=scripts/logs/eval_libri2mix_%j.out

# Mask2Flow-TSE -- Libri2Mix cross-corpus evaluation (see eval/eval_libri2mix.py
# docstring for full design notes: reuses inference/infer.py's chunked
# overlap-add path since Libri2Mix mixtures have real variable length, and
# evaluates both source_1-as-target and source_2-as-target per mixture).
#
# Requires scripts/download_wham.sh and scripts/generate_libri2mix_test.sh
# to have both finished first.
#
# Currently points at checkpoints_v2/flow/flow_best.pt, which as of
# 2026-08-29 IS the hard-t0 fine-tuned checkpoint (promoted for real --
# see mask2flow-tse-overview memory) -- so this is the number to compare
# against the existing test-clean headline (77.7% median S2vsS1, 3.5%
# catastrophic rate, n=2620).
#
# Submit:  sbatch scripts/run_eval_libri2mix_gpu.sh
# Resume (if it times out before finishing all mixtures): resubmit this
#   same script unchanged -- eval_libri2mix.py's --resume flag picks up
#   from wherever outputs/results/eval_libri2mix_min.jsonl left off.
# Re-print stats without rerunning anything:
#   python3 eval/eval_libri2mix.py --output outputs/results/eval_libri2mix_min.jsonl --summarize_only

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

echo "=== REAL JOB: eval_libri2mix.py (16k, min, mix_clean, both directions) ==="
python3 eval/eval_libri2mix.py \
    --mask_ckpt checkpoints_v2/masking/mask_best.pt \
    --flow_ckpt checkpoints_v2/flow/flow_best.pt \
    --projection_ckpt checkpoints_speaker_encoder/projection_latest.pt \
    --librimix_dir data/raw/LibriMix_storage/Libri2Mix/wav16k/min \
    --librimix_input_csv data/raw/LibriMix/metadata/Libri2Mix_test_only/libri2mix_test-clean.csv \
    --librispeech_dir data/raw/LibriSpeech \
    --output outputs/results/eval_libri2mix_min.jsonl \
    --resume
