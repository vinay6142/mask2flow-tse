#!/bin/bash
#SBATCH --partition=gpupart_p100
#SBATCH --gres=gpu:1
#SBATCH --time=47:59:00
#SBATCH --job-name=eval_multispeaker
#SBATCH --output=scripts/logs/eval_multispeaker_%j.out

# Mask2Flow-TSE -- multi-speaker (>2 total speakers) generalization test.
# See eval/eval_multi_speaker.py docstring for full design notes: the model
# is only ever TRAINED on 2-total-speaker mixtures (1 target + 1 interferer),
# so this asks what happens as more simultaneous talkers are added.
#
# History: job 10862 (2h cap, killed mid-run) and job 10865 (47:59:00 cap,
# completed both conditions cleanly -- see mask2flow-tse-multi-speaker-test
# memory / docs/results_and_limitations.md Sec 5.5.3 for those numbers:
# 2 speakers 78.3% median mel S2vsS1 / 1.3% catastrophic; 3 speakers 58.8% /
# 2.7% -- real but graceful degradation) established the script's correctness
# and resolved an earlier throughput mystery (filesystem cache warm-up, not a
# bug -- steady-state is ~0.5s/sample once warm).
#
# THIS RUN adds two things:
#   1. A real corpus-wide speaker-verification ACCURACY/EER metric
#      (eval_multi_speaker.py now reuses eval/verify_eer.py's exact
#      genuine-vs-impostor-trial methodology), directly comparable to the
#      existing 86.2%-accuracy 2-speaker headline -- job 10865's numbers only
#      had raw per-sample cosine similarity, which is NOT the same scale as
#      an "accuracy" target and can't be compared to one.
#   2. A THIRD condition: n_interferers=3 (4 total speakers) -- the actual
#      new ask, never measured before.
#
# Old eval_multispeaker_2total.jsonl / _3total.jsonl records (from job 10865)
# predate the new embedding fields the accuracy metric needs, so they are
# deleted and fully regenerated below rather than resumed -- this is a
# one-time reset, not routine; after this, --resume is safe/idempotent again
# for all three conditions as usual.
#
# Currently points at checkpoints_v2/flow/flow_best.pt, which as of
# 2026-08-29 IS the hard-t0 fine-tuned checkpoint (promoted -- see
# mask2flow-tse-overview memory).
#
# Submit:  sbatch scripts/run_eval_multi_speaker_gpu.sh
# Resume (if it times out or otherwise gets cut off before finishing):
#   resubmit this exact script unchanged -- all three invocations below pass
#   --resume, which picks up from wherever each output JSONL left off (the
#   one-time delete above only runs once per submission of THIS script, so a
#   resubmit won't wipe partial progress from the SAME submission's retry --
#   it only re-deletes if you literally rerun the whole script from scratch,
#   which is intentional: this script's job is "regenerate these 3 files
#   correctly", not "append forever").
# Re-print stats without rerunning anything:
#   python3 eval/eval_multi_speaker.py --output outputs/results/eval_multispeaker_2total.jsonl --summarize_only
#   python3 eval/eval_multi_speaker.py --output outputs/results/eval_multispeaker_3total.jsonl --summarize_only
#   python3 eval/eval_multi_speaker.py --output outputs/results/eval_multispeaker_4total.jsonl --summarize_only
#
# NOTE: 47:59:00 is this partition's confirmed max. At the observed
# steady-state (~0.5s/sample once warm), 3 x 300 samples is ~10-20 min of
# real compute plus per-condition model-loading overhead -- comfortably
# within budget; the generous cap is insurance, not an expectation this
# takes anywhere near 48h.

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

echo "=== One-time reset: old 2total/3total records predate the accuracy-metric embedding fields ==="
rm -f outputs/results/eval_multispeaker_2total.jsonl outputs/results/eval_multispeaker_3total.jsonl
echo "removed (if present); regenerating both fresh below, plus the new 4-speaker condition."
echo ""

echo "=== [1/3] BASELINE: n_interferers=1 (2 total speakers, the trained condition) ==="
python3 -u eval/eval_multi_speaker.py \
    --n_interferers 1 \
    --n_samples 300 \
    --mask_ckpt checkpoints_v2/masking/mask_best.pt \
    --flow_ckpt checkpoints_v2/flow/flow_best.pt \
    --projection_ckpt checkpoints_speaker_encoder/projection_latest.pt \
    --output outputs/results/eval_multispeaker_2total.jsonl \
    --resume

echo ""
echo "=== [2/3] n_interferers=2 (3 total speakers) ==="
python3 -u eval/eval_multi_speaker.py \
    --n_interferers 2 \
    --n_samples 300 \
    --mask_ckpt checkpoints_v2/masking/mask_best.pt \
    --flow_ckpt checkpoints_v2/flow/flow_best.pt \
    --projection_ckpt checkpoints_speaker_encoder/projection_latest.pt \
    --output outputs/results/eval_multispeaker_3total.jsonl \
    --resume

echo ""
echo "=== [3/3] NEW: n_interferers=3 (4 total speakers) ==="
python3 -u eval/eval_multi_speaker.py \
    --n_interferers 3 \
    --n_samples 300 \
    --mask_ckpt checkpoints_v2/masking/mask_best.pt \
    --flow_ckpt checkpoints_v2/flow/flow_best.pt \
    --projection_ckpt checkpoints_speaker_encoder/projection_latest.pt \
    --output outputs/results/eval_multispeaker_4total.jsonl \
    --resume
