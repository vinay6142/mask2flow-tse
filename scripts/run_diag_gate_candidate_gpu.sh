#!/bin/bash
#SBATCH --partition=gpupart_p100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=diag_gate
#SBATCH --output=scripts/logs/diag_gate_%j.out

# Mask2Flow-TSE -- cheap onset-gate check for ONE candidate (2026-09-14).
#
# The onset gate (see scripts/run_diag_onset_ckpt_gpu.sh for the full
# diagnosis) is invisible to mel-MSE, SI-SDR and speaker accuracy, so a
# candidate can pass the whole battery and still sound wrong at utterance
# starts. This runs only the gate measurement, ~25min instead of ~2h, so a
# candidate that still gates can be rejected before spending the battery.
#
# Usage:
#   sbatch scripts/run_diag_gate_candidate_gpu.sh <flow_ckpt> <tag> [mask_ckpt] [cfg_scale] [onset_pad_frames]
#
# The optional 5th arg measures the INFERENCE-SIDE onset repair (models/flow.py
# inference_with_onset_splice) instead of a retrained checkpoint: it adds a
# spliced pad-N column ALONGSIDE the pad0 one, so both land in the same table,
# on the same samples, in a single job -- a paired comparison rather than two
# runs. e.g. the current system with a 12-frame lead-in:
#   sbatch scripts/run_diag_gate_candidate_gpu.sh checkpoints_v2/flow/flow_best.pt \
#       onsetpad12 checkpoints_v2/masking/mask_best.pt 2.5 12
#
# Reference points, same seed/samples/settings:
#   deployed (mask_best + flow_best, cfg 2.5)   7/24 gated, median onset -4.67dB
#   2026-09-04 pipeline                         0/12 gated, median onset -0.31dB
#   adoption bar for a candidate                <= 1/24

set -eo pipefail
# NOTE: deliberately NOT using `set -u` -- conda activation references unset vars.

CKPT="$1"
TAG="$2"
MASK_CKPT="${3:-checkpoints_v2/masking/mask_best.pt}"
CFG="${4:-2.5}"
ONSET_PAD="${5:-}"     # optional; empty = gate the checkpoint as-is at pad0

# Empty 5th arg keeps the original behavior exactly: pad0 only, splice off.
PADS="0"
SPLICE="0"
if [ -n "$ONSET_PAD" ]; then
    PADS="0,${ONSET_PAD}"
    SPLICE="100"
fi

if [ -z "$CKPT" ] || [ -z "$TAG" ]; then
    echo "ERROR: usage: sbatch $0 <flow_ckpt> <tag> [mask_ckpt] [cfg_scale] [onset_pad_frames]"
    exit 1
fi
for f in "$CKPT" "$MASK_CKPT"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: checkpoint not found: $f"
        exit 1
    fi
done

# Job 11183 spent its entire hour retrying a huggingface.co HEAD request for
# microsoft/wavlm-base-plus-sv and was killed by the time limit before running
# a single sample. The weights are already in the node's HF cache -- every
# earlier job loaded them -- so the network call was never needed. Forcing
# offline mode makes the load deterministic instead of dependent on the node
# having outbound HTTPS.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== DIAGNOSTIC ==="
nvidia-smi || echo "[Diag] nvidia-smi failed -- no GPU visible."
source /home/mtech1/25CS60R85/miniconda3/etc/profile.d/conda.sh
conda activate mask2flow
echo "which python3: $(which python3)"
echo "=== END DIAGNOSTIC ==="
echo ""
echo "=== Stage 2: $CKPT"
echo "=== Stage 1: $MASK_CKPT"
echo "=== cfg    : $CFG   tag: $TAG"
echo "=== pads   : $PADS   splice_frames: $SPLICE"
echo ""

python3 -u eval/diag_onset_padding.py \
    --mask_ckpt "$MASK_CKPT" --flow_ckpt "$CKPT" \
    --n_samples 8 --n_interferers 1,2,3 \
    --pads "$PADS" --splice_frames "$SPLICE" \
    --cfg_variants "${CFG}:0" \
    --output outputs/results/diag_gate_${TAG}.jsonl

echo ""
echo "Read 'gated samples (onset < -10dB)'. Deployed is 7/24; bar is <= 1/24."
echo "Passing here is necessary, not sufficient -- a passing candidate still"
echo "needs run_eval_candidate_gpu.sh and a re-export + listening pass."
