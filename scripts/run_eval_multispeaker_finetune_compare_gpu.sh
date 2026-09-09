#!/bin/bash
#SBATCH --partition=gpupart_p100
#SBATCH --gres=gpu:1
#SBATCH --time=47:59:00
#SBATCH --job-name=eval_multispeaker_ftcmp
#SBATCH --output=scripts/logs/eval_multispeaker_ftcmp_%j.out

# Mask2Flow-TSE -- decide whether the hard-multispeaker fine-tune (job
# 10875, completed 2026-09-03, 25000 steps/13.29h) should be promoted.
# Same decision process as the hard-t0 fine-tune: evaluate BOTH candidate
# checkpoints on the REAL target metric (eval/eval_multi_speaker.py's
# accuracy table), not the training-time val_loss proxy (which never
# improved past step 6000 -- expected, matches the hard-t0 precedent, not
# itself a verdict).
#
# Two candidates x two conditions = 4 runs, n=300 each (matches the n=300
# convention already used for every multi-speaker number in this project):
#   - flow_ft_ms_best_step6000.pt (the proxy's pick)
#   - flow_ft_ms_final.pt         (the completed run's last checkpoint)
#   at n_interferers=1 (2 total speakers -- must NOT have regressed below
#   the pre-fine-tune 86.3%) and n_interferers=3 (4 total speakers -- the
#   number to compare against the 72.6% pre-fine-tune baseline / 75-80%
#   target).
#
# Pre-fine-tune baseline (flow_best.pt, for reference, already measured):
#   2 speakers: 86.3% accuracy / 78.3% median mel S2vsS1
#   4 speakers: 72.6% accuracy / 53.7% median mel S2vsS1
#
# Submit:  sbatch scripts/run_eval_multispeaker_finetune_compare_gpu.sh
# Each of the 4 runs is independently resumable (--resume, distinct
# --output per run) -- safe to resubmit this unchanged if it gets cut off.

set -eo pipefail

echo "=== DIAGNOSTIC ==="
nvidia-smi || echo "[Diag] nvidia-smi failed or not found -- no GPU visible to this job."
source /home/mtech1/25CS60R85/miniconda3/etc/profile.d/conda.sh
conda activate mask2flow
echo "which python3: $(which python3)"
echo "=== END DIAGNOSTIC ==="
echo ""

MASK_ARGS="--mask_ckpt checkpoints_v2/masking/mask_best.pt \
    --projection_ckpt checkpoints_speaker_encoder/projection_latest.pt"

run_eval () {
    local label="$1" flow_ckpt="$2" n_interferers="$3"
    echo ""
    echo "=== $label  (flow_ckpt=$flow_ckpt, n_interferers=$n_interferers / $((n_interferers + 1)) total speakers) ==="
    python3 -u eval/eval_multi_speaker.py \
        --n_interferers "$n_interferers" \
        --n_samples 300 \
        --flow_ckpt "$flow_ckpt" \
        $MASK_ARGS \
        --output "outputs/results/eval_multispeaker_${label}.jsonl" \
        --resume
}

echo "=== [1/4] step6000 candidate, 2 speakers (regression check) ==="
run_eval "2total_ft_step6000" "checkpoints_v2/flow_finetune_multispeaker/flow_ft_ms_best_step6000.pt" 1

echo "=== [2/4] step6000 candidate, 4 speakers (the real test) ==="
run_eval "4total_ft_step6000" "checkpoints_v2/flow_finetune_multispeaker/flow_ft_ms_best_step6000.pt" 3

echo "=== [3/4] final candidate, 2 speakers (regression check) ==="
run_eval "2total_ft_final" "checkpoints_v2/flow_finetune_multispeaker/flow_ft_ms_final.pt" 1

echo "=== [4/4] final candidate, 4 speakers (the real test) ==="
run_eval "4total_ft_final" "checkpoints_v2/flow_finetune_multispeaker/flow_ft_ms_final.pt" 3

echo ""
echo "=== All 4 runs done -- see each run's own printed accuracy table above. ==="
echo "Pre-fine-tune baseline for reference: 2spk 86.3% / 4spk 72.6% (flow_best.pt, n=300 each)."
