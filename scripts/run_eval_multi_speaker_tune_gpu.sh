#!/bin/bash
#SBATCH --partition=gpupart_p100
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --job-name=eval_multispeaker_tune
#SBATCH --output=scripts/logs/eval_multispeaker_tune_%j.out

# Mask2Flow-TSE -- cheap cfg_scale/n_steps sweep for the 4-speaker
# (n_interferers=3) condition, BEFORE committing to a full training-side
# fix. Baseline already measured (job 10866, default cfg_scale=1.5,
# n_steps=4): 72.6% accuracy / 27.4% EER / 0.8072 AUC, n=300 -- only ~2.4pp
# short of a 75% target. This checks whether that gap closes for free via
# inference-time knobs alone before investing in a multi-speaker fine-tune.
#
# Precedent (see mask2flow-tse-cfg-warmup-fix memory /
# eval/KNOWN_LIMITATIONS.md): cfg_scale/n_steps tuning did NOT fix the
# closely-related t=0 training-coverage gap (n_steps=16 was strictly worse
# than n_steps=4 there). Expect the same here -- this is a cheap way to
# confirm/rule that out for the multi-speaker gap specifically, not an
# assumption it'll work. n_samples=100 (not 300) since this is an
# exploratory scan, not a final report -- rerun the winning config (if any)
# at n=300 to confirm before citing it anywhere.
#
# Submit:  sbatch scripts/run_eval_multi_speaker_tune_gpu.sh
# Each variant is resumable independently (--resume, distinct --output per
# combo) -- safe to resubmit this unchanged if it gets cut off.

set -eo pipefail

echo "=== DIAGNOSTIC ==="
nvidia-smi || echo "[Diag] nvidia-smi failed or not found -- no GPU visible to this job."
source /home/mtech1/25CS60R85/miniconda3/etc/profile.d/conda.sh
conda activate mask2flow
echo "which python3: $(which python3)"
echo "=== END DIAGNOSTIC ==="
echo ""

CKPT_ARGS="--mask_ckpt checkpoints_v2/masking/mask_best.pt \
    --flow_ckpt checkpoints_v2/flow/flow_best.pt \
    --projection_ckpt checkpoints_speaker_encoder/projection_latest.pt"

run_variant () {
    local label="$1" cfg_scale="$2" n_steps="$3"
    echo ""
    echo "=== Variant: $label  (cfg_scale=$cfg_scale, n_steps=$n_steps, n_interferers=3 / 4 total speakers) ==="
    python3 -u eval/eval_multi_speaker.py \
        --n_interferers 3 \
        --n_samples 100 \
        --cfg_scale "$cfg_scale" \
        --n_steps "$n_steps" \
        $CKPT_ARGS \
        --output "outputs/results/eval_multispeaker_4total_tune_${label}.jsonl" \
        --resume
}

# baseline repeat at n=100 (for a like-for-like comparison against the tuned
# variants below, since the n=300 baseline used a different sample count)
run_variant "cfg1.5_steps4_baseline" 1.5 4

# weaker guidance -- maybe less overconfident extrapolation with more interference
run_variant "cfg1.0_steps4" 1.0 4

# stronger guidance -- maybe sharper target/interferer separation
run_variant "cfg2.0_steps4" 2.0 4

# finer Euler integration (precedent suggests this likely won't help, cheap to confirm)
run_variant "cfg1.5_steps8" 1.5 8

echo ""
echo "=== All 4 variants done -- see each variant's own printed accuracy table above. ==="
