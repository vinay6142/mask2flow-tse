#!/bin/bash
#SBATCH --partition=gpupart_p100
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --job-name=sweep_cfg_new
#SBATCH --output=scripts/logs/sweep_cfg_new_%j.out

# Mask2Flow-TSE -- re-tune the inference operating point for the CURRENT
# pipeline (timeline entry 26). cfg_scale=1.5 / n_steps=4 was validated back
# when Stage 1 was the step-130000 original and Stage 2 the hard-multispeaker
# checkpoint. BOTH stages have changed since (Stage 1 promoted 2026-09-12,
# Stage 2 adapted 2026-09-13), so the operating point is no longer known-good.
#
# WHY guidance strength specifically: the accuracy lost in-distribution is
# concentrated on EASY samples. Paired per-sample against the previous Stage 2,
# the adapted one is BETTER at 1-3 dB (genuine +0.0087) and WORSE at 7-10 dB
# (-0.0074), and its extra mel-catastrophic cases are 64% audio-unharmed. That
# is over-correction on inputs that barely need correcting, and cfg_scale is
# the direct knob for how hard Stage 2 pushes.
#
# Each config is measured in BOTH regimes, because the expected trade is
# "less guidance helps easy inputs, hurts hard ones" -- the point is to find
# whether a setting keeps the low-SNR win while recovering in-distribution.
#
# Baselines at the current setting (cfg 1.5, 4 steps), from job 11106:
#   trained SNR [1,10]: 2-speaker accuracy 84.3%
#   low SNR [-10,1)   : accuracy 78.7%,  SI-SDR vs mixture +6.66dB
# Pre-promotion reference (old S1 + old S2): 87.3% trained / 61.3% low.
#
# ~10 runs x n=300 x ~5min = ~50min total.
#
# Submit:  sbatch scripts/run_sweep_cfg_newpipeline_gpu.sh

set -eo pipefail
# NOTE: deliberately NOT using `set -u` -- conda activation references unset vars.

echo "=== DIAGNOSTIC ==="
nvidia-smi || echo "[Diag] nvidia-smi failed -- no GPU visible."
source /home/mtech1/25CS60R85/miniconda3/etc/profile.d/conda.sh
conda activate mask2flow
echo "which python3: $(which python3)"
echo "=== END DIAGNOSTIC ==="

COMMON="--mask_ckpt checkpoints_v2/masking/mask_best.pt \
    --flow_ckpt checkpoints_v2/flow/flow_best.pt \
    --projection_ckpt checkpoints_speaker_encoder/projection_latest.pt \
    --n_interferers 1 --n_samples 300 --resume"
OUT="outputs/results"

run() {   # run <cfg_scale> <n_steps> <regime: trained|lowsnr>
    local c="$1" s="$2" regime="$3" extra=""
    [ "$regime" = "lowsnr" ] && extra="--snr_min -10 --snr_max 1"
    echo ""
    echo "=== cfg_scale=$c  n_steps=$s  regime=$regime ==="
    python3 -u eval/eval_multi_speaker.py $COMMON \
        --cfg_scale "$c" --n_steps "$s" $extra \
        --output "$OUT/eval_sweep_cfg${c}_steps${s}_${regime}.jsonl"
}

for c in 1.0 1.25 1.5 2.0; do
    run "$c" 4 trained
    run "$c" 4 lowsnr
done

# One step-count probe at the current guidance, to check the two knobs are not
# interchangeable here (n_steps=16 was already shown worse for the t~0 gap).
run 1.5 8 trained
run 1.5 8 lowsnr

echo ""
echo "======================================================================"
echo "  Compare the [Speaker-verification ACCURACY] table from each run."
echo "  Looking for: a setting that recovers trained-SNR accuracy toward the"
echo "  pre-promotion 87.3% WITHOUT giving back the low-SNR gain (78.7%)."
echo "  If none does, the operating point is not the lever and a gentler"
echo "  Stage-2 curriculum (~25-30% low-SNR, 13h) is the next option."
echo "======================================================================"
