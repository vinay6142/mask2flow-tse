#!/bin/bash
#SBATCH --partition=gpupart_p100
#SBATCH --gres=gpu:1
#SBATCH --time=47:59:00
#SBATCH --job-name=flow_adapt_loggain
#SBATCH --output=scripts/logs/flow_adapt_loggain_%j.out

# Mask2Flow-TSE -- adapt Stage 2 to the log_gain Stage 1 (timeline entry 30).
#
# WHY THIS IS NOW WORTH RUNNING. The log_gain formulation (X + log M, true
# deletion) was rejected in entry 22 for ONE reason: the frozen Stage 2 had
# never seen mask output below the mixture, so it destroyed that input in
# 97.7-98.7% of samples. Since then (entry 25) Stage 2 was successfully adapted
# to a changed Stage 1 using exactly this recipe -- and its val_loss improved,
# the first time that proxy moved the right way in four fine-tunes. The reason
# for the rejection has therefore been removed.
#
# And log_gain's Stage 1 is not merely better at low SNR, it is better
# EVERYWHERE (median masked mel error):
#     Stage 1                trained SNR     low SNR
#     original                  5.944          13.151
#     current (promoted)        5.939          11.180
#     log_gain                  2.296           3.809
# A 2.6x cleaner input in the trained regime is why this can RAISE
# in-distribution accuracy rather than trade it, which is the goal here: the
# target is to match or beat the pre-campaign best, not to trade regimes.
#
# TARGET (set before the run): every accuracy at or above the pre-campaign
# like-for-like best, while keeping the low-SNR gain.
#     corpus-wide (n=400)      >= 86.5%     (currently 86.2%)
#     2-speaker                >= 87.3%     (currently 85.7%)
#     3-speaker                >= 82.1%     (currently 80.0%)
#     4-speaker                >= 77.0%     (currently 77.0%)
#     low-SNR                  >= 80.3%     (do not give back the campaign)
#     in-domain SI-SDR vs mixture >= +1.75dB (currently +1.75dB)
# If it clears these, promote BOTH the log_gain Stage 1 and this Stage 2
# together -- they are a matched pair and neither works with the other's
# counterpart. If it clears the low-SNR bar but not the in-distribution ones,
# it is the same trade-off verdict as the Stage-2 curricula and nothing moves.
#
# Curriculum matches what the log_gain Stage 1 was trained on (50% from
# [-10,1) dB, 2/3/4-speaker mix), so Stage 2 sees the distribution that Stage 1
# actually produces. lr 4e-5, ~25k steps, ~13h on a P100.
#
# Stage 1 stays FROZEN and is read from the log_gain checkpoint directly --
# load_frozen_masking restores mask_mode=log_gain from the checkpoint itself,
# so no flag is needed. Neither promoted checkpoint is touched.
#
# Submit:  sbatch scripts/run_finetune_flow_adapt_loggain_gpu.sh

set -eo pipefail
# NOTE: deliberately NOT using `set -u` -- conda activation references unset vars.

MASK_LG="checkpoints_v2/mask_finetune_lowsnr_log_gain/mask_ft_log_gain_final.pt"
[ -f "$MASK_LG" ] || { echo "ERROR: log_gain Stage 1 not found: $MASK_LG"; exit 1; }

echo "=== DIAGNOSTIC ==="
nvidia-smi || echo "[Diag] nvidia-smi failed -- no GPU visible."
source /home/mtech1/25CS60R85/miniconda3/etc/profile.d/conda.sh
conda activate mask2flow
echo "which python3: $(which python3)"
df -h . | tail -1
echo "=== END DIAGNOSTIC ==="
echo ""

LATEST="checkpoints_v2/flow_finetune_adapt_loggain/flow_ft_lg_latest.pt"
RESUME_ARG=""
if [ -f "$LATEST" ]; then
    echo "=== resuming from $LATEST ==="
    RESUME_ARG="--resume $LATEST"
else
    echo "=== starting fresh from flow_best.pt ==="
fi
echo ""

# --low_snr_range MUST use '=': space-separated, argparse reads '-10,1' as an
# option name (this mistake killed job 10990).
python3 -u training/finetune_flow_hard_lowsnr.py \
    --config configs/default_v2.yaml \
    --mask_ckpt "$MASK_LG" \
    --flow_ckpt checkpoints_v2/flow/flow_best.pt \
    --low_snr_prob 0.5 \
    --low_snr_range=-10,1 \
    --interferer_count_probs 0.5,0.3,0.2 \
    --max_steps 25000 \
    --stage flow_finetune_adapt_loggain \
    --prefix flow_ft_lg \
    $RESUME_ARG

echo ""
echo "=== Done. Evaluate the PAIR (log_gain Stage 1 + this Stage 2): ==="
echo "  sbatch scripts/run_eval_candidate_gpu.sh \\"
echo "      checkpoints_v2/flow_finetune_adapt_loggain/flow_ft_lg_final.pt loggain_pair \\"
echo "      $MASK_LG"
echo "  (Stage 1 must be passed explicitly -- the promoted default is the"
echo "   multiplicative one, and these two only work together.)"
