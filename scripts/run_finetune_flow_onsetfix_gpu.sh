#!/bin/bash
#SBATCH --partition=gpupart_p100
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=ft_onsetfix
#SBATCH --output=scripts/logs/ft_onsetfix_%j.out

# Mask2Flow-TSE -- Stage-2 rebalance to remove the onset gate (2026-09-14).
#
# THE PROBLEM. A listening pass found the deployed pipeline gates the first
# ~0.5s of some multi-speaker utterances: the output sits on a flat ~-60dB
# floor regardless of the target, then switches on. 5/12 exported samples,
# 3/8 of the 4-speaker ones. The 2026-09-04 pipeline did not do this (0/12).
#
# WHAT IT IS NOT (all tested, jobs 11174-11177):
#   - not position-0 / missing left context: a silent lead-in does not fix it
#   - not the operating point: cfg_scale 1.5..2.5 x cfg_warmup 0..2 moves the
#     gate count only 6..7 of 24
#   - not either checkpoint on its own. The 2x2 (job 11177, all at deployed
#     cfg 2.5) is unambiguous:
#         old S1 + old S2   1/24 gated   (control)
#         new S1 + old S2   1/24
#         old S1 + new S2   2/24
#         new S1 + new S2   7/24   <-- only the pair
#     It is an emergent interaction of the two promoted stages, which is
#     consistent with timeline entry 30's finding that the stages are
#     co-adapted.
#
# WHY REVERTING IS NOT THE ANSWER. The new-S1 + old-S2 pairing has no gate,
# but its full battery already exists (job 11081): low-SNR accuracy 69.7% vs
# the deployed 80.3%, Libri2Mix +1.91dB vs +3.39dB, catastrophic 19.1% vs
# 7.7%. Removing the gate that way costs most of the low-SNR campaign.
#
# THE HYPOTHESIS THIS RUN TESTS. Stage 2 is not inheriting a Stage-1 error --
# it is over-correcting. Measured over the first 0.5s of the exported samples:
#         Stage 1 level   Stage 2 level   what Stage 2 ADDS
#   gated     -2.8dB         -27.4dB          -23.6dB
#   clean     +0.2dB          -3.3dB           -3.4dB
# Stage 1's onset level is comparable in both groups (in two gated samples it
# is +11 to +13dB, i.e. it LEAVES residual interference); Stage 2 then removes
# 22-33dB and lands on silence. The adaptation that produced this Stage 2 ran
# at --low_snr_prob 0.5, i.e. half its batches at [-10,1)dB, where suppressing
# hard is usually right. The guess is that this taught an over-aggressive
# suppression prior that misfires at trained SNR where several speakers start
# together.
#
# SO: continue from the CURRENT flow_best (keeping its low-SNR competence)
# with the low-SNR share cut 0.5 -> 0.15, and the interferer mix weighted
# toward the 3- and 4-speaker cases where the gate actually appears. Short
# (8000 steps) and low LR, because the aim is to pull back an over-correction,
# not to retrain: a long run risks simply undoing the adaptation and landing
# back on cell B's numbers.
#
# PRE-REGISTERED BARS -- fixed before any result is seen. The candidate is
# adopted ONLY if it clears the gate AND holds every headline:
#     gate         <= 1/24 on eval/diag_onset_padding.py (deployed: 7/24)
#     low-SNR acc  >= 80.3%
#     corpus-wide  >= 86.2%
#     2/3/4 spk    >= 85.7 / 80.0 / 77.0
#     Libri2Mix    >= +3.39dB vs mixture, catastrophic <= 7.7%
# Anything less and the honest outcome is to keep the deployed checkpoint and
# document the gate as a characterized limitation -- the user's standing
# constraint is that accuracy must not drop below the previous best.
#
# flow_best.pt is NOT touched; this writes to its own stage directory.
#
# Submit: sbatch scripts/run_finetune_flow_onsetfix_gpu.sh
# Then:   sbatch scripts/run_diag_onset_ckpt_gpu.sh   (gate check, cheap)
#         sbatch scripts/run_eval_candidate_gpu.sh \
#             checkpoints_v2/flow_finetune_onsetfix/flow_ft_onsetfix_final.pt \
#             onsetfix checkpoints_v2/masking/mask_best.pt 2.5

set -eo pipefail
# NOTE: deliberately NOT using `set -u` -- conda activation references unset vars.

# Guard against job 11183's failure: a huggingface.co HEAD request for
# microsoft/wavlm-base-plus-sv hung and retried until the job hit its time
# limit, having run nothing. The weights are already in the node's HF cache,
# so force offline loading rather than depending on outbound HTTPS.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== DIAGNOSTIC ==="
nvidia-smi || echo "[Diag] nvidia-smi failed -- no GPU visible."
source /home/mtech1/25CS60R85/miniconda3/etc/profile.d/conda.sh
conda activate mask2flow
echo "which python3: $(which python3)"
echo "=== END DIAGNOSTIC ==="
echo ""

STAGE_DIR=checkpoints_v2/flow_finetune_onsetfix
LATEST="$STAGE_DIR/flow_ft_onsetfix_latest.pt"
RESUME_ARG=""
if [ -f "$LATEST" ]; then
    echo "=== resuming from $LATEST ==="
    RESUME_ARG="--resume $LATEST"
else
    echo "=== starting fresh from checkpoints_v2/flow/flow_best.pt ==="
fi

# NOTE: --low_snr_range needs the = form. A bare `--low_snr_range -10,1` is
# parsed as an option name because of the leading '-' and kills the job
# instantly (this cost job 10990).
python3 -u training/finetune_flow_hard_lowsnr.py \
    --config configs/default_v2.yaml \
    --mask_ckpt checkpoints_v2/masking/mask_best.pt \
    --flow_ckpt checkpoints_v2/flow/flow_best.pt \
    --low_snr_prob 0.15 \
    --low_snr_range=-10,1 \
    --interferer_count_probs 0.3,0.4,0.3 \
    --max_steps 8000 \
    --lr 2e-5 \
    --stage flow_finetune_onsetfix \
    --prefix flow_ft_onsetfix \
    $RESUME_ARG

echo ""
echo "Done. Check the gate FIRST (cheap, ~25min) before spending the full"
echo "battery on this candidate -- if it still gates, the rest is moot."
