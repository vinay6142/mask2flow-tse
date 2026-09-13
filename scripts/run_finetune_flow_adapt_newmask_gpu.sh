#!/bin/bash
#SBATCH --partition=gpupart_p100
#SBATCH --gres=gpu:1
#SBATCH --time=47:59:00
#SBATCH --job-name=flow_adapt_newmask
#SBATCH --output=scripts/logs/flow_adapt_newmask_%j.out

# Mask2Flow-TSE -- adapt Stage 2 to the NEW Stage 1 (timeline entries 22-23).
#
# The Stage-1 low-SNR fine-tune was promoted to checkpoints_v2/masking/mask_best.pt
# on 2026-09-12 and passed every pre-registered bar. But Stage 2 was trained on
# the OLD Stage 1's outputs, and it shows: at low SNR Stage 2 now SUBTRACTS from
# Stage 1 (-1.40 dB, median mel -6.3%). Stage 1 moved and Stage 2 did not. This
# run closes that gap by fine-tuning Stage 2 on the new Stage 1's outputs.
#
# Curriculum deliberately MATCHES the one Stage 1 was just trained on -- 50% of
# examples from [-10,1) dB, 2/3/4-speaker mix 0.5/0.3/0.2 -- so Stage 2 sees the
# input distribution Stage 1 is now trained to produce. lr 4e-5, the established
# Stage-2 fine-tune rate. ~25k steps, ~13h on a P100.
#
# NOTE the separate --stage/--prefix: reusing flow_finetune_lowsnr/ would make
# auto-resume pick up the COMPLETED job 10991 at step 25000 and exit immediately.
#
# flow_best.pt and mask_best.pt are NOT modified -- this writes to
# checkpoints_v2/flow_finetune_adapt_newmask/ and a promotion decision follows
# the usual battery.
#
# Reference numbers to beat/hold -- the CURRENT system is now
# (promoted Stage 1 + unchanged flow_best.pt), measured in job 11081:
#     low-SNR in-domain accuracy      69.7%   (do-nothing 64.0%)
#     low-SNR S2 vs S1                -1.40dB  <- the thing this run should fix
#     corpus-wide accuracy (n=400)    86.5%
#     in-domain n=2620 catastrophic    4.6%    SI-SDR vs S1 +1.54dB
#     2/3/4-speaker accuracy      84.9 / 80.0 / 75.4
#     Libri2Mix ALL SI-SDR vs mixture +1.91dB, OOD mel +11.2%, catastrophic 28.2%
#
# Submit:  sbatch scripts/run_finetune_flow_adapt_newmask_gpu.sh

set -eo pipefail
# NOTE: deliberately NOT using `set -u` -- conda activation references unset vars.

echo "=== DIAGNOSTIC ==="
nvidia-smi || echo "[Diag] nvidia-smi failed -- no GPU visible."
source /home/mtech1/25CS60R85/miniconda3/etc/profile.d/conda.sh
conda activate mask2flow
echo "which python3: $(which python3)"
python3 -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
echo "=== END DIAGNOSTIC ==="
echo ""

LATEST="checkpoints_v2/flow_finetune_adapt_newmask/flow_ft_adapt_latest.pt"
RESUME_ARG=""
if [ -f "$LATEST" ]; then
    echo "=== resuming from $LATEST ==="
    RESUME_ARG="--resume $LATEST"
else
    echo "=== starting fresh from --flow_ckpt ==="
fi
echo ""

# --low_snr_range MUST use '=': space-separated, argparse reads '-10,1' as an
# option name (this exact mistake killed job 10990).
python3 -u training/finetune_flow_hard_lowsnr.py \
    --config configs/default_v2.yaml \
    --mask_ckpt checkpoints_v2/masking/mask_best.pt \
    --flow_ckpt checkpoints_v2/flow/flow_best.pt \
    --low_snr_prob 0.5 \
    --low_snr_range=-10,1 \
    --interferer_count_probs 0.5,0.3,0.2 \
    --max_steps 25000 \
    --stage flow_finetune_adapt_newmask \
    --prefix flow_ft_adapt \
    $RESUME_ARG

echo ""
echo "=== Done. Evaluate (Stage 1 is the promoted default, so no 3rd argument): ==="
echo "  sbatch scripts/run_eval_candidate_gpu.sh \\"
echo "      checkpoints_v2/flow_finetune_adapt_newmask/flow_ft_adapt_final.pt adapt_newmask"
