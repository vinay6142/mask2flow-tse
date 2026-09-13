#!/bin/bash
#SBATCH --partition=gpupart_p100
#SBATCH --gres=gpu:1
#SBATCH --time=47:59:00
#SBATCH --job-name=finetune_lowsnr25
#SBATCH --output=scripts/logs/finetune_flow_lowsnr25_%j.out

# Mask2Flow-TSE -- SECOND point on the low-SNR trade-off curve: a GENTLER
# 25% curriculum, versus the 50% run (job 10991) already completed.
#
# WHY: the 50% run closed the SNR gap decisively but at a real, consistent
# cost across every normal-SNR metric --
#
#   metric                          flow_best.pt   50% low-SNR FT
#   in-domain low-SNR accuracy        61.3%          71.3%     <- target, fixed
#   in-domain low-SNR mel S2vsS1     -45.1%         +50.1%     <- target, fixed
#   corpus-wide accuracy (n=400)      86.5%          82.0%     <- cost
#   in-domain n=2620 catastrophic      3.7%          15.5%     <- cost (4.2x)
#   2/3/4-speaker accuracy      86.7/81.9/76.7  81.4/77.3/73.3 <- cost
#
# Per-SNR-bucket catastrophic rates show genuine capacity reallocation, not
# noise: <0dB improved 45.0%->31.9% while [5,7)dB degraded 0.8%->4.7%. The
# open question is whether that trade is inherent or just an over-aggressive
# curriculum -- this run answers it by halving the low-SNR weight.
#
# IMPORTANT: starts from checkpoints_v2/flow/flow_best.pt -- the SAME starting
# point as the 50% run, NOT from the 50% result. Only --low_snr_prob differs
# between the two experiments, so they are directly comparable points on one
# trade-off curve rather than a sequential chain.
#
# ~25k steps, ~13h on a P100 (the 50% run took 12.98h). Auto-resumes from
# flow_ft_lowsnr25_latest.pt, so resubmitting after a timeout is safe.
# flow_best.pt is NOT touched -- writes to checkpoints_v2/flow_finetune_lowsnr25/.
#
# Submit:  sbatch scripts/run_finetune_flow_lowsnr25_gpu.sh

set -eo pipefail
# NOTE: deliberately NOT using `set -u` -- conda activation references unset vars.

echo "=== DIAGNOSTIC ==="
nvidia-smi || echo "[Diag] nvidia-smi failed -- no GPU visible to this job."
source /home/mtech1/25CS60R85/miniconda3/etc/profile.d/conda.sh
conda activate mask2flow
echo "which python3: $(which python3)"
python3 -c "import torch; print('torch.cuda.is_available():', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
echo "=== END DIAGNOSTIC ==="
echo ""

RESUME_ARG=""
LATEST="checkpoints_v2/flow_finetune_lowsnr25/flow_ft_lowsnr25_latest.pt"
if [ -f "$LATEST" ]; then
    echo "=== Found $LATEST -- resuming this fine-tune from it ==="
    RESUME_ARG="--resume $LATEST"
else
    echo "=== No existing checkpoint -- starting fresh from --flow_ckpt ==="
fi
echo ""

# NOTE: --low_snr_range MUST use the equals form. Space-separated fails,
# because argparse reads the leading '-' as the start of another option name
# (it only exempts plain negative numbers like -10; the comma breaks that
# match). This exact mistake killed job 10990 at arg parsing.
python3 -u training/finetune_flow_hard_lowsnr.py \
    --config configs/default_v2.yaml \
    --mask_ckpt checkpoints_v2/masking/mask_best.pt \
    --flow_ckpt checkpoints_v2/flow/flow_best.pt \
    --low_snr_prob 0.25 \
    --low_snr_range=-10,1 \
    --interferer_count_probs 0.5,0.3,0.2 \
    --max_steps 25000 \
    --stage flow_finetune_lowsnr25 \
    --prefix flow_ft_lowsnr25 \
    $RESUME_ARG

echo ""
echo "=== Done. Evaluate with the generic candidate evaluator: ==="
echo "  sbatch scripts/run_eval_candidate_gpu.sh \\"
echo "      checkpoints_v2/flow_finetune_lowsnr25/flow_ft_lowsnr25_final.pt lowsnr25"
