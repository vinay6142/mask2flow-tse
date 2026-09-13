#!/bin/bash
#SBATCH --partition=gpupart_p100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=sweep_cfg_ext
#SBATCH --output=scripts/logs/sweep_cfg_ext_%j.out

# Mask2Flow-TSE -- extend the guidance sweep past 2.0 (timeline entry 27).
#
# Job 11110 found accuracy and AUC rising monotonically with cfg_scale in BOTH
# regimes, with 2.0 -- the largest value tested -- best everywhere:
#     cfg   trained acc / AUC      low-SNR acc / AUC
#     1.0     84.3% / 0.9236         78.3% / 0.8584
#     1.25    84.3% / 0.9261         78.3% / 0.8632
#     1.5     84.3% / 0.9272         78.7% / 0.8673   <- currently deployed
#     2.0     85.0% / 0.9310         79.6% / 0.8762
# The trend had not turned by the edge of the sweep, so the optimum may lie
# beyond it. This tests 2.5 and 3.0, plus cfg 2.0 with 8 steps in case the two
# knobs combine (8 steps at cfg 1.5 gave 79.4% low-SNR, similar to cfg 2.0's
# 79.6% but for twice the compute).
#
# WATCH the catastrophic rate, not just accuracy: high guidance was implicated
# in the original t~0 outlier investigation (entry 4). It was flat through 2.0
# (6.0-6.3% trained, 11.0% low), but that is exactly what would break first.
# SI-SDR also drifts slightly DOWN as guidance rises (+2.03dB at cfg 1.0 ->
# +1.90dB at 2.0, trained), so a large accuracy gain must be weighed against it.
#
# ~6 runs x n=300 x ~5min = ~30min.
#
# Submit:  sbatch scripts/run_sweep_cfg_extend_gpu.sh

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

for c in 2.5 3.0; do
    run "$c" 4 trained
    run "$c" 4 lowsnr
done
run 2.0 8 trained
run 2.0 8 lowsnr

echo ""
echo "======================================================================"
echo "  Pick the setting with the best accuracy/AUC in BOTH regimes whose"
echo "  catastrophic rate has not risen above ~7% (trained) / ~12% (low SNR)"
echo "  and whose SI-SDR has not fallen more than ~0.3dB from cfg 1.5."
echo "  Then RE-VALIDATE it on the full battery before changing the config"
echo "  default -- this sweep is only n=300 2-speaker:"
echo "    sbatch scripts/run_eval_candidate_gpu.sh \\"
echo "        checkpoints_v2/flow/flow_best.pt cfgNEW"
echo "  (after editing cfg_scale in configs/default_v2.yaml, which every eval"
echo "   script reads as its default)"
echo "======================================================================"
