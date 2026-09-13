#!/bin/bash
#SBATCH --partition=gpupart_p100
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --job-name=stage1_oracle
#SBATCH --output=scripts/logs/stage1_oracle_%j.out

# Mask2Flow-TSE -- is low-SNR extraction capped by Stage 1's trained NETWORK,
# or by its mask FORMULATION? (docs/methodology_and_project_history.md, 18-19)
#
# Stage 1 multiplies log-mel values by a [0,1] mask (models/masking.py, as the
# paper specifies). With this project's log(mel + 1e-8), quiet bins are
# negative, so the mask can only leave them alone or make them LOUDER, and loud
# bins can only be pulled down to 0. eval_multi_speaker.py's --stage1_mode
# swaps Stage 1's output for a diagnostic oracle built from the clean target:
#   oracle_logmask -- best mask WITHIN the formulation, clip(Y/X, 0, 1)
#   oracle_energy  -- true energy deletion, min(X, Y) in log space (~ target)
# Network-mode references at low SNR already exist and are NOT recomputed:
#   eval_lowsnr_indomain_baseline.jsonl  (flow_best.pt)
#   eval_lowsnr_indomain_lowsnr_ft.jsonl (50% fine-tune)
#
# Submit:  sbatch scripts/run_eval_stage1_oracle_gpu.sh

set -eo pipefail
# NOTE: deliberately NOT using `set -u` -- conda activation references unset vars.

echo "=== DIAGNOSTIC ==="
nvidia-smi || echo "[Diag] nvidia-smi failed -- no GPU visible."
source /home/mtech1/25CS60R85/miniconda3/etc/profile.d/conda.sh
conda activate mask2flow
echo "which python3: $(which python3)"
echo "=== END DIAGNOSTIC ==="

BASE="checkpoints_v2/flow/flow_best.pt"
FT50="checkpoints_v2/flow_finetune_lowsnr/flow_ft_lowsnr_final.pt"
COMMON="--mask_ckpt checkpoints_v2/masking/mask_best.pt \
    --projection_ckpt checkpoints_speaker_encoder/projection_latest.pt \
    --n_samples 300 --resume"
LOWSNR="--snr_min -10 --snr_max 1"
OUT="outputs/results"

run() {   # run <label> <flow_ckpt> <n_interferers> <stage1_mode> <output> [extra args]
    local label="$1" ckpt="$2" n="$3" mode="$4" out="$5"; shift 5
    echo ""
    echo "=== $label ==="
    python3 -u eval/eval_multi_speaker.py $COMMON --flow_ckpt "$ckpt" \
        --n_interferers "$n" --stage1_mode "$mode" --output "$OUT/$out" "$@"
}

echo ""
echo "######## LOW SNR [-10, 1) dB -- the regime in question ########"
run "[1/9] DECISIVE: flow_best + oracle_logmask @ low SNR (network ref: eval_lowsnr_indomain_baseline.jsonl)" \
    "$BASE" 1 oracle_logmask eval_oracle_lowsnr_logmask_base.jsonl $LOWSNR
run "[2/9] flow_best + oracle_energy @ low SNR" \
    "$BASE" 1 oracle_energy eval_oracle_lowsnr_energy_base.jsonl $LOWSNR
run "[3/9] 50% fine-tune + oracle_logmask @ low SNR (network ref: eval_lowsnr_indomain_lowsnr_ft.jsonl)" \
    "$FT50" 1 oracle_logmask eval_oracle_lowsnr_logmask_ft50.jsonl $LOWSNR
run "[4/9] 50% fine-tune + oracle_energy @ low SNR" \
    "$FT50" 1 oracle_energy eval_oracle_lowsnr_energy_ft50.jsonl $LOWSNR

echo ""
echo "######## CONTROL: trained SNR range [1, 10] dB ########"
echo "# At high SNR few target bins should be out of the mask's reach, so the"
echo "# oracles should sit close to the network. A gap that opens only at low SNR"
echo "# is what ties the formulation limit to the SNR regime specifically."
run "[5/9] flow_best + network @ trained SNR (also the fresh 2-speaker baseline, trim fix applied)" \
    "$BASE" 1 network eval_multispeaker_2total_base_trimfix.jsonl
run "[6/9] flow_best + oracle_logmask @ trained SNR" \
    "$BASE" 1 oracle_logmask eval_oracle_insnr_logmask_base.jsonl
run "[7/9] flow_best + oracle_energy @ trained SNR" \
    "$BASE" 1 oracle_energy eval_oracle_insnr_energy_base.jsonl

echo ""
echo "######## CONFOUND FIX (timeline entry 18) ########"
echo "# The 2/3/4-speaker baselines (job 10890) predate the trim_trailing_silence"
echo "# fix; every candidate since was measured with it. Re-measure the current"
echo "# checkpoint with today's code so that comparison is like-for-like."
run "[8/9] flow_best + network, 3 total speakers" \
    "$BASE" 2 network eval_multispeaker_3total_base_trimfix.jsonl
run "[9/9] flow_best + network, 4 total speakers" \
    "$BASE" 3 network eval_multispeaker_4total_base_trimfix.jsonl

echo ""
echo "======================================================================"
echo "  Done. How to read it:"
echo "    oracle_logmask ~ network at low SNR  -> the formulation is the ceiling;"
echo "                                            retraining Stage 1 won't help"
echo "    oracle_logmask >> network at low SNR -> the trained network is the"
echo "                                            bottleneck; retrain Stage 1"
echo "    oracle_energy  >> oracle_logmask     -> prices a change of formulation"
echo "  Each summary also prints the formulation diagnostics: negative-bin share,"
echo "  target bins no [0,1] mask can reach, and the network's insert %."
echo "======================================================================"
