#!/bin/bash
#SBATCH --partition=gpupart_p100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=rerun_eer_libri2mix
#SBATCH --output=scripts/logs/rerun_eer_libri2mix_%j.out

# Mask2Flow-TSE -- rerun ONLY eval/verify_eer.py and eval/eval_libri2mix.py
# against the current checkpoints_v2/flow/flow_best.pt, now that both
# scripts have the trim_trailing_silence() fix applied to their reference
# clip (2026-09-09 -- see mask2flow-tse-fixed-bugs / mask2flow-tse-listening-
# samples memory: this was previously fixed in eval_multi_speaker.py /
# results_stage2.py / export_listening_samples.py, but left open in these
# two). Every previously-reported accuracy number from these two scripts
# (87.0% corpus-wide, 72.5%/73.4% Libri2Mix in-distribution, etc.) was a
# documented LOWER BOUND, not wrong -- this closes that gap for the
# tightest-possible final numbers.
#
# Deliberately does NOT touch eval_multi_speaker.py -- it already had the
# fix applied before its last run (job 10890), so rerunning it here would
# just burn GPU time reproducing the same numbers. See
# scripts/run_refresh_checkpoint_evals_gpu.sh for the full 5-step refresh
# this was split out of.
#
# Submit:  sbatch scripts/rerun_verify_eer_and_libri2mix_gpu.sh

set -eo pipefail
# NOTE: deliberately NOT using `set -u` -- conda's activation shell
# functions reference variables that can be unset, and `-u` silently
# kills the job right at `conda activate` on many conda versions.

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

echo "=== [1/2] verify_eer.py: corpus-wide speaker-verification accuracy (n=400) ==="
echo "    (was 87.0% accuracy / 13.0% EER pre-fix -- a lower bound; expect equal or higher now)"
python3 -u eval/verify_eer.py \
    $CKPT_ARGS \
    --seed 42 --n_steps 4 --max_samples 400

echo ""
echo "=== One-time reset: eval_libri2mix_min.jsonl predates today's reference-clip fix ==="
echo "    (--resume would otherwise silently keep the STALE pre-fix rows)"
rm -f outputs/results/eval_libri2mix_min.jsonl
echo "removed (if present); regenerating fresh below."
echo ""

echo "=== [2/2] eval_libri2mix.py: cross-corpus validation (n=6000) ==="
echo "    (was 72.5% mel / 5.1% catastrophic / +2.51dB SI-SDR, in-distribution slice, pre-fix)"
python3 -u eval/eval_libri2mix.py \
    $CKPT_ARGS \
    --librimix_dir data/raw/LibriMix_storage/Libri2Mix/wav16k/min \
    --librimix_input_csv data/raw/LibriMix/metadata/Libri2Mix_test_only/libri2mix_test-clean.csv \
    --librispeech_dir data/raw/LibriSpeech \
    --output outputs/results/eval_libri2mix_min.jsonl \
    --resume

echo ""
echo "=== Both reruns done -- compare each run's printed summary above against the"
echo "=== pre-fix numbers noted inline, and against docs/results_and_limitations.md ==="
