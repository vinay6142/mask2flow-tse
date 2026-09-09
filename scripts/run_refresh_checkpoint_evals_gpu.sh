#!/bin/bash
#SBATCH --partition=gpupart_p100
#SBATCH --gres=gpu:1
#SBATCH --time=47:59:00
#SBATCH --job-name=refresh_checkpoint_evals
#SBATCH --output=scripts/logs/refresh_checkpoint_evals_%j.out

# Mask2Flow-TSE -- refresh every headline number that's still measured
# against the PRE-promotion (hard-t0) checkpoint, now that
# checkpoints_v2/flow/flow_best.pt has been re-promoted to the
# hard-multispeaker fine-tune (2026-09-04 -- see
# mask2flow-tse-multi-speaker-test / mask2flow-tse-overview memory).
#
# Already fresh against the CURRENT flow_best.pt (no need to rerun):
#   - in-domain n=2620 full_eval.py (75.1% median mel, 3.7% catastrophic,
#     +1.71dB SI-SDR gain vs S1 -- from the promotion-decision run itself)
#   - 2-speaker and 4-speaker eval_multi_speaker.py (86.7% / 76.7% accuracy
#     -- from the same promotion-decision comparison, job 10887)
#
# STALE (measured against the hard-t0 checkpoint, this job refreshes them):
#   - eval/verify_eer.py corpus-wide speaker-verification accuracy (was 86.2%)
#   - 3-speaker eval_multi_speaker.py accuracy (was 76.7%, but that run
#     predates even the accuracy-metric addition on this checkpoint lineage
#     -- worth a clean re-measurement, not just an assumption it still holds)
#   - eval/eval_libri2mix.py cross-corpus validation (was 73.4% in-distribution)
#
# The canonical eval_multispeaker_{2,3,4}total.jsonl and eval_libri2mix_min.jsonl
# files hold PRE-promotion data -- deleted and regenerated fresh below (same
# one-time-reset pattern used in run_eval_multi_speaker_gpu.sh previously),
# so these filenames stay the definitive, current numbers matching what the
# docs/results_and_limitations.md appendix's repro commands reference.
#
# Submit:  sbatch scripts/run_refresh_checkpoint_evals_gpu.sh
# Each step is independently resumable (--resume) if this gets cut off --
# resubmitting this unchanged is always safe (the one-time deletes only run
# once per submission, matching the earlier established pattern).

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

echo "=== One-time reset: canonical multi-speaker + Libri2Mix outputs predate the current checkpoint ==="
rm -f outputs/results/eval_multispeaker_2total.jsonl \
      outputs/results/eval_multispeaker_3total.jsonl \
      outputs/results/eval_multispeaker_4total.jsonl \
      outputs/results/eval_libri2mix_min.jsonl
echo "removed (if present); regenerating all fresh below."
echo ""

CKPT_ARGS="--mask_ckpt checkpoints_v2/masking/mask_best.pt \
    --flow_ckpt checkpoints_v2/flow/flow_best.pt \
    --projection_ckpt checkpoints_speaker_encoder/projection_latest.pt"

echo "=== [1/5] eval_multi_speaker.py: n_interferers=1 (2 total speakers) ==="
python3 -u eval/eval_multi_speaker.py \
    --n_interferers 1 --n_samples 300 \
    $CKPT_ARGS \
    --output outputs/results/eval_multispeaker_2total.jsonl \
    --resume

echo ""
echo "=== [2/5] eval_multi_speaker.py: n_interferers=2 (3 total speakers) ==="
python3 -u eval/eval_multi_speaker.py \
    --n_interferers 2 --n_samples 300 \
    $CKPT_ARGS \
    --output outputs/results/eval_multispeaker_3total.jsonl \
    --resume

echo ""
echo "=== [3/5] eval_multi_speaker.py: n_interferers=3 (4 total speakers) ==="
python3 -u eval/eval_multi_speaker.py \
    --n_interferers 3 --n_samples 300 \
    $CKPT_ARGS \
    --output outputs/results/eval_multispeaker_4total.jsonl \
    --resume

echo ""
echo "=== [4/5] verify_eer.py: corpus-wide speaker-verification accuracy (n=400) ==="
python3 -u eval/verify_eer.py \
    $CKPT_ARGS \
    --seed 42 --n_steps 4 --max_samples 400

echo ""
echo "=== [5/5] eval_libri2mix.py: cross-corpus validation (n=6000) ==="
python3 -u eval/eval_libri2mix.py \
    $CKPT_ARGS \
    --librimix_dir data/raw/LibriMix_storage/Libri2Mix/wav16k/min \
    --librimix_input_csv data/raw/LibriMix/metadata/Libri2Mix_test_only/libri2mix_test-clean.csv \
    --librispeech_dir data/raw/LibriSpeech \
    --output outputs/results/eval_libri2mix_min.jsonl \
    --resume

echo ""
echo "=== All 5 refresh runs done -- see each run's own printed summary above. ==="
