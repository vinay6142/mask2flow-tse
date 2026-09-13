#!/bin/bash
#SBATCH --partition=gpupart_p100
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --job-name=eval_lowsnr_ft
#SBATCH --output=scripts/logs/eval_lowsnr_ft_%j.out

# Mask2Flow-TSE -- evaluate the low-SNR fine-tune (job 10991,
# checkpoints_v2/flow_finetune_lowsnr/flow_ft_lowsnr_final.pt, 25k steps,
# 12.98h) BEFORE any promotion decision. flow_best.pt is NOT touched here.
#
# Step 1 is the DECISIVE one and runs first on purpose: eval_libri2mix.py
# produces both the sub-1dB slice (the thing this fine-tune targeted, where
# Stage 2 previously made 94.8% of sub--5dB samples worse than Stage 1) and
# the >=1dB in-distribution slice (which must NOT regress) from the same
# 6000 records. If step 1 looks bad, cancel the job -- steps 2-4 are the
# standard "did we break anything else" re-validation that only matters if
# step 1 is promising.
#
# Every output goes to a NEW filename -- the canonical
# eval_libri2mix_min.jsonl / eval_multispeaker_*.jsonl files hold the CURRENT
# flow_best.pt numbers that docs/results_and_limitations.md cites, and must
# not be clobbered by a candidate that hasn't been promoted.
#
# Submit:  sbatch scripts/run_eval_lowsnr_ft_gpu.sh

set -eo pipefail
# NOTE: deliberately NOT using `set -u` -- conda activation references unset vars.

echo "=== DIAGNOSTIC ==="
nvidia-smi || echo "[Diag] nvidia-smi failed -- no GPU visible."
source /home/mtech1/25CS60R85/miniconda3/etc/profile.d/conda.sh
conda activate mask2flow
echo "which python3: $(which python3)"
echo "=== END DIAGNOSTIC ==="
echo ""

CKPT="checkpoints_v2/flow_finetune_lowsnr/flow_ft_lowsnr_final.pt"
BASE_ARGS="--mask_ckpt checkpoints_v2/masking/mask_best.pt \
    --flow_ckpt $CKPT \
    --projection_ckpt checkpoints_speaker_encoder/projection_latest.pt"

echo "=== Evaluating candidate: $CKPT ==="
echo ""

echo "=== [1/4] DECISIVE -- eval_libri2mix.py (n=6000, ~56min) ==="
echo "    Pre-fine-tune, for comparison:"
echo "      in-distribution (SNR>=1dB, n=2361): 72.5% median mel, 5.3% catastrophic, +2.51dB vs S1"
echo "      out-of-dist     (SNR<1dB,  n=3639): -19.0% median mel, 40.4% catastrophic, -1.50dB vs S1"
echo "    Target: OOD slice improves substantially, in-dist slice does NOT regress."
python3 -u eval/eval_libri2mix.py \
    $BASE_ARGS \
    --librimix_dir data/raw/LibriMix_storage/Libri2Mix/wav16k/min \
    --librimix_input_csv data/raw/LibriMix/metadata/Libri2Mix_test_only/libri2mix_test-clean.csv \
    --librispeech_dir data/raw/LibriSpeech \
    --output outputs/results/eval_libri2mix_min_lowsnr_ft.jsonl \
    --resume

echo ""
echo "=== [1b/4] CONTROLLED in-domain low-SNR test -- matched pair (~20min total) ==="
echo "    Libri2Mix confounds two variables at once: low SNR AND a different corpus/mixer."
echo "    eval_multi_speaker.py takes --snr_min/--snr_max, so the SAME corpus, mixer, speakers"
echo "    and seed can be run at low SNR against BOTH checkpoints -- isolating the SNR variable"
echo "    cleanly. No prior baseline exists for this, so the old checkpoint is measured too."
echo ""
echo "--- 1b(i) BASELINE: current flow_best.pt at [-10, 1) dB ---"
python3 -u eval/eval_multi_speaker.py \
    --n_interferers 1 --n_samples 300 \
    --snr_min -10 --snr_max 1 \
    --mask_ckpt checkpoints_v2/masking/mask_best.pt \
    --flow_ckpt checkpoints_v2/flow/flow_best.pt \
    --projection_ckpt checkpoints_speaker_encoder/projection_latest.pt \
    --output outputs/results/eval_lowsnr_indomain_baseline.jsonl \
    --resume

echo ""
echo "--- 1b(ii) CANDIDATE: low-SNR fine-tune at [-10, 1) dB (same seed/corpus/mixer) ---"
python3 -u eval/eval_multi_speaker.py \
    --n_interferers 1 --n_samples 300 \
    --snr_min -10 --snr_max 1 \
    $BASE_ARGS \
    --output outputs/results/eval_lowsnr_indomain_lowsnr_ft.jsonl \
    --resume

echo ""
echo "=== [2/4] in-domain n=2620 -- the untouched primary headline (~45min) ==="
echo "    Current flow_best.pt: +1.71dB SI-SDR gain vs S1 (primary), 75.1% median mel, 3.7% catastrophic"
echo "    This checks the fine-tune didn't buy low-SNR ability at the cost of the in-domain result."
python3 -u eval/full_eval.py \
    $BASE_ARGS \
    --output outputs/results/full_eval_audio_gpu_lowsnr_ft.jsonl \
    --resume

echo ""
echo "=== [3/4] corpus-wide speaker-verification accuracy, n=400 (~4min) ==="
echo "    Current flow_best.pt: 86.5% accuracy / 13.5% EER (mixture 71.0%, ceiling 90.8%)"
python3 -u eval/verify_eer.py \
    $BASE_ARGS \
    --seed 42 --n_steps 4 --max_samples 400

echo ""
echo "=== [4/4] speaker-count conditions, n=300 each ==="
echo "    Current flow_best.pt: 2spk 86.7%, 3spk 81.9%, 4spk 76.7%"
echo "    The speaker-count curriculum stayed ON during this fine-tune specifically"
echo "    so these should HOLD, not regress."
for N in 1 2 3; do
    TOTAL=$((N + 1))
    echo ""
    echo "--- n_interferers=$N ($TOTAL total speakers) ---"
    python3 -u eval/eval_multi_speaker.py \
        --n_interferers $N --n_samples 300 \
        $BASE_ARGS \
        --output outputs/results/eval_multispeaker_${TOTAL}total_lowsnr_ft.jsonl \
        --resume
done

echo ""
echo "======================================================================"
echo "  All four evaluations done. Decision checklist:"
echo "    (a) Did the Libri2Mix SNR<1dB slice improve substantially?  <- the point"
echo "    (b) Did the SNR>=1dB slice hold (was 72.5% / 5.3% / +2.51dB)?"
echo "    (c) Did in-domain n=2620 hold (was +1.71dB SI-SDR / 75.1% mel)?"
echo "    (d) Did accuracy hold (was 86.5% corpus-wide; 86.7/81.9/76.7 by speaker count)?"
echo "  Promote ONLY if the target improved and nothing else meaningfully regressed."
echo "======================================================================"
