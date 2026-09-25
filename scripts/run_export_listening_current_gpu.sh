#!/bin/bash
#SBATCH --partition=gpupart_p100
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=export_listening_current
#SBATCH --output=scripts/logs/export_listening_current_%j.out

# Mask2Flow-TSE -- refresh the by-ear check for the CURRENT system
# (timeline entry 31). The existing outputs/listening_samples/ was exported
# 2026-09-04 and documents a pipeline that no longer exists: since then Stage 1
# was fine-tuned and promoted, Stage 2 was adapted to it and promoted, and
# cfg_scale moved 1.5 -> 2.5. Results section 5.8's qualitative verification
# therefore describes a superseded system.
#
# THE OLD DIRECTORY IS NOT OVERWRITTEN. This writes to
# outputs/listening_samples_current/, and uses the SAME default seed, so the
# trained-SNR draws are the SAME underlying mixtures as the 2026-09-04 export.
# sample_00 in the old directory and sample_00 in the new one are the same
# audio through two different systems -- a direct A/B by ear.
#
# THREE QUESTIONS THIS ANSWERS, which metrics cannot:
#  1. Does the low-SNR improvement actually sound real? Accuracy went 61.3% ->
#     80.3% and SI-SDR -15.86 -> +6.51dB in a regime where extraction used to
#     be worse than doing nothing. Nobody has listened to it.
#  2. Are the extra mel-catastrophic cases genuinely benign? The in-domain rate
#     rose 3.7% -> 10.2% across the campaign, and the argument that this is a
#     metric artifact rests on SI-SDR severity (severe regressions only
#     2.1 -> 2.7%) plus 64% of those cases being audio-unharmed. That argument
#     justified overriding a pre-registered promotion bar (entry 25), so it is
#     worth confirming by ear rather than by inference alone.
#  3. Did the stronger guidance (cfg 2.5) introduce audible artifacts? Higher
#     CFG was the mechanism behind the original t~0 amplification (entry 4).
#
# Small job: 4 samples x 4 conditions. Minutes once models load.
#
# Submit:  sbatch scripts/run_export_listening_current_gpu.sh

set -eo pipefail
# NOTE: deliberately NOT using `set -u` -- conda activation references unset vars.

echo "=== DIAGNOSTIC ==="
nvidia-smi || echo "[Diag] nvidia-smi failed -- no GPU visible."
source /home/mtech1/25CS60R85/miniconda3/etc/profile.d/conda.sh
conda activate mask2flow
echo "which python3: $(which python3)"
echo "=== END DIAGNOSTIC ==="
echo ""

CKPTS="--mask_ckpt checkpoints_v2/masking/mask_best.pt \
    --flow_ckpt checkpoints_v2/flow/flow_best.pt \
    --projection_ckpt checkpoints_speaker_encoder/projection_latest.pt"

echo "=== [1/2] trained SNR [1,10] dB, 2/3/4 speakers -- same draws as the 2026-09-04 export ==="
python3 -u eval/export_listening_samples.py \
    $CKPTS \
    --n_interferers 1,2,3 \
    --n_samples_per_condition 4 \
    --output_dir outputs/listening_samples_current/trained_snr

echo ""
echo "=== [2/2] low SNR [-10,1) dB, 2 speakers -- the regime the campaign was about ==="
python3 -u eval/export_listening_samples.py \
    $CKPTS \
    --n_interferers 1 \
    --snr_min -10 --snr_max 1 \
    --n_samples_per_condition 4 \
    --output_dir outputs/listening_samples_current/low_snr

echo ""
echo "======================================================================"
echo "  Browse outputs/listening_samples_current/ on the Z:\ mount and play"
echo "  the .wav files directly; read the README.txt each run writes first."
echo "  Listen in this order per sample: mixture -> extracted -> target_clean."
echo "    - low_snr/: does 'extracted' sound like the target speaker at all?"
echo "      This is the regime where the old system was worse than doing nothing."
echo "    - trained_snr/: compare each sample against the SAME sample_NN in"
echo "      outputs/listening_samples/ (the 2026-09-04 export). Same audio,"
echo "      old system vs new."
echo "    - target_clean.wav is the ground truth through the SAME vocoder, so"
echo "      anything wrong with IT is a vocoder limit, not an extraction fault."
echo "======================================================================"
