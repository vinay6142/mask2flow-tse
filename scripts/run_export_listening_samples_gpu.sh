#!/bin/bash
#SBATCH --partition=gpupart_p100
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=export_listening_samples
#SBATCH --output=scripts/logs/export_listening_samples_%j.out

# Mask2Flow-TSE -- export real, listenable .wav files for 2/3/4-speaker
# mixtures so extraction quality can be judged by ear, not just by metric.
# See eval/export_listening_samples.py's docstring for the full rationale
# (built directly from user-reported listening observations: occasional
# metallic vocoder quality, and low speaker-similarity scores traced to
# untrimmed trailing silence in the reference clip -- both addressed here).
#
# Small job -- only 4 samples x 3 conditions = 12 samples total, should
# take a few minutes once the models are loaded. 1h budget is generous
# headroom, not an expectation it takes anywhere near that.
#
# Submit:  sbatch scripts/run_export_listening_samples_gpu.sh
# Output:  outputs/listening_samples/{2,3,4}speakers/sample_NN/*.wav
#          (mirrored to the Z:\ mount -- browse there directly in Windows
#          Explorer and play the .wav files locally; READ
#          outputs/listening_samples/README.txt FIRST, it explains what
#          each file is and exactly what to listen for)

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

echo "=== REAL JOB: export_listening_samples.py (2/3/4 speakers, 4 samples each) ==="
python3 -u eval/export_listening_samples.py \
    --mask_ckpt checkpoints_v2/masking/mask_best.pt \
    --flow_ckpt checkpoints_v2/flow/flow_best.pt \
    --projection_ckpt checkpoints_speaker_encoder/projection_latest.pt \
    --n_interferers 1,2,3 \
    --n_samples_per_condition 4 \
    --output_dir outputs/listening_samples
