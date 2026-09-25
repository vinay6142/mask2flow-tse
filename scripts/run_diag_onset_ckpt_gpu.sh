#!/bin/bash
#SBATCH --partition=gpupart_p100
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --job-name=diag_onset_ckpt
#SBATCH --output=scripts/logs/diag_onset_ckpt_%j.out

# Mask2Flow-TSE -- which promotion introduced the onset gate? (2026-09-14)
#
# BACKGROUND. A listening pass on the current pipeline reported the 4-speaker
# extractions "damped especially at the start". Measurement showed it is not
# damping but a GATE: the output sits on a flat ~-60dB floor for the first
# ~0.5s regardless of the target, then switches on and tracks correctly (lag
# 0 frames, Stage 1 clean throughout). It is not content-driven either --
# 3speakers/sample_01 and 4speakers/sample_01 share the same target utterance,
# and only the 4-speaker case gates.
#
# It is a REGRESSION. The surviving 2026-09-04 export (old checkpoints,
# cfg 1.5), measured identically at the same seed and indices: 0/12 gated,
# median onset -0.31dB. Current: 5/12 and -7.82dB.
#
# RULED OUT SO FAR:
#   - position-0 / missing left context (jobs 11174, 11175): silent lead-in
#     helps mild cases, leaves severe ones unchanged or worse.
#   - the operating point (job 11176): sweeping cfg_scale 1.5/2.0/2.5 and
#     cfg_warmup_steps 0/1/2 moves the gate count only 6..7 of 24, and the
#     worst sample by 1.4dB out of 35. Guidance is worth about a decibel here,
#     not thirty.
#
# So the cause is in the weights promoted during the low-SNR campaign:
# Stage 1 on 2026-09-12, Stage 2 (adapted to it) on 2026-09-13. This run is
# the 2x2 that separates them. Both pre-promotion backups still exist, so all
# four cells are runnable:
#     A  old S1 + old S2   -- must reproduce ~0 gated, else the method is wrong
#     B  new S1 + old S2   -- Stage 1 alone
#     C  old S1 + new S2   -- Stage 2 alone
#     D  new S1 + new S2   -- the deployed pipeline (reference: 7/24)
#
# Every cell runs at the DEPLOYED operating point (cfg_scale 2.5, no warmup)
# so the checkpoints are the only thing that varies. Cell A doubles as the
# method's control: if it does not come back near-zero, the comparison against
# the 2026-09-04 export is not valid and nothing else here can be trusted.
#
# WHY IT MATTERS BEYOND THE AUDIO: pairing C against D also says whether
# Stage 2's adaptation FIXED a gate that the new Stage 1 introduced, or
# CAUSED it. That bears directly on docs 5.10, which promoted both on metrics
# that cannot see this defect.
#
# Diagnostic only: writes no checkpoint, edits no config.
#
# Submit: sbatch scripts/run_diag_onset_ckpt_gpu.sh

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

S1_OLD=checkpoints_v2/masking/mask_best_prelowsnr_backup.pt
S1_NEW=checkpoints_v2/masking/mask_best.pt
S2_OLD=checkpoints_v2/flow/flow_best_preadaptnewmask_backup.pt
S2_NEW=checkpoints_v2/flow/flow_best.pt

for f in "$S1_OLD" "$S1_NEW" "$S2_OLD" "$S2_NEW"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: checkpoint not found: $f"
        exit 1
    fi
done

run_cell () {
    local tag="$1" mask="$2" flow="$3" label="$4"
    echo ""
    echo "######################################################################"
    echo "# CELL $tag -- $label"
    echo "#   Stage 1: $mask"
    echo "#   Stage 2: $flow"
    echo "######################################################################"
    python3 -u eval/diag_onset_padding.py \
        --mask_ckpt "$mask" --flow_ckpt "$flow" \
        --n_samples 8 --n_interferers 1,2,3 \
        --pads 0 --splice_frames 0 \
        --cfg_variants 2.5:0 \
        --output outputs/results/diag_onset_ckpt_${tag}.jsonl
}

run_cell A "$S1_OLD" "$S2_OLD" "old S1 + old S2  (CONTROL: expect ~0 gated)"
run_cell B "$S1_NEW" "$S2_OLD" "new S1 + old S2  (Stage 1 alone)"
run_cell C "$S1_OLD" "$S2_NEW" "old S1 + new S2  (Stage 2 alone)"
run_cell D "$S1_NEW" "$S2_NEW" "new S1 + new S2  (DEPLOYED -- reference 7/24)"

echo ""
echo "======================================================================"
echo " Read the 'gated samples (onset < -10dB)' line of each cell."
echo " A must be near 0/24 or the method is invalid and B/C/D mean nothing."
echo " Whichever of B or C carries the gate is the promotion to revisit."
echo " If BOTH are clean and only D gates, the two new stages interact and"
echo " neither is individually at fault -- a different and harder finding."
echo "======================================================================"
