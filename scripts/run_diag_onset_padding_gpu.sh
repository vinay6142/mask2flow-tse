#!/bin/bash
#SBATCH --partition=gpupart_p100
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --job-name=diag_onset_pad
#SBATCH --output=scripts/logs/diag_onset_pad_%j.out

# Mask2Flow-TSE -- onset-damping diagnostic (2026-09-14).
#
# WHY: listening to outputs/listening_samples_current, the 4-speaker samples
# sounded "suppressed ... damped especially at the start" (3-speaker
# sample_03 too). Frame-level measurement confirmed it and, importantly,
# localized it to POSITION 0 rather than to speech onsets:
#     first burst (at frame 0)  -4.08 dB median, worst -16 dB
#     interior speech onsets    +0.29 dB
#     sustained speech          +0.01 dB
# Stage 1 is flat (~0 dB) -- only Stage 2 shows it. No existing metric
# catches this: it is a fraction of a second in a 10s utterance, so it
# barely moves mel-MSE or SI-SDR, but it is clearly audible.
#
# HYPOTHESIS: Stage 2's DiT uses RoPE (relative) attention, so frame 0 is
# the only frame with no left context, and Stage 2 GENERATES the mel from
# noise -- Stage 1 only scales the mixture, so it inherits a plausible level
# even where its mask is wrong. If that is the cause, giving Stage 2 a
# silent lead-in and discarding it afterwards should fix the audible part
# at zero training cost.
#
# RUNS 11174 + 11175 SETTLED THE PADDING QUESTION: NO.
#   Padding roughly halves MILD onset damping, but on the samples actually
#   listened to it leaves the severe cases unchanged or worse (3spk sample_03
#   -30.72 -> -32.04dB; 4spk sample_03 -35.47 -> -35.99dB). The position-0
#   hypothesis is rejected as the cause of the audible defect. Splicing was
#   built and works as designed, but has nothing left to fix.
#
# WHAT THE DEFECT ACTUALLY IS: a GATE, not damping. In affected samples the
# output sits on a flat ~-60dB floor for the first ~0.5s, independent of what
# the target is doing, then switches on abruptly and tracks correctly. Best
# lag is 0 frames, so it is not misalignment, and Stage 1 is clean throughout.
# It is not a content effect either: 3speakers/sample_01 and 4speakers/sample_01
# share the SAME target utterance, and the 3-speaker case starts perfectly
# (-0.6dB) while the 4-speaker case is gated (-34.5dB).
#
# AND IT IS A REGRESSION. Re-measuring the surviving 2026-09-04 export (old
# checkpoints, cfg_scale 1.5) with the identical method, same seed and sample
# indices:
#     median onset damping   -0.31dB (old)  vs  -7.82dB (current)
#     samples below -10dB      0/12         vs     5/12
#     samples below -20dB      0/12         vs     3/12
# The gate did not exist before the low-SNR campaign (docs 5.10). Every metric
# in that campaign's battery scored the change as an improvement.
#
# THIS RUN decomposes the cause. Three candidates: new Stage 1 (promoted
# 2026-09-12), new Stage 2 (2026-09-13), and cfg_scale 1.5 -> 2.5 (2026-09-13).
# Guidance is tested first because it is free to change and is the prime
# suspect: this project already documented (see FlowMatchingModule.inference)
# that ||v_cond - v_uncond|| at t=0 runs 5-40x larger than at later steps, and
# cfg_scale multiplies exactly that into an oversized first Euler step the
# remaining steps cannot correct -- which is what a hard output floor looks
# like. Raising 1.5 -> 2.5 amplified it by 1.67x.
#
# Also tested: cfg_warmup_steps > 0, the mitigation already built into the
# model for this exact mechanism (first N steps at cfg_scale=1.0). It is
# currently passed by NO inference path -- only full_eval.py exposes the flag,
# defaulting to 0 -- so it has never been active in any reported number.
#
# READ THE GATED-SAMPLES COUNT, NOT THE MEDIAN. The defect is a gate: one
# sample at -35dB is plainly audible while a 1dB median shift is not.
# Reference: 0/12 gated in the old export, 5/12 in the current one.
#
# DECISION RULE: a candidate must clear the gate (target 0/24) AND hold
# SI-SDR and mel MSE. If lowering cfg_scale clears it, that trades against the
# accuracy that motivated 2.5 in the first place (corpus-wide 85.2 -> 86.2%,
# docs timeline 27-29), so the trade has to be re-measured on the full battery
# with run_eval_candidate_gpu.sh -- not decided from this diagnostic alone.
# If cfg_warmup_steps clears it at cfg 2.5, that is the better outcome: it
# targets the t=0 step specifically and leaves the operating point intact.
#
# Diagnostic only: writes no checkpoint, edits no config.
#
# Submit: sbatch scripts/run_diag_onset_padding_gpu.sh

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

# --seed and --snr_* are deliberately NOT passed: the script's defaults are
# now export_listening_samples.py's own (seed 123, cfg.data SNR [1,10]), so
# sample_00..03 below are the SAME draws as
# outputs/listening_samples_current/trained_snr/ -- the ones actually listened
# to. Samples 04-07 are unheard, and are there for statistics.
#
# Pad and splice lengths are in MEL FRAMES; cfg.mel.hop_length is 160 at
# 16kHz, i.e. 100 frames/s, so pad 25 = 0.25s of silent lead-in and
# splice_frames 100 = the first 1.0s taken from the padded output.
python3 -u eval/diag_onset_padding.py \
    --n_samples 8 --n_interferers 1,2,3 \
    --pads 0 --splice_frames 0 \
    --cfg_variants 1.5:0,2.0:0,2.5:0,2.5:1,2.5:2 \
    --output outputs/results/diag_onset_cfg.jsonl

echo ""
echo "Done. cfg2.5w0 is the DEPLOYED setting -- it is the row to beat."
echo "If a variant clears the gate, it still has to survive the full battery"
echo "(scripts/run_eval_candidate_gpu.sh) before anything is adopted, and a"
echo "re-export + listening pass before it is believed -- the whole point of"
echo "this investigation is that the averaged metrics cannot see this defect."
