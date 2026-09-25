---
name: mask2flow-tse-next-steps
description: Where Mask2Flow-TSE stands as of 2026-09-20 and what Phase Two does next — Phase One is complete and reported
metadata: 
  node_type: memory
  type: project
  originSessionId: 92e3d418-e0ac-4d2e-962a-e7f2f76ff0a1
  modified: 2026-09-20T13:05:18.883Z
---

Part of [[mask2flow-tse-overview]]. **Phase One is COMPLETE and reported** (2026-09-20). Full
chronology: [[mask2flow-tse-master-timeline]]. Current numbers: [[mask2flow-tse-eval-results]].

> This file previously said "all items resolved as of 2026-08-30", which went stale through the
> entire low-SNR campaign. Rewritten 2026-09-20.

## State of the system
Deployed = promoted Stage 1 (low-SNR fine-tune) + adapted Stage 2 + `cfg_scale: 2.5` = **job 11112**.
Corpus-wide 86.2%, 2/3/4spk 85.7/80.0/77.0, low-SNR 80.3%, in-domain SI-SDR vs mixture +1.75dB,
Libri2Mix +3.39dB. Config is clean: `cfg_warmup_steps: 0`, `onset_pad_frames: 0`, `solver: euler`,
`reference_length: 3.0` — every opt-in experiment left inert by default.

## No open experiments
Tier 0 is exhausted — five consecutive inference-side experiments rejected, which is exactly what
the ceiling analysis predicts for a saturated operating point. **Do not propose further
inference-time tweaks** (guidance, solver, steps, enrollment window, projection swap) without new
evidence; they have been measured.

## Phase Two, in dependency order
1. **Stage 1 retrain on the expanded data (~47h).** Training data is now ~600h / ~1,417 speakers,
   up from ~100h / 251. The oracle says ALL remaining extraction headroom (~4.3pp) is Stage 1.
2. **Stage 2 adaptation (~13h) — MANDATORY, not optional.** The log_gain result proved a better
   Stage 1 alone makes the pipeline WORSE if Stage 2 is not co-adapted to its output.
3. **Measurement chain (~1-2d).** The larger pool at ~9.7pp: the from-scratch vocoder and the frozen
   speaker encoder bound every number in the project. Note this raises the ceiling AND the system
   number together — legitimate to report, but not the same claim as a better extractor.
4. **Multi-utterance enrollment (hours).** Phase One rejected a longer window on ONE utterance
   (ref5, job 11216) — pooling d-vectors across SEVERAL utterances adds real speech instead. Untested,
   inference-only.
5. **Onset gate.** Two mitigations failed; the only remaining class of fix consistent with an
   interaction defect is co-adapting both stages or weighting the loss toward the failing frames.

## Blocked / carried
- **`train-clean-360` extraction is BLOCKED by an NFS storage fault** (jobs 11228/11229: EIO +
  partial writes with 450GB+ free). The tarball is downloaded and sha256-verified at
  `data/raw/train-clean-360.tar.gz` — **do not delete it**; extraction is ~20 min once the
  fileserver is healthy. It would add 360h on top of 600h, a smaller increment than the first
  expansion, so it is **not a blocker for step 1**. Worth reporting to HPC admins.
- Before any Stage 1 retrain: **grep the training log for
  `[Dataset] Warning: ... not found, skipping`** — `data/librispeech.py:184` skips a missing split
  silently, and `train_splits` still lists train-clean-360.

## Immediate next command (when ready)
Set up and launch the Stage 1 retrain on the 600h. Not yet built as of 2026-09-20 — the existing
`training/train_mask.py` handles it, but the launcher and the promotion/eval plan for Phase Two have
not been written.
