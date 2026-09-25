---
name: mask2flow-tse-fixed-bugs
description: "Bugs found and fixed in the Mask2Flow-TSE project — don't re-diagnose these"
metadata: 
  node_type: memory
  type: project
  originSessionId: 28e94c75-57ba-46d9-9dad-9d9b2db586de
  modified: 2026-09-09T09:38:21.178Z
---

Part of [[mask2flow-tse-overview]]. Bugs already found and fixed in this project, most recent/relevant
first. Should not need revisiting, but the user has asked (as of 2026-08-26) to re-confirm the fixes
are actually present on disk before continuing, since this is a fresh session.

1. **MAJOR — padding-dilution bug (fixed):** cfg.audio.segment_length=10s pads every short utterance
   with near-silence (test-clean averages only 7.4s; 37.7% mean padding fraction across the corpus).
   This silence was never masked out of any loss/eval computation, inflating every MSE-based
   "improvement %" reported project-wide.
   **Fix:** added `target_valid_frames` tracking through data/librispeech.py -> data/mel.py's
   `make_frame_mask()`/`masked_mse()` -> models/flow.py, training/train_mask.py, training/train_flow.py,
   eval/results_stage2.py — all now mask padding out of loss/eval. Also added
   `trim_trailing_silence()` to data/augment.py (handles both hard-zero and vocoder-noise-floor
   padding) for inference/speaker_similarity.py.

2. **Unseeded SpeakerEncoder projection bug (fixed):** the encoder's 768->512 projection layer was
   never included in any optimizer and was random on every script invocation.
   **Fix:** trained it properly via training/train_speaker_encoder.py (30000 steps), added a
   `--projection_ckpt` flag everywhere the encoder is constructed.
   **How to apply:** ALWAYS pass `--projection_ckpt checkpoints_speaker_encoder/projection_latest.pt`
   when running eval/inference scripts, or the fallback is an untrained random projection.

3. Earlier bugs (BatchNorm EMA buffers never loaded, checkpoint path collisions, RIR conv speed,
   determinism, etc.) — all fixed; see git history / code comments if detail is ever needed.

4. **NOT A BUG (investigated 2026-09-09, don't re-chase) — WavLM `pos_conv_embed` "newly
   initialized" warning.** `models/speaker_encoder.py`'s `WavLMModel.from_pretrained()` always
   prints `Some weights ... were not initialized ... newly initialized:
   ['wavlm.encoder.pos_conv_embed.conv.parametrizations.weight.original0'/'original1']` — reads
   like the positional-conv layer is randomly reinitialized every construction (would be a real,
   serious determinism bug if true, since it feeds the shared speaker embedding `d`). Directly
   verified FALSE: on `torch==2.12.1+cu126` / `transformers==4.44.2`, the live model's
   `original0`/`original1` are bit-identical across independently-seeded constructions (0/248
   tensors differ) AND match the raw checkpoint's legacy `weight_g`/`weight_v` exactly (max abs
   diff 0.0, `scripts/verify_wavlm_posconv_values.py`). The warning is a stale/cosmetic
   false-positive in this transformers version — its missing-key bookkeeping doesn't know about
   the internal rename path that actually succeeds. **Nothing to fix; ignore this warning
   whenever it appears in any log.** (Originally triggered by re-examining `terminal.txt`, an old
   2026-08-21 log from the seed-sensitivity test that led to bug #2 above — that swing was bug #2,
   not this warning.) Diagnostic scripts kept at `scripts/diagnose_wavlm_posconv.py` /
   `scripts/verify_wavlm_posconv_values.py` for reference if a future library upgrade changes this.
