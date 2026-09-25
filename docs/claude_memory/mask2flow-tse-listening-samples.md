---
name: mask2flow-tse-listening-samples
description: "Manual/audio verification tool for Mask2Flow-TSE, plus a real trailing-silence bug found via user listening"
metadata: 
  node_type: memory
  type: project
  originSessionId: 7e1688b4-2460-4349-9353-8129f100dee7
  modified: 2026-09-09T11:11:49.493Z
---

Part of [[mask2flow-tse-overview]]. On 2026-09-05, user reported listening-based observations the
automated metrics hadn't surfaced: (1) vocoder sometimes sounds "metallic" even when words are
correct; (2) low speaker-similarity scores despite a subjectively-correct-sounding voice, traced BY
THE USER to trailing silence in the reference clip.

**Confirmed real, via code inspection (not just plausible):** `trim_trailing_silence()` already
exists in `data/augment.py` specifically for this exact problem (part of the original padding-
dilution bug fix -- see [[mask2flow-tse-fixed-bugs]]) and IS used in
`inference/speaker_similarity.py` / `eval/results_stage2.py` -- but was MISSING from
`eval/eval_multi_speaker.py` (my own script, built this session) and is ALSO missing from
`eval/verify_eer.py` and `eval/eval_libri2mix.py` (pre-existing scripts, not mine). All three build
the reference clip via `segment_waveform(..., ref_seg_len=3.0, random_start=True)`, which zero-pads
any reference utterance shorter than 3.0s with no trimming afterward -- diluting the mean-pooled
WavLM embedding.

**Fixed:** `eval/eval_multi_speaker.py` now calls `trim_trailing_silence(ref_wav, sr)` right after
segmenting the reference clip, matching the established pattern. Syntax-checked, not yet re-run
against this fix specifically (the 2026-09-04 refresh numbers in [[mask2flow-tse-multi-speaker-test]]
PREDATE this fix).

**Important direction-of-bias note:** a diluted reference embedding makes genuine-vs-impostor
separability HARDER (blurrier reference = less discriminative), so this bug's effect is to
UNDERSTATE true accuracy, not inflate it. This means every accuracy/EER number reported in this
project so far (86.2%/86.3%/87.0% at 2 speakers, 76.7%/81.9% at 3, 72.6%/76.7% at 4, Libri2Mix's
71.8%/86.2%/90.8% mixture/extraction/ceiling, etc.) is a plausible LOWER BOUND on the real number, not
an inflated one -- the actual system likely performs at least as well as reported, probably slightly
better. `verify_eer.py` and `eval_libri2mix.py` still have this gap; NOT fixed there yet (would need
the same one-line `trim_trailing_silence()` addition to each script's reference-clip handling).
**Not yet done: decide whether to fix + rerun those two and refresh the headline again** -- a real,
scoped follow-up, not urgent (numbers are a lower bound either way, not wrong in the alarming
direction), but worth doing before finalizing thesis numbers if the user wants the tightest-possible
citation.

**Built: `eval/export_listening_samples.py`** (new tool, mirrors `eval_multi_speaker.py`'s mixing/
inference pipeline but SAVES real .wav files instead of only computing metrics) +
`scripts/run_export_listening_samples_gpu.sh`. For each of 2/3/4 total speakers (--n_interferers
1,2,3), saves 4 samples (configurable) of: `mixture.wav`, `reference.wav` (trimmed), `target_clean.wav`
(ground truth vocoded from ITS OWN real mel -- the vocoder-artifact diagnostic: if this ALSO sounds
metallic, it's a vocoder training issue; if only `extracted.wav` sounds metallic, it's Stage 2's mel
prediction specifically), `stage1_only.wav`, `extracted.wav` (the real system output), plus
`info.json` per sample with every relevant metric so what's heard can be cross-referenced against the
numbers. A `README.txt` is written into the output dir itself explaining all of this (listening order,
the two specific diagnostic questions, metric definitions) so the user has a self-contained reference,
not something they need this session's memory to re-explain. Uses seed=123 (different from every
other eval's seed=42) so it's a fresh spot-check draw, not literally re-scoring already-measured
samples. Output lands at `outputs/listening_samples/` on the Z:\ mount -- browsable/playable directly
in Windows Explorer, no download step needed.

**RESOLVED 2026-09-05 -- user ran it and listened.** Verdict: "voices are extracted correctly and
seems good for now" -- confirmed correct by ear across the 2/3/4-speaker samples, consistent with the
quantitative accuracy numbers (87.0%/81.9%/76.7%). User explicitly said "let's not disturb it" --
i.e. do NOT chase the metallic-vocoder question further or touch the checkpoint/vocoder, since the
qualitative outcome was already positive. Documented as new `## 5.8 Qualitative verification` in
`docs/results_and_limitations.md` (method, the trailing-silence-bug tie-in, and the confirmed-correct
result) + an appendix repro command. This closes the loop opened by the user's original listening
complaints -- no further action on vocoder/extraction quality unless new issues surface.

**FULLY RESOLVED 2026-09-09 -- fix applied, rerun (job 10988), and folded into docs.** Applied
`trim_trailing_silence()` to `eval/verify_eer.py`'s `collect_embeddings()` (per-sample, since
trimming yields variable lengths -- covers both extraction conditioning and the stored reference
embedding in one fix) and `eval/eval_libri2mix.py`'s `process_one()`. Reran via
`scripts/rerun_verify_eer_and_libri2mix_gpu.sh` (job 10988, ~60min total on P100).

**Result: essentially NO change beyond sampling noise** -- contrary to the "should only go up"
prediction:
| Metric | Pre-fix | Post-fix |
|---|---|---|
| 2spk accuracy, corpus-wide (n=400, verify_eer) | 87.0% | 86.5% (-0.5pp) |
| Libri2Mix in-dist (snr>=1dB, n=2361) median mel S2vsS1 | 72.5% | 72.5% (unchanged) |
| Libri2Mix in-dist SI-SDR gain vs S1 | +2.51dB | +2.51dB (unchanged) |
| Libri2Mix in-dist catastrophic rate | 5.1% | 5.3% (+0.2pp) |

(Libri2Mix in-dist numbers recomputed from the fresh `eval_libri2mix_min.jsonl` via a one-off Node.js
snr_db>=1.0 filter script, run locally against the Z:\ mount -- no python3 available on this Windows
session, node was.) Two of four metrics identical to the decimal, other two within known noise band
at these sample sizes -- NOT evidence the fix hurt anything, just confirms EER/accuracy is noisy at
n=400/40-speakers. **Conclusion: the previously-reported "lower bound" numbers were not meaningfully
understated in practice** -- unlike the much bigger original padding-dilution bug (mixture/target
side, affected every sample), reference-clip dilution only mattered for a small enough minority of
genuinely-short clips that its aggregate effect is negligible. Folded into
`docs/results_and_limitations.md` as new `## 5.9`, plus updated the §5.8 caveat's forward-pointer
(no numbers in §5.2-5.7 touched, per this project's standing "never overwrite, add new sections"
discipline). This closes the LAST open item from the original three-item "what's next" list
(citations verified, consistency pass still optional/low-priority, this one now done too).

**REOPENED 2026-09-14 by a second listening pass on the CURRENT pipeline (post-§5.10 low-SNR
campaign; export job 11167 -> `outputs/listening_samples_current/`).** Speaker isolation still
confirmed by ear, but the user reported a NEW defect the 2026-09-05 pass didn't: 4-speaker
extractions "suppressed ... damped especially at the start" (sample_00/01/03), and 3-speaker
sample_03 the one bad case in its condition. Frame-energy measurement against `target_clean.wav`
reproduced it exactly: every by-ear-damped sample is 10-14dB low in its first second and recovers
after; every clean one is within ~1dB throughout.

**Localized to POSITION 0, not acoustic onset** (this is the key finding -- interior speech onsets
after >=0.2s silence are the same acoustic event and are FINE):
| region | level vs ground truth |
|---|---|
| first burst (begins at frame 0) | **-4.08dB** median, worst -16 |
| interior speech onsets | +0.29dB |
| sustained speech | +0.01dB |
Stage 1 is flat (~0dB) over the same frames -> defect is Stage 2's. Mechanism hypothesis: Stage 2's
DiT uses RoPE (relative) attention so frame 0 is the only frame with NO left context, and Stage 2
GENERATES the mel from noise while Stage 1 merely scales the mixture (so S1 inherits a plausible
level even where its mask is wrong).

**Why no metric caught it:** ~0.5s of a 10s utterance; mel-MSE and SI-SDR are utterance-averaged so
a 12dB hole in ~5% of frames moves them less than their own run-to-run spread, and speaker accuracy
is insensitive (identity carried by the unaffected 9.5s). This is the project's clearest argument
for keeping a listening check in the protocol.

**Test built, NOT yet run:** `eval/diag_onset_padding.py` + `scripts/run_diag_onset_padding_gpu.sh`
-- prepend P frames of silence (pad_value -11.5) to the Stage-1 mel before `flow_model.inference`,
drop them after, so real frame 0 gets left context. Runs pads 0,0,12,25,50 (pad0 TWICE under
different flow-noise seeds = visible noise floor) on the flagged samples (seed 42, idx 0-3 reproduce
them exactly) plus 4 unheard each at 2/3/4 speakers. **Pre-registered rule: adopt only if onset
damping is removed AND neither SI-SDR nor mel-MSE degrades** -- it would touch every inference path
(`eval/export_listening_samples.py`, `full_eval.py`, `eval_multi_speaker.py`, `eval_libri2mix.py`,
`inference/infer.py`) and need re-validation via `run_eval_candidate_gpu.sh` before restating any
documented number. Documented as timeline entry 31 + `docs/results_and_limitations.md` §5.11, with a
forward-pointer note added to §5.8 (no historical numbers altered).

**CORRECTION 2026-09-14 (same day) -- the position-0 hypothesis above was TESTED AND REJECTED, and
the real finding is bigger.** Jobs 11174/11175/11176:
- **Padding does not fix it.** Silent lead-in halves MILD onset damping but leaves severe cases
  unchanged/worse (3spk s03 -30.72 -> -32.04dB; 4spk s03 -35.47 -> -35.99dB). Job 11174 looked
  positive only because I ran it at seed 42 / SNR[-5,5] while the listened-to export uses
  export_listening_samples.py's OWN defaults (**seed 123, cfg.data SNR [1,10]**) -- different,
  harder draw. Procedural lesson: a diagnostic aimed at a listening report MUST use the export's
  seed+SNR or it measures a different phenomenon.
- **It is a GATE, not damping:** output pinned at a flat ~-60dB floor for the first ~0.5s regardless
  of the target, then switches on and tracks correctly. Best lag 0 frames (not misalignment), Stage 1
  clean. NOT content-driven: 3speakers/sample_01 and 4speakers/sample_01 share the SAME target
  utterance; 3spk starts at -0.6dB, 4spk gates at -34.5dB. Local SNR predicts only the mild part
  (-0.4dB when target dominates vs -7dB when buried <-15dB); gated samples have onset TMR ~= -1dB.
- **It is a REGRESSION introduced by the §5.10 low-SNR campaign.** The 2026-09-04 export survives at
  `outputs/listening_samples/` -- measured identically (same seed/indices): median onset **-0.31dB,
  0/12 gated**; current: **-7.82dB, 5/12 gated** (3/12 below -20dB). Every metric in that campaign's
  battery scored the promotions as improvements.
- **NOT the operating point** (job 11176): cfg_scale 1.5/2.0/2.5 x cfg_warmup_steps 0/1/2 moves the
  gate count only 6..7 of 24 and the worst sample 1.4dB out of 35.
- **Side finding worth its own battery run:** `cfg_warmup_steps=2` at deployed cfg 2.5 gives
  mel-MSE 3.083 -> 2.751 (-11%) and SI-SDR -8.95 -> -8.47dB, nothing worse. Never active in ANY
  reported number -- only full_eval.py exposes the flag (default 0); no other inference path passes it.

**NEXT: `scripts/run_diag_onset_ckpt_gpu.sh`** (written, not yet run) -- 2x2 over the surviving
backups `checkpoints_v2/masking/mask_best_prelowsnr_backup.pt` and
`checkpoints_v2/flow/flow_best_preadaptnewmask_backup.pt` vs the promoted ones, all at deployed
cfg 2.5. Cell A (old+old) is the method's control and MUST come back ~0/24. Documented as timeline
entry 32 and a rewritten §5.11 (entry 31 / the original §5.11 text stated the now-rejected
hypothesis; §5.11 was corrected in place since it was explicitly "under test", entry 31 left as the
chronological record with entry 32 as its correction).
