---
name: mask2flow-tse-onset-gate
description: Onset-damping/gating defect in Mask2Flow-TSE — fix candidate flow_ft_onsetfix failed its pre-registered gate bar 2026-09-15; battery eval blocked by a GPU squatter on gnode2
metadata: 
  node_type: memory
  type: project
  originSessionId: 92e3d418-e0ac-4d2e-962a-e7f2f76ff0a1
  modified: 2026-09-17T14:30:21.866Z
---

Part of [[mask2flow-tse-overview]]. Line of work opened 2026-09-14, AFTER
[[mask2flow-tse-lowsnr-gap]] closed (log_gain rejected, job 11164).

**The defect:** speech ONSETS are damped — level vs ground truth over the first 0.5s of speech.
Found by ear (follows the listening pass in [[mask2flow-tse-listening-samples]]), then measured.
The `tail` measure (after 1.0s) is near-clean, so it is specifically an onset defect, not global
attenuation. Headline metric is **"gated samples" = onset < -10dB**, the count that tracks
audibility. Deployed system (promoted Stage 1 + adapted Stage 2 + cfg 2.5): **7/24 gated**
(2spk 1/8, 3spk 3/8, 4spk 3/8), median onset damping **-4.67dB**.

Diagnostics built 2026-09-14/15 (`scripts/run_diag_onset_padding_gpu.sh`,
`run_diag_onset_ckpt_gpu.sh`, `run_diag_gate_candidate_gpu.sh`; jobs 11171-11177). Note the
gate diag is DETERMINISTIC (Stage 2 Euler-steps from x_enh, no noise drawn) so differences are
exact, but n=8 per speaker-count condition — 7/24 vs 4/24 is three samples of evidence.

**ROOT CAUSE = AN INTERACTION BETWEEN THE TWO PROMOTIONS, not either one alone (job 11177,
4-cell ablation, 2026-09-14).** This is the branch the diagnostic's own footer called "a different
and harder finding".
| Cell | Stage 1 | Stage 2 | gated | onset ALL | 4spk gated |
|---|---|---|---|---|---|
| A | old (`mask_best_prelowsnr_backup`) | old (`flow_best_preadaptnewmask_backup`) | 1/24 | -1.00dB | 0/8 |
| B | **new** | old | 1/24 | -2.01dB | 0/8 |
| C | old | **new** | 2/24 | -1.95dB | 0/8 |
| D | **new** | **new** = DEPLOYED | **7/24** | **-4.67dB** | **3/8** |
Superadditive: additive prediction from B and C is ~2/24 and -2.96dB; actual 7/24 and -4.67dB.
4-speaker is starkest — A/B/C all 0/8, D 3/8. Same mechanism the log_gain rejection found from the
other direction ("pipeline quality is NOT a function of Stage 1 accuracy alone — the stages are
co-adapted", see [[mask2flow-tse-lowsnr-gap]]); two independent experiments converging.

**No free inference-side mitigation exists.** Padding sweep (job 11176, 5 variants) moved nothing:
6/24, 7/24, 7/24, 6/24, 6/24. cfg_scale already swept and tuned to 2.5 separately.

**Fix attempt — `flow_ft_onsetfix_final.pt` (job 11178, 2026-09-14):** 8000 steps, 4.13h, best
val_loss 1.0655 (vs 1.0764 for the deployed adapted Stage 2). Trained clean. **Why it only partly
worked, in hindsight: it fine-tuned Stage 2 ALONE, treating a coupling defect as Stage-2-local.**
Any future attempt must target the interaction (co-adapt both stages, or an onset-weighted loss
term) rather than repeat the Stage-2-only recipe.

**REJECTED on its own pre-registered gate (job 11184, 2026-09-15): 4/24 vs a bar of <=1/24.**
Real but partial progress: 2spk 1/8 -> **0/8** (clean), 3spk 3/8 -> 1/8, but **4spk 3/8 -> 3/8,
unmoved**; onset damping -4.67 -> -2.37dB. The launcher's own rule is "passing the gate is
necessary, not sufficient — if it still gates, the rest is moot", so the full battery should NOT
be spent on this candidate.

**Battery eval (job 11187) died on the SLURM TIME LIMIT after 1 of 5 stages — NOT a code bug.**
Root cause: a foreign process **PID 161647 squatting 14,994MiB of gnode2's 16,384MiB GPU 1**,
present in every job diagnostic from 2026-09-15 on (11182/11183/11184/11187) but NOT at
2026-09-14 16:02 (job 11178 shows GPU1 at 0MiB), so it appeared between those times. Effect:
per-sample time degraded 5.5s -> 76.8s (14x) with a ~2h stall between samples 241-246; stage 1
took 347min instead of ~5min; Libri2Mix projected 9,389min (~6.5 days) at 51/6000 when killed.
**Check `nvidia-smi` in the job diagnostic preamble before trusting any timing on this cluster.**
Fix: kill it if it is the user's own orphan, else `#SBATCH --exclude=gnode2` + tell HPC admins.

**Second, independent failure mode: jobs 11182 and 11183 burned their ENTIRE time limit retrying
HuggingFace downloads** of `microsoft/wavlm-base-plus-sv/preprocessor_config.json`
(ProtocolError / SSLEOFError from the compute node). Worth guarding with `HF_HUB_OFFLINE=1` over a
pre-warmed cache.

**Partial battery result that did complete** (stage 1/5, controlled low-SNR in-domain n=300) —
onsetfix buys nothing outside the onset metric: accuracy 79.3% vs deployed 80.3%, SI-SDR vs
mixture +6.01 vs +6.51dB, S2 vs S1 +1.45 vs +1.72dB, catastrophic 11.0% vs 11.0%.
Output JSONLs are resumable: `eval_lowsnr_indomain_onsetfix.jsonl` (complete, 300),
`eval_libri2mix_min_onsetfix.jsonl` (~51 records).

**CORRECTION 2026-09-16 — padding DOES work; an earlier note here saying it "moved nothing" was
WRONG.** That reading came from job 11176, which is the **cfg** sweep (cfg 1.5/2.0/2.5 + warmup
variants, all 6-7/24 gated — cfg genuinely cannot fix it). The actual **pad** sweep is job **11174**:
| condition | pad0 | pad0_b | pad12 | pad25 | pad50 |
|---|---|---|---|---|---|
| 2 speakers | -1.47 | -1.47 | -0.84 | -0.80 | -0.76 |
| **3 speakers** | **-9.37** | -9.37 | **-1.82** | -2.14 | -4.74 |
| 4 speakers | -6.25 | -6.25 | -4.29 | -4.12 | -4.42 |
| **ALL** | **-4.57** | -4.57 | **-2.11** | -2.14 | -2.84 |
The two pad0 columns are IDENTICAL to the decimal (Stage 2 never samples noise — it starts from
`x = x_enh.clone()` and Euler-integrates), so the noise floor is zero and every difference is real
signal. Root cause per `eval/diag_onset_padding.py`'s docstring: first speech burst -4.08dB, interior
onsets after silence +0.29dB, sustained +0.01dB ⇒ a **POSITION-0 effect** — frame 0 is the only frame
with no left context for the DiT's RoPE, and Stage 2 generates its mel where Stage 1 merely scales
the mixture. Global padding costs 0.24dB SI-SDR / 2.4% mel MSE spread over all 10s and fails the
promotion bar; **splicing** (job 11175, validated: splice25 ≡ pad25 in the onset region) keeps the
tail bit-identical so only the improved region can move the averages.

**IMPLEMENTED 2026-09-16 (code written + verified locally, NOT yet run on the cluster).**
Opt-in everywhere, default = existing behavior bit-identical — same convention as `--stage1_mode`
and `mask_mode`.
- `models/flow.py`: `inference_with_onset_splice(flow_model, x_enh, d_vector, ..., pad_frames=0,
  splice_frames=100, xfade_frames=20)` + `ONSET_PAD_VALUE=-11.5`. `pad_frames=0` returns
  `flow_model.inference(...)` unchanged and does ONE forward pass; >0 costs a second pass.
- Wired into all 6 call sites, each with `--onset_pad_frames` (default None →
  `cfg.inference.onset_pad_frames`, else 0): `eval/eval_multi_speaker.py` (also records the setting
  in each JSONL record), `eval/verify_eer.py`, `eval/full_eval.py`, `eval/eval_libri2mix.py`,
  `eval/export_listening_samples.py`, `inference/infer.py`.
- **Chunking subtlety:** in the two chunked callers (`eval_libri2mix.py`, `inference/infer.py`) the
  repair is applied ONLY when `start == 0`. Interior chunks have genuine left context and show no
  defect — padding them would be wrong.
- `configs/default_v2.yaml`: `inference.onset_pad_frames: 0`, documented, NOT adopted yet (raise it
  only after the battery confirms), so it can be adopted with a one-line edit like cfg_scale 2.5 was.
- `scripts/run_eval_candidate_gpu.sh`: optional **5th** positional arg → `--onset_pad_frames`.
- Verified locally: `test_onset_splice.py` (new, repo root, project convention) **30/30** — proves
  pad_frames=0 is bit-identical to plain inference with exactly one pass, the tail past the crossfade
  is bit-identical to the unpadded run, the onset comes from the padded run, the crossfade is the
  exact linear blend, plus edge cases (splice>T, short xfade room, xfade=0, splice=0, negative pad)
  and that the caller's tensor is not mutated. Also `ast.parse` + an undefined-name AST check on all
  7 changed files, `bash -n` on the launcher, YAML parse. Could NOT run the real eval scripts locally
  (no soundfile/omegaconf/transformers on this Windows session).
- Windows gotcha hit again: the Write tool fails with `EPERM mkdir` creating NEW files on the `Z:\`
  SSHFS mount. Workaround that worked: write to the scratchpad, then `cp` over with bash. Editing
  EXISTING files on the mount works fine.

**MEASURED 2026-09-16 (job 11202, tag `onsetpad12`, pad_frames=12) — battery ran CLEAN end to end,
all 5 stages, 0 errors. VERDICT: the onset fix is an AUDIBLE-QUALITY change, NOT an accuracy change.
Answers the question it was built to answer: padding/splice does NOT raise accuracy.**
Validity confirmed — a genuinely paired comparison: the checkpoint-independent probes are IDENTICAL
between 11112 and 11202 (low-SNR mixture 64.0%/AUC 0.7117 + ceiling 90.3%/0.9677; corpus-wide
mixture 71.0%/0.7958 + ceiling 90.8%/0.9704; 2spk mixture 70.3%/0.7899; full_eval ceiling 79.8%/
margin +0.1344). Same samples scored, and only Stage 2's output changed — the wiring is correct.
| metric | pad0 (11112) | pad12 (11202) | delta |
|---|---|---|---|
| low-SNR acc / AUC | 80.3% / 0.8827 | 79.3% / 0.8798 | -1.0pp / -0.0029 |
| corpus-wide n=400 acc / AUC | 86.2% / 0.9361 | 85.8% / 0.9348 | -0.4pp / -0.0013 |
| 2spk acc / AUC | 85.7% / 0.9342 | 85.3% / 0.9328 | -0.4pp / -0.0014 |
| 3spk acc / AUC | 80.0% / 0.8870 | 80.3% / 0.8848 | +0.3pp / -0.0022 |
| 4spk acc / AUC | 77.0% / 0.8511 | 77.0% / 0.8509 | 0.0pp / -0.0002 |
| in-domain n=2620 SI-SDR vs mix | +1.75dB | **+1.76dB** | +0.01 |
| in-domain catastrophic | 10.2% | **9.9%** | -0.3pp |
| low-SNR SI-SDR vs mixture | +6.51dB | **+6.60dB** | +0.09 |
| Libri2Mix ALL SI-SDR vs mix | +3.39dB | **+3.42dB** | +0.03 |
Pattern: **SI-SDR consistently very slightly BETTER, accuracy/AUC consistently very slightly WORSE**
— every individual delta inside n=300/n=400 noise, but AUC falls in 5/5 conditions (sign test
p=0.031, and it is a PAIRED comparison so the effective noise is smaller than the unpaired SE), so
there is probably a real but tiny verification cost.
**WHY accuracy did not move (the caveat flagged before the run, now confirmed):** the WavLM speaker
embedding is mean-pooled over the WHOLE utterance, so a damped first ~0.5s of a 4-10s utterance
barely shifts it. The metric is structurally insensitive to this defect — which is exactly why the
defect was found BY EAR and not by the battery.
**DECISION STILL OPEN (user's call):** adopt for perceptual quality (fixes what the user hears,
costs a 2nd Stage-2 pass per utterance and ~1 AUC point in the 3rd decimal) vs leave at 0 (every
number in docs/ was measured at pad0; adopting means restating them). The user's stated standing
goal has been "accuracy at or ABOVE the previous best, never below", which this does NOT meet.
**NOT YET RUN:** the gate diagnostic at pad12 — the battery does not measure gated-sample count, so
the 7/24 -> ? improvement is still unconfirmed at the operating point, and a re-export + listening
pass is what would actually settle "is it fixed".
**Job 11204 (2026-09-16) failed instantly (103-byte log, usage error) — MY fault:** I gave the user
`sbatch scripts/run_diag_gate_candidate_gpu.sh` with NO arguments; it requires `<flow_ckpt> <tag>`.
Lesson: always hand over the fully-formed command with every required positional arg.
**FIXED 2026-09-16:** `run_diag_gate_candidate_gpu.sh` gained an optional **5th** arg
`[onset_pad_frames]`. It previously hardcoded `--pads 0 --splice_frames 0`, so it could only gate a
retrained CHECKPOINT and had no way to measure the inference-side repair. With the 5th arg it passes
`--pads 0,N --splice_frames 100`, putting the pad0 reference and the spliced pad-N column in the SAME
table on the SAME samples — a paired within-run comparison. Empty 5th arg reproduces the old command
line exactly (verified by dry run); `bash -n` clean. `eval/diag_onset_padding.py` needed no change —
it already had `--pads`/`--splice_frames`/`--xfade_frames` natively (that is how jobs 11174/11175 ran).
NOTE: `diag_onset_padding.py` keeps its OWN `run_stage2()`/`make_splice()`, which my
`models/flow.py inference_with_onset_splice` was written to mirror — deliberate duplication left
alone because the diag script produced the reference numbers and should not be perturbed.
**GATE MEASURED 2026-09-16 (job 11205, pad0 vs spliced pad12, paired in one table) — THE PADDING FIX
DOES NOT FIX THE DEFECT. 7/24 -> 6/24 gated. Bar is <=1/24.**
| condition | pad0 | pad12 | splice12 |
|---|---|---|---|
| onset 2spk | -1.85 | -1.57 | -1.57 |
| onset 3spk | -4.14 | -3.58 | -3.58 |
| **onset 4spk** | **-7.75** | **-7.06** | **-7.06** |
| onset ALL | -4.67 | **-3.36** | -3.36 |
| tail ALL (validity check) | -1.30 | -1.20 | **-1.29** |
| gated 2/3/4spk | 1/8, 3/8, 3/8 | 1/8, **2/8**, 3/8 | 1/8, 2/8, 3/8 |
| **gated ALL** | **7/24** | **6/24** | **6/24** |
Only ONE gated sample rescued, and it is a 3-speaker one. **4-speaker — the condition the user
actually complained about — does not move at all (3/8 -> 3/8).** Same holdout as the onsetfix
fine-tune. Note the FINE-TUNE did better on this metric (4/24) than padding (6/24); neither passes.
Splice mechanism CONFIRMED correct in production: its onset column equals pure pad12 to the decimal
while its tail matches pad0 (-1.29 vs -1.30, vs pure pad's -1.20) — exactly the built-in validity
check the script documents.

**MY ERROR, corrected here: the -4.57 -> -2.11dB pad-sweep figures from job 11174 that justified
building this were measured on a DIFFERENT, HARDER sample draw — seed 42 at SNR [-5,5]** (stated
explicitly in `eval/diag_onset_padding.py`'s argparse NOTE; the launcher defaults were changed
afterwards, on 2026-09-15, precisely so later runs would match the listened-to samples). Job 11205
uses the correct operating point (seed 123, SNR [1,10]) and shows a much smaller effect. Its pad0
column (-4.67 ALL; 2/3/4spk -1.85/-4.14/-7.75) differs from 11174's (-4.57; -1.47/-9.37/-6.25),
which is the tell. **Always check that two diagnostic runs share a seed/SNR draw before comparing.**
Minor, unchased: on THIS draw pure pad12 beats splice12 on whole-utterance SI-SDR (-8.17 vs -8.90)
and mel (2.975 vs 3.050) — the splice's original rationale (global padding costs 0.24dB SI-SDR) came
from the seed-42 draw and does not hold here. n=24 though, and battery 11202 at n=2620 showed
splice ~ baseline, so not worth pursuing.

**DOCUMENTED 2026-09-16 — this line of work is now CLOSED and written up.** Added timeline entries
**33** (the 2x2 interaction result), **34** (the onsetfix fine-tune + its rejection, plus the
HF-offline and GPU-squatter infrastructure lessons) and **35** (the inference-side repair: what was
built, the paired battery 11202, the gate 11205) to `docs/methodology_and_project_history.md`;
updated its closing "Where this leaves the project" paragraph; replaced §5.11's "Status: cause being
localized" in `docs/results_and_limitations.md` with the resolved cause + the two-mitigation table +
an honest statement of the current system. Verified: entries 33/34/35 in sequence, 0 malformed
tables. Both docs now state the defect is a characterized limitation, same framing as the
SNR-coverage gap.

**PROCESS LESSON worth keeping (my error, 2026-09-16):** `docs/` entry 32 ALREADY said "Padding does
not fix it (jobs 11174, 11175)" AND already carried the seed-42/SNR[-5,5] caveat about job 11174's
numbers. I recommended building the padding fix without reading the docs first, re-derived a
conclusion the project had already reached, and spent a full battery (11202) plus a gate run (11205)
confirming it. **Read `docs/methodology_and_project_history.md` and `docs/results_and_limitations.md`
BEFORE proposing any experiment on this project — they are the authoritative record and are usually
ahead of memory.** What the episode did leave behind of value: a validated opt-in implementation, and
the first measurement of the repair at the deployed operating point at n=2620/6000 scale rather than
on 24 samples at a different draw.

**`cfg_warmup_steps=2` — MEASURED AND REJECTED 2026-09-17 (job 11210). Documented as docs timeline
entry 36; the two places that called it a promising lead (methodology entry 32, results §5.11) now
point at the rejection.** Clean battery run, warm-up confirmed active in all 5 stages.
- **The quality half REPLICATED at full scale** (it was only ever 24 samples before): in-domain mel
  72.9->73.6%, catastrophic 10.2->9.6%, SI-SDR vs mix +1.75->+1.83dB / vs S1 +1.30->+1.39dB,
  hurts-rate 25.1->23.2%; Libri2Mix catastrophic 13.1->12.5%, SI-SDR vs mix +3.39->+3.46dB. Every
  signal-quality axis improved.
- **But ACCURACY FELL EVERYWHERE — the axis nobody had measured:** corpus-wide 86.2->85.5%
  (AUC .9361->.9327), 2/3/4spk 85.7/80.0/77.0 -> **85.0/78.4/74.6** (AUC down in all four). 4spk
  gives back 2.4pp = the entire gain the cfg 2.5 re-tune bought there.
- **MECHANISM: warm-up is just WEAKER GUIDANCE re-parameterized.** 2 of 4 Euler steps at cfg 1.0 =
  time-averaged cfg ~1.75, on the exact axis the cfg sweep already mapped (less guidance -> better
  SI-SDR, worse accuracy). It reproduces cfg 2.0's 2spk accuracy (85.0%) almost exactly. NOT a free
  win — the same trade the project already optimized in the opposite direction. "No metric worse"
  held only because job 11176 reported mel and SI-SDR and never accuracy.
- **GENERALIZABLE LESSON (now in the docs):** a change that improves every metric you happen to be
  looking at has not been shown to be free. Third instance of this trap in the project, after the
  mel-vs-SI-SDR divergence and the Stage-1-accuracy-vs-pipeline-quality result.
- Keep `inference.cfg_warmup_steps: 0`. Flag + plumbing stay in, inert, so a future checkpoint can be
  retested in one command.

**(superseded) WIRED 2026-09-17, READY TO RUN, not yet measured.** Flagged in docs entries
32 AND 35 as the only change in the whole campaign that improved averaged metrics with nothing worse
(mel-MSE 3.083 -> 2.751, -11%; SI-SDR -8.95 -> -8.47dB; it also nudges the gate 7->6/24) — but
measured on only 24 samples in job 11176 and NEVER on the accuracy battery, because only
`full_eval.py` exposed the flag. Now plumbed everywhere, same opt-in convention as onset_pad_frames:
- `configs/default_v2.yaml`: `inference.cfg_warmup_steps: 0`, documented, NOT adopted.
- All 6 inference paths take `--cfg_warmup_steps` (default None -> config -> 0). `full_eval.py` was
  ALSO changed — it had `default=0` hardcoded, so leaving it would have made a config change apply to
  4 of 5 battery stages and silently skew the comparison.
- `models/flow.py inference_with_onset_splice` already forwarded it; no change needed there.
- **Design difference vs the onset pad:** warm-up is applied to EVERY chunk in the chunked callers,
  not just `start == 0` — it concerns the Euler trajectory at t=0, which each chunk's inference
  restarts, not position within the utterance.
- `scripts/run_eval_candidate_gpu.sh`: optional **6th** positional arg.
- Verified: ast.parse + undefined-name check on 7 files, `bash -n`, YAML parse, launcher arg dry run
  (4/5/6-arg forms all correct), `test_onset_splice.py` still 30/30 (it already asserted
  cfg_warmup_steps is forwarded to both passes).
**COMMAND (~2h):**
`sbatch scripts/run_eval_candidate_gpu.sh checkpoints_v2/flow/flow_best.pt warmup2 checkpoints_v2/masking/mask_best.pt 2.5 0 2`
Compare against job 11112: 2/3/4spk 85.7/80.0/77.0, corpus-wide 86.2%, low-SNR 80.3%, in-domain
SI-SDR vs mixture +1.75dB / catastrophic 10.2%, Libri2Mix ALL +3.39dB.
**CAVEAT:** CFG warm-up was investigated and ruled out earlier in the project for the t~0
catastrophic outliers (see [[mask2flow-tse-cfg-warmup-fix]]) — but that was a different pipeline
(both stages since replaced, cfg 1.5 -> 2.5), so it is not the same test. It has failed once before.

**BOTTOM LINE: keep `inference.onset_pad_frames: 0`.** The onset defect resists BOTH levers —
retraining (onsetfix 7->4/24, failed) and inference-side padding (7->6/24, failed) — which is
consistent with the entry above diagnosing it as an S1xS2 INTERACTION that neither lever alone
reaches. The code stays in as validated, opt-in, zero-cost-when-off. That combination (a
well-localized position-0 defect, a clean 4-cell ablation showing it is an interaction, and two
independent failed mitigations) is the honest thesis writeup.

**(superseded) CORRECT COMMAND (~25min):**
`sbatch scripts/run_diag_gate_candidate_gpu.sh checkpoints_v2/flow/flow_best.pt onsetpad12 checkpoints_v2/masking/mask_best.pt 2.5 12`

**(superseded) NEXT COMMAND (nothing has been measured yet):**
`sbatch scripts/run_eval_candidate_gpu.sh checkpoints_v2/flow/flow_best.pt onsetpad12 checkpoints_v2/masking/mask_best.pt 2.5 12`
Compare against job 11112 (current system): 2/3/4spk 85.7/80.0/77.0, corpus-wide 86.2%, low-SNR
80.3%, in-domain SI-SDR vs mixture +1.75dB, Libri2Mix ALL +3.39dB. Cheap pre-check first if
preferred: `run_diag_gate_candidate_gpu.sh` (~25min) to see the gated count move off 7/24.
**Clear the gnode2 GPU squatter before submitting** or timings will be ~14x inflated.
**Open question the eval settles:** onset damping is PROVEN to improve; whether that converts into
speaker-verification accuracy is NOT — the damaged region is ~0.5s of a 4-10s utterance. pad12 vs
pad25 is also unresolved (pad12 best overall/3spk, pad25 marginally better at 4spk).

**SUPERSEDED RECOMMENDATION (2026-09-16, before the pad sweep was re-read): STOP fixing, document the
interaction finding, put remaining effort into the thesis.** Still the right call for RETRAINING —
see the trade-curve evidence below — but the inference-side padding fix is free and was wrongly
dismissed at the time. Rationale: every cheap lever is spent
(padding no effect, cfg already tuned, Stage-2-only FT partial and costs ~1pp low-SNR accuracy);
what remains is a speculative co-adaptation run at 4-13h/attempt, and the low-SNR campaign already
established these trades come out roughly proportional rather than free; and the 4-cell ablation is
a stronger thesis result written up than silently fixed. Counter-argument if the 4spk onset
clipping was clearly AUDIBLE in the listening pass: one targeted attempt at the interaction is
defensible. Clear the GPU squatter and the HF fetch before spending GPU time either way, and do
NOT resubmit the battery for `onsetfix` — already rejected on its own gate.
