---
name: mask2flow-tse-cfg-warmup-fix
description: "Root-cause diagnosis of Mask2Flow-TSE's catastrophic Stage-2 MSE outliers; the training-side fix (hard-t0 fine-tune) was later executed and shows real improvement"
metadata: 
  node_type: memory
  type: project
  originSessionId: 28e94c75-57ba-46d9-9dad-9d9b2db586de
  modified: 2026-08-28T06:29:03.682Z
---

**UPDATE 2026-08-28 — Option 2 ("real fix") below WAS executed and WORKED on the real target
metric.** `training/finetune_flow_hard_t0.py` (t_hard_prob=0.5, t_hard_max=0.25, lr=4e-5) fine-tuned
Stage 2 from `flow_best.pt` (step 298000) for 25000 steps / 11.58h on a P100 GPU via
`scripts/run_finetune_flow_hardt0_gpu.sh` (job 10606, log `scripts/logs/finetune_flow_hardt0_10606.out`).
The script's own training-time `val_loss` proxy got WORSE, not better (0.7546 → best 1.0909 at step
16000, 1.1131 at final step 25000) — expected and not a red flag, since hard-t0 reweighting
deliberately oversamples the harder near-t=0 region the base checkpoint was never scored on, so it's
not comparable to the pre-fine-tune val_loss number. The script explicitly warns not to trust this
proxy and to check the real target metric instead.
That real check was done with a NEW script, `eval/verify_eer.py` (added 2026-08-27, supersedes the
noisy in-batch verification metric in `full_eval.py` — see that script's docstring), which computes
corpus-wide speaker-verification EER/AUC (genuine = probe vs. own reference; impostor = probe vs.
every other speaker's reference; n=400 samples, 155540 impostor trials) for mixture / Stage 2 / oracle
target probes against the same reference set. Result, evaluating `flow_ft_final.pt` (=step 25000,
the LATEST/final checkpoint, not `flow_ft_best.pt`/step 16000 which was "best" only by the untrusted
proxy metric): mixture baseline EER 28.2% (acc 71.8%, AUC 0.8029) → Stage 2 EER 13.8% (acc 86.2%, AUC
0.9353), vs. ground-truth ceiling EER 9.2% (acc 90.8%, AUC 0.9691). That's ~76% of the mixture→ceiling
EER gap closed — a genuine, real-metric win from the retrain, not a proxy-metric artifact.
Caveats / not yet done: (1) `flow_ft_best.pt` (step 16000) has NOT been run through `verify_eer.py`
for comparison — the sbatch script's own header comment suggests checking it too, since the proxy
plateaued there while training continued 9000 more steps; worth a quick comparison before picking
which checkpoint to promote. (2) EER is a corpus-wide aggregate and can look excellent even if the
~12% catastrophic-outlier subset (samples 3/20/27/29/30/32/40/46 from the diagnosis below) is
unchanged — this result does NOT by itself confirm the catastrophic-outlier problem is fixed; rerun
`eval/diagnose_catastrophic.py` or `eval/full_eval.py` against the new checkpoint to check that
specifically if it matters for the thesis writeup. (3) the pasted eval log's shell prompt
(`gpu114`, the login node per [[mask2flow-tse-overview]]) and its pace (~3.5s/sample) suggest
`verify_eer.py` was run interactively on CPU, not via sbatch — fine for correctness, just slow; submit
it as a GPU job if extending to the full 2620-sample test-clean set.
Given this success, the "do not propose retraining Stage 2 again" guidance below is now superseded —
the user reopened it and it paid off. See [[mask2flow-tse-eval-results]] for the corpus EER numbers
recorded as a headline result, and [[mask2flow-tse-next-steps]] for the follow-up items above.

**UPDATE 2026-08-28 (same day, later) — full n=2620 validation CONFIRMS the root cause was fixed,
not just papered over by aggregate EER.** Two follow-ups from the caveats above, both done:
(1) `eval/verify_eer.py` against `flow_ft_best.pt` (step 16000): EER=13.2%/acc=86.8%/AUC=0.940 —
marginally better than step 25000's 13.8%/86.2%/0.935, but the 0.6pp gap is likely within noise at
n=400; not a strong signal either way. (2) `scripts/run_full_eval_audio_gpu_hardt0.sh` (job 10661,
GPU, `flow_ft_final.pt`/step 25000, full test-clean n=2620) — this is the one that matters. Compared
to the pre-finetune full-set numbers recorded above (`flow_best.pt`, n=2620): median mel S2vsS1
67.3%→**77.7%**; **catastrophic rate 12.7%→3.5%** (92/2620) — a 3.6x reduction; median SI-SDR gain
+0.93dB→**+1.58dB**; SI-SDR-hurts rate 28.5%→18.2%; catastrophic rate at the worst SNR bucket
(1-3dB) 22.1%→**9.9%**, at the best bucket (7-10dB) 5.5%→0.6%. The reduction holds across every SNR
bucket, hardest ones improving the most — this is a real, validated fix of the t≈0
training-coverage gap, not an artifact of the EER metric being insensitive to outliers.
**Conclusion upgraded:** this is no longer "characterized limitation, not fixed" — it's "diagnosed
and substantially mitigated via targeted (hard-t0) retraining." Update `eval/KNOWN_LIMITATIONS.md`
to reflect this before citing the old "12% catastrophic, not fixable at inference time" framing
anywhere (thesis, report) — the old framing was correct as of 2026-08-26 but is now stale.
Caveats still open: the worst ~10 samples in the n=2620 sweep are still badly broken (SI-SDR crashes
to -75dB, -59dB) — the tail shrank a lot (12.7%→3.5%) but did not disappear; don't overclaim "solved,"
say "reduced 3.6x." Also, `flow_ft_best.pt` (step 16000) has NOT had the same full n=2620
catastrophic/SI-SDR sweep run against it — only EER at n=400. Given step 25000 already clears the
bar and has the full rigorous validation, it's the recommended checkpoint to promote; running the
same full_eval sweep on step 16000 too would be the thorough symmetric check but isn't blocking.

---

**RESOLVED as of 2026-08-26 — decision made, investigation closed (historical; see UPDATE above for what happened next).** Three inference-time
mechanisms (CFG amplification, CFG warm-up, Euler step count) were tested and all ruled out;
n_steps=16 was strictly worse than n_steps=4 (median 71.5%/68.6% vs 75.8%/71.9%) at 4x compute, so
n_steps=4 stays the default going forward — do not retest higher step counts, this is closed.
Root cause: a genuine t=0 training-coverage gap in Stage 2 (see below), not fixable at inference
time. User explicitly chose "document as a characterized limitation" over "scope a targeted
Stage-2 fine-tune" given time/compute constraints — do not propose retraining Stage 2 again unless
the user reopens this. Full writeup committed to the repo at `eval/KNOWN_LIMITATIONS.md` (this is
the citable summary for the thesis/report; read it first if this topic comes up again instead of
re-deriving from the raw notes below).

**Audio-domain follow-up (same day, also folded into KNOWN_LIMITATIONS.md):** built
`eval/audio_domain_quality.py` (vocodes mixture/Stage1/Stage2/target through the trained HiFi-GAN,
same seed=42/batch=60 samples, measures waveform MSE + SI-SDR). Findings: (1) real improvement is
much smaller in audio than mel-MSE implied — median SI-SDR gain only +0.95 dB vs Stage 1, waveform
MSE improvement 22.1%/6.5% (not 75.8%/71.9%); report BOTH domains together in the thesis, mel-MSE
alone overstates audible quality gain. (2) Of the 8 known mel-catastrophic samples, 3 (27, 30, 32)
are actually smoothed out by the vocoder (positive SI-SDR gain despite catastrophic mel-MSE) but 5
(3, 20, 29, 40, 46) remain bad or get worse — samples 40 and 46 are the worst concrete failures
found in the whole investigation: SI-SDR crashes of -42dB and -14dB vs Stage 1 (effectively
garbage/destroyed audio, not just "high MSE"). (3) A third independent metric (SI-SDR) confirms the
~25-30% "Stage 2 sometimes hurts" pattern already seen in mel-MSE (~12% catastrophic) and
speaker-similarity (30% reduced) — this is now a robust, cross-metric-confirmed finding, not an
artifact of any one metric.

**Full-test-set validation (same day, also folded into KNOWN_LIMITATIONS.md):** built
`eval/full_eval.py` (resumable, incremental-JSONL-checkpointed, iterates the whole
`LibriSpeechTSEDataset(seed=42)` deterministically — needed since a full CPU pass takes ~75 min).
Ran full mel-domain (n=2620, all of test-clean) and audio-domain (n=400, vocoding is too slow on
CPU for the full 2620). **Every n=60 number holds up within noise at scale** — median mel S2vsS1
67.3% (n=2620) vs 71.9% (n=60); catastrophic rate 12.7% (n=2620) vs ~12-13% (n=60); median SI-SDR
gain +0.93dB (n=400) vs +0.95dB (n=60); SI-SDR "hurts" rate 28.5% (n=400) vs 26.7% (n=60). This is
no longer a small-sample artifact — cite the n=2620/n=400 numbers as the thesis headline, not n=60.
Two new findings only visible at scale: (a) catastrophic rate is strongly, monotonically
SNR-dependent — 22.1% at 1-3dB SNR down to 5.5% at 7-10dB SNR (a stronger, more specific claim than
"~12% catastrophic"); (b) mel-MSE ranking does NOT predict audio-domain severity at n=400 either
(confirms the n=8 anecdote generalizes) — do not select "worst samples" for qualitative/listening
discussion by mel-MSE, resort by `sisdr_gain_vs_s1` in the audio-domain JSONL instead.
`eval/full_eval.py --output <file> --summarize_only` reprints stats from a saved JSONL without
rerunning the model — use this for any further slicing (e.g. re-deriving worst samples by SI-SDR).

Part of [[mask2flow-tse-overview]]. As of 2026-08-26, actively investigating the catastrophic
Stage-2 MSE regressions noted in [[mask2flow-tse-eval-results]] (samples where flow-matching makes
MSE dramatically WORSE than Stage 1 alone, e.g. -2465%, -1347%, -430% on the seed=42/batch=60 run).

**Root cause found (confirmed via instrumented diagnostic, not yet fixed+reverified on GPU):**
At t=0, `x_t = x_enh` exactly (rectified-flow interpolation has no signal toward the target yet), so
`v_θ` must extrapolate the entire trajectory from one frozen point. Instrumenting
`FlowMatchingModule.inference()`'s Euler loop showed `‖v_cond − v_uncond‖` (the CFG delta) spikes
5–40x higher at step 0 than at any later step, for exactly the 8 samples that end up catastrophic
(samples 3, 32, 20, 27, 29, 46, 30, 40 in `eval_newproj_10555`) — e.g. sample 32: delta=2133 at t=0
vs ~110–173 after; sample 46: delta=2813 at t=0 vs ~70–87 after. `cfg_scale=1.5` amplifies that raw
t=0 disagreement into an oversized first Euler step; steps 1–3 can't correct it, they just keep
extrapolating from an already-bad `x`.

**Ruled out:** speaker-embedding degeneracy. Built a proper unseen-speaker verification metric
(same/different-speaker cosine similarity + separation AUC on held-out test-clean speakers — see
below on why the existing `val_acc` is meaningless) and got AUC=0.93 — the trained projection
separates unseen speakers well. Not the cause.

**Second bug found this session (measurement bug, not a model bug):**
`training/train_speaker_encoder.py`'s `val_acc` (reported as 0.0083 in eval logs, looked alarming)
is structurally meaningless: `validate()` compares a classifier head trained only on
train-clean-100 speaker classes against `val_ds` labels, which are test-clean speaker identities
independently re-numbered by a separate `SpeakerClassificationDataset` instance — label 7 in train
means nothing next to label 7 in val. Training log confirms: train `batch_acc` climbs to ~95–100%
(projection genuinely separates train speakers) while `val_loss` monotonically INCREASES the whole
run (5.86→11.97) and `val_acc` stays ~0–0.8%, chance level by construction. This metric should be
replaced (see `eval/diagnose_catastrophic.py`'s `verify_speaker_generalization()` for the correct
approach) if `train_speaker_encoder.py` is ever rerun; not urgent since the AUC check above already
shows the current `projection_latest.pt` generalizes fine.

**`cfg_warmup_steps` fix — TESTED on GPU-node (actually ran on CPU, see below), RESULT: only
partial, does NOT fix the worst offenders.** Added `cfg_warmup_steps` param to
`FlowMatchingModule.inference()` in `models/flow.py` (runs first N Euler steps at `cfg_scale=1.0`),
plumbed through `eval/results_stage2.py` / `eval/diagnose_catastrophic.py`. Confirmation run
(`--cfg_warmup_steps 1`, seed=42/batch=60/n_steps=4) result: sample 46 crossed the -30% threshold
(fixed), others (20, 27, 29, 30, 40) improved moderately (e.g. sample 20: -430%→-311%), but the
worst 2 (**sample 3: -2555%, sample 32: -1230%**) barely moved or got slightly worse. Overall
median actually similar/unchanged (76.2%/72.0% vs prior ~75.8%/71.9%) — no regression, but no real
fix either. **Root-cause correction:** the `x_norm` trajectories were nearly IDENTICAL with and
without CFG at step 0 (e.g. sample 3: 4016.81→4560.03 warmup vs →4547.06 baseline), and
`v_applied_norm` at step 0 was already close to `v_cond_norm` even with full CFG (1565.75 vs
1573.62) — meaning CFG was never actually inflating magnitude much. **CFG amplification is NOT the
(sole) mechanism** — the original diagnostic's framing ("CFG spike causes overshoot") was
incomplete/wrong. The real signal: MSE explodes (76 vs S1's 2.87) while velocity norm growth stays
mild — so `v_cond` itself is pointing in a systematically wrong DIRECTION for these specific
inputs, independent of guidance.

**[C] Velocity direction check — RUN, CONCLUSIVE. Root cause confirmed: genuine model limitation,
not a bug.** Cosine similarity between the model's raw t=0 prediction (`v_cond`, no CFG) and the
true required velocity (`target - stage1_out`), flagged vs baseline samples (seed=42/batch=60):
baseline samples 1/2/4/5 all score 0.77-0.92 (clearly correlated); flagged samples show a clean
monotonic relationship between cos_sim and MSE damage — samples 27/30/40 (mild, -37% to -77%) score
0.43-0.62; sample 29 (-37%) scores 0.32; the worst offenders **sample 3 (-2555%), sample 32
(-1230%), sample 20 (-311%) score only 0.08-0.15 — essentially orthogonal/uncorrelated**. Magnitude
checks (`true_vel_norm`, `tgt_max_abs`, `s1out_max_abs`) for flagged samples are unremarkable, same
range as baseline — rules out data corruption. SNR doesn't explain it either (sample 3 has a decent
7.52dB SNR but is the worst offender) — it's something about the specific speaker/content pairing
landing in a region the model never learned well.

**Conclusion:** at t=0 the model must predict the ENTIRE correction in one shot from Stage 1's
output + speaker embedding alone (interpolation gives zero progress signal at t=0). For ~88-90% of
inputs it does this well; for a specific hard/OOD subset its single-shot guess is close to random.
This is a training-coverage gap, not an inference-time bug — CFG scale, CFG warmup, and the
earlier-rejected safety-net are all inference-time patches that cannot fix a direction the model
itself got wrong.

**Two paths forward, by cost (neither executed yet):**
1. Cheap, no retraining: test more Euler steps (`--n_steps 8` or `16` vs current `4`) to see if the
   trajectory can partially self-correct after a bad first guess — `n_steps=4` currently locks in
   the t=0 error with little room to recover. Run:
   ```
   python3 eval/results_stage2.py \
       --mask_ckpt checkpoints_v2/masking/mask_best.pt \
       --flow_ckpt checkpoints_v2/flow/flow_best.pt \
       --projection_ckpt checkpoints_speaker_encoder/projection_latest.pt \
       --batch_size 60 --seed 42 --n_steps 16
   ```
   Check specifically whether samples 3/20/32's `S2 vs S1%` improve, and whether the overall median
   holds ≥ the ~75.8%/71.9% baseline.
2. Real fix, more expensive: continue training Stage 2 from `flow_best.pt` with importance-sampled/
   harder-weighted t≈0 batches, since this is fundamentally a training-coverage gap.

Do NOT set `cfg_warmup_steps=1` as a config default — confirmed marginal, not a fix (see above).
If the n_steps test doesn't help, this should be written up as an honest, characterized limitation
(median improvement remains strong at ~76%/72%; a specific ~10-13% hard-example subset degrades)
rather than chased further without a training-time fix.
