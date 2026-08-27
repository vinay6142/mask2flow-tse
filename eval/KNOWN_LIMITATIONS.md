# Known limitation: catastrophic Stage-2 outliers on hard extraction cases

**Status:** characterized and documented, not fixed. Root cause isolated to a training-coverage
gap in Stage 2 (flow matching), not a bug, and independently confirmed to persist (with different
severity per-sample) in actual post-vocoder audio, not just mel-domain MSE — see "Audio-domain
verification" below. Validated at full-test-set scale (n=2620 mel-domain, n=400 audio-domain) —
see "Full-set validation" below — every number from the original n=60 investigation holds up. See
`eval/diagnose_catastrophic.py`, `eval/audio_domain_quality.py`, and `eval/full_eval.py` for the
diagnostic tooling used to reach this conclusion; rerun all three against any future Stage-2
checkpoint to check whether a retrain has closed the gap.

## Summary

On real LibriSpeech test-clean (seed=42, batch=60, `checkpoints_v2/masking/mask_best.pt` +
`checkpoints_v2/flow/flow_best.pt`, `n_steps=4`, `cfg_scale=1.5`), the full two-stage pipeline's
**median** MSE improvement is strong: 75.8% vs mixture, 71.9% on top of Stage 1 alone. However,
**7–8 of 60 samples (~12%)** show Stage 2 making MSE dramatically *worse* than Stage 1 alone —
regressions from -51% to -2454% — pulling the *mean* improvement negative (-10.6%) even though the
median stays high. Speaker-similarity shows a parallel but partially distinct pattern: median
delta is net positive (+0.13), but 18–30% of samples have extraction *reduce* speaker similarity
vs. doing nothing.

## Methodology: three plausible mechanisms tested and ruled out

1. **Speaker-embedding degeneracy.** Hypothesis: the trained d-vector projection
   (`checkpoints_speaker_encoder/projection_latest.pt`) doesn't generalize to unseen speakers,
   producing degenerate conditioning for the flagged samples. Built a proper unseen-speaker
   verification metric (same/different-speaker cosine similarity + separation AUC on held-out
   test-clean speakers — replacing `train_speaker_encoder.py`'s `val_acc`, which is measured
   against a disjoint, incompatible label space and is chance-level by construction, not a real
   generalization signal). Result: **AUC = 0.93–0.97** across three independent runs — the
   projection separates unseen speakers well. Ruled out.

2. **CFG amplification at t=0.** Hypothesis: at t=0 the rectified-flow interpolation gives zero
   progress signal (`x_t = x_enh` exactly), so `v_cond` and `v_uncond` disagree sharply there, and
   `cfg_scale=1.5` amplifies that raw disagreement into an oversized first Euler step. Added
   `cfg_warmup_steps` to `FlowMatchingModule.inference()` (runs the first N steps at
   `cfg_scale=1.0`, no guidance amplification). Result: only 1 of 8 flagged samples crossed the fix
   threshold; the worst two (samples 3, 32) were essentially unaffected. Direct trace evidence:
   `x_norm` trajectories were nearly identical with and without CFG at step 0, because
   `v_applied_norm` was already close to `v_cond_norm` even with full guidance active. Ruled out as
   the primary mechanism (the CFG delta spike is real but not what's driving the damage).

3. **Euler discretization too coarse.** Hypothesis: `n_steps=4` is too few steps for the model to
   self-correct after a bad first guess; more steps would let the trajectory recover. Reran at
   `n_steps=16`. Result: median *dropped* slightly (71.5%/68.6% vs 75.8%/71.9%) at 4x the inference
   compute — some flagged samples improved marginally, others got worse (sample 3: -2454% →
   -2568%; sample 30: -60% → -100%). A finer discretization of the same learned vector field does
   not recover; it just integrates the same bad direction more faithfully. Ruled out.

## Root cause: t=0 training-coverage gap

Directly measured cosine similarity between the model's raw t=0 prediction (`v_cond`, no CFG) and
the true required velocity (`target - stage1_out`), for flagged vs. healthy baseline samples:

| Sample | cos_sim(pred, true) | S2 vs S1 |
|---|---|---|
| baseline (4 samples) | 0.77 – 0.92 | positive |
| mild regressions (3 samples) | 0.32 – 0.62 | -37% to -77% |
| worst offenders (3 samples) | **0.08 – 0.15** | -311% to -2555% |

This is a clean, monotonic relationship, and it isn't explained by input magnitude — `true_vel_norm`,
`tgt_max_abs`, and `s1out_max_abs` for flagged samples sit in the same range as healthy baseline
samples, ruling out data corruption or an anomalous mixing draw. SNR doesn't explain it either
(the worst offender, sample 3, has a middling 7.52dB SNR).

**Conclusion:** at t=0, Stage 2 must predict the entire correction in a single shot from Stage 1's
output and the speaker embedding alone — there is no interpolation signal yet. For the large
majority of inputs the trained model does this well; for a specific ~12% subset of hard
speaker/content configurations, its one-shot prediction is close to orthogonal to what's actually
needed. This is a gap in what the 298,000-step training run covered well, not a numerical,
guidance, or data-pipeline defect — and as such it cannot be fixed by any inference-time lever
(guidance scale, guidance scheduling, or step count all confirmed ineffective above).

## Audio-domain verification (2026-08-26): the picture is real but more nuanced

Everything above is measured in the mel-spectrogram domain — the space the models are trained in,
not what a listener hears. `eval/audio_domain_quality.py` vocodes mixture/Stage1/Stage2/target
through the trained HiFi-GAN (same seed=42/batch=60 samples, so directly index-comparable to
everything above) and measures waveform MSE and SI-SDR (Le Roux et al. 2019 — the standard,
scale-invariant metric in the source-separation/speech-enhancement literature).

**The improvement is real but much smaller in audio terms than mel-MSE implied:** median waveform
improvement is 22.1% vs mixture / 6.5% vs Stage 1 (not 75.8%/71.9%), and median SI-SDR gain is only
**+1.33 dB vs mixture, +0.95 dB vs Stage 1**. Mel-MSE percentages are measured relative to a
sometimes-small Stage-1-MSE denominator and are unbounded, so they overstate perceptual severity in
both directions.

Cross-referencing the 8 known mel-catastrophic samples against actual audio quality splits them
into two distinct groups:

- **Genuinely smoothed out by the vocoder (3 of 8):** samples 27, 30, 32 — all show *positive*
  SI-SDR gains in audio (+3.77, +0.47, +1.14 dB) despite catastrophic mel-MSE. Sample 32 was the
  *second-worst* mel-domain failure (-1345%) but is fine as audio.
- **Still bad, or worse than the mel numbers implied (5 of 8):** samples 3, 20, 29, 40, 46. Two are
  dramatically worse than mel-MSE suggested: **sample 40 crashes to -49.67 dB SI-SDR (a 42 dB loss
  vs Stage 1 — output is effectively garbage/noise for that segment)**, and **sample 46 drops to
  -39.84 dB (a 14 dB loss)**. These are the single most severe concrete failures found across this
  entire investigation — audibly destroyed output, not just a high MSE number.

**Broader pattern confirmed by a third independent metric:** 16/60 (26.7%) of samples show Stage 2
*reducing* SI-SDR vs Stage 1 alone — consistent with the mel-domain pattern (~12% catastrophic) and
the speaker-similarity check (18/60, 30% reduced). Three independent metrics (mel-MSE,
speaker-similarity, audio SI-SDR) now agree there is a robust ~25-30% "Stage 2 sometimes hurts"
phenomenon, not an artifact of any single metric's sensitivity.

## Full-set validation (2026-08-26): n=60/400 numbers hold up at scale

Everything above came from small batches (n=60, or n=4-11 for the diagnostic deep-dives) — thin for
a thesis headline. `eval/full_eval.py` iterates the entire `LibriSpeechTSEDataset(seed=42)`
deterministically (resumable, incremental JSONL checkpointing, since a full CPU pass takes ~75
minutes). Two runs: full mel-domain pass (n=2620, all of test-clean, no vocoding) and an
audio-domain pass (n=400, mel + vocoding + SI-SDR — vocoding is the expensive part on CPU so a full
2620-sample audio-domain pass wasn't practical this session).

**Every earlier n=60 number holds up within noise at 7-44x the sample size** — this was not a
lucky/unlucky single batch:

| Metric | n=60 | n=2620 (mel) / n=400 (audio) |
|---|---|---|
| Median mel S2 vs S1 | 71.9% | 67.3% (n=2620) |
| Catastrophic rate (mel S2vsS1 < -30%) | ~12-13% | **12.7%** (n=2620) |
| Median SI-SDR gain vs S1 | +0.95 dB | **+0.93 dB** (n=400) |
| SI-SDR "hurts" rate | 26.7% | **28.5%** (n=400) |

**New finding — catastrophic rate is strongly, monotonically SNR-dependent** (invisible at n=60;
the n=2620 SNR-bucket breakdown makes it clear):

| SNR bucket | Catastrophic rate |
|---|---|
| 1–3 dB (hardest) | 22.1% |
| 3–5 dB | 16.2% |
| 5–7 dB | 10.0% |
| 7–10 dB (easiest) | 5.5% |

Harder mixtures fail catastrophically ~4x more often than easy ones. This refines, not
contradicts, the t=0 training-coverage-gap root cause: the gap isn't triggered purely at random,
it correlates with mixture difficulty in aggregate — even though individual counterexamples exist
(sample 3 from the original n=60 batch had a decent 7.52dB SNR yet was the single worst failure;
SNR predicts *rate*, not any one case).

**New finding — mel-MSE ranking does not predict audio-domain severity, confirmed at n=400 (not
just the original n=8 anecdote):** the n=400 worst-10-by-mel-MSE list shows almost no correlation
with actual SI-SDR damage. idx=243 has the single worst mel-MSE (-3885%) but only a mild -3.12dB
SI-SDR loss; idx=174 has much milder mel-MSE (-953%, 7th on the list) but the worst audio
destruction in the whole list (-45.32dB); idx=89 and idx=290 are mel-catastrophic yet *gain* SI-SDR
(smoothed out by the vocoder). **Do not rank or select "worst samples" by mel-MSE alone for
qualitative discussion/listening-test purposes in the thesis — rerun by SI-SDR instead**, e.g. via
`python3 eval/full_eval.py --output <file> --summarize_only` on the audio-domain JSONL and sorting
its records by `sisdr_gain_vs_s1`.

## Decision (2026-08-26)

Documented as a characterized limitation rather than pursued further for now, given the median
result is strong and the fix (continued Stage-2 training with hard-example reweighting /
importance-sampled t≈0 batches) is a substantial additional investment with uncertain payoff.
`n_steps=4` remains the default — `n_steps=16` was tested and is strictly worse at 4x the cost, so
there is no reason to change it.

For the thesis/report, use the **full-set numbers as the citable headline** (median mel S2-vs-S1
improvement 67.3% on n=2620; median SI-SDR gain +0.93 dB vs Stage 1 on n=400; 12.7% catastrophic
rate on n=2620), not the original n=60 figures — report the audio-domain SI-SDR number as primary
and the mel-domain number as a training-objective-space secondary figure, not standalone, since
mel-MSE improvement alone overstates the system's actual audible quality gain. The SNR-bucket
breakdown is worth a table/figure of its own — it's a much stronger, more specific claim than
"~12% of samples are catastrophic" (e.g. "catastrophic-failure rate falls monotonically from 22% at
1-3dB SNR to 5.5% at 7-10dB SNR").

**If revisited:** the concrete fix path is continuing Stage 2 training from `flow_best.pt` with
loss reweighting toward the hardest t≈0 predictions (e.g. per-batch importance sampling weighted
by current loss, or oversampling low-cosine-similarity examples identified via this same
diagnostic). `eval/diagnose_catastrophic.py` part [C] is the tool to verify whether a future
checkpoint has closed the gap — rerun it and check whether the cosine-similarity floor for hard
samples has risen out of the 0.08–0.15 range. Samples 40 and 46 (near-total audio failures, not
just high MSE) would be the highest-priority cases to fix first if any further work is done here.
