# Known limitation: catastrophic Stage-2 outliers on hard extraction cases

**Status (updated 2026-08-28): diagnosed AND substantially mitigated via targeted retraining.**
Everything below through "Decision (2026-08-26)" describes the original investigation against
`checkpoints_v2/flow/flow_best.pt` (step 298000) — kept verbatim as the record of how the root
cause was found. See "Update 2026-08-28: hard-t0 retrain closes most of the gap" at the end of this
file for what happened next: continuing Stage-2 training with the t≈0 hard-example reweighting this
document's own "If revisited" section proposed cut the catastrophic rate from 12.7% to 3.5% (a 3.6x
reduction, holding across every SNR bucket) on the full n=2620 test-clean set. The fine-tuned
checkpoint (`checkpoints_v2/flow_finetune_hardt0/flow_ft_final.pt`, step 25000) has since been
promoted to `checkpoints_v2/flow/flow_best.pt`, so it is now what every script's default checkpoint
path loads. The original pre-finetune checkpoint is preserved at
`checkpoints_v2/flow/flow_best_step298000.pt` for reproducing the numbers below. The tail did not
fully disappear (see the update section) — do not describe this as "fixed," describe it as
"reduced 3.6x." See `eval/diagnose_catastrophic.py`, `eval/audio_domain_quality.py`, and
`eval/full_eval.py` for the diagnostic tooling used throughout; rerun all three against any future
Stage-2 checkpoint to check whether further retraining closes the remaining gap.

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

## Update 2026-08-28: hard-t0 retrain closes most of the gap

The "If revisited" fix path above was executed: `training/finetune_flow_hard_t0.py`
(`t_hard_prob=0.5`, `t_hard_max=0.25`, `lr=4e-5`) continued Stage-2 training from the checkpoint
above for 25,000 steps (11.58h on a P100, `scripts/run_finetune_flow_hardt0_gpu.sh`, job 10606).
The script's own training-time `val_loss` proxy got *worse* (0.75 to roughly 1.1) — expected, since
hard-t0 reweighting deliberately oversamples the region the base model scored worst on, so it isn't
comparable to the pre-finetune number. The real check used a new script, `eval/verify_eer.py`
(corpus-wide speaker-verification EER/AUC, replacing an earlier unreliable in-batch metric), and
`eval/full_eval.py` rerun against the fine-tuned checkpoints at full scale (n=2620):

| Metric (n=2620) | `flow_best.pt` (pre-finetune, step 298000) | `flow_ft_final.pt` (step 25000) | `flow_ft_best.pt` (step 16000) |
|---|---|---|---|
| Median mel S2-vs-S1 | 67.3% | **77.7%** | 77.2% |
| Catastrophic rate (S2 vs S1 < -30%) | 12.7% | **3.5%** (92/2620) | 4.0% (105/2620) |
| Median SI-SDR gain vs S1 | +0.93 dB | 1.58 dB | **1.64 dB** |
| SI-SDR-hurts rate | 28.5% | 18.2% | 18.1% |
| Catastrophic @ 1-3dB SNR (hardest) | 22.1% | **9.9%** | 10.5% |
| Catastrophic @ 7-10dB SNR (easiest) | 5.5% | **0.6%** | 1.0% |
| Corpus-wide speaker-verification EER (n=400, `verify_eer.py`) | not measured | 13.8% | 13.2% |

Both fine-tuned checkpoints are effectively tied with each other and both cut the catastrophic rate
roughly 3.6x, with the reduction holding across every SNR bucket (hardest bucket improves the most,
in absolute terms). `flow_ft_final.pt` (step 25000, the final checkpoint of the completed run) was
chosen over `flow_ft_best.pt` (step 16000, "best" only by the untrusted training-time proxy) since
neither difference between them is likely meaningful at this sample size, and the final checkpoint
is the more defensible choice to cite. **`flow_ft_final.pt` has been promoted to
`checkpoints_v2/flow/flow_best.pt`** — every script's default checkpoint path now loads the
retrained model. The original pre-finetune checkpoint is preserved at
`checkpoints_v2/flow/flow_best_step298000.pt` if the old numbers ever need reproducing.

**This confirms the root-cause diagnosis above was correct and actionable** — the t≈0
training-coverage gap was a real, fixable-by-retraining limitation, not an inherent ceiling. It is
NOT fully eliminated: the worst individual samples in the n=2620 sweep still crash to -75dB/-59dB
SI-SDR, and the catastrophic rate, while much lower, is not zero. Report this as "reduced 3.6x by
targeted retraining," not "solved." If revisited again, the same tooling (`diagnose_catastrophic.py`
part [C], sorting `full_eval.py`'s JSONL by `sisdr_gain_vs_s1`) would be the way to characterize
what's left in the remaining ~3.5%.

## Update 2026-08-30: cross-corpus validation on Libri2Mix — fix generalizes, plus a second,
## distinct, well-characterized SNR-coverage limitation

`eval/eval_libri2mix.py` (see also `scripts/generate_libri2mix_test.sh`,
`scripts/run_eval_libri2mix_gpu.sh`) ran the current `checkpoints_v2/flow/flow_best.pt` (the
hard-t0 fine-tuned checkpoint above) against Libri2Mix (JorisCos/LibriMix), a 2-speaker benchmark
built independently of this project's own on-the-fly LibriSpeech mixer — `mix_clean` condition,
16kHz, `min` mode, both directions per mixture (n=6000 samples from 3000 mixtures).

**Raw aggregate looks bad at first glance:** median mel S2-vs-S1 23.9%, catastrophic rate 26.7%,
median SI-SDR gain vs Stage 1 only +1.17dB, vs mixture **-0.61dB** (negative — extraction looks
worse than doing nothing, in aggregate). This is NOT a generalization failure and NOT an eval-script
bug (checked: source_1-as-target vs source_2-as-target split 27.2%/26.2% catastrophic, no meaningful
asymmetry that would flag a target-index mix-up). The real cause: **Libri2Mix's `mix_clean`
generation has no SNR floor** — each source is loudness-normalized to an independent random target
level, so mixture SNR (for whichever source is treated as target) comes out roughly symmetric around
0dB (this eval set: range -11.6 to +11.6dB). This project's OWN mixer (`data/augment.py`,
`configs/default_v2.yaml`: `snr_min=1.0, snr_max=10.0`) never once trains or evaluates the model on
a mixture where the target is quieter than the interferer — that entire regime (60.6% of this
Libri2Mix eval set, `snr_db < 1.0`) is out-of-distribution by construction of the training data, not
a property of Libri2Mix "being harder" in general.

**Splitting exactly on that boundary (snr_db >= 1.0, matching the trained snr_min) recovers the
expected result:**

| | In-distribution (SNR≥1dB, n=2361, 39.4%) | Out-of-distribution (SNR<1dB, n=3639, 60.6%) |
|---|---|---|
| Median mel S2-vs-S1 | 73.4% | -19.0% |
| Catastrophic rate | 5.7% | 40.3% |
| Median SI-SDR gain vs S1 | +2.37dB | -1.50dB |
| Median SI-SDR gain vs mixture | +2.59dB | -8.16dB |

The in-distribution slice (73.4%/5.7%) is consistent with — and SI-SDR-wise actually exceeds — the
existing test-clean headline above (77.7%/3.5%/+1.58dB), on a completely independently-constructed
corpus. The full SNR-bucket breakdown (<0dB: 45.1% catastrophic → [0,1): 18.0% → [1,3): 10.1% →
[3,5): 2.7% → [5,7): 0.8% → [7,10)+: 0.0%) is smoothly monotonic and lines up with the SAME
SNR-dependence pattern already documented above for this project's own corpus (22.1%→5.5% across
1-3dB to 7-10dB) — this is the same phenomenon extended into a harder regime Libri2Mix happens to
generate and this project's own synthetic mixer structurally cannot (its `snr_min=1.0` floor
excludes it by construction).

**Conclusion:** the hard-t0 fix's benefit is confirmed on an independent external corpus, within the
SNR range the model was actually trained for. Separately, this surfaces a second, distinct,
well-explained limitation — no training coverage below 1dB SNR (i.e., the model has never been
asked to extract the quieter of two overlapping speakers) — worth naming explicitly in the thesis as
future work (extending `snr_min` downward, potentially negative, in a future training/fine-tuning
run) rather than conflating it with the t≈0 catastrophic-outlier issue above, which is a distinct
mechanism (single-shot prediction error at the start of the flow trajectory, not an SNR-coverage
gap). Repro:
```
sbatch scripts/download_wham.sh                 # once; ~52GB WHAM noise (unused in mix_clean audio
                                                  # itself, but required by LibriMix's own generation
                                                  # script regardless of --types)
sbatch scripts/generate_libri2mix_test.sh        # after WHAM finishes; ~1.4GB output
sbatch scripts/run_eval_libri2mix_gpu.sh         # after generation finishes
python3 eval/eval_libri2mix.py --output outputs/results/eval_libri2mix_min.jsonl --summarize_only
```
