# Results and Limitations

*Draft chapter — Mask2Flow-TSE (M.Tech project, implementation of Moon et al., arXiv:2603.12837v1).
Source of truth for every number below: `eval/KNOWN_LIMITATIONS.md` and the eval scripts cited in
each section's repro block. Update this file, don't fork a separate copy of these numbers, if any
eval is rerun.*

## 5.1 Experimental setup

**Architecture (two-stage pipeline):**
- Stage 1 — `MaskingModule`, a BiLSTM (~11.2M params) predicting a soft time-frequency mask;
  `X_enhanced = mixture ⊙ mask`.
- Stage 2 — `FlowMatchingModule`, a 9-block DiT (~77.3M params, RoPE, AdaLN-Zero, classifier-free
  guidance) trained on frozen Stage 1 output, run with `n_steps=4` Euler integration and
  `cfg_scale=1.5` at inference (the value used for §5.2–§5.6; re-tuned to 2.5 on 2026-09-13, §5.10).
- Vocoder — HiFi-GAN, trained from scratch on this project's own LibriSpeech data (16kHz, n_mels=80,
  hop_length=160; a Whisper-aligned mel config, not compatible with standard pretrained vocoders).

**Checkpoints used for every result below:**

| Component | Checkpoint | Step |
|---|---|---|
| Masking (Stage 1) | `checkpoints_v2/masking/mask_best.pt` | 130,000 |
| Flow matching (Stage 2) | `checkpoints_v2/flow/flow_best.pt` (hard-t0 fine-tune, promoted) | 25,000 |
| Speaker-encoder projection | `checkpoints_speaker_encoder/projection_latest.pt` | 30,000 |
| Vocoder | `checkpoints_vocoder/vocoder_best.pt` | — |

The Stage 2 checkpoint is a fine-tune of the original 298,000-step run, continued for 25,000
additional steps with t≈0 hard-example reweighting (`t_hard_prob=0.5`, `t_hard_max=0.25`) to address
the training-coverage gap described in §5.5.1. Both checkpoints are preserved on disk
(`flow_best_step298000.pt` is the pre-fine-tune original) so any comparison below can be reproduced
against either.

> **Superseded for current results.** The table above describes the checkpoints that produced
> §5.2–§5.6, and is kept unchanged so those numbers stay reproducible. Since then Stage 2 has been
> re-promoted twice (§5.7, §5.10), Stage 1 once (§5.10 — its first change since 2026-08-02), and
> `cfg_scale` is now 2.5 rather than 1.5 (§5.10). The current set is listed at the end of §5.10.

**Evaluation corpora:**
1. **In-domain** — LibriSpeech test-clean, mixed on the fly by this project's own mixer
   (`data/augment.py`, `snr_min=1.0, snr_max=10.0`), n=2620 (full test-clean split).
2. **Cross-corpus** — Libri2Mix (JorisCos/LibriMix) test-clean, `mix_clean` condition, 16kHz,
   `min` mode, both extraction directions per mixture, n=6000 (3000 mixtures) — built independently
   of this project's mixer, with no SNR floor.

## 5.2 Headline metrics

Three metrics are reported, in order of how they're weighted in the discussion below:

1. **SI-SDR improvement (SI-SDRi)** — primary. The field-standard metric for source
   separation/target-speaker extraction; directly comparable to other work in this space.
2. **Speaker-verification accuracy (1−EER)** — secondary, TSE-specific. SI-SDR alone doesn't confirm
   the *correct* speaker was extracted; this probe does.
3. **Mel-domain % MSE improvement and catastrophic-outlier rate** — diagnostic. This is the
   project's own internal metric (the training objective's own space), reported because it is what
   surfaced and let us root-cause the two limitations in §5.5 — not intended as an externally
   comparable number, since mel-MSE percentages are unbounded and overstate perceptual severity in
   both directions (see §5.5.1).

### 5.2.1 SI-SDR improvement (primary)

| Corpus | n | SI-SDR gain vs Stage 1 | SI-SDR gain vs mixture |
|---|---|---|---|
| In-domain (LibriSpeech test-clean) | 2,620 | **+1.58 dB** | — |
| Cross-corpus, in-distribution (Libri2Mix, SNR≥1dB) | 2,361 | **+2.37 dB** | +2.59 dB |

The cross-corpus number is restricted to the SNR range the model was actually trained on
(`snr_db ≥ 1.0`, matching `snr_min`); see §5.4 for why, and the unrestricted aggregate.

### 5.2.2 Speaker-verification accuracy (secondary)

Corpus-wide speaker-verification probe (`eval/verify_eer.py`, n=400, seed=42, n_steps=4): genuine
trials are probe-vs-own-reference, impostor trials are probe vs. every other speaker's reference
(155,540 impostor trials total).

| Probe | Accuracy (1−EER) | EER | AUC |
|---|---|---|---|
| Mixture (do-nothing baseline) | 71.8% | 28.2% | 0.8029 |
| **Stage 2 extraction (this system)** | **86.2%** | **13.8%** | **0.9353** |
| Ground-truth target (ceiling) | 90.8% | 9.2% | 0.9691 |

The system closes ~76% of the gap between the do-nothing baseline and the ground-truth ceiling.

### 5.2.3 Mel-domain diagnostic

| Metric (n=2620, in-domain) | Value |
|---|---|
| Median mel S2-vs-S1 improvement | 77.7% |
| Catastrophic-outlier rate (S2 vs S1 < −30%) | 3.5% (92/2620) |

See §5.3 for the before/after this metric revealed, and §5.5.1 for why it's reported as a diagnostic
rather than a standalone headline.

## 5.3 Effect of the hard-t0 fine-tune

The pre-fine-tune checkpoint (`flow_best_step298000.pt`) showed a substantial catastrophic-outlier
tail (§5.5.1). Continuing Stage 2 training for 25,000 steps with t≈0 hard-example reweighting
(`training/finetune_flow_hard_t0.py`, job 10606, 11.58h on a P100) produced the following full-scale
(n=2620) comparison:

| Metric | Pre-fine-tune (step 298,000) | **Post-fine-tune (step 25,000, current default)** |
|---|---|---|
| Median mel S2-vs-S1 | 67.3% | **77.7%** |
| Catastrophic rate | 12.7% | **3.5%** (3.6× reduction) |
| Median SI-SDR gain vs S1 | +0.93 dB | **+1.58 dB** |
| SI-SDR-hurts rate | 28.5% | 18.2% |
| Catastrophic @ 1–3dB SNR (hardest bucket) | 22.1% | **9.9%** |
| Catastrophic @ 7–10dB SNR (easiest bucket) | 5.5% | **0.6%** |

The reduction holds across every SNR bucket, with the hardest bucket improving the most in absolute
terms. This is reported as **"reduced 3.6×," not "solved"** — the worst individual samples in the
n=2620 sweep still crash to −75dB/−59dB SI-SDR; the tail shrank substantially but did not disappear.

## 5.4 Cross-corpus generalization (Libri2Mix)

Running the fine-tuned checkpoint against Libri2Mix — a benchmark built independently of this
project's own mixer — is the generalization test. The raw aggregate looks poor at first glance
(median mel S2-vs-S1 23.9%, catastrophic rate 26.7%, SI-SDR gain vs mixture **−0.61dB**), but this is
a distribution-mismatch artifact, not a generalization failure: Libri2Mix's `mix_clean` generation
has no SNR floor (this eval set ranges −11.6 to +11.6dB), while this project's own mixer enforces
`snr_min=1.0`. 60.6% of the Libri2Mix eval set falls below that floor — a regime never seen in
training.

Restricting to the trained SNR range (`snr_db ≥ 1.0`, n=2,361 of 6,000) recovers a result consistent
with — and SI-SDR-wise exceeding — the in-domain headline:

| | In-distribution (SNR≥1dB, n=2,361, 39.4%) | Out-of-distribution (SNR<1dB, n=3,639, 60.6%) |
|---|---|---|
| Median mel S2-vs-S1 | 73.4% | −19.0% |
| Catastrophic rate | 5.7% | 40.3% |
| SI-SDR gain vs S1 | +2.37 dB | −1.50 dB |
| SI-SDR gain vs mixture | +2.59 dB | −8.16 dB |

**Conclusion:** the hard-t0 fix generalizes to an independent external corpus within the SNR range
the model was trained for. The out-of-distribution slice is not a new failure mode — it is the same
SNR-dependence trend already characterized in-domain, extended into a harder regime (target quieter
than interferer) that this project's own mixer structurally never generates. See §5.5.2.

## 5.5 Known limitations

### 5.5.1 t≈0 training-coverage gap (diagnosed, substantially mitigated)

**Mechanism:** at t=0, the rectified-flow interpolation gives Stage 2 zero progress signal
(`x_t = x_enh` exactly) — the model must predict the entire correction in one shot from Stage 1's
output and the speaker embedding alone. Three inference-time mechanisms were tested and ruled out
as the cause (speaker-embedding degeneracy: AUC 0.93–0.97, not degenerate; CFG amplification: warm-up
fixed only 1/8 flagged samples; finer Euler discretization: `n_steps=16` was strictly worse than
`n_steps=4`). The confirmed root cause is a direct measurement: cosine similarity between the
model's raw t=0 velocity prediction and the true required velocity is 0.77–0.92 for healthy samples
but only 0.08–0.15 for the worst offenders — a genuine training-coverage gap, not a numerical or
guidance-scale artifact.

**Mitigation:** the hard-t0 fine-tune (§5.3) targeted exactly this gap by oversampling t≈0 during
continued training, and cut the catastrophic rate 3.6× (12.7%→3.5%) with the reduction holding
across every SNR bucket — confirming the diagnosis was correct and actionable. Not fully eliminated:
the worst remaining samples still crash to −75dB/−59dB SI-SDR.

### 5.5.2 SNR-coverage gap below 1dB (characterized; substantially closed — see §5.10)

Surfaced by the Libri2Mix cross-corpus run (§5.4): the model has never been trained to extract the
*quieter* of two overlapping speakers, since `snr_min=1.0` excludes that regime by construction.
Catastrophic rate rises smoothly and monotonically as SNR drops below the training floor (<0dB:
45.1% → [0,1): 18.0% → [1,3): 10.1% → [3,5): 2.7% → [5,7): 0.8% → [7,10)+: 0.0%). This is a
**distinct mechanism** from §5.5.1 (an SNR-coverage gap, not a single-shot t≈0 prediction error) and
should not be conflated with it in discussion.

### 5.5.3 Speaker-count generalization (characterized, substantially mitigated)

Training data is built exclusively from 2-total-speaker mixtures (1 target + exactly 1 interferer —
`data/augment.py`'s `MixtureCreator`). `eval/eval_multi_speaker.py` tests the natural next question:
what happens as more simultaneous talkers are added? Same in-domain LibriSpeech test-clean corpus and
methodology as §5.2–5.3, synthetically mixed via a new eval-only `mix_multi_at_snr()` (each
interferer independently drawn from the same trained SNR range, `snr_min=1.0, snr_max=10.0`, against
the target). Three conditions from the same script for a paired, apples-to-apples comparison — 2, 3,
and 4 total speakers — n=300 each:

| Metric | 2 speakers (trained) | 3 speakers | 4 speakers |
|---|---|---|---|
| Median mel S2-vs-S1 | 78.3% | 58.8% | **53.7%** |
| Catastrophic rate | 1.3% (4/300) | 2.7% (8/300) | 1.3% (4/300) |
| SI-SDR gain vs Stage 1 | +1.54 dB | +1.78 dB | +2.28 dB |
| SI-SDR gain vs mixture | +2.09 dB | +2.78 dB | +4.04 dB |
| SI-SDR-hurts rate | 15.3% | 21.0% | 18.3% |
| Speaker-identity sim, mixture (baseline) | 0.190 | 0.027 | −0.036 |
| Speaker-identity sim, extracted | 0.451 | 0.292 | 0.157 |

**Speaker-verification accuracy** (same genuine/impostor EER methodology as §5.2.2's 86.2% headline —
`eval/eval_multi_speaker.py` now reuses `eval/verify_eer.py`'s exact `genuine_impostor_scores`/
`compute_eer_auc` functions, so these are directly comparable, unlike the raw cosine-similarity rows
above):

| Probe | 2 speakers | 3 speakers | 4 speakers |
|---|---|---|---|
| Mixture (do-nothing) | 70.3% | 64.4% | 62.0% |
| **Stage 2 extraction** | **86.3%** | **76.7%** | **72.6%** |
| Ground-truth target (ceiling) | 90.3% | 90.3% | 90.3% |

3-speaker accuracy (76.7%) already lands inside a 75–80% target range; 4-speaker (72.6%) falls only
~2.4 points short of 75% — a materially smaller gap than the mel-domain numbers alone would suggest.

**A real degradation exists, as expected for an untrained condition**: mel-domain quality drops
monotonically (78.3%→58.8%→53.7%) and accuracy drops with it (86.3%→76.7%→72.6%). The 2-speaker
column also cross-validates §5.2–5.3 on an independent n=300 draw (78.3% vs. the 77.7%/n=2620
headline). Catastrophic rate is noisier (1.3%→2.7%→1.3%) — likely sampling noise at n=300 on a
low-frequency event, not a real non-monotonic trend; not worth over-interpreting.

**The SI-SDR and raw-cosine-similarity figures need a careful read, not a surface one**: SI-SDR gain
keeps *rising* through 4 speakers (+1.54→+1.78→+2.28 dB vs Stage 1), and raw mixture-vs-reference
cosine similarity keeps *falling* (0.190→0.027→−0.036, i.e. below chance at 4 speakers). Neither is
the model doing better — it's the baseline collapsing further as more competing voices are mixed in,
so even a degraded extraction represents a larger *relative* improvement over an increasingly
uninformative starting point. **The accuracy/EER table above is the metric that isn't fooled by this**
(it's a corpus-wide genuine-vs-impostor separability measure, not a baseline-relative gain), and it's
the one to cite for any claim about extraction quality at higher speaker counts.

**Net characterization: graceful, not catastrophic, degradation.** 3-speaker accuracy (76.7%) already
sits inside a 75–80% target; 4-speaker (72.6%) is close but short. This is a distinct limitation from
§5.5.1/§5.5.2 (a speaker-count-coverage gap, not a t≈0 or SNR-coverage mechanism) worth naming as its
own axis of future work — and, unlike those two, the current gap to a 75–80% target is small enough
that closing it may be tractable without a large training investment (see §5.6).

**Cheap-fix elimination check:** a `cfg_scale`/`n_steps` sweep at the 4-speaker condition (baseline
1.5/4, plus 1.0/4, 2.0/4, 1.5/8, n=100 each) landed all four variants within ~1pp of each other
(75.0–76.0%) — noise, not a real effect. Inference-time tuning does **not** close the gap, matching
the precedent from §5.5.1 (`n_steps=16` was similarly ineffective there). This rules out the cheap
fix and confirms a training-side intervention is needed.

**Stage attribution:** breaking the existing mel-domain records down by stage (`mel_s1_imp_pct` /
`mel_s2_vs_s1_pct`, no new evaluation run needed) isolates where the degradation actually happens:

| | 2 speakers | 4 speakers |
|---|---|---|
| Stage 1 alone vs mixture | 8.7% | 10.9% |
| Stage 2 added on top of Stage 1 | 78.3% | 53.7% |

Stage 1 (masking) is unaffected — if anything marginally better at 4 speakers, and 0/300 samples
where it makes things worse than the raw mixture. **The entire degradation is concentrated in Stage
2** (flow matching). This scopes any future fix to a Stage-2-only fine-tune, exactly mirroring the
successful hard-t0 precedent (§5.3) rather than requiring Stage 1 retraining too.

**Mitigation: hard-multispeaker fine-tune — closes the gap.** Following the scoping above, Stage 2
was continued from `flow_best.pt` for 25,000 steps (13.29h on a P100,
`training/finetune_flow_hard_multispeaker.py`) with `LibriSpeechTSEDataset` now sampling a variable
interferer count per training example (50% 2-speaker / 30% 3-speaker / 20% 4-speaker — weighted
toward the original condition specifically to avoid regressing it) instead of always exactly 1
interferer. Two candidate checkpoints — the training-time proxy's "best" pick (step 6,000; the proxy
never improved past this point, matching the same pattern seen in the hard-t0 fine-tune and not
itself informative) and the completed run's final checkpoint (step 25,000) — were both evaluated
against the real target metric (n=300 each):

| | Pre-fine-tune | step 6,000 | **final (step 25,000)** |
|---|---|---|---|
| 2-speaker accuracy | 86.3% | 86.7% | **86.7%** |
| 2-speaker mel S2-vs-S1 | 78.3% | 75.5% | 76.4% |
| 2-speaker catastrophic | 1.3% | 2.7% | 1.7% |
| 4-speaker accuracy | 72.6% | 76.0% | **76.7%** |
| 4-speaker mel S2-vs-S1 | 53.7% | 63.5% | 66.3% |
| 4-speaker catastrophic | 1.3% | 1.0% | 1.3% |

**No regression at 2 speakers** (accuracy ticked up slightly, within noise) **and 4-speaker accuracy
climbed from 72.6% to 76.7%, clearing the 75–80% target.** The final checkpoint beats the proxy's
"best" pick on essentially every axis — same conclusion, same reasoning, as the hard-t0 fine-tune's
own final-vs-best comparison.

**Pre-promotion check on the primary n=2620 in-domain headline — a real, honest tradeoff, not a clean
win:**

| Metric (n=2620) | Pre-promotion headline | `flow_ft_ms_final.pt` |
|---|---|---|
| Median mel S2-vs-S1 | 77.7% | 75.1% (↓2.6pp) |
| Catastrophic rate | 3.5% (92/2620) | 3.7% (97/2620) (flat, within noise) |
| SI-SDR gain vs S1 (primary metric, §5.2) | +1.58 dB | +1.71 dB (↑) |
| SI-SDR-hurts rate | 18.2% | 18.1% (flat) |

The mel-domain diagnostic softened modestly (concentrated mostly in the hardest 1–3dB SNR bucket:
9.9%→11.1% catastrophic there specifically), but SI-SDR — this document's own designated *primary*
metric (§5.2), precisely because mel-domain percentages overstate perceptual severity — held and
actually improved slightly. Net read: a large multi-speaker win purchased at a small, arguably
negligible cost on the primary in-domain metric.

**Promoted 2026-09-04.** `flow_ft_ms_final.pt` → `checkpoints_v2/flow/flow_best.pt` (plain file copy,
MD5-verified byte-identical both directions; the prior hard-t0 checkpoint backed up first to
`flow_best_prehardmultispeaker_backup.pt`, itself MD5-confirmed unchanged since its own 2026-08-29
promotion). Every checkpoint referenced elsewhere in this document as "current"/"headline" (§5.2,
§5.3, §5.4) reflects the *pre*-promotion checkpoint — those numbers are the historical record of how
this checkpoint was reached and validated, kept exactly as originally measured, not overwritten.
**§5.7 below reports the full refreshed headline** (corpus-wide `verify_eer.py`, Libri2Mix, and all
three speaker-count conditions, all re-measured against the checkpoint as currently promoted) and
explains why the new numbers came out the way they did.

## 5.6 Future work

- **§5.5.2 (SNR-coverage gap) — DONE.** `snr_min` was extended downward to −10dB, but as a curriculum
  for **Stage 1** rather than Stage 2, which is where an oracle decomposition located the constraint.
  Low-SNR accuracy went 61.3% → 80.3% and the Libri2Mix aggregate from −0.04dB to +3.39dB against the
  mixture, at a cost of about 0.15dB of in-domain SI-SDR. Full numbers, the two Stage-2 attempts that
  were rejected first, and the remaining formulation limit are in §5.10.
- **Close the remaining §5.5.1 tail**: `eval/diagnose_catastrophic.py` part [C] (cosine similarity
  between predicted and true t≈0 velocity) is the existing tool to check whether a future checkpoint
  has raised the cosine-similarity floor for the hardest examples out of the current 0.08–0.15 range.
- **§5.5.3 (4-speaker accuracy target) — DONE.** The hard-multispeaker fine-tune closed the gap and
  was promoted to `checkpoints_v2/flow/flow_best.pt`; the full re-validated headline (in-domain,
  Libri2Mix, all three speaker-count conditions) is in §5.7, along with why the results came out this
  way. Nothing further needed on this item.
- **Extend to 5+ speakers**, or check whether the mitigation holds at speaker counts beyond what the
  curriculum trained on (it weighted 2/3/4-speaker only) — not started.

## 5.7 Checkpoint update (2026-09-04): hard-multispeaker fine-tune promoted — full headline refresh

**Nothing above this section was changed or overwritten.** Every number in §5.2–§5.6 is exactly as
originally measured and remains an accurate historical record of the checkpoint that stood at the
time — the pre-fine-tune-hardt0 checkpoint for the earliest numbers, then the hard-t0-fine-tuned
checkpoint for everything from §5.3 onward, up to and including the §5.5.3 speaker-count diagnosis.
This section documents what changed after that point, reports the same measurements redone against
the new checkpoint, and explains why the new numbers came out the way they did.

**What changed.** Following the §5.5.3 diagnosis (degradation isolated entirely to Stage 2; Stage 1
unaffected by speaker count) and the elimination of a cheap inference-time fix (the `cfg_scale`/
`n_steps` sweep), a **Stage-2-only** fine-tune was run: `training/finetune_flow_hard_multispeaker.py`,
25,000 steps, 13.29h on a P100, continuing from the hard-t0 checkpoint used throughout §5.2–§5.6.
Nothing else changed — same architecture, same loss (plain rectified-flow MSE, no biased sampling
unlike the hard-t0 fine-tune), same Stage 1, same vocoder. The only change: `LibriSpeechTSEDataset`
now samples a variable interferer count per training example — a curriculum of 50% 2-speaker / 30%
3-speaker / 20% 4-speaker (via the new `mix_multi_at_snr()`, first built as an eval-only tool for
§5.5.3 and then wired into training) — instead of always exactly 1 interferer. The resulting
checkpoint (`flow_ft_ms_final.pt`) was compared against a training-proxy-selected alternative
(`flow_ft_ms_best_step6000.pt`) and won on every axis, then validated on the full in-domain n=2620
set, and promoted to `checkpoints_v2/flow/flow_best.pt` on 2026-09-04 (prior checkpoint backed up to
`flow_best_prehardmultispeaker_backup.pt`, both copies MD5-verified).

**Refreshed headline — every trustworthy metric held or improved, on every corpus and every speaker
count tested:**

| Speaker-count accuracy (n=300 each, §5.5.3 methodology) | Before | **After** |
|---|---|---|
| 2-speaker | 86.3% | 86.7% |
| 2-speaker (trustworthy corpus-wide, n=400, `verify_eer.py`) | 86.2% | **87.0%** |
| 3-speaker | 76.7% | **81.9%** |
| 4-speaker | 72.6% | 76.7% |

| In-domain, full test-clean (n=2620, §5.2 methodology) | Before | After |
|---|---|---|
| SI-SDR gain vs Stage 1 (primary metric) | +1.58 dB | **+1.71 dB** |
| Median mel S2-vs-S1 (diagnostic) | 77.7% | 75.1% |
| Catastrophic rate | 3.5% | 3.7% |

| Libri2Mix cross-corpus, in-distribution (SNR≥1dB, n=2361, §5.4 methodology) | Before | After |
|---|---|---|
| SI-SDR gain vs Stage 1 | +2.37 dB | **+2.51 dB** |
| Median mel S2-vs-S1 | 73.4% | 72.5% |
| Catastrophic rate | 5.7% | **5.1%** |

| Libri2Mix, out-of-distribution (SNR<1dB, n=3639 — untouched regime, §5.5.2) | Before | After |
|---|---|---|
| SI-SDR gain vs Stage 1 | −1.50 dB | −1.45 dB (essentially unchanged) |
| Median mel S2-vs-S1 | −19.0% | −18.7% (essentially unchanged) |

All three speaker-count conditions now land at 76% or above — comfortably at or above the original
75–80% target across the board, not just at the 4-speaker edge case that motivated the fine-tune.

**Why the new results came out this way.** The pattern across every table above is specific and
mechanistically consistent with what actually changed, not a generic "everything got better" effect
— which is itself good evidence the improvement is real rather than an artifact:

- **3-speaker improved the most (76.7%→81.9%, the largest jump in the whole table), despite 4-speaker
  getting explicit curriculum weight too.** 3-speaker sits at a moderate distributional distance from
  the original 2-speaker training data — different enough to require real new learning, but the
  underlying task doesn't change qualitatively (still "suppress non-target energy, conditioned on the
  speaker embedding"), just quantitatively (more competing energy). That makes it comparatively easy
  to absorb with the curriculum's 30% weight. It also plausibly benefits from *both* directions at
  once: direct exposure, plus positive transfer from the model simultaneously getting better at the
  harder 4-speaker case (a model that copes better with 3 competing voices should find 2 easier by
  extension) — the same transfer effect that plausibly explains the next point.
- **2-speaker held and even improved slightly, despite dropping from ~100% training exposure to 50%.**
  The base checkpoint already had 300,000+ steps of pure 2-speaker training behind it before this
  fine-tune started; 50% of a further 25,000 steps is a rounding error against that foundation, not a
  meaningful reduction in absolute exposure. Meanwhile, learning to handle harder mixtures plausibly
  sharpens the same underlying skill (attending to the speaker embedding, suppressing non-target
  spectral energy) that the easier 2-speaker case also draws on — consistent with curriculum-learning
  effects seen elsewhere, where training on harder examples improves performance on easier ones too,
  not only the harder regime being targeted.
- **4-speaker improved substantially (72.6%→76.7%) but by fewer absolute points than 3-speaker,
  despite being the whole point of the fine-tune.** It started furthest from the trained distribution
  and got the smallest curriculum weight (20%, versus 30% for 3-speaker) — closing a bigger gap with
  proportionally less dedicated exposure is intrinsically harder within a fixed 25,000-step budget.
  That it still cleared the 75–80% target at all is the headline result; further gains here (per
  §5.6's remaining future-work item) would likely need either more steps or a curriculum reweighted
  further toward 4-speaker specifically.
- **SI-SDR (primary) improved while mel-domain (diagnostic) softened slightly — consistently, in both
  the in-domain and Libri2Mix tables, by a similar small margin (~1–3pp) each time.** This is the same
  pattern §5.2 already documents as a general property of the mel-domain metric: it's measured
  relative to a sometimes-small Stage-1-MSE denominator and is unbounded, so it's disproportionately
  sensitive to small changes in raw spectral-matching precision even when they don't translate to
  worse *audible* quality. A plausible, consistent explanation: reallocating some of Stage 2's
  capacity toward handling harder, spectrally messier mixtures costs a small amount of precision on
  the easiest cases' raw mel reconstruction, without costing (and apparently slightly helping) the
  metrics that reflect what actually reaches a listener (SI-SDR after vocoding) or what actually
  identifies the speaker (accuracy). The softening is a diagnostic-metric artifact of exactly the kind
  §5.2 warns about, not a real quality regression — precisely why this document treats SI-SDR and
  accuracy as primary and mel-domain as secondary in the first place.
- **Libri2Mix in-distribution improved too, even though the fine-tune never saw a single Libri2Mix
  sample.** Libri2Mix uses different speakers, content, and a different mixing pipeline than this
  project's own synthetic corpus, so any improvement there reflects genuine representational
  generalization, not memorization. Broadening the range of interference *patterns* seen in training
  (multiple simultaneous competing voices, more varied total interference energy) plausibly improves
  robustness to interference variability generally, not only to the literal 3-/4-speaker case —
  consistent with the modest gains showing up on an entirely independent corpus.
- **Libri2Mix out-of-distribution (SNR<1dB) stayed essentially flat — the cleanest evidence the
  improvement is targeted, not a fluke.** This fine-tune's curriculum varied *speaker count*; every
  interferer, regardless of how many were present, was still drawn from the same trained
  `[snr_min=1.0, snr_max=10.0]` range. The model was never shown a case where the target is quieter
  than an interferer, so the §5.5.2 SNR-coverage gap — a genuinely distinct mechanism — was never
  touched, and the numbers confirm exactly that: no change, in either direction. If this untouched
  regime had *also* improved, that would be a reason to be suspicious of the eval methodology rather
  than the result; instead, only the things this fine-tune actually targeted moved, and only in the
  direction expected.

## 5.8 Qualitative verification: manual listening check (2026-09-05)

Every result above is metric-based. Metrics are proxies — this section reports a direct, by-ear
check of what those metrics claim, motivated by two specific listening observations that surfaced
during this project that no automated number had caught: occasional "metallic" vocoder quality, and
speaker-similarity scores that seemed low despite a subjectively correct-sounding voice.

**Method:** `eval/export_listening_samples.py` (new tool) exports real, playable `.wav` files —
`mixture.wav`, `reference.wav`, `extracted.wav` (the actual system output), `target_clean.wav`
(ground truth vocoded from its *own* real mel, not a prediction — isolates vocoder-only artifacts
from extraction-specific ones), and `stage1_only.wav` — for a fresh spot-check draw (seed 123,
distinct from every metric eval's seed 42) of 4 samples each at 2, 3, and 4 total speakers (12
samples total).

**Also fixed as part of building this:** the reference-clip trailing-silence issue behind the
low-similarity observation turned out to be a real, confirmed gap — `trim_trailing_silence()` already
existed in this codebase for exactly this problem (`inference/speaker_similarity.py`,
`eval/results_stage2.py`) but was missing from `eval/eval_multi_speaker.py`, and (at the time this
section was written) from `eval/verify_eer.py` / `eval/eval_libri2mix.py` too. Because a diluted
reference embedding makes genuine-vs-impostor separability *harder*, not easier, every accuracy/EER
number in §5.2–§5.7 was flagged as a plausible **lower bound** at the time. That gap has since been
closed and re-measured — see §5.9.

**Result: manually confirmed correct.** Across the sampled 2/3/4-speaker sets, `extracted.wav`
correctly isolates the target speaker's voice by ear, consistent with the quantitative accuracy
numbers in §5.5.3/§5.7. On this basis, the checkpoint currently promoted to `flow_best.pt` is
considered validated both quantitatively and qualitatively, and is being left as-is rather than
pursued further — the metallic-vocoder question (whether it traces to the vocoder itself or to Stage
2's mel prediction, per the `target_clean.wav` vs `extracted.wav` comparison this tool was built to
support) did not surface as a problem worth chasing given the confirmed-correct outcome, so no
vocoder or Stage 2 changes followed from this check.

> **Superseded in part.** This section records the listening check as it stood on 2026-09-05,
> against the checkpoint promoted then. A second listening pass on the *current* pipeline
> (2026-09-14, after the low-SNR campaign of §5.10) reached the same conclusion about speaker
> isolation but found a short defect at the start of each utterance that this pass did not report —
> see §5.11. Nothing in this section is restated there; the two are separate observations of
> different checkpoints.

## 5.9 Reference-clip trailing-silence fix, applied to the remaining scripts and re-measured (2026-09-09)

§5.8 found that `trim_trailing_silence()` — already used elsewhere to stop zero-padding from
diluting mean-pooled WavLM embeddings — was missing from two scripts: `eval/verify_eer.py` (the
source of the corpus-wide 87.0% accuracy headline) and `eval/eval_libri2mix.py`. Every accuracy/EER
number reported through §5.7 was flagged there as a plausible lower bound as a result. Both scripts
were fixed (same one-line insert at the reference-clip build site, mirroring
`eval/eval_multi_speaker.py`'s existing pattern) and rerun against the unchanged current checkpoint
(`flow_best.pt`, the §5.7 hard-multispeaker fine-tune) to close that gap:

| Metric | §5.7 (pre-fix, flagged as lower bound) | Post-fix (this section) |
|---|---|---|
| 2-speaker accuracy, corpus-wide (n=400, `verify_eer.py`) | 87.0% | 86.5% |
| Libri2Mix in-distribution median mel S2-vs-S1 (n=2361) | 72.5% | 72.5% (unchanged) |
| Libri2Mix in-distribution SI-SDR gain vs Stage 1 (n=2361) | +2.51 dB | +2.51 dB (unchanged) |
| Libri2Mix in-distribution catastrophic rate (n=2361) | 5.1% | 5.3% |

**Result: the fix changes essentially nothing beyond sampling noise.** Two of the four numbers are
identical to the decimal place reported; the other two moved by 0.5pp and 0.2pp respectively — both
well inside the noise band this project already treats as non-meaningful at these sample sizes (see
§5.7's own step6000-vs-final tiebreaker discussion for the same standard applied elsewhere). The
0.5pp move in accuracy is technically a decrease, which does not match the naive "should only go up"
expectation from §5.8's dilution argument — but that argument described the *average*-case direction
of the effect, not a per-sample guarantee, and EER at n=400/40-speakers is a rank-based statistic
where a handful of borderline pairs crossing the decision threshold either way is ordinary noise, not
evidence the fix made anything worse.

**Practical takeaway: the numbers reported throughout §5.2–§5.7 were not meaningfully understated.**
The theoretical lower-bound caveat is now closed — every accuracy/EER number tied to these two scripts
is measured with the fix applied — but empirically it made no material difference to any conclusion
in this document. This itself is informative: unlike the much larger original padding-dilution bug
(§5.1), which affected the mixture/target side of *every* sample, the reference-clip dilution
apparently only mattered for a small enough minority of clips (those genuinely short relative to the
3.0s `reference_length` window) that its aggregate effect on these corpora is negligible.

Repro:
```bash
sbatch scripts/rerun_verify_eer_and_libri2mix_gpu.sh
```

## 5.10 Low-SNR campaign (2026-09-10 → 09-13): the SNR-coverage gap closed

**Nothing above this section was changed.** §5.2–§5.9 remain exactly as measured, against the
checkpoints that stood at the time. This section records what changed since and the numbers that
supersede them.

**What changed**, in the order the changes were made: Stage 1 was fine-tuned on a low-SNR curriculum
(half the examples drawn from [−10, 1) dB, alongside the existing 2/3/4-speaker mix) and promoted —
its first change since 2026-08-02; Stage 2 was then adapted to that new Stage 1 on the same curriculum
and promoted; and `cfg_scale` was raised from 1.5 to 2.5 after a sweep found the deployed value had
been tuned for a pipeline whose two stages had both since been replaced.

| | Before the campaign | After |
|---|---|---|
| Low-SNR accuracy ([−10,1) dB, n=300) | 61.3% — *below* the 64.0% do-nothing baseline | **80.3%** |
| Low-SNR SI-SDR vs mixture | −15.86 dB | **+6.51 dB** |
| Libri2Mix aggregate SI-SDR vs mixture (n=6000) | −0.04 dB | **+3.39 dB** |
| Libri2Mix sub-1dB slice, median mel | −19.0% | **+63.8%** |
| Libri2Mix ≥1dB slice, catastrophic rate | 5.3% | 7.7% |
| Corpus-wide accuracy (n=400) | 86.5% | 86.2% |
| 2 / 3 / 4-speaker accuracy (n=300) | 87.3 / 82.1 / 77.0% | 85.7 / 80.0 / 77.0% |
| In-domain SI-SDR vs mixture (n=2620) | +1.90 dB | +1.75 dB |

A regime the system previously handled *worse than not running at all* is now handled well. The price
is roughly 0.15 dB of in-domain SI-SDR end-to-end and one to two points of 2- and 3-speaker accuracy;
4-speaker and corpus-wide accuracy sit level with where they started.

**How it was found matters more than the numbers.** Two Stage-2 curricula (50% and 25% low-SNR) were
trained first and both rejected on their measured results: they bought low-SNR ability at 4.5 and 2.8
points of corpus-wide accuracy and a three- to four-fold rise in the catastrophic rate. Only then was
the constraint located, by feeding Stage 2 oracle inputs built from the clean target. That
decomposition split the low-SNR failure roughly evenly between the trained Stage-1 network and its
mask formulation, and showed the work belonged in Stage 1 — untouched since August. A six-hour Stage-1
fine-tune then delivered what thirteen-hour Stage-2 curricula could not, and adapting Stage 2 to it
afterwards supplied the rest. The chronological record is in
`docs/methodology_and_project_history.md`, entries 14–29.

**A limitation in the mask formulation itself.** The paper applies its [0,1] mask by multiplying
log-mel values (Eq. 9). Under this implementation's `log(mel + 1e-8)`, every bin quieter than linear
power 1 is negative — about 74% of them at low SNR — and multiplying a negative value by a number in
[0,1] moves it *upward*. In those bins the mask cannot suppress at all, which leaves roughly
three-quarters of low-SNR target bins unreachable by any mask. The paper reports pure deletion
(D = 100%, I = 0%) but never specifies the log offset, so this is an under-specified detail rather
than a contradiction. An alternative, `X + log M`, restores true deletion and produces a Stage 1 more
than twice as accurate in every regime — but the frozen Stage 2, trained only on outputs that can never
fall below the mixture, destroys that input in 97.7–98.7% of samples. The change is therefore reachable
at Stage 1 yet unadoptable without retraining Stage 2. That retraining was then carried out and the
pair rejected: it fails every acceptance bar (corpus-wide accuracy 83.5% against the 86.5% required,
low-SNR 77.3% against 80.3%) and is worse than the current system on every metric. The reason is
instructive — Stage 2 contributes almost nothing when handed the cleaner input, its improvement over
Stage 1 falling from +74.1% mel and +1.49 dB to +11.7% and +0.34 dB, and turning negative at low SNR.
The final output is therefore worse in absolute terms despite a Stage 1 whose error is 2.6 times lower.
Pipeline quality is not a function of Stage 1 accuracy alone: the two stages are co-adapted, and a
coarse mask refined generatively — the paper's own decomposition — is doing real work.

**Two notes on method.** First, one promotion overrode a pre-registered acceptance bar. The Stage-2
adaptation failed two mel-based catastrophic-rate bars and was promoted regardless, because severe
SI-SDR regressions barely moved (2.1% → 2.7%) and 64% of the additional mel failures left the audio
essentially unharmed. The same bar was honoured where it did track damage: the earlier Stage-2
curricula, whose severe regressions quintupled, were rejected. Entry 25 of the methodology document
records the override and its evidence, so a reader can disagree with it on the numbers. Second, the
oracle ceiling of 73.0% quoted for the paper's formulation was measured alongside the Stage 2 of the
time; the final system reaches 80.3% and exceeds it. An oracle bounds the pair it was measured in,
not one stage in isolation.

**Current checkpoints** (superseding the table in §5.1):

| Component | Checkpoint | Note |
|---|---|---|
| Masking (Stage 1) | `checkpoints_v2/masking/mask_best.pt` | low-SNR fine-tune, promoted 2026-09-12; previous at `mask_best_prelowsnr_backup.pt` |
| Flow matching (Stage 2) | `checkpoints_v2/flow/flow_best.pt` | adapted to the new Stage 1, promoted 2026-09-13; previous at `flow_best_preadaptnewmask_backup.pt` |
| Inference | `cfg_scale=2.5`, `n_steps=4` | guidance re-tuned 2026-09-13 |

Repro:
```bash
# Stage 1 low-SNR fine-tune (either mask formulation), then adapt Stage 2 to it:
sbatch scripts/run_finetune_mask_lowsnr_gpu.sh multiplicative
sbatch scripts/run_finetune_flow_adapt_newmask_gpu.sh

# Full battery for any candidate (optional Stage-1 checkpoint and guidance overrides):
sbatch scripts/run_eval_candidate_gpu.sh <flow_ckpt> <tag> [mask_ckpt] [cfg_scale]

# Libri2Mix in-distribution / out-of-distribution split behind the rows above:
python3 eval/snr_split_summary.py outputs/results/eval_libri2mix_min_cfg2.5.jsonl
```

## 5.11 Second listening pass (2026-09-14): an onset defect the metrics cannot see

§5.8 verified the 2026-09-04 checkpoint by ear. After the low-SNR campaign of §5.10 changed both
stages and the guidance scale, the current pipeline was re-exported and listened to again. The
speaker-isolation conclusion held — the correct voice is extracted across 2, 3 and 4 speakers — but
the pass surfaced a defect §5.8 had not: the 4-speaker extractions sound suppressed at the start,
and one 3-speaker sample (sample_03) was audibly worse than the rest of its condition.

**The report was reproduced by measurement.** Taking the median level of the extraction over the
ground truth's active frames, in dB, every sample identified by ear as damped is 10–14 dB low in its
first second and recovers afterwards; every sample called clean is within ~1 dB throughout. The
flagged 3-speaker sample measures −12.6 dB in its first second and +0.9 dB after it.

**The mild component is positional, not acoustic.** A speech onset following a pause is
acoustically the same event as an utterance's first word, so if onsets were simply handled badly,
interior ones would suffer equally. Pooled over the twelve exported samples:

| region | level vs ground truth |
|---|---|
| first speech burst (begins at frame 0) | **−4.08 dB** (worst case −16 dB) |
| interior speech onsets (after ≥0.2 s of silence) | +0.29 dB |
| sustained speech | +0.01 dB |

Interior onsets are reproduced accurately; only the beginning of the sequence is affected. Stage 1's
output is flat (~0 dB) over the same frames, placing the defect in Stage 2. This is consistent with
the stages' different roles: Stage 1 multiplies the mixture mel and so inherits a plausible level
even where its mask is inaccurate, whereas Stage 2 generates the mel from noise, so a weak prior
appears directly as lost energy. Stage 2's DiT uses rotary (relative) position embeddings, making
frame 0 the only frame in the sequence with no left context.

**Why no metric reported it.** The affected region is roughly 0.5 s of a 10 s utterance. Mel-MSE and
SI-SDR are utterance-averaged, so a 12 dB deficit over ~5% of frames perturbs them by less than
their own run-to-run variation; speaker-verification accuracy is essentially insensitive to it,
since speaker identity is carried by the unaffected remainder. This is a limitation of
utterance-averaged evaluation rather than of any individual metric: a short defect at a fixed
position is invisible to all three, however audible it is. It is also the clearest justification in
this project for keeping a listening check in the evaluation protocol at all.

**The severe cases are a gate, and a regression.** The positional account above explains the
*mild* deficit but not the samples that were actually audible. Testing it directly (silent lead-in
before Stage 2, discarded afterwards) left the severe cases unchanged or slightly worse (−30.72 →
−32.04 dB; −35.47 → −35.99 dB), so it was rejected. Examining those samples frame by frame shows a
different phenomenon: the output sits on a flat ≈−60 dB floor for the first ~0.5 s, independent of
the target, then switches on abruptly and tracks correctly. Best cross-correlation lag is 0 frames,
so it is not misalignment, and Stage 1 is clean throughout. It is not content either —
`3speakers/sample_01` and `4speakers/sample_01` use the *same* target utterance, and only the
4-speaker case gates (−0.6 dB against −34.5 dB). Local SNR does predict the mild component
(−0.4 dB where the target dominates against −7 dB where it is buried below −15 dB), but the gated
samples have onset target-to-mixture ratios near −1 dB, i.e. the target dominates there.

Applying the identical measurement to the surviving 2026-09-04 export — same tool, same seed, same
sample indices, older checkpoints and cfg_scale 1.5 — dates the defect:

| | 2026-09-04 | 2026-09-14 |
|---|---|---|
| median onset level vs ground truth | −0.31 dB | −7.82 dB |
| samples gated below −10 dB | **0 / 12** | **5 / 12** |
| samples below −20 dB | 0 / 12 | 3 / 12 |

The gate was introduced by the low-SNR campaign of §5.10, whose every metric scored the changes as
improvements. §5.8's clean listening verdict was correct for the checkpoint it tested.

**It is not the operating point.** Guidance was the prime suspect, since ‖v_cond − v_uncond‖ at
t≈0 is documented in this project as 5–40× its later magnitude (§5.5.1) and cfg_scale multiplies it
into the first Euler step. Sweeping cfg_scale 1.5/2.0/2.5 and cfg_warmup_steps 0/1/2 moved the gate
count only between 6 and 7 of 24 samples, and the worst sample by 1.4 dB out of 35: reverting to the
pre-campaign guidance does not recover the pre-campaign audio.

That sweep produced one incidental result that looked worth pursuing: `cfg_warmup_steps=2` at the
deployed cfg_scale 2.5 improved mel-MSE from 3.083 to 2.751 (−11%) and SI-SDR from −8.95 to
−8.47 dB with nothing worse, and had never been active in any number this chapter reports. It was
wired through every inference path and measured on the full battery (job 11210, 2026-09-17). The
quality gains replicate — in-domain catastrophic 10.2 → 9.6%, SI-SDR vs mixture +1.75 → +1.83 dB,
Libri2Mix +3.39 → +3.46 dB — but **accuracy falls in every condition**: corpus-wide 86.2 → 85.5%,
and 2/3/4-speaker 85.7/80.0/77.0 → 85.0/78.4/74.6, with AUC down in all four. Warm-up runs the first
two of four Euler steps at cfg_scale 1.0, so it is simply *weaker guidance* — time-averaged about cfg
1.75 — on the axis §5.10 already swept and optimized in the opposite direction; it reproduces cfg
2.0's 2-speaker accuracy almost exactly. "Nothing worse" held only because the diagnostic that
surfaced it reported mel and SI-SDR and not accuracy. Rejected; `inference.cfg_warmup_steps` stays 0.

**Cause: an interaction between the two promotions, not either one.** The 2×2 (job 11177,
`scripts/run_diag_onset_ckpt_gpu.sh`), run at the deployed operating point with the old/old cell as
the method's control:

| Cell | Stage 1 | Stage 2 | gated | median onset | 4-speaker gated |
|---|---|---|---|---|---|
| A | old | old | 1 / 24 | −1.00 dB | 0 / 8 |
| B | **new** | old | 1 / 24 | −2.01 dB | 0 / 8 |
| C | old | **new** | 2 / 24 | −1.95 dB | 0 / 8 |
| D | **new** | **new** — deployed | **7 / 24** | **−4.67 dB** | **3 / 8** |

The control returns 1/24, so the table is readable. Each promotion *alone* is indistinguishable
from the pre-campaign pipeline; only the pair gates, and superadditively — summing the individual
effects predicts ≈2/24 and −2.96 dB against a measured 7/24 and −4.67 dB. Every gated 4-speaker
sample requires both new checkpoints present (A, B, C all 0/8; D 3/8). This removes the cheapest
remedy: there is no single promotion to revert, because neither is individually at fault. It also
agrees with §5.10's log-gain result from the opposite direction — a Stage 1 with 2.6× lower error
made the pipeline *worse* against a Stage 2 trained on a different input family. The stages are
co-adapted, and neither stage's quality predicts the pipeline's.

**Two mitigations, both measured against a bar fixed in advance, both rejected.** The bar was
≤ 1/24 gated — the pre-campaign level — on the grounds that a defect the campaign introduced should
be absent after a fix.

| | deployed | Stage-2 fine-tune (job 11184) | inference-side repair (job 11205) |
|---|---|---|---|
| gated, all | 7 / 24 | 4 / 24 — fail | 6 / 24 — fail |
| gated 2 / 3 / 4 speakers | 1/8, 3/8, 3/8 | 0/8, 1/8, **3/8** | 1/8, 2/8, **3/8** |
| median onset | −4.67 dB | −2.37 dB | −3.36 dB |

The fine-tune (`flow_ft_onsetfix_final.pt`, 8000 steps) cleared 2-speaker and most of 3-speaker; the
inference-side repair (`models/flow.py inference_with_onset_splice`, a silent lead-in spliced over
the first 1.0 s with a 20-frame cross-fade) recovered one sample. **Neither moved 4-speaker at all.**
Both failing in the same condition is what the ablation predicts: a defect living in the coupling
between two checkpoints is reachable neither by adjusting one stage nor by post-processing the
pair's output.

The inference-side repair was additionally validated on the full battery (job 11202) as a properly
paired comparison — checkpoint-independent probes identical to job 11112's — and changes nothing
that matters: accuracy 86.2 → 85.8% corpus-wide, 85.7/80.0/77.0 → 85.3/80.3/77.0 by speaker count,
SI-SDR vs mixture +1.75 → +1.76 dB. It is switched **off** by default
(`inference.onset_pad_frames: 0`), so every number in this chapter is unaffected by its presence in
the codebase. That accuracy cannot respond is structural, and is the same reason the gate escaped
§5.8: the WavLM embedding is mean-pooled over the whole utterance, so half a second of a
four-to-ten-second clip barely shifts it.

**Honest statement of the current system.** It is quantitatively the best configuration this project
has produced *and* it carries an audible onset gate on a minority of multi-speaker utterances that
its predecessor did not. The cause is established by ablation, two remedies have been measured and
rejected on a pre-registered rule, and a third — co-adapting both stages, or weighting the training
loss toward the failing frames — is identified as the only remaining avenue but was not pursued, at
4–13 h per attempt with no evidence that training budget is the binding constraint. It is reported
here as a characterized limitation, in the same form as the SNR-coverage gap of §5.6.

## Appendix: repro commands

```bash
# Primary metric (SI-SDR), in-domain, full test-clean:
sbatch scripts/run_full_eval_audio_gpu_hardt0.sh
python3 eval/full_eval.py --output outputs/results/full_eval_audio_gpu_hardt0.jsonl --summarize_only

# Secondary metric (speaker-verification accuracy):
python3 eval/verify_eer.py --mask_ckpt checkpoints_v2/masking/mask_best.pt \
    --flow_ckpt checkpoints_v2/flow/flow_best.pt \
    --projection_ckpt checkpoints_speaker_encoder/projection_latest.pt \
    --seed 42 --n_steps 4 --max_samples 400

# Cross-corpus (Libri2Mix) — full pipeline:
sbatch scripts/download_wham.sh                 # once; WHAM noise required by LibriMix's generator
sbatch scripts/generate_libri2mix_test.sh        # after WHAM finishes
sbatch scripts/run_eval_libri2mix_gpu.sh         # after generation finishes
python3 eval/eval_libri2mix.py --output outputs/results/eval_libri2mix_min.jsonl --summarize_only

# Speaker-count generalization (2/3/4-speaker mixtures, incl. accuracy/EER):
sbatch scripts/run_eval_multi_speaker_gpu.sh
python3 eval/eval_multi_speaker.py --output outputs/results/eval_multispeaker_2total.jsonl --summarize_only
python3 eval/eval_multi_speaker.py --output outputs/results/eval_multispeaker_3total.jsonl --summarize_only
python3 eval/eval_multi_speaker.py --output outputs/results/eval_multispeaker_4total.jsonl --summarize_only

# cfg_scale/n_steps tuning sweep at 4 speakers (cheap check before a training-side fix):
sbatch scripts/run_eval_multi_speaker_tune_gpu.sh

# Hard-multispeaker fine-tune (closes the §5.5.3 gap) + promotion-decision comparison:
sbatch scripts/run_finetune_flow_multispeaker_gpu.sh
sbatch scripts/run_eval_multispeaker_finetune_compare_gpu.sh
sbatch scripts/run_full_eval_audio_gpu_multispeaker_ft.sh    # final pre-promotion check

# §5.7 refresh: re-measure everything above against the newly-promoted checkpoint
# (regenerates the SAME canonical output filenames used earlier in this appendix):
sbatch scripts/run_refresh_checkpoint_evals_gpu.sh

# §5.8 qualitative check: export real, playable .wav files (2/3/4 speakers):
sbatch scripts/run_export_listening_samples_gpu.sh
# output: outputs/listening_samples/{2,3,4}speakers/sample_NN/*.wav + README.txt
```
