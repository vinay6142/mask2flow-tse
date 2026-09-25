---
name: mask2flow-tse-lowsnr-gap
description: "Closing the SNR-coverage gap (the last open limitation) in Mask2Flow-TSE — diagnosis done, fine-tune built 2026-09-10, not yet run"
metadata: 
  node_type: memory
  type: project
  originSessionId: 314006b1-f0de-4a42-b9fe-887c5480a146
  modified: 2026-09-14T08:18:16.201Z
---

Part of [[mask2flow-tse-overview]]. The third and last characterized limitation (see
[[mask2flow-tse-eval-results]]): the mixer's `snr_min=1.0` floor means the model has NEVER trained on
a mixture where the target is quieter than the interferer. 60.6% of Libri2Mix falls in that regime.

**Stage attribution done 2026-09-10 (zero GPU cost, from the existing `eval_libri2mix_min.jsonl`
via a local Node script splitting `mel_s1_imp_pct`/`mel_s2_vs_s1_pct` by `snr_db`).** Result differs
importantly from the speaker-count case:

| SNR | S1 vs mix (mel) | S1 hurts | S2 vs S1 | **S2 hurts** | S1 SI-SDR gain |
|---|---|---|---|---|---|
| <-5dB | +1.1% | 20.3% | -30.6% | **94.8%** | **-7.83 dB** |
| [-5,-2) | +2.4% | 17.4% | -32.2% | 78.8% | -7.08 dB |
| [-2,0) | +4.4% | 15.4% | -7.8% | 53.8% | -3.52 dB |
| [0,1) | +6.2% | 10.0% | +53.6% | 29.3% | -1.51 dB |
| [1,3) trained | +7.5% | 8.0% | +67.2% | 15.0% | ~0.00 dB |
| [5,10]+ trained | +8.1% | 2.6% | +75.7% | 1.4% | +0.57 dB |

Three findings: (1) Stage 2 is the dominant failure and it's near-total below -5dB (94.8% of samples
made WORSE than Stage 1) — a training-distribution gap, not an architectural flaw. (2) **Unlike the
speaker-count gap, Stage 1 degrades too** (8.1%→1.1% mel, hurts-rate 2.6%→20.3%), so the
"Stage-2-only is obviously sufficient" precedent does NOT transfer — Stage 1 is a plausible ceiling.
(3) Most interesting: **Stage 1's metrics disagree in DIRECTION at low SNR** — mildly positive in mel
(+1.1%) while severely negative in SI-SDR (-7.83dB), because a deletion-only mask forced to remove
most of the spectrum takes target energy with it. Sharper than §5.2's existing "mel understates
severity" caveat; here mel inverts the sign.

**BUILT 2026-09-10, NOT YET RUN.** User chose the 50% low [-10,1) / 50% keep [1,10] curriculum
(mirrors the multi-speaker fine-tune's preserve-the-validated-condition weighting):
- `data/augment.py`: `create_mixture()` gained optional `snr_db` override (default None = unchanged).
- `data/librispeech.py`: `low_snr_prob`/`low_snr_range` opt-in params + `_draw_snr_value()` /
  `_draw_snr_override()` helpers, applied to BOTH single- and multi-interferer paths.
  `build_dataloaders_multispeaker()` passes them through to TRAIN only; val stays 2-speaker/[1,10].
- `training/finetune_flow_hard_lowsnr.py`: thin script REUSING `finetune_flow_multispeaker()` rather
  than duplicating the 330-line loop (that loop is curriculum-agnostic). Writes to
  `checkpoints_v2/flow_finetune_lowsnr/`, never touches `flow_best.pt`.
- `scripts/run_finetune_flow_lowsnr_gpu.sh` + `scripts/smoke_test_lowsnr_curriculum.py` (pre-flight).

**Key design choice: the speaker-count curriculum stays ON (0.5/0.3/0.2) during this fine-tune** —
the starting checkpoint IS the hard-multispeaker result (3spk 81.9%, 4spk 76.7%), so single-interferer
-only training would risk handing those back.

**Bug caught in review** (worth remembering as a pattern): the SNR helpers were first inserted
mid-`__init__`, orphaning `self.speakers = ...` after a `return`. Always re-read the region after an
Edit that inserts methods into a class — anchoring on a line that isn't the true end of a block
silently swallows the rest.

**RUN 1 FAILED INSTANTLY (job 10990) — my bug, fixed.** `--low_snr_range -10,1` space-separated:
argparse reads the leading `-` as an option name (it only exempts plain negative numbers like `-10`;
the comma breaks that match). Fixed to `--low_snr_range=-10,1` in the launcher, with the reason
written into both `--help` and the docstring. Died at arg parsing, no GPU time wasted.

**RUN 2 COMPLETED (job 10991, 2026-09-10) — 25000/25000 steps, 12.98h on P100, no crashes.**
Curriculum confirmed active in the log: "50% of train examples from [-10.0, 1.0) dB ... 50% from
[1.0, 10.0] dB" plus "P(2 total speakers)=0.5, P(3)=0.3, P(4)=0.2". val_loss bounced 1.12–1.33 with
no trend (best 1.1254 @ step13000, final 1.2268) — exactly as predicted and as both prior fine-tunes
behaved; the proxy is measured on a 2-speaker/[1,10] val set while training on a harder mix, so it
says nothing either way. Two candidates exist, same as every prior fine-tune:
`flow_ft_lowsnr_best_step13000.pt` (proxy pick) and `flow_ft_lowsnr_final.pt` (step 25000).
Precedent: "final" won both previous times.

**EVALUATED 2026-09-10 (job 11032) — GAP CLOSES, BUT AT A REAL COST. NOT PROMOTED.**

Target regime (the win). Controlled in-domain test at [-10,1)dB, n=300, same corpus/mixer/seed —
added because Libri2Mix confounds low-SNR with a different corpus; `eval_multi_speaker.py` has
`--snr_min`/`--snr_max` flags that make this clean:
| | flow_best.pt | 50% low-SNR FT |
|---|---|---|
| accuracy | **61.3%** (mixture do-nothing = 64.0%, i.e. extraction ACTIVELY HARMFUL) | **71.3%** |
| median mel S2vsS1 | -45.1% | +50.1% |
| SI-SDR vs S1 | -5.94dB | +4.11dB |
Libri2Mix OOD (<1dB) agrees: mel -19.0%→+8.5%, catastrophic 40.4%→29.2%, SI-SDR vs S1 -1.50→+1.42dB.
Raw aggregate turns positive first time: SI-SDR vs mixture -0.04→+1.09dB.

The cost (why it was rejected) — NOT confined to the mel diagnostic, unlike the two prior fine-tunes:
| | flow_best.pt | 50% low-SNR FT |
|---|---|---|
| corpus-wide accuracy n=400 | 86.5% | 82.0% |
| in-domain n=2620 catastrophic | 3.7% | **15.5%** (4.2x) |
| in-domain n=2620 SI-SDR vs S1 | +1.71dB | +1.16dB |
| 2/3/4-spk accuracy | 86.7/81.9/76.7 | 81.4/77.3/73.3 |
| Libri2Mix in-dist catastrophic | 5.3% | 8.6% |
Per-bucket shows real capacity reallocation, not noise: <0dB 45.0%→31.9% while [5,7)dB 0.8%→4.7%.

**FOLLOW-UP IN FLIGHT: 25% curriculum** (`scripts/run_finetune_flow_lowsnr25_gpu.sh`, written
2026-09-10, bash -n clean). **TRAINED 2026-09-11 (job 11033): 25000/25000 steps, 13.84h, best val_loss
1.0791 @ step11000, final 1.2530 — uninformative as always.** Log confirms 25%/75% split, speaker-
count curriculum on, fresh start from flow_best.pt. Candidates: `flow_ft_lowsnr25_final.pt` (use this,
precedent) and `flow_ft_lowsnr25_best_step11000.pt`. **NOT YET EVALUATED.** Next:
`sbatch scripts/run_eval_candidate_gpu.sh checkpoints_v2/flow_finetune_lowsnr25/flow_ft_lowsnr25_final.pt lowsnr25`.
**PRE-REGISTERED DECISION RULE (fixed 2026-09-11 before seeing results — apply it as written, don't
move the bars after results land; timeline entry 17 has the table):** primary = in-domain low-SNR
accuracy >= ~67% (clearly above the 64.0% do-nothing mixture line); retention = corpus-wide acc
>= 85% (was 86.5%), in-domain n=2620 catastrophic <= 5% (was 3.7%), 2/3/4-spk acc each within ~3pp
of 86.7/81.9/76.7. Thresholds derived from binomial SE (~1.7pp @n=400 acc, ~0.4pp @n=2620 cat,
~2.5pp @n=300 acc). (a) all met → promote w/ backup+MD5; (b) lands between the two prior points
without meeting retention → trade is inherent, STOP curriculum tuning, keep flow_best.pt, document
the curve; (c) costs vanish but primary bar missed → curriculum isn't the lever, Stage-1-ceiling
hypothesis (finding 2 above) becomes leading explanation.

**EVALUATED 2026-09-11 (job 11054) — OUTCOME (b), applied exactly as pre-registered. flow_best.pt
KEPT, nothing promoted. Curriculum-weight tuning STOPPED — don't propose a 10%/35% run.**
| | old flow_best | 25% FT | 50% FT |
|---|---|---|---|
| low-SNR in-domain acc (primary, >=67) | 61.3 | **68.9 PASS** | 71.3 |
| corpus-wide acc n=400 (>=85) | 86.5 | **83.7 FAIL** | 82.0 |
| in-domain n=2620 mel catastrophic (<=5) | 3.7 | **11.3 FAIL** | 15.5 |
| in-domain SI-SDR vs S1 (primary metric) | +1.71 | +1.31 | +1.16 |
| in-domain severe SI-SDR < -10dB vs S1 | 2.0 | 7.5 | 10.2 |
| 2/3/4spk acc (within 3pp) | 86.7/81.9/76.7 | 84.7/79.0/75.3 PASS | 81.4/77.3/73.3 |
| Libri2Mix OOD mel / catastrophic | -19.0/40.4 | -8.5/34.6 | +8.5/29.2 |
| Libri2Mix in-dist catastrophic | 5.3 | 7.8 | 8.6 |
Every metric sits between the two earlier points. Curve is roughly proportional, no knee: 25% keeps
38-76% of the 50% run's gain (76% controlled low-SNR acc, only 38% Libri2Mix OOD mel) but 62-73%
of its in-domain cost. External-corpus gain shrinks fastest.
Mel-artifact objection pre-empted: share of mel-catastrophic cases whose audio is ~fine (SI-SDR
> -3dB vs S1) is 46/97=47% for old ckpt but only ~25% for both fine-tunes (98/405, 75/297) — their
failures are mostly REAL damage, so the mel rate understates their cost rather than inflating it.

**STAGE-1 BOTTLENECK DIAGNOSTIC (2026-09-11, zero GPU, local Node over Libri2Mix JSONLs).** Stage 1
SI-SDR identical across files (max diff 0.0000, join valid). Within FIXED SNR bands (removes SNR as
confound), quartiles by Stage-1 damage (sisdr_s1 - sisdr_mix), 50% checkpoint:
- [-5,-2)dB: NET system SI-SDR vs mixture -15.25/-13.95/-7.66/+3.92 dB (most->least S1 damage);
  system beats doing-nothing 26/22/32/64%.
- [-2,0)dB: NET -6.31/-5.19/+0.67/+6.28 dB; beats-mix 33/34/54/84%.
Stage 2 is NOT passive (its gain over S1 is largest exactly where S1 is worst: +7.02 / +11.59 dB in
Q1) but recovery is partial, so final quality is set largely by Stage 1. Mechanism for why the trade
is inherent: Stage 2 capacity spent regenerating mask-deleted target energy at low SNR is taken from
high-SNR precision. CAVEAT: correlational — intrinsically hard samples (similar voices) could make
both stages fail (common cause).
**Proposed causal test (NOT built, awaiting user):** oracle deletion-only mask at low SNR, fed to
Stage 2 — eval-only, no training. If Stage 2 then succeeds, Stage 1 is causally the bottleneck and
improving Stage 1 is the right investment; if not, it isn't. BEFORE building: read models/masking.py
+ data/mel.py to see exactly how the mask is applied to this project's (log-)mel — a naive
clip(Y/X,0,1) oracle may be wrong if mel values can be negative (multiplying a negative log value by
M<1 moves it toward 0, i.e. LOUDER, not quieter).

**CHECKED 2026-09-11 (user approved the oracle test) — STRUCTURAL FINDING, verified in code AND paper.**
- `data/mel.py:72`: `log_mel = torch.log(mel + log_offset)`; config `log_offset: 1e-8`. Natural log,
  silence ≈ -18.4, every bin with linear power < 1 is NEGATIVE.
- `models/masking.py:241`: `x_enhanced = x_mel * mask`, sigmoid mask in [0,1], applied to the
  ORIGINAL (non-normalized) log-mel.
- ⇒ X·M always lies between X and 0. Positive (loud) bins can only be lowered to 0 = linear power 1,
  never below. Negative bins can only stay (M=1) or get LOUDER (M<1). Suppressing a louder interferer
  to reveal a quieter target — the low-SNR task — is structurally impossible in negative bins. The
  masking.py docstring claim "X_enh <= X everywhere -> pure deletion" is only true for X >= 0.
- Paper (arXiv 2603.12837v1, WebFetch): SAME formulation. Eq. 9 `X_enh = X ⊙ M` on log-mel, M in
  [0,1] via sigmoid; Eq. 11 `L_mask = ||X_enh - Y||^2`; Table 3 reports D=100%, I=0%; Stage 2 starts
  from x0 = X_enh (Eq. 12). Paper does NOT specify log offset / base / dB. So NOT an implementation
  bug — an under-specified paper detail. "Pure deletion" only holds if log-mel values are
  non-negative (e.g. log(1+mel)); this implementation's log(mel+1e-8) makes quiet bins negative.
- Unverified: the WebFetch summarizer said the paper applies the mask to the l2-normalized input,
  but its quote only says the input is normalized. Irrelevant to this finding either way
  (l2 normalization divides by a positive scalar, sign unchanged).
- NOT YET MEASURED: fraction of real LibriSpeech log-mel bins that are negative; whether the trained
  Stage 1 actually raises bins (older memory claims compute_di_proportion ~100% delete, which would
  be consistent with the network learning M≈1 on negative bins).

**ORACLE EXPERIMENT BUILT + VERIFIED 2026-09-11, NOT YET RUN.** Code in `eval/eval_multi_speaker.py`:
`STAGE1_MODES`, `oracle_stage1()`, `formulation_stats()`, `--stage1_mode` (default `network`). Network
path unchanged: stage1_out is the same tensor, the new stats are post-inference tensor ops with no
RNG, and every sample reseeds anyway. Verified locally: whole-file ast.parse + `test_stage1_oracle.py`
at repo root, 13/13 (logmask can't push X=-10 to Y=-15 while energy reaches it; logmask per-bin
optimal vs 201-pt grid over 4000 bins; stats exclude padded frames). Launcher
`sbatch scripts/run_eval_stage1_oracle_gpu.sh` (9 runs x n=300, ~1h):
- low SNR [-10,1): {flow_best, 50% FT} x {oracle_logmask, oracle_energy} ->
  `eval_oracle_lowsnr_{logmask,energy}_{base,ft50}.jsonl`. Network refs NOT rerun, already exist:
  `eval_lowsnr_indomain_baseline.jsonl` (flow_best), `eval_lowsnr_indomain_lowsnr_ft.jsonl` (50%).
- control at trained SNR, flow_best: network `eval_multispeaker_2total_base_trimfix.jsonl`, plus
  `eval_oracle_insnr_{logmask,energy}_base.jsonl`.
- confound fix, flow_best network: `eval_multispeaker_{3,4}total_base_trimfix.jsonl` (2spk = the
  control's network run). Replace 86.7/81.9/76.7 with these for like-for-like comparisons.
Timeline entry 19 has the writeup.
Design: `--stage1_mode {network, oracle_logmask, oracle_energy}` on
eval_multi_speaker.py, default `network` = byte-identical. `oracle_logmask` = best mask within the
paper's formulation, M* = clip(Y/X, 0, 1) per bin (per-bin least squares under Eq. 11; M=1 where
|X|≈0). `oracle_energy` = true energy deletion = min(X, Y) in log domain (≡ IRM on linear power).
Reading: logmask ≈ network → network already near its formulation's best, Stage-1 retraining won't
help, formulation is the ceiling; logmask >> network → network is the bottleneck, retrain Stage 1;
energy >> logmask → prices a formulation change (mask in linear domain). Plus per-sample stats:
frac negative mixture bins, frac target bins unreachable by ANY [0,1] mask, network-S1 frac bins
raised + insert %.

**ORACLE RESULTS (job 11075, 2026-09-11) — all 9 runs completed.** Sample alignment with the older
network runs confirmed: at low SNR the oracle runs' mixture probe (64.0%, AUC 0.7117) and ceiling
(90.3%, AUC 0.9677) are identical to job 11032's network run — those depend only on the drawn audio.

Low SNR [-10,1), n=300 each (accuracy / SI-SDR vs mixture / catastrophic):
| Stage 2 input | flow_best | 50% FT |
|---|---|---|
| trained Stage 1 (job 11032) | 61.3% / -15.86dB / 57.0% | 71.3% / +0.00 / 22.3% |
| oracle_logmask (best mask within paper formulation) | **73.0%** / +7.98 / 24.3% | 71.6% / +6.06 / 33.3% |
| oracle_energy (true deletion) | **89.9%** / +20.21 / 0.3% | 89.7% / +20.15 / 0.3% |
Trained-SNR control, flow_best: network 87.3% / +2.12 / 1.7%; logmask 88.7% / +4.35 / 0.0%; energy
90.0% / +11.77 / 0.3%. Ceiling 90.3% throughout.

**Decomposition (flow_best accuracy):** low SNR network→logmask +11.7pp (learnable gap),
logmask→energy +16.9pp (formulation gap). Trained SNR: +1.4pp and +1.3pp. BOTH gaps open
specifically at low SNR — control confirms. Answer to the causal question: it's BOTH.
Paired per-sample check (Node over the JSONLs, exact): same sample_id → max |mix mse diff| 0.0 and
max |network S1 mse diff| 0.0 in every join (300/300). Paired median final SI-SDR: low SNR flow_best
logmask−network **+16.23dB (better in 89%)**, energy−logmask +11.11dB (100%); 50% FT logmask−network
+4.47dB (70%); trained SNR logmask−network +2.00dB (91%), energy−logmask +7.51dB (99%). System beats
doing nothing at low SNR: network 25% → logmask 74% → energy 100%. Stage 1 alone vs mixture at low
SNR: network −5.77dB vs logmask +8.30dB.
**METRIC-DEPENDENT RANKING — don't call either gap dominant:** at low SNR accuracy says the
formulation gap is bigger (16.9 vs 11.7pp) but SI-SDR says the learnable gap is (16.2 vs 11.1dB). At
trained SNR the SI-SDR formulation gap (+7.5dB) is NOT small, so an additive log-gain's upside isn't
confined to low SNR (upper bound, though).
**Stage 1 training cost** (from checkpoints_v2/masking timestamps, 2026-08-01→03): ~4,300 steps/h
(~0.84 s/step, ~2x faster than Stage 2); original run 200k steps ≈ 47h, best at 130k ≈ 30h; a 25k-step
fine-tune ≈ 6h. MaskingModule is constructed in 5 places — eval/results.py, eval/results_stage2.py
(load_masking, used by every eval), training/train_flow.py, training/train_mask.py, diag_test.py — so a
formulation flag needs plumbing through these and the checkpoint config.
**RECOMMENDED to user 2026-09-11 (awaiting choice):** A/B Stage-1 fine-tune — paper formulation vs
additive log-gain, identical curriculum (50% low SNR + speaker-count), identical ~25k steps from
mask_best.pt (~6h each, two parallel jobs possible), both evaluated with flow_best Stage 2 (NOT the 50%
FT). Warm-starting the additive arm from multiplicative weights is plausible (same keep/delete
decisions, different magnitudes) but unproven.

**USER CHOSE THE A/B (2026-09-11). BUILT + VERIFIED, NOT YET RUN.** Timeline entry 21 has the writeup.
- `models/masking.py`: `mask_mode` — "multiplicative" (default, paper Eq. 9, behavior unchanged) or
  "log_gain" (x + logsigmoid(logits) = true deletion). `MASK_MODES`, `set_mask_mode()`. Identical
  parameters in both modes, so mask_best.pt loads into either. Corrected the docstring's false
  "X_enh <= X everywhere -> pure deletion" claim.
- mask_mode travels IN THE CHECKPOINT: `training/train_mask.py save_checkpoint(..., extra=None)`
  merges extra into the payload (omitted = identical payload). Loaders read
  `ckpt.get("mask_mode", "multiplicative")`: `eval/results_stage2.py load_masking` (all evals +
  inference/infer.py), `training/train_flow.py load_frozen_masking` (Stage-2 training + flow
  fine-tunes), `eval/results.py load_checkpoint`. No CLI flag needed anywhere.
- `training/finetune_mask_lowsnr.py --mask_mode {multiplicative,log_gain}`: warm start mask_best.pt
  (raw state_dict then EMA overlay), 50% [-10,1) + 0.5/0.3/0.2 speakers, 25k steps, lr 1e-4 (between
  Stage-2 FT 4e-5 and Stage-1 base 2e-4 — log_gain must re-learn magnitudes; both arms share it),
  warmup 1000, ema 0.999, accum 2, val_every 1000, seed 42 (same construction + data order in both
  arms). Checkpoints `checkpoints_v2/mask_finetune_lowsnr_{mode}/mask_ft_{mode}_{best,latest,final}.pt`
  with mask_mode stored; resume refuses a mismatched mode.
- Launch both: `sbatch scripts/run_finetune_mask_lowsnr_gpu.sh multiplicative` and `... log_gain`
  (~6h each at 0.84 s/step, longer if they share a node's CPUs; --time 20h; auto-resume).
- Evaluate each with flow_best as Stage 2 via the evaluator's NEW optional 3rd arg:
  `sbatch scripts/run_eval_candidate_gpu.sh checkpoints_v2/flow/flow_best.pt mask_{mode} checkpoints_v2/mask_finetune_lowsnr_{mode}/mask_ft_{mode}_final.pt`.
  Its header now shows like-for-like refs + oracle ceilings.
- Verified locally: `test_mask_formulation.py` 14/14; syntax + undefined-name AST check OK on all 7
  changed/new .py files (checker lives in the scratchpad, not the repo).
**A/B TRAINED (jobs 11076 multiplicative / 11077 log_gain, 2026-09-11/12).** Both 25k steps, 7.69h /
8.52h, no crashes, identical warm start + curriculum + seed; logs confirm the config for both.
- Stage-1 val loss (same MSE on the same 2-speaker trained-SNR val set, so comparable BETWEEN arms):
  multiplicative best **6.3662** @step19000; log_gain best **2.9515** @step12000 — 2.2x lower. Train
  loss 13-20 vs 5-7.
- D% (deletion share) during validation: multiplicative drifts **99.8 -> 87.3** as training proceeds,
  i.e. it learns to RAISE bins to fit low-SNR targets — the move the paper's D=100% claim says
  masking never makes. log_gain is exactly 100.0 throughout, by construction.
- mask_mode plumbing CONFIRMED in production: `mask_ft_{mode}_{best,final}.pt` carry
  mask_mode=multiplicative/log_gain; mask_best.pt has none -> multiplicative.
- **CAUTION 1:** mask_best.pt's stored val_loss=24.5692 (step 130000, written 2026-08-02) is NOT
  comparable to these — it predates the padding-dilution fix that added frame masking to the loss.
  The only clean pre-fine-tune Stage-1 reference on today's metric is job 11075's median masked mel
  MSE: 5.944 at trained SNR, 13.151 at low SNR (per-sample median vs mean-over-batches, so still
  approximate).
- **CAUTION 2:** for tag=final/latest this project's save_checkpoint stores the TRAIN loss in the
  "val_loss" field (inherited from train_mask.py). final's 14.69 / 5.70 are last train losses, not
  val. Only `best` carries a real validation number.
- **CHECKPOINT CHOICE: evaluate `final` for BOTH arms** (Stage-2 fine-tune precedent). The val set is
  2-speaker/trained-SNR only, so "best by val" selects for preserving the easy regime rather than
  learning low SNR — the same mismatched-proxy trap. `best` is the fallback if final disappoints.
- **NEXT:** `sbatch scripts/run_eval_candidate_gpu.sh checkpoints_v2/flow/flow_best.pt mask_multiplicative checkpoints_v2/mask_finetune_lowsnr_multiplicative/mask_ft_multiplicative_final.pt`
  and the same with `mask_log_gain` / the log_gain final (~2h each), then apply the rule below.

**PRE-REGISTERED A/B RULE (fixed 2026-09-11 before results — apply as written):** an arm is promotable
only if low-SNR in-domain acc >= ~67% (baseline 61.3%, do-nothing 64.0%) AND corpus-wide >= 85%
(86.5%) AND in-domain catastrophic <= 5% (3.7%) AND 2/3/4spk each within ~3pp of 87.3/82.1/77.0. If
BOTH qualify: prefer log_gain ONLY if its low-SNR acc beats multiplicative by > ~3.5pp (≈ noise in a
difference of two n=300 accuracies); otherwise pick the paper-faithful multiplicative arm. If NEITHER
qualifies: learnable gap not closable by Stage-1 fine-tuning at this budget — record as the result.

**A/B RESULTS (jobs 11081 multiplicative / 11082 log_gain, 2026-09-12) — MULTIPLICATIVE WINS every
bar; log_gain fails 3 of 4. Rule applied as written; tie-break never needed. Timeline entry 22.**
| Bar | baseline | multiplicative | log_gain |
|---|---|---|---|
| low-SNR acc (>=~67) | 61.3% | **69.7% PASS** | 69.4% PASS |
| corpus-wide (>=85) | 86.5% | **86.5% PASS** | 76.5% FAIL |
| in-domain catastrophic (<=5) | 3.7% | **4.6% PASS** | 97.7% FAIL |
| 2/3/4spk (within ~3pp) | 87.3/82.1/77.0 | **84.9/80.0/75.4 PASS** | 76.6/69.3/66.0 FAIL |

Multiplicative arm detail: low-SNR system SI-SDR vs mixture -15.86 -> **+1.08dB**; Libri2Mix ALL
SI-SDR vs mix -0.04 -> **+1.91dB**, mel 32.3 -> 54.3%, catastrophic 26.6 -> 19.1%; OOD <1dB mel
-19.0 -> **+11.2%**, catastrophic 40.4 -> 28.2%, SI-SDR vs S1 -1.50 -> +1.05dB; in-dist slice also
IMPROVED (catastrophic 5.3 -> 4.9%, SI-SDR vs S1 +2.51 -> +2.75dB). Costs: in-domain n=2620
catastrophic 3.7 -> 4.6%, SI-SDR vs S1 +1.71 -> +1.54dB, mel 75.1 -> 74.1%, ~2pp speaker-count.
**KEY COMPARISON:** Stage-1 fine-tuning ≈ matches the 50% Stage-2 curriculum's low-SNR accuracy
(69.7 vs 71.3%) while giving back almost none of its cost (corpus-wide 86.5 vs 82.0; in-domain
catastrophic 4.6 vs 15.5). Entry 18's "trade is inherent" held only for STAGE 2; at Stage 1 it is
nearly free.
**log_gain:** Stage 1 is the most accurate ever measured here (mel MSE 2.296 @trained SNR vs 5.939
mult / 5.944 original; 3.809 @low SNR vs 11.180 / 13.151), D=100.0% as designed — but the FROZEN
Stage 2 destroys it (worsens its input in 97.7-98.7% of samples, in-distribution included). Exactly
the entry-21 risk: Stage 2 never saw outputs below the mixture. Formulation gap is REACHABLE at
Stage 1 but needs Stage-2 retraining to adopt.
**Also:** multiplicative arm now raises 18-23% of bins (was 13-16%), insert 0.4-0.8% (was 0.1-0.3%) —
continuing the training-time D% drift 99.8 -> 87.3.
**NOTE:** even in the winning arm Stage 2 SUBTRACTS at low SNR (S2 vs S1 -1.40dB, mel -6.3%) because
Stage 1 moved and Stage 2 didn't → a light Stage-2 adaptation to the new Stage 1 is the obvious next
step, and would also be required to adopt log_gain.
**PROMOTED 2026-09-12 (user chose "promote, then adapt Stage 2").** `mask_ft_multiplicative_final.pt`
-> `checkpoints_v2/masking/mask_best.pt`; previous file backed up to `mask_best_prelowsnr_backup.pt`
FIRST. MD5-verified both ways: backup c346d82339e00dd9687c7e02f67a4486 (178,615,061 B, == the file it
replaced); promoted 7db6352dd010592d11d734173c7c881d (178,618,495 B, == source). FIRST Stage-1
promotion in the project — Stage 1 had been frozen since 2026-08-02.
**CONSEQUENCE — every "baseline" output file now describes the PRE-promotion system**
(`eval_lowsnr_indomain_baseline.jsonl`, `eval_multispeaker_*_base_trimfix.jsonl`, etc.). The CURRENT
system = promoted Stage 1 + unchanged flow_best.pt = **job 11081's numbers**: low-SNR acc 69.7%,
corpus-wide 86.5%, in-domain n=2620 catastrophic 4.6% / SI-SDR vs S1 +1.54dB, 2/3/4spk 84.9/80.0/75.4,
Libri2Mix ALL +1.91dB vs mixture / in-dist catastrophic 4.9%. `scripts/run_eval_candidate_gpu.sh`'s
reference header was updated to these.
**NEXT — Stage-2 adaptation, built + bash -n clean, NOT yet run:**
`sbatch scripts/run_finetune_flow_adapt_newmask_gpu.sh` (~13h). Reuses
`training/finetune_flow_hard_lowsnr.py` with `--stage flow_finetune_adapt_newmask --prefix
flow_ft_adapt` (its OWN checkpoint dir — reusing flow_finetune_lowsnr/ would auto-resume the COMPLETED
job 10991 at step 25000 and exit instantly). Curriculum deliberately matches Stage 1's (50% [-10,1) +
0.5/0.3/0.2 speakers) so Stage 2 sees the distribution Stage 1 now produces; lr 4e-5. Target: fix the
-1.40dB S2-vs-S1 deficit at low SNR while holding the job-11081 numbers above. Evaluate afterwards with
`sbatch scripts/run_eval_candidate_gpu.sh checkpoints_v2/flow_finetune_adapt_newmask/flow_ft_adapt_final.pt adapt_newmask`
(no 3rd arg needed — Stage 1 is the promoted default now).

**ADAPTATION RUN 1 DIED ON QUOTA (job 11083, 2026-09-13):** `OSError: [Errno 28] No space left on
device` at step 21,874/25,000 after 15.5h. NOT a code bug. Config was correct in the log (50%/50% SNR
split, speaker mix on, `[Masking] mask_mode=multiplicative` from the promoted Stage 1, started from
flow_best.pt EMA step 25000). val_loss ran 1.17-1.24 (uninformative as always).
**Resumable, nothing lost:** `flow_ft_adapt_latest.pt` (step 20000, full 1,264,657,795 B) was written
before the failure, and save_checkpoint is atomic (tmp + os.replace). Just resubmit
`sbatch scripts/run_finetune_flow_adapt_newmask_gpu.sh` — it auto-resumes from latest.pt; ~3.5h for the
remaining 5000 steps.
**Cleanup done 2026-09-13 (user approved):** deleted `checkpoints_v2/flow_corrupted_discard/` (41GB,
abandoned corrupt Aug run, referenced nowhere) and 101 `checkpoints_vocoder/vocoder_step*.pt` (84GB;
vocoder frozen since 2026-08-20, all scripts default to vocoder_best.pt which was kept, as was
vocoder_latest.pt). Freed ~125GB; checkpoints_v2 157G -> 116G, checkpoints_vocoder 86G -> 1.7G.
~80GB more is available if needed from superseded `flow_best_step*.pt` (keep step298000) and the
fine-tune dirs' intermediate `*_best_step*.pt`.

**ADAPTATION COMPLETED (job 11103, 2026-09-13):** resumed from step 20000, finished the remaining 5000
steps in 2.59h, no errors. Config confirmed again in the log (50/50 SNR, speaker mix on,
`mask_mode=multiplicative` from the promoted Stage 1). **Best val_loss 1.0764 @step24000 vs the
starting flow_best.pt's 1.2941** — notable because in ALL THREE prior Stage-2 fine-tunes this proxy
stayed flat or worsened; here it improved, consistent with Stage 2 adapting to a Stage 1 that now
produces cleaner input. Still a mismatched proxy (2-speaker/trained-SNR only) — the battery decides.
Candidates: `flow_ft_adapt_final.pt` (step 25000) and `flow_ft_adapt_best_step24000.pt`.
**EVALUATE `final`** (Stage-2 precedent; proxy mismatched), no 3rd arg needed since Stage 1 is the
promoted default:
`sbatch scripts/run_eval_candidate_gpu.sh checkpoints_v2/flow_finetune_adapt_newmask/flow_ft_adapt_final.pt adapt_newmask`

**PRE-REGISTERED RULE for this candidate (fixed 2026-09-13 BEFORE evaluation — apply as written).**
Baseline = the CURRENT system (promoted Stage 1 + flow_best.pt) = job 11081's numbers.
- TARGET (the reason this run exists): low-SNR S2-vs-S1 must rise from **-1.40dB** to >= 0dB, i.e.
  Stage 2 must stop degrading Stage 1's output; low-SNR accuracy should hold or beat **69.7%**.
- RETENTION: corpus-wide acc >= 85% (86.5%); in-domain n=2620 catastrophic <= 5% (4.6%) and SI-SDR
  vs S1 >= ~+1.4dB (+1.54dB); 2/3/4spk each within ~3pp of 84.9/80.0/75.4; Libri2Mix in-dist
  catastrophic <= ~6% (4.9%) and ALL-slice SI-SDR vs mixture >= ~+1.7dB (+1.91dB).
- If TARGET met and RETENTION holds -> promote to flow_best.pt (backup + MD5, as always).
- If TARGET met but retention fails -> same trade-off situation as the Stage-2 curricula: keep
  flow_best.pt, record the curve.
- If TARGET missed -> Stage 2 cannot be adapted to the new Stage 1 at this budget; the pipeline stays
  as promoted (Stage 1 improved, Stage 2 unchanged) and that is the result.

**ADAPTATION EVALUATED (job 11106, 2026-09-13) — TARGET MET DECISIVELY; 2 retention bars FAIL, both
on the mel-catastrophic proxy, which the SI-SDR check shows is NOT tracking audio damage here.**
| | new S1 + flow_best (current) | new S1 + adapted S2 | bar |
|---|---|---|---|
| low-SNR S2-vs-S1 | -1.40dB | **+1.79dB** | >=0 PASS |
| low-SNR accuracy | 69.7% | **78.7%** | >=69.7 PASS |
| low-SNR SI-SDR vs mixture | +1.08dB | **+6.66dB** | — |
| corpus-wide acc n=400 | 86.5% | 85.2% | >=85 PASS (just) |
| in-domain n=2620 mel catastrophic | 4.6% | **9.7%** | <=5 **FAIL** |
| in-domain SI-SDR vs S1 | +1.54dB | +1.38dB | >=~1.4 borderline |
| 2/3/4spk acc | 84.9/80.0/75.4 | 84.3/78.3/74.9 | within 3pp PASS |
| Libri2Mix ALL SI-SDR vs mix | +1.91dB | **+3.47dB** | >=~1.7 PASS |
| Libri2Mix in-dist catastrophic | 4.9% | **7.3%** | <=~6 **FAIL** |
| Libri2Mix OOD mel / catastrophic | 11.2% / 28.2% | **65.0% / 16.2%** | — |
| Libri2Mix in-dist mel / SI-SDR vs mix | 72.5% / +2.78dB | 75.0% / +3.01dB | — |

**SI-SDR severity (in-domain n=2620), the decisive context:** severe regressions (SI-SDR < -10dB vs
S1) 2.0% (old S1) -> 2.1% (new S1) -> **2.7%** (adapted) — barely moved; median SI-SDR vs S1 1.71 ->
1.54 -> 1.38dB; hurts-rate 18.1 -> 21.7 -> 23.3%. Share of mel-catastrophic cases whose audio is
~fine (SI-SDR > -3dB) RISES 47% -> 56% -> **64%**. Contrast the Stage-2 low-SNR curricula, where
severe regressions jumped 2.0 -> 10.2% and only ~25% of mel failures were benign (real damage).
So the two failing bars are the mel proxy breaking down, not audio getting worse.
**DECISION PUT TO USER 2026-09-13** (promoting would override a pre-registered bar after seeing
results — must be stated explicitly in the thesis if done, citing the SI-SDR severity evidence).
Alternative not yet tried: evaluate `flow_ft_adapt_best_step24000.pt` (val 1.0764) instead of final.

**USER CHOSE "promote + document the override" (2026-09-13).** Promoting
`flow_ft_adapt_final.pt` -> `checkpoints_v2/flow/flow_best.pt`, backup
`flow_best_preadaptnewmask_backup.pt`; pre-check confirmed the outgoing flow_best.pt MD5 is
7382688b841cb81106a4d9d3620562a4 (= hard-multispeaker), as expected.

**ACCURACY-LOSS DIAGNOSTIC (2026-09-13, zero GPU, paired over the n=300 2-speaker embeddings that
eval_multi_speaker.py already stores).** The in-distribution loss is concentrated on EASY samples.
Adapted minus previous (new S1 + flow_best), split by that sample's own SNR:
| SNR | d(genuine sim) | d(margin) |
|---|---|---|
| [1,3) | **+0.0087** | +0.0051 |
| [3,5) | +0.0025 | +0.0008 |
| [5,7) | -0.0068 | -0.0014 |
| [7,10] (n=104, biggest) | **-0.0074** | **-0.0061** |
Mean genuine: 0.4399 (old S1+old S2) -> 0.4331 (new S1) -> 0.4313 (adapted). So the adapted Stage 2
is BETTER at the bottom of the trained range and WORSE at the top = over-correction on inputs that
barely need correcting — consistent with its extra mel-catastrophic cases being 64% audio-unharmed.

**NEXT, CHEAP FIRST: `sbatch scripts/run_sweep_cfg_newpipeline_gpu.sh`** (~50min, 10 runs):
cfg_scale 1.0/1.25/1.5/2.0 x {trained SNR, low SNR}, plus n_steps=8 at cfg 1.5 in both regimes.
Rationale: cfg_scale=1.5 / n_steps=4 was validated when Stage 1 was the step-130000 original and
Stage 2 the hard-multispeaker checkpoint — BOTH have changed since — and guidance strength is the
direct knob for over-correction. Target: recover trained-SNR accuracy (84.3% now vs 87.3%
pre-promotion) without giving back the low-SNR gain (78.7%). If no setting does, the operating point
is not the lever and a gentler Stage-2 curriculum (25-30% low SNR, 13h) is the next option.
NOTE: earlier cfg sweeps were ruled out for the t~0 gap and the 4-speaker gap, but this is a
different failure mode (over-correction after a curriculum shift) on a materially different pipeline.

**SWEEP RESULTS (job 11110, 2026-09-13) — HYPOTHESIS REFUTED, but the operating point IS mis-tuned.**
| cfg | steps | trained acc / AUC | low-SNR acc / AUC | trained catastrophic | SI-SDR vs mix (trained) |
|---|---|---|---|---|---|
| 1.0 | 4 | 84.3% / 0.9236 | 78.3% / 0.8584 | 6.0% | +2.03dB |
| 1.25 | 4 | 84.3% / 0.9261 | 78.3% / 0.8632 | 6.0% | +2.00dB |
| 1.5 (deployed) | 4 | 84.3% / 0.9272 | 78.7% / 0.8673 | 6.0% | +1.96dB |
| **2.0** | 4 | **85.0% / 0.9310** | **79.6% / 0.8762** | 6.3% | +1.90dB |
| 1.5 | 8 | 84.4% / 0.9286 | 79.4% / 0.8714 | 6.7% | +1.98dB |
- **Lowering cfg does NOTHING** (84.3% at 1.0/1.25/1.5, catastrophic flat) → the over-correction is in
  the WEIGHTS, not the guidance strength. The ~1.3pp corpus-wide loss is NOT recoverable at inference.
- Accuracy AND AUC rise monotonically with cfg in BOTH regimes; 2.0 (the edge of the sweep) is best.
  Individual accuracy deltas are within n=300 noise, but the monotone AUC trend across 4 settings in 2
  independent regimes is real signal (AUC uses all pairs, not one threshold).
- Cost of higher cfg is small and in SI-SDR (+2.03 -> +1.90dB trained; +6.67 -> +6.49 low).
- n_steps=8 @cfg1.5 gives a similar low-SNR gain (79.4%) for 2x compute → guidance is the cheaper knob.
**NEXT: `sbatch scripts/run_sweep_cfg_extend_gpu.sh`** (~30min): cfg 2.5 and 3.0 at 4 steps, plus cfg
2.0 @ 8 steps, both regimes. The trend hadn't turned at 2.0. WATCH the catastrophic rate — strong
guidance was implicated in the original t~0 outlier investigation.
**THEN, if a better setting is found:** edit `cfg_scale` in `configs/default_v2.yaml` (every eval
script reads it as the default) and RE-VALIDATE on the full battery
(`sbatch scripts/run_eval_candidate_gpu.sh checkpoints_v2/flow/flow_best.pt cfgNEW`) — the sweep is
only n=300 2-speaker. Timeline entry 27.

**EXTENDED SWEEP (job 11111, 2026-09-13) — pick cfg_scale 2.5. Timeline entry 28.**
| cfg | trained acc / AUC | trained cat | trained SI-SDR | low acc / AUC | low cat |
|---|---|---|---|---|---|
| 1.5 (deployed) | 84.3% / 0.9272 | 6.0% | +1.96dB | 78.7% / 0.8673 | 11.0% |
| 2.0 | 85.0% / 0.9310 | 6.3% | +1.90dB | 79.6% / 0.8762 | 11.0% |
| **2.5** | **85.7%** / 0.9342 | 7.0% | +1.88dB | 80.3% / 0.8827 | 11.0% |
| 3.0 | 85.6% / **0.9370** | 7.0% | +1.84dB | **80.7%** / **0.8863** | 12.0% |
| 2.0 @ 8 steps | 85.4% / 0.9326 | 6.7% | +1.96dB | 80.7% / 0.8789 | 10.7% |
Accuracy PLATEAUS at 2.5 (3.0 is -0.1pp trained, noise) while catastrophic climbs 6.0->7.0% and
low-SNR 11.0->12.0% at 3.0. AUC still rises at 3.0 but plateau + rising outliers = stop; strong
guidance was the mechanism in the t~0 amplification (entry 4). SI-SDR declines monotonically across
the whole range (+2.03 -> +1.84dB trained) — that's guidance's real cost here.
**CHOSEN: cfg_scale 2.5** — +1.4pp trained / +1.6pp low-SNR accuracy vs the deployed 1.5, for -0.08dB
SI-SDR and +1pp catastrophic. Should recover most of the 1.3pp the Stage-2 adaptation cost, for free.
**VALIDATE BEFORE CHANGING THE DEFAULT** (sweep is only n=300 2-speaker). All four eval scripts accept
`--cfg_scale`, so `scripts/run_eval_candidate_gpu.sh` gained an optional 4th arg — validate WITHOUT
editing configs/default_v2.yaml, so a failed validation leaves nothing half-changed:
`sbatch scripts/run_eval_candidate_gpu.sh checkpoints_v2/flow/flow_best.pt cfg2.5 checkpoints_v2/masking/mask_best.pt 2.5`
Only if that holds up, set `inference.cfg_scale: 2.5` in configs/default_v2.yaml (line 109).

**VALIDATED + ADOPTED (job 11112, 2026-09-13). `configs/default_v2.yaml` now has cfg_scale: 2.5.**
| metric | cfg 1.5 | cfg 2.5 |
|---|---|---|
| corpus-wide acc (n=400) | 85.2% | **86.2%** (pre-adaptation was 86.5% — recovered) |
| 2/3/4spk acc | 84.3/78.3/74.9 | **85.7/80.0/77.0** |
| low-SNR acc | 78.7% | **80.3%** |
| in-domain catastrophic | 9.7% | 10.2% |
| in-domain SI-SDR vs mixture | +1.84dB | +1.75dB |
| Libri2Mix ALL SI-SDR vs mix | +3.47dB | +3.39dB |
Accuracy up 1-2pp everywhere for ~0.1dB SI-SDR and +0.5pp catastrophic. Validated via the evaluator's
new 4th arg (no config edit until it passed), then adopted.

**END-TO-END COST OF THE WHOLE LOW-SNR CAMPAIGN (in-domain n=2620, vs DOING NOTHING — the fair view,
since "vs Stage 1" flatters later systems once Stage 1 itself improved):**
SI-SDR vs mixture +1.90 -> +1.85 -> +1.84 -> **+1.75dB** across original / new S1 / adapted S2 / cfg2.5;
share of samples beating do-nothing 74% -> 71% -> 71% -> **70%**. So the campaign cost ~0.15dB and 4pp
of that share in-domain, in exchange for: low-SNR acc 61.3 -> 80.3%, low-SNR SI-SDR -15.86 -> +6.51dB,
Libri2Mix aggregate -0.04 -> +3.39dB, 4spk accuracy back to its original 77.0%, 2/3spk within ~2pp.
Timeline entries 27-29.

**NEXT EXPERIMENT — log_gain revisited (built 2026-09-13, NOT yet run).** User's goal: accuracy at or
ABOVE the pre-campaign best, never below.
Why it's now worth running, when entry 22 rejected it: that rejection had ONE cause — the FROZEN
Stage 2 destroyed log_gain's output (97.7-98.7% of samples), having never seen mask output below the
mixture. Entry 25 then PROVED Stage 2 adapts to a changed Stage 1 with this exact recipe (its val_loss
even improved, the first time that proxy moved the right way in four fine-tunes). The cause of the
rejection is gone.
And log_gain's Stage 1 is better EVERYWHERE, not only at low SNR — median masked mel error 2.296 at
trained SNR (vs 5.939 current promoted, 5.944 original) and 3.809 at low SNR (vs 11.180 / 13.151).
A 2.6x cleaner input in the TRAINED regime is why this can RAISE in-distribution accuracy instead of
trading it, which is exactly what the goal requires.
Run: `sbatch scripts/run_finetune_flow_adapt_loggain_gpu.sh` (~13h). Same recipe/curriculum as the
successful adaptation; `--mask_ckpt` = the log_gain Stage 1; own stage/prefix
(flow_finetune_adapt_loggain / flow_ft_lg). load_frozen_masking auto-restores mask_mode=log_gain from
the checkpoint, so no flag is needed.
Then evaluate the PAIR — Stage 1 must be passed explicitly, since the promoted default is the
multiplicative one and these two only work together:
`sbatch scripts/run_eval_candidate_gpu.sh checkpoints_v2/flow_finetune_adapt_loggain/flow_ft_lg_final.pt loggain_pair checkpoints_v2/mask_finetune_lowsnr_log_gain/mask_ft_log_gain_final.pt`

**PRE-REGISTERED TARGET (fixed 2026-09-13 before the run — the user's ">= previous best, not below"):**
corpus-wide >= 86.5% (now 86.2), 2spk >= 87.3% (85.7), 3spk >= 82.1% (80.0), 4spk >= 77.0% (77.0),
low-SNR >= 80.3% (don't give back the campaign), in-domain SI-SDR vs mixture >= +1.75dB.
- Clears ALL -> promote BOTH checkpoints together (matched pair; neither works with the other's
  counterpart), backup + MD5 as always. Note this ADOPTS a deviation from the paper's Eq. 9 — document
  it as evidence-backed, citing the entry-20 oracle decomposition.
- Clears low-SNR but not the in-distribution bars -> same trade-off verdict as the Stage-2 curricula:
  nothing moves, record the curve.
- Fallback if it fails: a gentler Stage-2 curriculum (25-30%) would recover some in-distribution
  accuracy, but entry 26 showed the over-correction lives in the weights, so expect a proportional
  trade rather than a free win.
Disk: cleanup freed ~125GB and this run adds ~10GB; if ENOSPC recurs, ~80GB more is reclaimable from
superseded `*_step*.pt` snapshots.

**log_gain ADAPTATION TRAINED (job 11137, 2026-09-14): 25000 steps, 12.80h, no errors. Plumbing
confirmed in production — log shows `[Masking] mask_mode=log_gain` loaded from the checkpoint.**
**WARNING SIGN in the proxy:** best val_loss **1.3718** vs **1.0764** for the multiplicative pairing —
same loss, same 2-speaker/trained-SNR val set, only the Stage 1 differs, so directly comparable.
Trajectory: 3.2679 @step1000 (Stage 2 initially cannot handle log_gain input at all — confirms entry
22's finding quantitatively), 1.4749 @9000, plateau ~1.39-1.44, best 1.3718 @ ~19000-24000. Converged
but did NOT reach parity: even with a 2.6x more accurate Stage 1, the adapted Stage 2's output is
worse on the easy regime.
Still evaluate rather than judge on the proxy — mel/val has repeatedly failed to predict accuracy here
(the FROZEN Stage 2 scored 69.4% low-SNR accuracy while its mel looked catastrophic), and the target is
written in accuracy.
**NEXT:** `sbatch scripts/run_eval_candidate_gpu.sh checkpoints_v2/flow_finetune_adapt_loggain/flow_ft_lg_final.pt loggain_pair checkpoints_v2/mask_finetune_lowsnr_log_gain/mask_ft_log_gain_final.pt`
(Stage 1 MUST be passed explicitly — promoted default is multiplicative; the two only work together.)
Apply the pre-registered target above as written.
**log_gain PAIR EVALUATED (job 11164, 2026-09-14) — FAILS EVERY BAR. REJECTED, nothing promoted.**
| bar | required | log_gain pair | current system |
|---|---|---|---|
| corpus-wide | >=86.5% | **83.5%** | 86.2% |
| 2/3/4spk | >=87.3/82.1/77.0 | **82.7/80.0/76.0** | 85.7/80.0/77.0 |
| low-SNR | >=80.3% | **77.3%** | 80.3% |
| in-domain SI-SDR vs mix | >=+1.75dB | **+1.15dB** | +1.75dB |
Worse than the current system on EVERY metric (Libri2Mix ALL SI-SDR vs mix +2.83 vs +3.39dB too).
**WHY — the real finding:** Stage 2 contributes almost nothing on log_gain input. Its improvement over
Stage 1 collapses from +74.1% mel / +1.49dB (current pairing, 2spk) to **+11.7% / +0.34dB**, and goes
NEGATIVE at low SNR (-16.2% / -0.12dB). So the final output is WORSE in absolute terms (~2.0 mel MSE)
than the current pipeline (~1.54) despite Stage 1's error being 2.6x lower (2.296 vs 5.939).
**Pipeline quality is NOT a function of Stage 1 accuracy alone** — the stages are co-adapted; 300k
steps of Stage-2 training against multiplicative-mask outputs can't be re-pointed at a different input
family in 25k, and the residual it learned to predict (Y - X_enh) is much smaller when X_enh is already
near Y. Recasts entry 22's failure as structural, not mere distribution shift.
**Contingency NOT taken** (another 25k steps): shortfalls are 2.7-4.6pp and the mechanism is structural,
not convergence. Documented in timeline entry 30 and folded into results §5.10.
**STATUS: the low-SNR line of work is CLOSED.** Current system stands: Stage 1 low-SNR fine-tune +
adapted Stage 2 + cfg_scale 2.5.

**(superseded) CONTINGENCY if it falls short but shows promise:** val was still creeping down slowly at 25k
(1.4749@9000 -> 1.3718 best), so another 25k steps is the obvious extension — resume is built in
(`flow_ft_lg_latest.pt`, same launcher). If it falls short with no promise, the verdict is that the
formulation change is not adoptable within this two-stage design at reasonable cost, which is itself
the finding entry 22 predicted and §5.10 already describes.

Formulation stats (Stage 1 frozen, so identical across checkpoints): negative mixture bins 74.2% low
SNR / 79.5% trained; target bins unreachable by ANY [0,1] mask 73.5% / 66.7% — pervasive, not just low
SNR (a negative bin is reachable only if Y >= X, and additive interference makes Y < X nearly
everywhere; at high SNR the gap is small so it costs little). Network raises 13.5% / 15.7% of bins yet
insert proportion is only 0.1% / 0.2% — the magnitude-weighted D/I metric hides it (raises are tiny),
so the paper's I≈0% "holds" by that metric while ~1 in 7 bins is made louder.
Mel vs SI-SDR: at low SNR oracle_logmask's Stage-1 mel MSE is only 17% better than the network
(10.94 vs 13.15) yet final SI-SDR is ~24dB better — unreachable bins are mostly QUIET bins that dominate
log-MSE but carry little energy; the mask is exact in loud bins where the target is present.
Stage 2 is well-behaved on near-perfect input: oracle_energy gives S2-vs-S1 +0.03dB, 0.3%
catastrophic, reaches ceiling. The 50% FT's Stage 2 gains nothing from a better mask (71.3→71.6%)
while flow_best gains +11.7pp → the FT learned to compensate for Stage 1. Pair any improved Stage 1
with flow_best, NOT the 50% FT.
**CONFOUND FIX:** flow_best 2/3/4spk WITH trim fix = **87.3 / 82.1 / 77.0** (old pre-fix
86.7/81.9/76.7, +0.2 to +0.6pp). USE THESE as like-for-like baselines from now on. Re-checking the 25%
run: 84.7/79.0/75.3 → -2.6/-3.1/-1.7pp; 3spk now 0.1pp past the ~3pp bar (within noise). Verdict (b)
unchanged — it rested on two clean failed bars.

**NEXT (awaiting user):** (1) retrain Stage 1 within the paper's formulation (low-SNR ceiling 73.0%),
(2) switch Stage 1 to an additive log-gain `x_mel + log(mask)` = true deletion (oracle_energy is
EXACTLY its per-bin ceiling since X + log M can reach any value <= X; log-mel representation and vocoder
unchanged; Stage 2 handles near-target input fine, but an imperfect new Stage 1's error profile is
untested for Stage 2; deviates from paper Eq. 9), or (3) stop and document the decomposition.

**CONFOUND FOUND (2026-09-11):** the 2/3/4spk baselines 86.7/81.9/76.7 (job 10890; JSONLs written
2026-09-04 10:17-10:24) PREDATE the trim_trailing_silence fix to eval_multi_speaker.py (made while
building the listening export tool; job 10902 ran 13:56 same day). All candidates were measured WITH
the fix. My 2026-09-09 comment in scripts/rerun_verify_eer_and_libri2mix_gpu.sh claiming job 10890
"already had the fix" was WRONG. Verdict unaffected: the two FAILED bars are clean comparisons
(verify_eer baseline re-measured with the fix on 09-09; in-domain uses unchanged eval code), and the
trim fix moved every other metric <=0.5pp. A ~15min rerun of current flow_best.pt on 2/3/4spk would
make the curve's baseline point airtight for the thesis.

Original note: Starts from the SAME `flow_best.pt` as the 50% run Starts from the SAME `flow_best.pt` as the 50% run
(not chained off it), so `--low_snr_prob` is the only variable → two comparable points on one
trade-off curve. Then evaluate with the NEW generic
`scripts/run_eval_candidate_gpu.sh <ckpt> <tag>` (supersedes per-experiment eval scripts; skips the
checkpoint-independent in-domain low-SNR baseline, which already exists at
`outputs/results/eval_lowsnr_indomain_baseline.jsonl`). Also new: `eval/snr_split_summary.py` — the
SNR-split computation behind every §5.5.2 table, previously an ad-hoc snippet, now a real repo tool.

Timeline entry 16 in `docs/methodology_and_project_history.md` has the full writeup including the
"first fine-tune rejected on its measured results" framing.

**SUPERSEDED (kept for context) — evaluation plan, before it was run:** `scripts/run_eval_lowsnr_ft_gpu.sh` (written 2026-09-10, bash -n
clean, ~8h budget) evaluates `flow_ft_lowsnr_final.pt` in 4 steps, decisive one FIRST so the job can
be cancelled early if it looks bad: (1) `eval_libri2mix.py` n=6000 — yields BOTH the sub-1dB slice
(the target) and the >=1dB slice (must not regress) from the same records; (2) in-domain n=2620;
(3) `verify_eer.py` n=400; (4) 2/3/4-speaker n=300 each. All outputs go to NEW `*_lowsnr_ft.jsonl`
filenames so the canonical files the docs cite aren't clobbered by an unpromoted candidate.
Baselines to beat/hold are printed inline in the script.
If results plateau well short, that's the signal Stage 1 needs retraining too (see finding 2 above),
not a surprise.

Timeline entries 14 (diagnosis) and 15 (implementation) in
`docs/methodology_and_project_history.md` — the user wants that running record maintained for later
viva prep, see [[mask2flow-tse-thesis-writeup]].
