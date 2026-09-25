---
name: mask2flow-tse-master-timeline
description: "Chronological master record of every process run in Mask2Flow-TSE Phase One — what was run, why, job IDs, and the outcome. Start here to reconstruct project history."
metadata: 
  node_type: memory
  type: project
  originSessionId: 92e3d418-e0ac-4d2e-962a-e7f2f76ff0a1
  modified: 2026-09-20T13:04:23.117Z
---

Part of [[mask2flow-tse-overview]]. **This is the chronological index. For depth on any item, follow
the linked topic memory, and above all read `docs/methodology_and_project_history.md` (36-entry
timeline) and `docs/results_and_limitations.md` — those repo docs are the authoritative record and
are usually AHEAD of these memory files (see [[feedback-keep-memory-current]]).**

Phase One ran roughly 2026-07 to 2026-09-20. ~40 evaluated SLURM jobs; **15 candidate changes
formally evaluated, 4 promoted, 11 rejected.**

---
## Phase 0 — build (2026-07 → 2026-08-22)
| What | Detail |
|---|---|
| Data | LibriSpeech train-clean-100 + test-clean (fetched outside `data/download.py`, whose main path never worked until fixed 2026-09-19) |
| Stage 1 training | 200k steps, best @130k, ~4,300 steps/h ≈ 47h (2026-08-01→03) |
| Stage 2 training | 300k steps on FROZEN Stage 1 output — the origin of the stage co-adaptation that explains later findings |
| Vocoder | HiFi-GAN from scratch, step 170,000, val 0.4452, frozen since 2026-08-20. Needed because the Whisper-aligned mel (80/1024/160/400) fits no pretrained vocoder |
| Speaker projection | step 30,000 (job 10548, 2026-08-22). 393,728 trainable params on frozen WavLM |

## 2026-08-13/14 — first eval sweeps (jobs 10332-10345)
Baseline evals, plus two rejected ideas worth remembering so they are not re-proposed:
- **Checkpoint tail-averaging** (`flow_avg_tail5.pt`, job 10336): average of steps 228k/268k/270k/
  298k/300k → 75.5% full-pipeline / 73.0% S2-vs-S1 vs a 75.8%/71.9% baseline. **A wash, never
  promoted.** NOTE: that was a tail average from ONE run; souping the five *different* fine-tunes is
  still untested.
- **Safety-net fallback** (energy-collapse detector → fall back to Stage 1): **REJECTED** — removed
  the outliers but collapsed the median 75.8→46.6% because it fired on ~16% of all frames.

## 2026-08-26/28 — the t≈0 catastrophic-outlier campaign → [[mask2flow-tse-cfg-warmup-fix]]
1. **Diagnosis** (`eval/diagnose_catastrophic.py`): for the failing samples ‖v_cond − v_uncond‖ at
   t=0 ran **5-40x** its magnitude at later steps. CFG amplifies that into an oversized first Euler
   step the rest cannot correct.
2. Ruled out, in order: CFG amplification, CFG warm-up, Euler step count. None were the cause.
3. **hard-t0 fine-tune** (job 10606) — over-weight t≈0 during training.
4. **Full n=2620 validation** (10661): catastrophic **12.7% → 3.5%** (3.6x), median mel 67.3→77.7%,
   SI-SDR +0.93→+1.58dB, improving in EVERY SNR bucket.
5. step16000 vs step25000 compared (10738) — a tie; final chosen as the more defensible citation.
6. **PROMOTED 2026-08-29.** Also created `eval/verify_eer.py`, adding corpus-wide speaker-
   verification accuracy as the secondary metric.

## 2026-08-29/30 — Libri2Mix cross-corpus validation → [[mask2flow-tse-eval-results]]
Why: every number so far came from one in-house mixer. Needed an independently built corpus.
- WHAM noise download: 10790 (killed by a **munge auth daemon failure** — cluster fault), 10816 OK.
- Generation: 10802 (blocked), 10817 (**upstream LibriMix bug** — `write_noise()` called
  unconditionally for `mix_clean`, crashing every worker; patched in the vendored copy), 10819 OK.
- Eval: 10805/10818/10820 (last one a path bug of mine — LibriMix writes metadata as a SIBLING of
  the split dir), **10821 OK: 6000 samples, ~45 min**.
- **Result:** hard-t0 fix confirmed on an external corpus. Also surfaced the **sub-1dB SNR coverage
  gap** (60.6% of Libri2Mix) that drove the entire September campaign.

## 2026-09-02/05 — speaker-count generalization → [[mask2flow-tse-multi-speaker-test]]
- **First ever 3/4-speaker measurement** (10866): 86.3 / 76.7 / **72.6** — the model had only trained
  on 2-speaker mixtures.
- cfg/n_steps tuning at 4 speakers (10872) — an early hint that guidance mattered.
- **hard-multispeaker fine-tune** (10875), 0.5/0.3/0.2 speaker-count curriculum.
- Comparison (10887) + refresh (10890): **86.7 / 81.9 / 76.7, PROMOTED.**
- **Listening export** (10902) — found a **trailing-silence bug** that had been UNDERSTATING results.
- Thesis docs drafted: `docs/results_and_limitations.md` + `docs/methodology_and_project_history.md`.

## 2026-09-09 — measurement correction + citations
- `trim_trailing_silence` applied to the remaining scripts and everything re-measured (10988).
- Related Work citations verified against arXiv/publisher records; one real fix (SpEx+ is Ge et al.,
  not Xu et al.).

## 2026-09-10/13 — the low-SNR campaign → [[mask2flow-tse-lowsnr-gap]]
The largest single line of work. Sequence and reasoning:
1. **Stage attribution** (local, zero GPU): Stage 2 dominant failure below −5dB (94.8% of samples
   made worse), but Stage 1 degrades too — so "Stage 2 only" did not transfer from the speaker-count
   case.
2. **50% low-SNR curriculum, Stage 2** (10990 arg bug → 10991 trained; eval 11032): target won
   (61.3→71.3%) but corpus-wide 86.5→82.0% and in-domain catastrophic 3.7→**15.5%**. **REJECTED.**
3. **25% curriculum** (11033, eval 11054): proportional trade, no knee. **REJECTED — and
   curriculum-weight tuning STOPPED here.**
4. **Stage-1 bottleneck diagnostic** (local): quality set largely by Stage 1.
5. **STRUCTURAL FINDING**, verified in code AND paper: `log_mel = log(mel + 1e-8)` makes quiet bins
   NEGATIVE, and `X·M` with M∈[0,1] moves a negative bin TOWARD ZERO — i.e. LOUDER. "Pure deletion"
   holds only for non-negative bins. The paper (Eq. 9, D=100%/I=0%) does not specify its log offset,
   so this is an **under-specified paper detail, not an implementation bug.**
6. **Oracle decomposition** (11075) — replace Stage 1 with two oracles. Low SNR: network 61.3% →
   best-mask-in-formulation 73.0% → true-energy-deletion 89.9% → ceiling 90.3%. Trained SNR:
   87.3 / 88.7 / 90.0 / 90.3. **Both a learnable gap (+11.7pp) and a formulation gap (+16.9pp) are
   real, both open at low SNR.** This is the measurement that later defined Phase Two.
7. **Stage-1 A/B** — multiplicative (11076) vs log_gain (11077), identical curriculum/seed/steps.
   Evals 11081 / 11082: **multiplicative passed every bar; log_gain FAILED 3 of 4** because the
   FROZEN Stage 2 destroyed its output (97.7-98.7% of samples worsened).
8. **Stage 1 PROMOTED 2026-09-12** — first Stage 1 change since 2026-08-02. Low-SNR 61.3→69.7% at
   NO corpus-wide cost (86.5→86.5), speaker-count −2.4/−2.1/−1.6pp, inside the ±3pp bar fixed
   beforehand.
9. **Stage 2 adaptation** to the new Stage 1 (11083 died ENOSPC at step 21,874 → resumed 11103;
   eval 11106): target met decisively (low-SNR S2-vs-S1 −1.40→**+1.79dB**, accuracy 69.7→**78.7%**)
   but **two retention bars FAILED** (mel catastrophic 4.6→9.7%, Libri2Mix in-dist 4.9→7.3%).
   **PROMOTED WITH A DOCUMENTED OVERRIDE** — SI-SDR severity showed severe regressions barely moved
   (2.1→2.7%) while 64% of the new mel-"catastrophic" cases had audio that was fine.
10. **cfg_scale sweep** (11110) + extended (11111) → **2.5 chosen**; validated (11112) and ADOPTED.
    Recovered 2/3/4spk to **85.7/80.0/77.0** and corpus-wide to 86.2% for ~0.1dB SI-SDR. **Free.**
11. Disk cleanup: ~125GB freed (abandoned corrupt run + 101 vocoder step snapshots).

## 2026-09-13/14 — log_gain revisited, then closed
`flow_ft_lg` trained (11137), pair evaluated (11164): **FAILED EVERY BAR.** The finding that matters:
Stage 2's improvement over Stage 1 collapsed from +74.1% mel to **+11.7%** — the final output was
worse in absolute terms despite Stage 1's error being 2.6x lower. **Pipeline quality is not a
function of Stage 1 accuracy alone; the stages are co-adapted.** Low-SNR line CLOSED.

## 2026-09-14/16 — the onset gate → [[mask2flow-tse-onset-gate]]
- Listening export (11167) → **a defect found BY EAR that no metric could see.**
- Diagnostics 11171-11176: localized to **sequence position 0** (interior onsets clean at +0.29dB);
  ruled out the operating point; pad/splice sweeps.
- **2x2 ablation** (11177): old/old 1/24, newS1 1/24, newS2 2/24, **deployed 7/24** —
  **an INTERACTION, superadditive, neither promotion at fault alone.** Removes "revert one" as an option.
- onsetfix fine-tune (11178), gate check (11184): 7/24→4/24 vs a ≤1/24 bar. **REJECTED.**
- Battery 11187 killed by the **GPU squatter** (14,994MiB of a 16,384MiB P100; 5.5s→76.8s/sample).
- Inference-side padding/splice implemented; battery 11202 (no accuracy gain), gate 11205
  (7/24→6/24). **REJECTED.** Docs entries 33-35 written.

## 2026-09-17 — CFG warm-up → docs entry 36
Flagged twice as "improves everything, nothing worse" but never measured on accuracy (only
`full_eval.py` exposed the flag). Wired through all five scripts, evaluated (11210): quality gains
**replicated at full scale**, but accuracy fell everywhere; 4spk gave back 2.4pp. **REJECTED —
warm-up is simply WEAKER GUIDANCE re-parameterized** (2 of 4 steps unguided ≈ cfg 1.75), on an axis
already swept and optimized in the other direction.

## 2026-09-17/18 — ceiling analysis + Tier 0 → [[mask2flow-tse-accuracy-headroom]]
**Ceiling established:** the ceiling probe passes the CLEAN TARGET through the same
mel→HiFi-GAN→WavLM chain, so **90.3% is what perfect extraction of perfect speech scores.**
Extraction headroom ~4.6pp (oracle says ~4.3 of it is Stage 1); **measurement chain costs ~9.7pp.**
Three findings on disk: only train-clean-100 present (trained on 100h not 960h), no real RIRs,
`projection_best.pt` never evaluated.
Then five inference-side experiments, **ALL REJECTED**: onset padding (11202), cfg warm-up (11210),
reference_length 5.0 (11216), Heun solver (11218), projection_best (11220).
- **reference_length**: the key did not exist; everything had used a hardcoded 3.0s. Longer was
  worse AND **the ceiling itself fell 90.3→89.6%** — a worse speaker reference, not worse extraction.
- **Heun @2 vs Euler @4** (equal compute): worse. **Rectified flow trains trajectories to be
  STRAIGHT, and Euler is exact on a straight line** — no integration error to recover, while halving
  steps halved CFG applications.
- **projection_best**: identical 86.2% accuracy but LOWER AUC, and mixture rose while ceiling fell —
  a less discriminative embedding space, not a better system.
**Conclusion: the system is SATURATED at the trained operating point. Stop proposing inference-side
tweaks.**

## 2026-09-18/20 — data expansion campaign
Goal: the Stage 1 retrain the ceiling analysis points to.
- Guard blocked (11215 @32G free — the Windows-side `df` had shown 1.2T, the CLUSTER `df` showed 32G
  at 100%; **always trust the cluster-side reading**).
- 11217: crashed on a **pre-existing `data/download.py` bug** — summing human-readable size strings;
  that path had never worked.
- **11219: train-clean-360 FAILED (connection reset); train-other-500 + dev/test + RIRs SUCCEEDED.**
  Training data went **~100h/251 speakers → ~600h/~1,417 speakers** (5.6x speakers — arguably the
  more relevant axis for a speaker-conditioned task).
- 11221/11222 blocked by my own guard (fixed: it now sizes from the splits actually MISSING).
- 11223: 360 failed again — truncated stream, caught by sha256.
- Built `scripts/download_split_resumable.sh` (wget -c + sha256 + flac-count verification).
- **11224: download + sha256 SUCCEEDED; extraction hit ENOSPC** leaving **83,232 of 104,014 files
  (80%)** — and because tar extracts by path, the missing fifth was WHOLE SPEAKER DIRECTORIES.
  Cleaned up (~46GB freed).
- **11228 / 11229: extraction failed with `Input/output error` + partial writes at 450GB+ free —
  an NFS storage-layer fault, NOT space and NOT our code. train-clean-360 REMAINS BLOCKED.**

## 2026-09-20 — Phase One report
Published as an artifact: **https://claude.ai/artifact/NthjvPgADEfYETqh86gN3P** (12 sections,
4 Mermaid architecture diagrams, 4 SVG charts, 13 process cards, 17 decision-log entries, print
styles for PDF export). Local copy at `docs/phase1_report.html`. See [[mask2flow-tse-thesis-writeup]].

---
## Cross-cutting lessons (each learned the hard way, each cost real time)
1. **A change that improves every metric you are looking at has NOT been shown to be free.** Three
   instances: the mel-vs-SI-SDR divergence, log_gain (2.6x better Stage 1 → worse pipeline), and CFG
   warm-up. Ask which axis the current diagnostic cannot see, and measure that one.
2. **The stages are co-adapted.** Neither stage's quality predicts the pipeline's. Any improved
   Stage 1 REQUIRES a Stage 2 adaptation.
3. **Check the checkpoint-independent probes.** When mixture/ceiling move between runs, the
   comparison is not paired — caught ref5 and projection_best.
4. **Existence is not completeness.** Three separate failures left partial artifacts that existence
   checks would have accepted; verify by count.
5. **Validation loss never predicted anything here** — the val set is 2-speaker/trained-SNR while
   fine-tunes train on harder mixtures.
6. **Listening finds what metrics structurally cannot.** The embedding is mean-pooled over the whole
   utterance, so a 0.5s defect is invisible to it.
