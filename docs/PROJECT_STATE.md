# Mask2Flow-TSE — Project State and Handoff

**Last updated: 2026-09-20. Phase One complete.**

This is the *operational* handoff document: where the project stands, what is deployed, what was
tried, and what comes next. It is deliberately different from the two thesis documents:

| Document | Purpose |
|---|---|
| `docs/methodology_and_project_history.md` | Thesis narrative + 36-entry development timeline. **Authoritative record.** |
| `docs/results_and_limitations.md` | Thesis results chapter. **Authoritative for every number.** Never overwrites an old figure. |
| `docs/phase1_report.html` | Presentation report — diagrams, charts, process cards, decision log. |
| **`docs/PROJECT_STATE.md`** (this file) | Operational state: what is deployed, what is blocked, what to run next. |

---

## 1. Current system

**Deployed = job 11112**: promoted Stage 1 (low-SNR fine-tune, 2026-09-12) + adapted Stage 2
(2026-09-13) + `cfg_scale: 2.5` (2026-09-13).

| Condition | Do nothing | This system | Ceiling |
|---|---|---|---|
| Corpus-wide (n=400) | 71.0% | **86.2%** | 90.8% |
| 2 / 3 / 4 speakers (n=300 ea.) | 70.3 / 64.3 / 61.4% | **85.7 / 80.0 / 77.0%** | 90.3% |
| Low SNR [−10,1) dB (n=300) | 64.0% | **80.3%** | 90.3% |
| In-domain SI-SDR vs mixture (n=2620) | — | **+1.75 dB** | — |
| Libri2Mix SI-SDR vs mixture (n=6000) | — | **+3.39 dB** | — |

Accuracy = 1 − EER over genuine/impostor trials.

### Checkpoints

| Path | What |
|---|---|
| `checkpoints_v2/masking/mask_best.pt` | Stage 1, low-SNR fine-tuned, promoted 2026-09-12 |
| `checkpoints_v2/masking/mask_best_prelowsnr_backup.pt` | the Stage 1 it replaced |
| `checkpoints_v2/flow/flow_best.pt` | Stage 2, adapted to the new Stage 1, promoted 2026-09-13 |
| `checkpoints_v2/flow/flow_best_preadaptnewmask_backup.pt` | the Stage 2 it replaced |
| `checkpoints_speaker_encoder/projection_latest.pt` | speaker projection (use this; `_best` was evaluated and is worse) |
| `checkpoints_vocoder/vocoder_best.pt` | HiFi-GAN, frozen since 2026-08-20 |

There is **no central config field for checkpoint paths** — ~13 scripts name them as argparse
defaults. "Promoting" means physically replacing the file, with a backup and an MD5 check.

### Config state — every experimental flag is inert by default

```yaml
inference:
  flow_steps: 4
  cfg_scale: 2.5          # re-tuned 2026-09-13, the only adopted inference change
  solver: "euler"         # heun/midpoint implemented, measured, rejected
  cfg_warmup_steps: 0     # implemented, measured, rejected
  onset_pad_frames: 0     # implemented, measured, rejected
audio:
  reference_length: 3.0   # 5.0 measured, rejected
data:
  rir_path: "data/raw/RIRS/RIRS_NOISES/simulated_rirs"   # NOT the RIRS root — see §5
```

---

## 2. Data on disk

| Split | Utterances | Note |
|---|---|---|
| train-clean-100 | 28,539 | |
| train-other-500 | 148,688 | downloaded 2026-09-19 |
| **train-clean-360** | **absent** | **blocked — see §6** |
| dev-clean / dev-other / test-clean / test-other | 2,703 / 2,864 / 2,620 / 2,939 | |

**Training pool: ~600 h, ~1,417 speakers** (was ~100 h / 251 before 2026-09-19). For a
speaker-conditioned task the 5.6× increase in speaker count is arguably the more relevant axis.

RIRs: 60,000 simulated + 417 real + **843 pointsource NOISE files**. `rir_path` must point at the
`simulated_rirs` subtree — `RIRLoader` globs recursively, so the root would convolve speech with
noise recordings as if they were impulse responses.

---

## 3. What was tried — 15 candidates, 4 promoted, 11 rejected

| Job | Change | Verdict | Reason |
|---|---|---|---|
| 10555 | Safety-net fallback | rejected | Fired on ~16% of frames; median 75.8→46.6% |
| 10606 | **hard-t0 fine-tune** | **promoted** | Catastrophic 12.7→3.5%, improving in every SNR bucket |
| 10875 | **hard-multispeaker FT** | **promoted** | 3/4-spk +5.2/+4.1 pp, 2-spk flat |
| 11032 | 50% low-SNR curriculum (S2) | rejected | Corpus-wide −4.5 pp, catastrophic 3.7→15.5% |
| 11054 | 25% low-SNR curriculum (S2) | rejected | Proportional trade, no knee — curriculum tuning stopped |
| 11075 | *oracle decomposition* | diagnostic | Located the ceiling; defined Phase Two |
| 11082 | log_gain Stage 1, frozen S2 | rejected | Best Stage 1 ever measured; frozen S2 destroyed it |
| 11081 | **Stage 1 low-SNR FT** | **promoted** | Low-SNR +8.4 pp at zero corpus-wide cost |
| 11106 | **Stage 2 adaptation** | **promoted (override)** | Target met; 2 bars failed on the mel proxy — see §4 |
| 11112 | **cfg 1.5 → 2.5** | **adopted** | +1.4/+1.7/+2.1 pp, free |
| 11164 | log_gain pair (both stages) | rejected | Failed every bar; stages are co-adapted |
| 11184 | onset fine-tune | rejected | 7/24→4/24 vs a ≤1/24 bar; 4-spk unmoved |
| 11202 | onset padding/splice | rejected | 7/24→6/24; no accuracy gain |
| 11210 | cfg_warmup_steps = 2 | rejected | Weaker guidance re-parameterized; accuracy fell |
| 11216 | reference_length 5.0 | rejected | Ceiling itself fell — worse speaker reference |
| 11218 | Heun solver @ equal compute | rejected | Rectified flow is straight; Euler already exact |
| 11220 | projection_best | rejected | Same accuracy, lower AUC, compressed scale |

---

## 4. Known limitations

1. **t≈0 coverage gap** — diagnosed, mitigated 3.6× (catastrophic 12.7→3.5%). Not solved.
2. **SNR coverage below 1 dB** — closed. Low-SNR accuracy 61.3→80.3%, SI-SDR −15.86→+6.51 dB.
3. **Speaker-count generalization** — mitigated. 4-speaker 72.6→77.0%.
4. **Onset gate — OPEN.** First ~0.5 s of affected utterances sits on a ≈−60 dB floor then switches
   on. A **position-0** effect (interior onsets are clean at +0.29 dB). The 2×2 ablation proves it
   is an **interaction** between the two promotions — each is clean alone (1/24, 2/24 vs a 1/24
   control), only the pair gates (7/24), superadditively. Two mitigations measured and both failed,
   both in the 4-speaker condition.

**One documented override.** Job 11106 was promoted despite failing two pre-registered retention
bars (in-domain mel catastrophic 4.6→9.7%, Libri2Mix in-dist 4.9→7.3%). Justification: both failing
bars were the *mel proxy*, while SI-SDR severity showed severe regressions barely moved (2.1→2.7%)
and 64% of the new mel-"catastrophic" cases had audio that was fine. Recorded as an override because
moving a bar after seeing results is what pre-registration exists to prevent.

---

## 5. Engineering notes that will bite again

- **Existence is not completeness.** Three separate failures left partial artifacts that existence
  checks accepted. Worst case: an interrupted extraction left **83,232 of 104,014 files (80%)** —
  and because `tar` extracts by path, the missing fifth was *whole speaker directories*.
  `data/librispeech.py:184` prints a warning for a missing split and carries on — **grep the
  training log for `[Dataset] Warning: ... not found, skipping` before trusting any run.**
- **Trust the cluster-side `df`, never the `Z:` SSHFS mount's** — they disagreed by 1.2 TB vs 32 GB.
- **`torchaudio.load` is unusable here** (needs `torchcodec`, absent). Everything reads via
  `soundfile`; `data/augment.py load_audio` documents why.
- **Check the checkpoint-independent probes** (mixture, ceiling) in every comparison. If they move,
  the runs are not paired — this caught two false results.
- **Validation loss has never predicted anything** in this project; the val set is 2-speaker at
  trained SNR while fine-tunes train on harder mixtures.

---

## 6. Blocked

**`train-clean-360` extraction.** The 21.5 GB tarball is downloaded and **sha256-verified** at
`data/raw/train-clean-360.tar.gz` — *do not delete it*. Extraction fails with `Input/output error`
and partial writes at 450 GB+ free (jobs 11228, 11229): an **NFS storage-layer fault**, not capacity
and not code. Worth reporting to HPC admins alongside the earlier munge-daemon and GPU-contention
faults. Once the fileserver is healthy:

```bash
sbatch scripts/download_split_resumable.sh train-clean-360   # ~20 min, skips the download
```

It would add 360 h on top of 600 h — a smaller increment than the first expansion, so it is **not a
blocker for the Stage 1 retrain.**

---

## 7. Phase Two

| # | Step | Why | Cost |
|---|---|---|---|
| 1 | **Stage 1 retrain on ~600 h** | The oracle says all remaining extraction headroom (~4.3 pp) is here | ~47 h |
| 2 | **Stage 2 adaptation** | **Mandatory** — log_gain proved a better Stage 1 alone makes the pipeline worse | ~13 h |
| 3 | Measurement chain (vocoder, speaker encoder) | The larger pool at ~9.7 pp; bounds every number in the project | 1–2 d |
| 4 | Multi-utterance enrollment | Phase One rejected a longer window on *one* utterance; pooling several is untested | hours |
| 5 | Onset gate | Only remaining fix class consistent with an interaction defect: co-adapt both stages, or weight the loss toward failing frames | 4–13 h/attempt |

**Do not propose further inference-side tuning** — guidance, solver, step count, enrollment window
and projection swap have all been measured and rejected. That is what a saturated operating point
looks like, and the ceiling analysis predicted it.

---

## 8. Reproduce anything

```bash
# Full promotion battery (6 positionals, plus arbitrary extra flags forwarded to all stages)
sbatch scripts/run_eval_candidate_gpu.sh <flow_ckpt> <tag> [mask_ckpt] [cfg] [onset_pad] [warmup] [--extra ...]

# Current system
sbatch scripts/run_eval_candidate_gpu.sh checkpoints_v2/flow/flow_best.pt repro \
    checkpoints_v2/masking/mask_best.pt 2.5 0 0

# Cheap onset-gate screen (~25 min) — invisible to every battery metric
sbatch scripts/run_diag_gate_candidate_gpu.sh checkpoints_v2/flow/flow_best.pt gate \
    checkpoints_v2/masking/mask_best.pt 2.5

# Libri2Mix in-distribution / out-of-distribution split
python3 eval/snr_split_summary.py outputs/results/eval_libri2mix_min_<tag>.jsonl

# Playable audio for a listening check
python3 eval/export_listening_samples.py
```

Local unit tests (plain CPU torch, no cluster needed):
`test_onset_splice.py` (30 checks), `test_flow_solvers.py` (23), `test_mask_formulation.py` (14),
`test_stage1_oracle.py` (13), `test_chunk_overlap_add.py` (41).
