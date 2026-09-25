---
name: mask2flow-tse-eval-results
description: CURRENT headline numbers for Mask2Flow-TSE (job 11112 = promoted Stage 1 + adapted Stage 2 + cfg 2.5) plus the ceiling analysis and repro commands
metadata: 
  node_type: memory
  type: project
  originSessionId: 92e3d418-e0ac-4d2e-962a-e7f2f76ff0a1
  modified: 2026-09-20T13:04:52.400Z
---

Part of [[mask2flow-tse-overview]]. **Current deployed system = job 11112** (Stage 1 low-SNR
fine-tune promoted 2026-09-12, Stage 2 adaptation promoted 2026-09-13, `cfg_scale: 2.5` adopted
2026-09-13). Chronology of how these came about: [[mask2flow-tse-master-timeline]].

> Earlier versions of this file carried the 2026-08-28 hard-t0 numbers as the headline. Those are
> now THREE promotions out of date and were replaced 2026-09-20. Historic tables live in
> `docs/results_and_limitations.md`, which never overwrites an old number.

## Headline (job 11112)
| Condition | Do nothing | **This system** | Ceiling | SI-SDR vs mix |
|---|---|---|---|---|
| Corpus-wide, n=400 | 71.0% | **86.2%** (AUC .9361) | 90.8% | — |
| 2 speakers, n=300 | 70.3% | **85.7%** (AUC .9342) | 90.3% | +1.88 dB |
| 3 speakers, n=300 | 64.3% | **80.0%** (AUC .8870) | 90.3% | +2.31 dB |
| 4 speakers, n=300 | 61.4% | **77.0%** (AUC .8511) | 90.3% | +2.37 dB |
| Low SNR [-10,1)dB | 64.0% | **80.3%** (AUC .8827) | 90.3% | +6.51 dB |
| In-domain n=2620 | — | catastrophic 10.2% | — | **+1.75 dB** |
| Libri2Mix n=6000 | — | in-dist catastrophic 7.7% | — | **+3.39 dB** |

Accuracy = 1 − EER over genuine/impostor trials across the whole evaluated set.

## The ceiling — read this before proposing any improvement
`eval/eval_multi_speaker.py:272` builds the ceiling probe by passing the GROUND-TRUTH target mel
through HiFi-GAN and then the speaker encoder. **So 90.3% is what perfect extraction of perfectly
clean speech scores.** Oracle decomposition (job 11075, 2 speakers, trained SNR):
| Stage 1 input | Accuracy |
|---|---|
| current system | 85.7% |
| oracle: best mask within the paper's formulation | 88.7% |
| oracle: true energy deletion | **90.0%** |
| ceiling (clean target) | 90.3% |
⇒ **extraction headroom ~4.6pp and ~4.3 of it is Stage 1; the measurement chain (vocoder + speaker
encoder) costs ~9.7pp**, more than twice what is left in the extractor. Stage 2 and the operating
point are SATURATED — confirmed by five consecutive rejected inference-side experiments.

## Cross-corpus (Libri2Mix, n=6000, split at the trained SNR floor)
| Slice | Share | Median mel S2vsS1 | Catastrophic | SI-SDR vs S1 |
|---|---|---|---|---|
| in-distribution (≥1dB) | 39.4% | 73.4% | 5.7% | +2.37 dB |
| out-of-distribution (<1dB) | 60.6% | −19.0% → later +11.2% | 40.3% → 28.2% | −1.50 → +1.05 dB |
The in-distribution slice is consistent with (and SI-SDR-wise better than) the in-house headline, on
a corpus this project did not build — that is the cross-corpus evidence.

## Accuracy over the whole project (2/3/4 speakers)
| Job | Change | 2spk | 3spk | 4spk |
|---|---|---|---|---|
| 10866 | first 3/4-spk measurement | 86.3 | 76.7 | 72.6 |
| 10890 | hard-multispeaker FT **promoted** | 86.7 | 81.9 | 76.7 |
| 11075 | trim fix (MEASUREMENT, not model) | **87.3** | **82.1** | 77.0 |
| 11032 | 50% low-SNR curriculum — rejected | 81.4 | 77.3 | 73.3 |
| 11054 | 25% low-SNR curriculum — rejected | 84.7 | 79.0 | 75.3 |
| 11082 | log_gain S1 + frozen S2 — rejected | 76.6 | 69.3 | 66.0 |
| 11081 | Stage 1 low-SNR FT **promoted** | 84.9 | 80.0 | 75.4 |
| 11106 | Stage 2 adaptation **promoted (override)** | 84.3 | 78.3 | 74.9 |
| **11112** | **cfg 2.5 — CURRENT** | **85.7** | **80.0** | **77.0** |
| 11164 | log_gain pair — rejected | 82.7 | 80.0 | 76.0 |
| 11202 | onset padding — rejected | 85.3 | 80.3 | 77.0 |
| 11210 | cfg_warmup 2 — rejected | 85.0 | 78.4 | 74.6 |
| 11216 | reference_length 5.0 — rejected | 84.4 | 79.7 | 75.3 |
| 11218 | Heun solver — rejected | 84.4 | 80.0 | 76.0 |
| 11220 | projection_best — rejected | 86.0 | 80.3 | 76.7 |
Net vs first measurement: 2spk −0.6, **3spk +3.3, 4spk +4.4, low-SNR +19.0**. The 2/3spk shortfall
against the 11075 peak is the residual price of low-SNR capability; 4spk is at its all-time best.

## Repro
```
# full battery for any candidate (6 positionals + arbitrary extra flags):
sbatch scripts/run_eval_candidate_gpu.sh <flow_ckpt> <tag> [mask_ckpt] [cfg] [onset_pad] [warmup] [--extra ...]
# current system, reproduce the table above:
sbatch scripts/run_eval_candidate_gpu.sh checkpoints_v2/flow/flow_best.pt repro \
    checkpoints_v2/masking/mask_best.pt 2.5 0 0
# Libri2Mix SNR split behind the cross-corpus rows:
python3 eval/snr_split_summary.py outputs/results/eval_libri2mix_min_<tag>.jsonl
# cheap onset-gate screen (~25min), invisible to every battery metric:
sbatch scripts/run_diag_gate_candidate_gpu.sh <flow_ckpt> <tag> [mask] [cfg] [onset_pad]
```
