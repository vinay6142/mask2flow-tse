START HERE for the Mask2Flow-TSE project: [Master timeline](mask2flow-tse-master-timeline.md) for what happened when, then [Eval results](mask2flow-tse-eval-results.md) for current numbers, then [Next steps](mask2flow-tse-next-steps.md) for what Phase Two does. The repo docs (`docs/methodology_and_project_history.md`, 36-entry timeline, and `docs/results_and_limitations.md`) are AUTHORITATIVE and usually ahead of these files — read them before proposing experiments.

- [Mask2Flow-TSE master timeline](mask2flow-tse-master-timeline.md) — **chronological index of every process run in Phase One**: what, why, job IDs, outcome, plus six cross-cutting lessons
- [Mask2Flow-TSE next steps](mask2flow-tse-next-steps.md) — Phase One COMPLETE 2026-09-20; no open experiments; Phase Two = Stage 1 retrain on 600h → mandatory Stage 2 adaptation → measurement chain
- [Mask2Flow-TSE eval results](mask2flow-tse-eval-results.md) — CURRENT headline (job 11112): corpus-wide 86.2%, 2/3/4spk 85.7/80.0/77.0, low-SNR 80.3%; the 90.3% ceiling analysis; accuracy across all 15 candidates
- [Mask2Flow-TSE overview](mask2flow-tse-overview.md) — goal, architecture, HPC/SLURM environment, checkpoint paths and promotion mechanics
- [Mask2Flow-TSE accuracy headroom](mask2flow-tse-accuracy-headroom.md) — ceiling is vocoder+encoder limited, not extraction; ~4.3pp left and all of it Stage 1; Tier 0 exhausted; the data-expansion campaign and its blocked train-clean-360
- [Mask2Flow-TSE low-SNR gap](mask2flow-tse-lowsnr-gap.md) — the largest campaign: two rejected Stage-2 curricula, the log-mel formulation finding, oracle decomposition, Stage 1 promotion, Stage 2 adaptation with a documented override, cfg 2.5
- [Mask2Flow-TSE onset gate](mask2flow-tse-onset-gate.md) — position-0 defect found by ear; 2x2 ablation proves it is an S1xS2 INTERACTION; both mitigations failed; cfg_warmup also measured and rejected
- [Mask2Flow-TSE CFG warmup fix](mask2flow-tse-cfg-warmup-fix.md) — root cause of the t≈0 catastrophic outliers and the hard-t0 retrain that cut them 3.6x
- [Mask2Flow-TSE multi-speaker test](mask2flow-tse-multi-speaker-test.md) — 3/4-speaker generalization gap found and closed by the speaker-count curriculum
- [Mask2Flow-TSE listening samples](mask2flow-tse-listening-samples.md) — qualitative verification; found a trailing-silence bug that was understating results, and later the onset gate
- [Mask2Flow-TSE fixed bugs](mask2flow-tse-fixed-bugs.md) — padding-dilution, unseeded projection, and the WavLM warning red herring; don't re-diagnose
- [Mask2Flow-TSE thesis writeup](mask2flow-tse-thesis-writeup.md) — the two repo docs plus the published Phase One report artifact
- [Mask2Flow-TSE git authorship](mask2flow-tse-git-authorship.md) — user commits/pushes themselves, Claude never a git contributor
- [Keep memory current](feedback-keep-memory-current.md) — update memory as work proceeds; read the repo docs before proposing work
