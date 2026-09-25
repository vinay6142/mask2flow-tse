---
name: mask2flow-tse-overview
description: "Goal, architecture, and environment for the Mask2Flow-TSE M.Tech project"
metadata: 
  node_type: memory
  type: project
  originSessionId: 28e94c75-57ba-46d9-9dad-9d9b2db586de
  modified: 2026-09-13T08:48:34.505Z
---

User (vinayreddy6142@gmail.com) is implementing the Mask2Flow-TSE paper (Moon et al., arXiv:2603.12837v1,
2026 — target speaker extraction) from scratch as an M.Tech project. No public reference code exists,
so all design/debugging decisions are the user's own.

**Architecture:** Two-stage pipeline.
- Stage 1 = MaskingModule (BiLSTM, ~11.2M params) predicts a soft mask; X_enhanced = mixture ⊙ mask.
  Since 2026-09-11 it has `mask_mode`: "multiplicative" (default, paper Eq. 9) or "log_gain"
  (x + log(mask), true deletion). The mode is stored in checkpoints and restored by every loader;
  checkpoints without it are multiplicative. See [[mask2flow-tse-lowsnr-gap]].
- Stage 2 = FlowMatchingModule (9-block DiT, ~77.3M params, RoPE, AdaLN-Zero, CFG), trained on frozen
  Stage 1 output, single-step Euler inference.
- Vocoder = HiFi-GAN, trained from scratch on the project's own LibriSpeech data (16kHz, n_mels=80,
  hop_length=160 — a Whisper-aligned mel config, NOT compatible with standard pretrained vocoders).

**Environment:** HPC cluster, login node gpu114, SLURM (sbatch only — never run training in an
interactive shell, it silently falls back to CPU otherwise), conda env `mask2flow`.
Code root on HPC: /home/mtech1/25CS60R85/code/mask2flow_tse (locally mounted/mirrored at
z:\code\mask2flow_tse). Active config: configs/default_v2.yaml.

**Local Windows session (verified 2026-09-11):** the Z:\ SSHFS mount gives file read/write only, no
cluster exec. `python3` is a Microsoft Store stub, but `python` is a real C:\Python313 WITH CPU torch
— usable for pure-function tests on repo code; pass Windows paths (`Z:/code/...`), not `/z/...`.
Don't assume the rest of the stack (omegaconf/torchaudio/transformers untested locally): extract the
functions under test with ast instead of importing the module (see `test_stage1_oracle.py`). Node is
also available. The Write tool fails with EPERM on the Z:\ mount — write files there with a Bash
heredoc, kept under ~5KB each (a ~9KB heredoc broke mid-command).

**Storage/quota (learned the hard way 2026-09-13):** this account has a per-user QUOTA, not just the
shared filesystem's capacity — a training job died with `ENOSPC` while `df` still showed 1.1 TB free
of 22 TB, so `df` is NOT a reliable check. Checkpoints had reached ~244GB. Check headroom before
launching anything long. Flow checkpoints are ~1.26GB each, vocoder ~0.89GB, masking ~0.18GB, and the
training loops keep every periodic `*_step*.pt`, so directories grow fast. Cleanup 2026-09-13 freed
~125GB. Keep: every promoted checkpoint, its `*_backup.pt` lineage, each run's `final`/`best`, and
`vocoder_best.pt`; the periodic step snapshots are the disposable part.

Checkpoints:
- checkpoints_v2/masking/mask_best.pt — **RE-PROMOTED 2026-09-12** (first Stage-1 promotion; it had
  been the step-130000 original, frozen since 2026-08-02). Now the low-SNR Stage-1 fine-tune
  (`mask_finetune_lowsnr_multiplicative/mask_ft_multiplicative_final.pt`, step 25000, MD5
  7db6352dd010592d11d734173c7c881d, 178,618,495 B). The pre-promotion original is at
  `mask_best_prelowsnr_backup.pt` (MD5 c346d82339e00dd9687c7e02f67a4486). Passed every pre-registered
  bar: low-SNR accuracy 61.3%→69.7% with corpus-wide accuracy unchanged at 86.5%. See
  [[mask2flow-tse-lowsnr-gap]]. (Before this, Stage 1 was untouched throughout — including the
  hard-multispeaker fine-tune, which was Stage-2-only; see [[mask2flow-tse-multi-speaker-test]]'s
  stage-attribution finding for why Stage 1 didn't need it then.)
- checkpoints_v2/flow/flow_best.pt — **RE-PROMOTED 2026-09-13** to the Stage-2 adaptation
  (`flow_finetune_adapt_newmask/flow_ft_adapt_final.pt`, step 25000, MD5
  f4f99bc31df441b25bc5669b0e69af20, 1,264,658,719 B), trained on the NEW Stage 1's outputs after that
  was promoted 2026-09-12. Outgoing hard-multispeaker checkpoint backed up to
  `flow_best_preadaptnewmask_backup.pt` (MD5 7382688b841cb81106a4d9d3620562a4). Promoted DESPITE
  failing two pre-registered mel-catastrophic bars: severe SI-SDR regressions stayed flat (2.1→2.7%)
  and 64% of the extra mel failures were audio-unharmed, so the mel proxy wasn't tracking damage —
  override documented in timeline entry 25. Current system: low-SNR accuracy 78.7% (was 61.3%),
  corpus-wide 85.2% (was 86.5%), 2/3/4spk 84.3/78.3/74.9, Libri2Mix aggregate +3.47dB vs mixture (was
  -0.04dB). See [[mask2flow-tse-lowsnr-gap]]. Older history below.
- checkpoints_v2/flow/flow_best.pt — **RE-PROMOTED 2026-09-04.** Promoted
  `checkpoints_v2/flow_finetune_multispeaker/flow_ft_ms_final.pt` (step 25000, hard-multispeaker
  curriculum fine-tune, itself continued FROM the hard-t0 checkpoint — see
  [[mask2flow-tse-multi-speaker-test]] for the full backstory/numbers) to `checkpoints_v2/flow/
  flow_best.pt`, same plain-file-copy-through-`Z:\`-mount procedure as every prior promotion (avoids
  the login-node OOM risk). Verified byte-identical via MD5 (`7382688b841cb81106a4d9d3620562a4`,
  1,264,655,539 bytes) matching source and destination. The prior (hard-t0) `flow_best.pt` was backed
  up FIRST to `flow_best_prehardmultispeaker_backup.pt` (MD5 `926f8d51a24fec8bd682c2e33162336a`,
  1,264,655,303 bytes — confirmed identical to the hard-t0 promotion's own originally-recorded MD5,
  i.e. nothing had drifted since 2026-08-29). Older lineage untouched: `flow_best_prehardt0_backup.pt`
  / `flow_best_step298000.pt` (the pre-hard-t0 original) still there too.
  **Every script defaulting to `flow_best.pt` now uses the hard-multispeaker fine-tune**: no
  regression at 2 speakers (86.3%→86.7% accuracy), 4-speaker accuracy 72.6%→76.7% (cleared the
  75-80% target), and on the primary in-domain n=2620 headline SI-SDR gain vs S1 held/improved
  slightly (+1.58dB→+1.71dB) with a modest mel-domain-diagnostic softening (77.7%→75.1% median,
  concentrated mostly in the hardest 1-3dB SNR bucket) — see [[mask2flow-tse-multi-speaker-test]] for
  full numbers and the promotion decision writeup. NOT yet re-validated: the trustworthy corpus-wide
  EER (`verify_eer.py`) and the Libri2Mix cross-corpus check, both optional extra diligence, not
  blocking. `eval/KNOWN_LIMITATIONS.md` has NOT been updated yet to reflect this new promotion.
- checkpoints_speaker_encoder/projection_latest.pt (step 30000, trained d-vector projection)
- checkpoints_vocoder/vocoder_best.pt

Note: there is no central config field for checkpoint paths — `configs/default_v2.yaml` has none;
every eval/inference/training script hardcodes `checkpoints_v2/flow/flow_best.pt` etc. as an
argparse default (13 files reference `flow_best.pt` by name). "Promoting" a checkpoint means
physically replacing the file at that path (with a backup), not editing a config.

See also [[mask2flow-tse-fixed-bugs]], [[mask2flow-tse-eval-results]], [[mask2flow-tse-next-steps]].
