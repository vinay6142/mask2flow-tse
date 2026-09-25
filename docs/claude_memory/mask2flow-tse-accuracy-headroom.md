---
name: mask2flow-tse-accuracy-headroom
description: "What is actually left to raise Mask2Flow-TSE accuracy — the measured ceiling, three untapped findings (only train-clean-100 on disk, no real RIRs, projection_best never evaluated), and the tiered option list"
metadata: 
  node_type: memory
  type: project
  originSessionId: 92e3d418-e0ac-4d2e-962a-e7f2f76ff0a1
  modified: 2026-09-20T07:04:27.652Z
---

Part of [[mask2flow-tse-overview]]. Opened 2026-09-17 when the user asked "are there no more ways to
increase accuracy" after every prior line closed (see [[mask2flow-tse-onset-gate]]).

**THE HEADROOM IS BOUNDED AND ALREADY MEASURED.** At the trained operating point (2spk, SNR [1,10]):
| | accuracy |
|---|---|
| current system | 85.7% |
| with a PERFECT Stage 1 (oracle_energy, job 11075) | 90.0% |
| **ceiling** — clean target through the same chain | **90.3%** |
| perfect | 100% |
Verified in code 2026-09-17: `eval/eval_multi_speaker.py:272` builds the ceiling probe by passing the
GROUND-TRUTH target mel through HiFi-GAN and then the speaker encoder (`:287`). So **90.3% is what
perfect extraction of perfectly clean speech scores** — the remaining 9.7pp is lost by the vocoder +
speaker encoder before extraction quality is involved. Two separate pools:
- **Extraction: ~4.6pp**, and the oracle says ~4.3 of it is Stage 1 (85.7 -> 90.0 with a perfect
  mask). Stage 2 and the operating point are effectively SATURATED at trained SNR — consistent with
  the cfg plateau, the warm-up trade, and the onset repair all yielding nothing.
- **Measurement chain: ~9.7pp** — more than twice what is left in the extractor. Improving it raises
  the ceiling AND the system number without making extraction better; legitimate to report as
  "the measurement apparatus bounds the result", but it is not the same claim.

**THREE UNTAPPED FINDINGS (all verified on disk 2026-09-17):**
1. **Only `train-clean-100` is on disk.** `configs/default_v2.yaml` lists train-clean-100 +
   train-clean-360 + train-other-500 with the comment "with 200GB use all three", but the other two
   were never downloaded. **Every checkpoint in this project — Stage 1, Stage 2, vocoder, speaker
   projection — trained on ~100h instead of ~960h.** Single largest untapped lever, and it is a
   download, not an idea.
2. **No real RIRs.** `data/raw/RIRS` does not exist; `data/augment.py RIRLoader` silently falls back
   to SYNTHETIC RIRs, and `cfg.data.conditions.reverb_prob` is 0.34 — so a third of training samples
   have synthetic room acoustics. (Note: the multi-speaker path uses `mix_multi_at_snr()`, which is
   additive-only and bypasses the clean/reverb branch entirely.)
3. **`projection_best.pt` has NEVER been evaluated.** Every script defaults to `projection_latest.pt`.
   Both exist in `checkpoints_speaker_encoder/`. Also: the projection's logged `val_acc` is 0.00833 =
   exactly 1/120 = chance — most likely a broken val metric (the embeddings clearly work), but nobody
   has ever looked.

**DISK (2026-09-17):** filesystem 22T, 1.2T free, 95% full — but that is the SHARED volume, and job
11083 died with ENOSPC anyway, so a per-user quota is the real constraint and is NOT visible from the
Windows session. `checkpoints_v2` is back to **144G**; `data/raw` 50G of which `wham_noise` is **36G**
— reclaimable, its only purpose was generating Libri2Mix which already exists in
`LibriMix_storage` (1.9G). ~80G more in superseded `*_step*.pt` snapshots. **Check quota on the
cluster before committing to the ~53GB download.**

**TIERED OPTIONS (full list in the 2026-09-17 conversation):**
- Tier 0, free/inference-only: projection_best vs latest; **enrollment length** (strongest cheap
  lever — conditions BOTH stages); Stage-2 checkpoint weight averaging / model soup; higher-order ODE
  solver (only Euler ever used); cfg sweep at 3/4spk (both sweeps were 2spk only).
- Tier 1, hours: retrain the 393k-param speaker projection; download the data + RIRs.
- Tier 2, half-day to days: **Stage 1 retrain from scratch on full data** (~47h, 200k @ 4300 steps/h)
  — where the oracle says the headroom is; Stage 2 long fine-tune (25k ~13h; full 300k ~150h = 6+
  days); **joint Stage1+Stage2 fine-tune** (never tried, and entry 33's interaction finding argues
  for exactly this); vocoder continue/retrain (lifts the ceiling for every probe).
- Tier 3: log_gain done properly (retrain Stage 2 from scratch on log_gain input — entry 30 only ever
  gave it 25k steps against a 300k-step Stage 2, which was never the real test); stronger speaker
  encoder (ECAPA-TDNN / WavLM-large) to lift the ceiling; scale Stage 2; Stage 1 BiLSTM -> Conformer.
**MANDATORY SEQUENCING:** any improved Stage 1 must be followed by a Stage-2 adaptation — log_gain
proved a better Stage 1 ALONE makes the pipeline worse (entry 30).

**BUILT 2026-09-17 (not yet run):**
- `scripts/download_librispeech_full.sh` — sbatch wrapper over the EXISTING
  `data/download.py --splits all --rirs` (which already had `download_librispeech()` and
  `download_rirs()`). Guards on >=110GB free before starting (2x transient during extraction) and
  prints `quota -s`. Idempotent, but skips on directory EXISTENCE not completeness — delete a
  partial split's dir before resubmitting (the trap that bit Libri2Mix generation in job 10817).
  bash -n clean, 0 CR bytes.
- **`reference_length` plumbing.** The key did NOT exist in the config — `data/librispeech.py:271`
  and the standalone eval scripts all fell back to a hardcoded 3.0, so no number in docs/ ever used
  anything else. Added `audio.reference_length: 3.0` to the config, plus `--reference_length` to all
  five eval scripts. Implemented as an IN-PLACE `cfg.audio.reference_length = ...` override right
  after `OmegaConf.load`, so it reaches both the script and `LibriSpeechTSEDataset` (which reads
  `self.cfg.audio`) with one assignment instead of threading a parameter.
- `run_eval_candidate_gpu.sh` now forwards **arbitrary extra flags** after the 6 positionals, rather
  than growing a 7th and 8th (the list was already long enough to mis-invoke — see job 11204). A
  `--projection_ckpt` passed as an extra lands AFTER the default, and argparse takes the last
  occurrence, so it overrides cleanly. Verified at every arity.

**DISK — THE WINDOWS-SIDE `df` WAS WRONG (2026-09-18, job 11215).** The download script's guard ran
`df` ON THE CLUSTER and got `fileserver2:/fileserver2/people/mtech1 22T 21T 32G 100%` — **32G free,
100% full**, not the 1.2T the `Z:` SSHFS mount reported the day before. The guard refused to start
(needs >=110GB) instead of dying partway, and this also explains job 11083's ENOSPC cleanly.
**Always trust a cluster-side `df`, never the SSHFS mount's.** `quota -s` returns nothing on this
cluster, so there is NO per-user quota — it is a SHARED 22T volume, meaning free space can be
consumed by other users at any time. Check with `df -h /home/mtech1` (shared free) and
`du -sh /home/mtech1/25CS60R85` (own usage, slow). Reclaimable inventory at the time: 117
`*_step*.pt` snapshots = **109.5GB**, `wham_noise` 36G, plus `train-clean-100.tar.gz` (6.4G) and
`test-clean.tar.gz` (347M) which are redundant since both are already extracted. User freed space;
161GB free as of 2026-09-18, so the download guard now passes.

**TIER 0 RESULT — reference_length 5.0 REJECTED (job 11216, `ref5`).** Longer enrollment is worse
across the board vs job 11112 (ref 3.0):
| | ref 3.0 | ref 5.0 |
|---|---|---|
| low-SNR acc | 80.3% / .8827 | 80.0% / .8770 |
| corpus-wide | 86.2% / .9361 | 85.5% / .9302 |
| 2/3/4spk | 85.7/80.0/77.0 | 84.4/79.7/75.3 |
**THE DIAGNOSTIC DETAIL: the CEILING itself dropped, 90.3% -> 89.6%** (and the mixture probe moved
too, 70.3 -> 70.0). Those probes depend on the enrollment embedding, so this is NOT a paired
comparison the way the onset/warmup ones were — and it localizes the cause precisely: **longer
enrollment produced a WORSE speaker reference embedding**, and everything downstream just followed.
Extraction did not get worse; the reference did. Plausible causes (untested): WavLM-base-plus-sv is
tuned for ~3s segments, and/or `segment_waveform` zero-pads utterances shorter than the request.
NOTE this does NOT reject "more enrollment audio" in general — only "a longer window on ONE
utterance". Pooling d-vectors over SEVERAL utterances adds real speech instead of asking one clip
for more than it has, and remains untested.

**CORRECTION: checkpoint averaging was NOT untried.** `checkpoints_v2/flow/flow_avg_tail5.pt` exists
— an average of steps 228k/268k/270k/298k/300k, evaluated 2026-08-14 (job 10336): 75.5% full-pipeline
/ 73.0% S2-vs-S1 against a 75.8%/71.9% baseline. A wash, never promoted. That was a TAIL AVERAGE FROM
ONE RUN though, which is not the same as souping the five different fine-tunes (hard-t0, multispeaker,
adapt, lowsnr, onsetfix) — that variant is still untested.

**ODE SOLVERS IMPLEMENTED 2026-09-18 (Tier 0.4), inert by default, not yet run.**
- `models/flow.py`: `SOLVERS = ("euler", "heun", "midpoint")` and a `solver=` arg on
  `inference()`. The velocity computation (incl. the CFG branch) is factored into a closure so the
  2nd-order methods can evaluate it at intermediate points; the euler path calls it once per step in
  the original order and is **bit-identical** to before. Also forwarded through
  `inference_with_onset_splice`, all five eval scripts (`--solver`), `inference/infer.py`, and
  `configs/default_v2.yaml` (`inference.solver: "euler"`, NOT adopted).
- **Budget note that matters for interpreting results:** heun/midpoint cost 2 velocity evals per
  step, so `--solver heun --n_steps 2` is the same compute as the deployed `euler --n_steps 4`.
  Comparing heun@4 to euler@4 would be a 2x-compute comparison, not a fair one.
- `test_flow_solvers.py` (new, repo root) **23/23**: builds a tiny real FlowMatchingModule and stubs
  forward() with analytic fields whose integrals are known by hand — constant field exact for all
  solvers; v=2t where euler@4 provably gives 0.75 while heun@2 and midpoint@2 are EXACT at 1.0 on the
  same 4 evaluations; v=x (exact x0*e) where heun@2 beats euler@4; a bit-for-bit regression of the
  default path against the original hand-rolled Euler loop; eval-count checks confirming cfg warmup
  and guidance still apply per step under every solver; and ValueError on an unknown solver.
  `test_onset_splice.py` still 30/30 (its stub gained the new kwarg).

**DOWNLOAD BLOCKED THEN FIXED (jobs 11215, 11217, 2026-09-18).** 11215 hit the disk guard (32G free).
After the user freed space, 11217 passed the guard at 161G and then crashed on a **pre-existing bug in
`data/download.py`**: `download_librispeech()` did `sum(SPLITS[s] ...)` over values that are
human-readable STRINGS ("23.1 GB"), so it raised `TypeError: unsupported operand type(s) for +:
'int' and 'str'` before touching a single split — and `total_size` was never even used afterwards.
That function therefore crashed on EVERY call including the "minimal" mode, meaning the LibriSpeech
already on disk was fetched some other way and job 11217 was the first real run of it. **FIXED**:
added `_size_gb()` to parse "23.1 GB"/"337 MB" -> float GB, and the total is now actually printed
(all splits = 61.0 GB, of which ~6.3 already on disk). Verified by parsing each value.

**SECOND ISSUE FOUND, NOT YET TRIGGERED — the RIR download would silently corrupt reverb training.**
openslr resource 28 is RIRS_NOISES, which ships `simulated_rirs/` and `real_rirs_isotropic_noises/`
(genuine impulse responses) ALONGSIDE `pointsource_noises/` (ordinary noise recordings).
`data/augment.py RIRLoader` does `rglob("*.wav")` over whatever `cfg.data.rir_path` points at, so
pointing it at `data/raw/RIRS` would convolve speech with NOISE CLIPS as if they were impulse
responses — on the 34% of training samples that draw the reverb condition. Added a post-download
layout check to `scripts/download_librispeech_full.sh` that counts wavs per subtree and says to set
`rir_path` to `data/raw/RIRS/RIRS_NOISES/simulated_rirs` instead. **Verify the counts in the job log
before enabling reverb training.**

**TIER 0 IS EXHAUSTED — ALL FIVE INFERENCE-SIDE EXPERIMENTS REJECTED (2026-09-16..18).**
onset padding (11202/11205), cfg_warmup (11210), reference_length 5.0 (11216), heun solver (11218),
projection_best (11220). Nothing moved the system. **This is exactly what job 11075's oracle
predicted: a perfect Stage 1 reaches 90.0% against a 90.3% ceiling, so Stage 2 and the operating
point are SATURATED at trained SNR — no inference-time knob can move a saturated system.** Stop
proposing inference-side tweaks; the remaining headroom is Stage 1 (~4.3pp, needs the retrain) and
the measurement chain (~9.7pp, vocoder + speaker encoder).

**0.4 heun solver REJECTED (job 11218, `heun2`, heun@2 == euler@4 in compute).** Valid paired
comparison (mixture 71.0/64.0 + ceiling 90.8/90.3 identical to 11112).
| | euler@4 | heun@2 |
|---|---|---|
| corpus-wide | 86.2% / .9361 | 85.7% / .9353 |
| low-SNR | 80.3% / .8827 | **77.7% / .8674** |
| 2/3/4spk | 85.7/80.0/77.0 | 84.4/80.0/76.0 |
| mel S2vsS1 2spk | 74.1% | 62.5% |
**WHY — worth remembering before anyone proposes a fancier solver again:** Stage 2 is trained with a
RECTIFIED FLOW objective, which optimizes for STRAIGHT trajectories, and **Euler is exact on a
straight line**. There was no integration error left for a 2nd-order method to recover. Meanwhile
halving n_steps to hold compute fixed also halved the number of CFG applications, and this pipeline
is demonstrably guidance-sensitive (the 1.5->2.5 retune was worth +1.4/+1.7/+2.1pp). So it traded
real guidance resolution for a near-zero integration gain. `test_flow_solvers.py` proves the solver
IS more accurate on a curved field — the model just does not produce one. (Untested variant that
would separate the two effects: heun@4, i.e. 2x compute. Given the mechanism, not worth it.)

**0.1 projection_best REJECTED (job 11220).** Identical headline accuracy, LOWER AUC:
corpus-wide 86.2% both, but AUC .9361 -> .9308; 2/3/4spk 85.7/80.0/77.0 -> 86.0/80.3/76.7;
low-SNR 80.3 -> 80.0. **The tell is the reference probes: mixture rose 71.0 -> 71.8 AND ceiling fell
90.8 -> 89.7**, narrowing the do-nothing-to-ground-truth spread from 19.8pp to 17.9pp. So
`projection_best` is a LESS DISCRIMINATIVE embedding space — not scoring better, measuring on a
squashed scale, which the threshold-free AUC confirms. Keep `projection_latest` (what every script
already defaults to). Same lesson as ref5: when a change moves the enrollment embedding, the mixture
and ceiling probes move too, so check them before reading the system number.

**DOWNLOAD FINISHED 2026-09-19 (job 11219) — PARTIAL, AND NOT THE PART WE WANTED.**
| split | result |
|---|---|
| train-clean-100 | already present, skipped (28,539 flac) |
| **train-clean-360** | **FAILED — `[Errno 104] Connection reset by peer`** after ~7.5GB / 2h43m |
| train-other-500 | **OK, 148,688 utterances** |
| dev-clean / dev-other / test-other | OK (2703 / 2864 / 2939) |
| RIRs | OK, extracted |
So the split recommended for DROPPING is the one that landed, and the high-value one died. **Training
data now = train-clean-100 + train-other-500 ≈ 600h / ~1,417 speakers, vs 100h / 251 speakers
before** — 6x hours and 5.6x SPEAKERS, which for target-speaker extraction is arguably the more
important axis. (LibriSpeech "other" is split by ASR difficulty, not by noise, so the acoustic
mismatch against test-clean is milder than the name suggests.)

**THREE CODE PROBLEMS THIS EXPOSED, ALL FIXED 2026-09-19:**
1. `download_librispeech()` swallowed the failure — the except block printed and continued, so a
   13MB log still ended in "Download complete!" and the missing split was one line. Now it collects
   failures, returns them, and main `sys.exit(1)` so a 10h sbatch job cannot report success while a
   split is silently absent.
2. `verify_downloads()` crashed on `torchaudio.load` -> `ModuleNotFoundError: torchcodec`. Switched
   to soundfile. **`data/augment.py load_audio` already read via soundfile for exactly this reason
   and says so in its docstring — so training/eval were never affected, only the verify step.**
3. **`cfg.data.rir_path` was an active hazard the moment the RIRs landed.** Counted on disk:
   `pointsource_noises` **843 wav** (ordinary noise), `real_rirs_isotropic_noises` 417 (mixed),
   `simulated_rirs` (3 room-size categories, ~60k). RIRLoader rglobs whatever the path points at, so
   `data/raw/RIRS` would have used 843 noise clips as impulse responses on the 34% reverb condition.
   **Changed to `data/raw/RIRS/RIRS_NOISES/simulated_rirs`.** No existing checkpoint is affected —
   the path did not exist before, so RIRLoader had been falling back to synthetic RIRs all along.

**MY OWN GUARD THEN BLOCKED THE RETRY (jobs 11221 @38G, 11222 @93G free, 2026-09-19).** `NEED_GB`
was a hardcoded **110**, sized for BOTH large splits, but only train-clean-360 (23.1GB, ~46GB peak)
was still missing — so it refused to start with ample space. **FIXED**: the guard now computes the
requirement from the splits actually MISSING on disk (per-split size table, x2 for the extraction
transient, +6GB margin). Verified against the real disk state: missing=train-clean-360,
NEED_GB=**52**, so 93GB free now passes. Lesson: a fixed resource guard goes stale the moment the
job becomes partially complete — derive it from remaining work.

**ATTEMPT 2 ALSO FAILED (job 11223, 2026-09-19) — train-clean-360 is 0 for 2.** Failure mode was
DIFFERENT and more informative than attempt 1's connection reset:
`ERROR: invalid hash value (expected "146a56496217e96c14334a160df97fffedd6e0a04e66b9c5af0d40be3c792ecf",
got "0c2f3a09...")` — the stream ended early (~9GB of 21.5GB) and torchaudio hashed the truncated
file. **The openslr link is not reliable for a 21.5GB single-shot transfer on this cluster; it drops
around 7.5-9GB.** Both my download.py fixes proved out here: the verify step ran cleanly via
soundfile (listed every split with sample durations), and the run ended
`INCOMPLETE -- these splits FAILED: ['train-clean-360']` with exit 1 instead of "Download complete!".
Data state unchanged: train-clean-100 + train-other-500 (~600h, ~1,417 speakers), no 360.
torchaudio left NO partial file behind (it cleans up its temp), so nothing to resume from.

**ATTEMPT 3 (job 11224, 2026-09-19, the resumable wget path): DOWNLOAD SUCCEEDED, EXTRACTION DIED ON
ENOSPC.** `saved [23049477885/23049477885]` and **`sha256 OK`** — wget -c fetched the full 21.5GB
where torchaudio had failed twice. Then `tar` hit 704 x `Cannot close: No space left on device`.
Cause: the job started at 83GB free, spent ~3h downloading, and **other users consumed the shared
volume during the transfer** (this volume swung 161GB -> 38GB overnight once already). The verified
tarball survived, so this cost extraction time only, not another download.
**THE PARTIAL EXTRACT WAS THE WORST-SHAPED FAILURE YET: 83,232 of 104,014 flac = 80% complete.**
80% looks healthy at a glance and every existence check in the project would have accepted it — and
because tar extracts sequentially by path, the missing 20% is **whole speaker directories from the
end of the archive**, not a random sample. For a speaker-conditioned task, silently losing entire
speaker identities is worse than losing scattered utterances, and nothing would have reported it.
**CLEANED UP 2026-09-20 (user approved after verification):** deleted the partial
`data/raw/LibriSpeech/train-clean-360` and the redundant `data/raw/train-other-500.tar.gz` (~46GB
freed). Safety checked first: train-other-500's extracted copy is complete (148,688 flac, confirmed
by job 11223's own verify pass with a successful sample load, matching the published count), and the
only `tar.gz` references anywhere in `scripts/` are the download script building its own URL.
**`data/raw/train-clean-360.tar.gz` (21.5GB, sha256-verified) is KEPT — it is the source for the
re-extract. Do not delete it until the split extracts to the full 104,014.**
Still reclaimable if needed: `wham_noise` (~36GB, Libri2Mix already built from it), and
dev-clean/dev-other/test-other `.tar.gz` (~0.9GB). train-clean-100.tar.gz and test-clean.tar.gz were
already removed by the user earlier.

**ATTEMPT 4 (job 11228, 2026-09-20) — EXTRACTION FAILED AGAIN, BUT THIS ONE IS INFRASTRUCTURE, NOT
SPACE AND NOT OUR CODE.** Everything up to `tar` worked perfectly: wget saw the complete local
tarball and returned `416 Requested Range Not Satisfiable -> "already fully retrieved"` (resume logic
validated), `sha256 OK`, and the new space check passed with **501GB free** (someone freed a lot;
83GB -> 501GB, 98% not 100%). Then tar produced **361 x "Cannot close: Input/output error" + 30 x
"Bad file descriptor"**. EIO at close() on an NFS mount with 501GB free is a storage-layer fault —
same category as the munge daemon dying in job 10790 or the gnode2 GPU squatter. **The hardening did
its job**: partial directory auto-removed, tarball kept and still verified, so the state is clean and
a retry costs only extraction time.
**Mitigation added**: `tar ... --no-same-owner --no-same-permissions`. Extracting ~104k small files
onto shared NFS as non-root means tar's ownership/mode restores are extra metadata round-trips NFS
can reject; dropping them removes a failure class and speeds extraction. It will NOT fix a genuinely
sick fileserver — **if this recurs, it is one for the HPC admins.**
CURRENT STATE 2026-09-20: no partial dir, `data/raw/train-clean-360.tar.gz` (21,982M) intact and
sha256-verified. Retry is just `sbatch scripts/download_split_resumable.sh train-clean-360`.

**`scripts/download_split_resumable.sh` HARDENED 2026-09-19/20 with what jobs 11223 AND 11224 taught:**
- **Space checked immediately BEFORE `tar`, not just at job start** — the 11224 bug exactly. Refuses
  to extract rather than leaving a partial dir, and lists what is safe to reclaim.
- **Completeness verified by FLAC COUNT against a per-split table** (train-clean-360 = 104,014), not
  by directory existence. A short extract is deleted automatically; a pre-existing partial dir is
  rejected with expected-vs-actual rather than silently skipped.
- Failed extraction auto-removes its partial directory.
- **sha256 verification** against `146a5649...792ecf` — the authoritative value, learned from
  11223's own rejection message. Falls back to `tar -tzf` for splits with no known hash.
- **Distinguishes partial from corrupt**: wget exit != 0 means incomplete -> keep the file and say
  "resubmit to resume"; wget exit 0 but hash mismatch means corruption -> say to delete and restart,
  because resuming cannot repair it.
- **Dropped `-O`** — with `-O`, `wget -c` cannot resume correctly; `-P` plus default naming is what
  makes `-c` work. Verified the resulting path matches what the script expects.
Nothing is extracted until verification passes, so a bad download can never leave a half-extracted
split dir that downstream existence checks read as "done".

**(historical) RETRY RUNNING BUT AT RISK (job 11223, started 2026-09-19 ~11:57).** Guard fix confirmed working
("Missing splits: train-clean-360 (~23.1GB)", "Need >=52GB", 93GB passed). Progress 9.11G/21.5G at
4h42m = **1.94 GB/h, SLOWER than attempt 1's 2.8 GB/h**, with the instantaneous rate swinging
229-577 kB/s. Against the 12h limit that is 6h (at 577kB/s, just fits) to 15h (at 229kB/s, times
out). **torchaudio does NOT resume** — `torch.hub.download_url_to_file` writes a temp file, so a
timeout restarts all 21.5GB from zero, exactly as job 11219 did.

**BUILT 2026-09-19: `scripts/download_split_resumable.sh <split>` (defaults train-clean-360).**
Uses `wget -c` straight from `https://www.openslr.org/resources/12/<split>.tar.gz` instead of
torchaudio, so every resubmission resumes and a timeout costs time but never bytes — the same reason
`download_wham.sh` survived a mid-job kill. Also **verifies the archive with `tar -tzf` BEFORE
extracting**, because a truncated tarball extracts partially and then the split dir EXISTS, which
every downstream existence check reads as "done" (the job-10817 trap). Skips instantly if the split
dir is already there; keeps the tarball so a re-extract needs no re-download. bash -n clean, 0 CR.
**Use this if job 11223 times out** — do NOT just resubmit the torchaudio path, it starts over.

**GOTCHA for the Stage-1 retrain:** `data/librispeech.py:184` prints
`[Dataset] Warning: <path> not found, skipping.` for a missing split and carries on. `train_splits`
still lists train-clean-360, so if it is still absent the retrain will quietly train on less data
than intended — **grep that warning in the training log before trusting the run.**

**(historical) DOWNLOAD IN PROGRESS (job 11219, started 2026-09-18 ~16:19).** train-clean-100 correctly skipped
(28,539 flac). On train-clean-360: 7.53G/21.5G after 2h43m = **~0.8 MB/s, 2.8 GB/h**. Projection
against the 12h limit: 360 finishes (~5h more) but it will **time out ~4h into train-other-500**
(needs ~10.8h). **RECOMMENDED: drop train-other-500** — the whole evaluation is test-clean and
Libri2Mix mix_clean, while "other" is noisier/accented speech from a distribution never measured;
train-clean-360 alone is the 100h -> 460h (4.6x) win. `download.py` already defines
`--splits recommended` as exactly "everything except train-other-500".
**BEFORE ANY RESUBMIT: delete a partial `train-other-500/` dir and its tarball** — the skip-check is
`split_path.exists() and any(rglob("*.flac"))`, so a half-extracted split is treated as complete
(the trap that silently broke Libri2Mix generation in job 10817).

**COMMANDS:**
```
sbatch scripts/download_librispeech_full.sh                       # Step 1, ~53GB, gates Tier 2
# solver at EQUAL compute to the deployed euler@4 -- the only fair comparison:
sbatch scripts/run_eval_candidate_gpu.sh checkpoints_v2/flow/flow_best.pt heun2 \
    checkpoints_v2/masking/mask_best.pt 2.5 0 0 --solver heun --n_steps 2
sbatch scripts/run_eval_candidate_gpu.sh checkpoints_v2/flow/flow_best.pt projbest \
    checkpoints_v2/masking/mask_best.pt 2.5 0 0 \
    --projection_ckpt checkpoints_speaker_encoder/projection_best.pt
```
Baseline to beat, job 11112: 2/3/4spk 85.7/80.0/77.0, corpus-wide 86.2%, low-SNR 80.3%, in-domain
SI-SDR vs mixture +1.75dB, Libri2Mix ALL +3.39dB.
**CAVEAT on the enrollment sweep:** a longer reference means more zero-padding for short utterances;
`trim_trailing_silence` handles it, but check the printed `reference_length` in the log and be alert
for the padding-dilution failure mode in [[mask2flow-tse-fixed-bugs]].
