---
name: mask2flow-tse-thesis-writeup
description: Status of the Mask2Flow-TSE thesis writeup draft (Results and Limitations chapter)
metadata: 
  node_type: memory
  type: project
  originSessionId: 7e1688b4-2460-4349-9353-8129f100dee7
  modified: 2026-09-20T13:05:49.150Z
---

Part of [[mask2flow-tse-overview]]. On 2026-09-02, started drafting the thesis Results/Limitations
chapter at `docs/results_and_limitations.md` (new `docs/` dir — none existed before). Covers:
setup/checkpoints table, the three-tier metric framing decided that session (SI-SDRi primary,
speaker-verification accuracy secondary, mel-MSE/catastrophic-rate as diagnostic only — not a
headline, since it's unbounded and non-standard outside this project), the hard-t0 fine-tune
before/after table, the Libri2Mix cross-corpus section, both known limitations (t≈0
training-coverage gap — diagnosed+3.6x-mitigated; SNR<1dB coverage gap — characterized, not yet
addressed) written up as textbook framing (not the raw investigative notes), a Future Work section,
and a repro-commands appendix. Source of truth for every number in it is
`eval/KNOWN_LIMITATIONS.md` + [[mask2flow-tse-eval-results]] — this file is meant to be the single
copy, not forked; update it in place if any eval reruns.

**Update 2026-09-02:** added §5.5.3 (speaker-count generalization / 3-speaker mixtures — see
[[mask2flow-tse-multi-speaker-test]] for the full backstory) and a matching §5.6 future-work bullet
and appendix repro command. The chapter now documents THREE distinct characterized limitations
(t≈0 gap, SNR-coverage gap, speaker-count-coverage gap), each with its own root cause and none
conflated with the others.

**Update 2026-09-04/05:** added §5.7 (checkpoint re-promotion — hard-multispeaker fine-tune — full
refreshed headline + six-point "why" analysis, nothing prior overwritten) and §5.8 (qualitative
listening verification — see [[mask2flow-tse-listening-samples]]: user confirmed extraction sounds
correct across 2/3/4 speakers, explicitly said not to touch the pipeline further). The
qualitative/listening-test discussion flagged as future work below is now DONE, at §5.8.

**Update 2026-09-05: all remaining chapters drafted**, in a new file
`docs/methodology_and_project_history.md` (companion to `docs/results_and_limitations.md`, which
stays the source of truth for numbers): Introduction/Motivation, Related Work (citations from memory,
flagged for the user to verify before formal submission -- did NOT fabricate specific paper details,
named real well-known works: VoiceFilter, SpeakerBeam, Conv-TasNet, SpEx, WavLM, rectified
flow/flow matching, DiT, classifier-free guidance, HiFi-GAN), Methodology/Architecture (5 Mermaid
diagrams -- pipeline overview, Stage 1 masking, Stage 2 network function, Stage 2 Euler-integration
loop, vocoder -- built to REPLACE two user-supplied hand-drawn PNG diagrams after auditing them
against the actual code), a 13-entry chronological Development Timeline (the "what we did, why,
result, what's next" narrative the user explicitly asked for), a one-paragraph Results Summary
pointing to the results doc, and a Conclusion.

**Diagram audit findings (this is why the images needed replacing, not just reformatting):**
checked line-by-line against `models/masking.py`, `models/flow.py`, `models/speaker_encoder.py`,
`vocoder_train/hifigan.py`. Real errors found and corrected: (1) the reference-audio path was drawn
going through the mixture's mel extractor before the Speaker Encoder -- wrong, WavLM consumes the
RAW waveform directly via its own internal feature extraction, never touches this project's mel
pipeline; (2) Stage 2 inference was drawn as a SINGLE Euler step -- correct only for the paper's
`n_steps=1` default, but this project's actual deployed config (`configs/default_v2.yaml`,
every eval command throughout the whole project) runs n_steps=4 in a loop, each step re-running the
full DiT stack twice (CFG cond+uncond) -- redrawn as two diagrams (network function + outer loop)
matching how the code itself is structured; (3) a final AdaLN modulation layer (γ,β from c) between
the last DiT block and the output projection was missing from the diagram entirely (small but
structurally real -- confirmed via exact parameter-count match against the training log's "Output:
1,242,704 params" line); (4) vocoder's "Element-wise Sum" is actually an average (÷3,
`num_kernels`) not a raw sum. Masking-module diagram (Stage 1) had NO errors -- matched the code
exactly as drawn.

Mermaid chosen over trying to edit the pasted PNGs directly (not possible with available tools) --
renders natively in GitHub/VS Code Markdown preview, and stays a live, checked-into-the-repo text
artifact instead of a static image that can silently drift from the code again.

The results/limitations chapter (§5, in the OTHER doc) remains the single source of truth for every
number; this new doc's Results Summary section explicitly does not duplicate those tables. Thesis
docs are now essentially complete: both files exist, cross-reference each other, and cover intro
through conclusion. Nothing else queued unless the user asks for something new (e.g. converting to
a different format like LaTeX/Word).

**Update 2026-09-09 — §2 Related Work citations verified via WebSearch against arXiv/publisher
records (previously flagged as "from memory, unverified").** All checked out, including the core
Mask2Flow-TSE paper itself (arXiv:2603.12837, Junwon Moon et al., Sungkyunkwan/Seoul, submitted
Interspeech 2026 — confirms this isn't a hallucinated foundation; paper's own ~85M-param TSE network
matches this project's Stage1+Stage2 ~88.5M closely, a good independent scale sanity-check). One
real fix applied: SpEx+ was misattributed to "Xu et al." — actual first author is Ge (Meng Ge,
Chenglin Xu as co-author, Interspeech 2020, arXiv:2005.04686); corrected in the doc. Also softened
the section's caveat line to note the verification and flag the arXiv-vs-venue-year convention
choice (VoiceFilter 2018/2019, WavLM 2021/2022) for whichever bibliography style the target venue
wants. Every other citation (SpeakerBeam/TD-SpeakerBeam, Conv-TasNet, SpEx, WavLM, rectified flow,
flow matching, Voicebox, DiT, classifier-free guidance, HiFi-GAN) matched author/venue/year exactly
as written. This closes next-steps item "verify Related Work citations" — no longer open.

**Update 2026-09-13 — results chapter brought back in line after the low-SNR campaign.** It had
drifted: §5.5.2 was still titled "characterized, not yet addressed" and §5.6 still listed "extend
snr_min downward" as future work, both false since 2026-09-12/13, and its headline stopped at the
2026-09-04 promotion (two promotions and a guidance re-tune out of date). Added `## 5.10 Low-SNR
campaign` before the Appendix — what changed, the before/after table, how the constraint was located
(two Stage-2 curricula rejected, then the oracle decomposition pointing at Stage 1), the mask-formulation
limitation and why log_gain isn't adoptable without retraining Stage 2, the two method notes (the
pre-registered bar that was overridden with its evidence; an oracle bounding the PAIR not one stage),
and a current-checkpoint table. Status-only edits elsewhere: §5.5.2 heading, the §5.6 bullet marked
DONE, and a note under §5.1's checkpoint table saying it is kept for reproducing §5.2-§5.6 only. NO
earlier number was altered — same discipline as §5.7 and §5.9. Timeline entries 14-29 in
methodology_and_project_history.md remain the chronological record.

**Update 2026-09-20 — PHASE ONE REPORT PUBLISHED, for the user's presentation/submission.**
Artifact: **https://claude.ai/artifact/NthjvPgADEfYETqh86gN3P** — republish the same scratchpad path
(or pass that URL as `url` from a new conversation) to update it; publishing without the URL creates
a SEPARATE artifact. Local copy kept at **`docs/phase1_report.html`** (94KB, self-contained).
Contents: 12 sections — summary, problem, architecture, training method, evaluation method, results,
**processes run** (13 cards: what each pipeline does, why it exists, what it produced), decision log
(all 15 candidates with reason-for-attempt and reason-for-verdict), ceiling analysis, limitations,
engineering record, infrastructure, Phase Two. **4 Mermaid architecture diagrams** (the repo's own
verified ones, with two staleness fixes: the "pure deletion" claim and cfg 1.5 -> 2.5) and **4 SVG
charts** (accuracy chronology, oracle decomposition, SNR-bucket coverage gap, the low-SNR SI-SDR
swing). Print CSS included so Ctrl+P -> Save as PDF gives a clean document.
**NOTE on downloads:** the artifact sandbox BLOCKS page-initiated downloads (`<a download>`, blob/data
URLs, script-driven saves are all inert for viewers) — so never add a download button; print-to-PDF
plus a local repo copy is the working pattern.
Editorial stance chosen deliberately and worth preserving if it is revised: the REJECTIONS are
foregrounded as the substance (11 of 15), and the §7-K promotion that overrode two pre-registered
bars is stated plainly with its evidence rather than smoothed away — more credible to an examiner
than a clean record.

**Update 2026-09-16 — the onset-gate investigation written up and CLOSED; both docs are current as
of this date.** Timeline grew from the original 13 entries to **35**; entries 31-32 (written
2026-09-14) covered the discovery, and this session added:
- **Entry 33** — the 2x2 ablation result. Docs had described the 2x2 as merely *runnable*; the result
  is that the gate is an INTERACTION (A old/old 1/24, B newS1 1/24, C newS2 2/24, D deployed 7/24,
  superadditive; 4spk 0/8,0/8,0/8 -> 3/8). Removes the "revert the guilty promotion" remedy.
- **Entry 34** — the onsetfix Stage-2 fine-tune, rejected on its pre-registered <=1/24 bar at 4/24,
  plus the two infrastructure lessons of those days (HF_HUB_OFFLINE now exported by the launchers
  after jobs 11182/11183 burned their allocations on huggingface HEAD retries; the gnode2 GPU
  squatter that made job 11187 14x slower, caught via the nvidia-smi preamble).
- **Entry 35** — the inference-side repair: what was built, the paired battery (job 11202, no
  accuracy change), the gate (job 11205, 7->6/24), and why speaker-verification accuracy is
  structurally blind to a 0.5s defect (WavLM embedding is mean-pooled over the utterance — the same
  reason §5.8's battery missed it).
- Closing "Where this leaves the project" paragraph updated; §5.11 of `results_and_limitations.md`
  had its "Status: cause being localized" replaced with the resolved cause, the two-mitigation
  comparison table, and an honest statement of the current system.
Framing throughout: a characterized limitation with an established cause and two pre-registered
rejections — deliberately the same shape as the SNR-coverage gap, not papered over. Verified
structurally (entries 33/34/35 in sequence, 0 malformed tables). NO earlier number altered — same
discipline as §5.7/§5.9/§5.10.
NOTE for future sessions: these docs are AHEAD of the memory files and are the authoritative record —
read them before proposing experiments (see [[feedback-keep-memory-current]] for why that rule exists).

**Update 2026-09-09 — added a "parameter efficiency" bullet to §6 Future Work** (in
`docs/methodology_and_project_history.md`), from a real per-module param audit prompted by the
user's "reduce params / improve the model" question: WavLM-base-plus-sv frozen 94.4M (52% of the
~183M system total, the single biggest lever since it's untrained) vs. Stage 2's 9 DiT blocks at
74.4M/77.3M (96% of that module). Named concrete techniques (ECAPA-TDNN swap, ALBERT-style
block-sharing, pruning/low-rank factorization, CFG guidance-distillation, and the contemporaneous
MeanFlow-TSE arXiv:2512.18572 as a one-step-inference precedent) but deliberately NOT pursued as
actual experiments — nothing in the results chapter calls for it, and any of these would require
re-validating the full headline. Same session also chased down a red herring: the WavLM
`pos_conv_embed` "newly initialized" warning (visible in an old 2026-08-21 `terminal.txt` the user
reopened) looked like a live non-determinism bug but was directly verified as a harmless
transformers==4.44.2 cosmetic false-positive (values match the checkpoint exactly) — see new item 4
in [[mask2flow-tse-fixed-bugs]], not a bug, don't re-chase.
