---
name: feedback-keep-memory-current
description: "User wants memory updated as work proceeds, not just at the end, so any session can resume mid-task"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 92e3d418-e0ac-4d2e-962a-e7f2f76ff0a1
  modified: 2026-09-16T13:03:52.047Z
---

Update the memory files AS WORK PROCEEDS — after each meaningful step, not only when a task
finishes. The user asked for this explicitly on 2026-09-16 ("update your memory with each prompt so
that we can continue anytime from where we left off").

**Why:** this project runs on an HPC cluster where a single experiment is a 4-13h training job plus
a 2-6h eval battery, and the user works across many separate sessions. Context that lives only in a
conversation is lost when that session ends, and the cost of re-deriving it is a wasted GPU day.
It has already bitten us: a whole line of work (the onset-damping investigation, jobs 11171-11187)
was missing from memory and a fresh session had to reconstruct it from SLURM logs.

**How to apply:** when a step lands — code written, job submitted, result read, decision made —
write it to the relevant project memory before moving on. Record enough that a cold session can pick
up: job IDs, what was measured, what the pre-registered bar was, what is still pending, and the exact
next command. Prefer updating the existing topic file (e.g. [[mask2flow-tse-onset-gate]]) over
creating new ones. Keep `MEMORY.md` one line per topic.

**Corollary — READ the project's own docs before proposing work.** On this project
`docs/methodology_and_project_history.md` (a numbered timeline, entry 35+ as of 2026-09-16) and
`docs/results_and_limitations.md` are the authoritative record, and they are usually AHEAD of these
memory files. On 2026-09-16 I recommended building an onset-padding fix without reading them; entry
32 already said "padding does not fix it" and already carried the caveat that the numbers I was
citing came from a different sample draw. The result was a re-derived conclusion at the cost of a
full eval battery. Check the timeline first, then memory, then propose.

Related: [[mask2flow-tse-overview]] for the environment this constraint comes from,
[[mask2flow-tse-onset-gate]] for the episode this corollary came from.
