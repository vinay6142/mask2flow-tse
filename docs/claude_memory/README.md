# Claude memory bundle

A snapshot of the assistant's project memory, committed to the repo so it travels with the code.
Taken **2026-09-24**, end of Phase One.

## Why this exists

Claude Code stores memory per machine, at `~/.claude/projects/<project-slug>/memory/`. A session on
a different machine — the web client at claude.ai/code, another laptop, a fresh checkout — starts
with an empty memory directory and cannot see this one. Committing the files here is how the context
moves.

## For a new session picking this up

Read in this order:

1. **`MEMORY.md`** — the index. One line per topic.
2. **`mask2flow-tse-master-timeline.md`** — chronological record of every process run in Phase One:
   what, why, job ID, outcome, plus the cross-cutting lessons.
3. **`mask2flow-tse-eval-results.md`** — current headline numbers and the ceiling analysis.
4. **`mask2flow-tse-next-steps.md`** — where Phase Two starts.

Then, to adopt this as your own memory, copy these files into your session's memory directory:

```bash
mkdir -p ~/.claude/projects/<project-slug>/memory
cp docs/claude_memory/*.md ~/.claude/projects/<project-slug>/memory/
```

The exact slug is derived from the working directory; on the machine this snapshot came from it was
`z--`. If unsure, ask Claude to check where its memory directory is before copying.

## Important: the repo docs outrank these files

These memory notes are working context, not the record. The authoritative documents are:

| File | Role |
|---|---|
| `docs/methodology_and_project_history.md` | 36-entry development timeline; thesis narrative |
| `docs/results_and_limitations.md` | Authoritative for every reported number |
| `docs/PROJECT_STATE.md` | Operational state: what is deployed, blocked, and next |
| `docs/phase1_report.html` | Presentation report with diagrams and charts |

The memory files have drifted behind those docs before, and it cost a full evaluation battery to
re-derive a conclusion `docs/` already contained. **Read the docs before proposing experiments.**

## Staleness warning

This is a point-in-time snapshot. Anything here about checkpoint state, disk contents, job results or
open work was true on 2026-09-24 and may not be now. Verify against the repo and the cluster before
acting on it — particularly:

- which checkpoints are promoted (`checkpoints_v2/`),
- what data is on disk (`data/raw/LibriSpeech/`),
- whether `train-clean-360` was ever extracted (it was blocked by an NFS fault at snapshot time).
