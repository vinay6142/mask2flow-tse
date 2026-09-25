---
name: mask2flow-tse-git-authorship
description: "User's git workflow preference for mask2flow_tse -- they commit/push themselves, Claude never appears as contributor"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 314006b1-f0de-4a42-b9fe-887c5480a146
  modified: 2026-09-09T16:52:12.124Z
---

For the mask2flow_tse repo (and treat as the default for the user's own repos generally unless
told otherwise): the user runs `git commit`/`git push` themselves — Claude should not execute git
commit/push on their behalf, and should not suggest itself as a co-author (no `Co-Authored-By:
Claude` trailer, no Claude identity in author/committer fields). Commits should show the user's own
name as sole contributor.

**Why:** stated directly 2026-09-09 after Claude found 81 uncommitted files (including the entire
[[mask2flow-tse-overview]] project's bug fixes, fine-tune training code, and thesis docs — only one
commit existed in the whole repo's history) and was about to propose grouped commits. The user
interrupted specifically to say they'll do the committing personally.

**How to apply:** if git state is worth flagging (uncommitted work, risk of loss, a sensible commit
grouping), still surface that — the finding itself is useful. But stop short of running `git add`/
`commit`/`push`, and don't dump a step-by-step command list either; describe what's going on and
let the user drive the actual git operations. If asked to draft commit messages as reference text
(not to execute), that's fine — just don't include Claude as an author/co-author anywhere in them.
