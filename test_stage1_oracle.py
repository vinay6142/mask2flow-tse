"""
Checks for the Stage-1 oracle helpers in eval/eval_multi_speaker.py
(oracle_stage1, formulation_stats) -- see docs/methodology_and_project_history.md
timeline entry 19 for why they exist.

Needs only torch: the two functions are extracted from the module with ast, so
the module's other imports (omegaconf, torchaudio, transformers, ...) are never
loaded. Same spirit as test_chunk_overlap_add.py.

Run from the repo root, locally or on the cluster:
  python test_stage1_oracle.py
"""
import os
import ast
import sys

import torch

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval", "eval_multi_speaker.py")
tree = ast.parse(open(SRC, encoding="utf-8").read(), filename=SRC)
wanted = {"oracle_stage1", "formulation_stats"}
funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
assert {f.name for f in funcs} == wanted, f"missing: {wanted - {f.name for f in funcs}}"
ns = {"torch": torch}
exec(compile(ast.Module(body=funcs, type_ignores=[]), SRC, "exec"), ns)
oracle_stage1, formulation_stats = ns["oracle_stage1"], ns["formulation_stats"]

fails = 0


def check(name, ok, detail=""):
    global fails
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail and not ok else ""))
    fails += 0 if ok else 1


# One mel bin, 8 frames, covering every sign case. Expected values derived by hand:
# logmask = X * clip(Y/X, 0, 1), which can only land between X and 0;
# energy  = min(X, Y).
X = torch.tensor([[4.0, 4.0, 4.0, -10.0, -10.0, -10.0, 0.0, 1e-9]])
Y = torch.tensor([[2.0, 6.0, -3.0, -5.0, -15.0, 2.0, -5.0, 0.0]])
lm = oracle_stage1(X, Y, "oracle_logmask")
en = oracle_stage1(X, Y, "oracle_energy")
check("output shape is (1, F, T)", lm.shape == (1, 1, 8) and en.shape == (1, 1, 8))
check("oracle_logmask matches hand-derived values",
      torch.allclose(lm[0], torch.tensor([[2.0, 4.0, 0.0, -5.0, -10.0, 0.0, 0.0, 0.0]]), atol=1e-6), lm[0].tolist())
check("oracle_energy matches hand-derived values",
      torch.allclose(en[0], torch.tensor([[2.0, 4.0, -3.0, -10.0, -15.0, -10.0, -5.0, 0.0]]), atol=1e-6), en[0].tolist())
check("logmask CANNOT make a negative bin quieter (X=-10, Y=-15 stays -10)", abs(lm[0, 0, 4].item() + 10) < 1e-6)
check("energy oracle CAN (same bin reaches -15)", abs(en[0, 0, 4].item() + 15) < 1e-6)
zero = torch.zeros_like(X)
check("logmask output always lies between X and 0",
      bool(((lm[0] >= torch.minimum(X, zero) - 1e-6) & (lm[0] <= torch.maximum(X, zero) + 1e-6)).all()))
try:
    oracle_stage1(X, Y, "bogus")
    check("unknown mode raises ValueError", False)
except ValueError:
    check("unknown mode raises ValueError", True)

# The logmask oracle must be the best [0,1] mask in every bin, i.e. the minimiser of
# the Stage 1 loss (X*M - Y)^2. Compare against a fine grid of masks on random bins.
g = torch.Generator().manual_seed(0)
Xr, Yr = torch.randn(80, 50, generator=g) * 8, torch.randn(80, 50, generator=g) * 8
best = (oracle_stage1(Xr, Yr, "oracle_logmask")[0] - Yr) ** 2
grid = torch.linspace(0, 1, 201).view(-1, 1, 1)
grid_best = ((Xr.unsqueeze(0) * grid - Yr.unsqueeze(0)) ** 2).min(dim=0).values
check("logmask is per-bin optimal over [0,1] (vs 201-point grid, 4000 bins)",
      bool((best <= grid_best + 1e-4).all()), f"max excess {(best - grid_best).max().item():.2e}")

# formulation_stats: 4 real frames, then 2 padded frames chosen so that counting
# them would change every statistic.
Xs = torch.tensor([[4.0, 4.0, -10.0, -10.0, -10.0, -10.0]])
Ys = torch.tensor([[2.0, 4.0, -5.0, -15.0, 0.0, 0.0]])
Ss = torch.tensor([[[2.0, 4.0, -5.0, -10.0, 0.0, 0.0]]])   # network: halves a loud bin, raises a quiet one
fm = torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0]])
st = formulation_stats(Xs, Ys, Ss, fm)
check("frac_mix_bins_negative = 2/4 (padding excluded)", abs(st["frac_mix_bins_negative"] - 0.5) < 1e-6, st)
check("frac_target_unreachable = 1/4 (only X=-10, Y=-15)",
      abs(st["frac_target_unreachable_by_any_mask"] - 0.25) < 1e-6, st)
check("network_frac_bins_raised = 1/4", abs(st["network_frac_bins_raised"] - 0.25) < 1e-6, st)
check("network_insert_pct = 5/(2+5) = 71.43%", abs(st["network_insert_pct"] - 500 / 7) < 1e-3, st)
st0 = formulation_stats(Xs, Ys, Ss, torch.zeros_like(fm))
check("all-padding frame mask falls back to all frames, no crash",
      abs(st0["frac_mix_bins_negative"] - 4 / 6) < 1e-6, st0)

print(f"\n{'ALL PASSED' if fails == 0 else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
