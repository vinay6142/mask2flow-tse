"""Unit test for the ODE solvers in models/flow.py FlowMatchingModule.inference().

Exercises the REAL solver code: builds a tiny FlowMatchingModule and replaces
its forward() with an analytic velocity field whose exact integral is known by
hand, so each solver's output can be checked against the true answer rather than
against another implementation.

The properties that matter:
  1. solver="euler" (the default) is unchanged -- same arithmetic, same number
     of forward passes, in the same order, as before solvers were added. Every
     number in docs/ was produced by this path.
  2. heun/midpoint cost 2 velocity evaluations per step, so heun at n_steps=2
     uses the SAME budget as euler at n_steps=4 -- that is the only fair
     comparison, and the launcher/config comments say so.
  3. On a field where Euler is provably wrong, the 2nd-order solvers are
     provably exact -- which is the entire reason to offer them.

Run: python test_flow_solvers.py
"""
import sys
import torch

from models.flow import FlowMatchingModule, SOLVERS

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def tiny_model():
    # small enough to build instantly on CPU; the solver code under test does
    # not depend on width/depth
    return FlowMatchingModule(n_mels=8, hidden_dim=32, n_heads=2, n_blocks=1,
                              ffn_mult=1, embed_dim=16, dropout=0.0, cfg_dropout=0.0)


class Field:
    """Analytic velocity field + a call counter, substituted for forward()."""

    def __init__(self, fn):
        self.fn = fn
        self.n_calls = 0
        self.seen_t = []

    def __call__(self, x, t, d_vector, force_cfg_drop=False):
        self.n_calls += 1
        self.seen_t.append(float(t[0]))
        return self.fn(x, t)


def run(model, field, x0, **kw):
    model.forward = field
    d = torch.zeros(x0.shape[0], 16)
    return model.inference(x0, d, **kw)


print("=" * 70)
print("  FlowMatchingModule.inference() -- ODE solver test")
print("=" * 70)

m = tiny_model()
x0 = torch.zeros(2, 8, 5)

# ── 1. constant field: every solver is exact ────────────────────────────────
print("\n[1] constant field v=3 -> exact answer is x0 + 3 for ANY solver/steps")
for s in SOLVERS:
    for n in (1, 2, 4):
        f = Field(lambda x, t: torch.full_like(x, 3.0))
        out = run(m, f, x0, n_steps=n, solver=s)
        check(f"{s} n_steps={n} integrates a constant exactly",
              torch.allclose(out, torch.full_like(x0, 3.0), atol=1e-6),
              f"got {float(out[0,0,0]):.6f}")

# ── 2. v = 2t : Euler is provably wrong, 2nd-order provably exact ───────────
# exact: x(1) = x0 + int_0^1 2t dt = x0 + 1
# euler with n steps: sum_k 2(k/n)(1/n) = (n-1)/n   -> 0.75 at n=4
# heun/midpoint: exact for a field linear in t, any n
print("\n[2] v=2t -> exact integral is 1.0; Euler must undershoot, others must not")
f = Field(lambda x, t: 2.0 * t.view(-1, 1, 1).expand_as(x))
out_e4 = run(m, f, x0, n_steps=4, solver="euler")
euler_calls = f.n_calls
check("euler n_steps=4 gives the hand-derived 0.75",
      abs(float(out_e4[0, 0, 0]) - 0.75) < 1e-6, f"got {float(out_e4[0,0,0]):.6f}")
check("euler used 4 velocity evals", euler_calls == 4, f"got {euler_calls}")

f = Field(lambda x, t: 2.0 * t.view(-1, 1, 1).expand_as(x))
out_h2 = run(m, f, x0, n_steps=2, solver="heun")
heun_calls = f.n_calls
check("heun n_steps=2 is EXACT (1.0)",
      abs(float(out_h2[0, 0, 0]) - 1.0) < 1e-6, f"got {float(out_h2[0,0,0]):.6f}")
check("heun n_steps=2 used the SAME 4 evals as euler n_steps=4",
      heun_calls == euler_calls, f"got {heun_calls} vs {euler_calls}")

f = Field(lambda x, t: 2.0 * t.view(-1, 1, 1).expand_as(x))
out_m2 = run(m, f, x0, n_steps=2, solver="midpoint")
mid_calls = f.n_calls
check("midpoint n_steps=2 is EXACT (1.0)",
      abs(float(out_m2[0, 0, 0]) - 1.0) < 1e-6, f"got {float(out_m2[0,0,0]):.6f}")
check("midpoint n_steps=2 used 4 evals", mid_calls == 4, f"got {mid_calls}")

check("at equal budget, both 2nd-order solvers beat euler",
      abs(float(out_h2[0, 0, 0]) - 1.0) < abs(float(out_e4[0, 0, 0]) - 1.0))

# ── 3. x-dependent field: v = x, exact x(1) = x0 * e ────────────────────────
print("\n[3] v=x -> exact is x0*e; 2nd-order should land closer at equal budget")
x1 = torch.ones(1, 8, 3)
E = float(torch.exp(torch.tensor(1.0)))
f = Field(lambda x, t: x)
e4 = float(run(m, f, x1, n_steps=4, solver="euler")[0, 0, 0])
f = Field(lambda x, t: x)
h2 = float(run(m, f, x1, n_steps=2, solver="heun")[0, 0, 0])
check("heun n=2 closer to e than euler n=4 (same 4 evals)",
      abs(h2 - E) < abs(e4 - E), f"euler {e4:.5f} heun {h2:.5f} exact {E:.5f}")

# ── 4. euler path unchanged (regression against the pre-solver arithmetic) ──
print("\n[4] the default path still matches the original Euler loop exactly")
f = Field(lambda x, t: 2.0 * t.view(-1, 1, 1).expand_as(x) + 0.5 * x)
got = run(m, f, x0 + 0.3, n_steps=4, solver="euler")

# hand-rolled original loop
x = (x0 + 0.3).clone()
dt = 1.0 / 4
for step in range(4):
    t = torch.full((x.shape[0],), step * dt)
    v = 2.0 * t.view(-1, 1, 1).expand_as(x) + 0.5 * x
    x = x + v * dt
check("default solver == original Euler iteration, bit for bit",
      torch.equal(got, x))

import inspect
# inference() is wrapped by @torch.no_grad(), so read the signature rather than
# __defaults__ (which lives on the wrapper and is None).
check("default argument really is euler",
      inspect.signature(FlowMatchingModule.inference).parameters["solver"].default == "euler")

# ── 5. cfg_warmup still applies per step under every solver ────────────────
print("\n[5] cfg_warmup_steps and guidance still work under each solver")
for s in SOLVERS:
    f = Field(lambda x, t: torch.full_like(x, 1.0))
    run(m, f, x0, n_steps=2, cfg_scale=2.0, cfg_warmup_steps=1, solver=s)
    # step0 unguided = 1 eval for euler / 2 for 2nd-order;
    # step1 guided = 2 evals (cond+uncond) for euler, 4 for 2nd-order
    expect = 3 if s == "euler" else 6
    check(f"{s}: warmup step unguided, later step guided ({expect} evals)",
          f.n_calls == expect, f"got {f.n_calls}")

# ── 6. bad input rejected ──────────────────────────────────────────────────
print("\n[6] an unknown solver is rejected, not silently ignored")
try:
    run(m, Field(lambda x, t: x), x0, n_steps=1, solver="rk4")
    check("unknown solver raises ValueError", False, "no exception")
except ValueError:
    check("unknown solver raises ValueError", True)

print("\n" + "=" * 70)
print(f"  {PASS}/{PASS + FAIL} checks passed")
print("=" * 70)
sys.exit(1 if FAIL else 0)
