"""
Checks for Stage 1's mask_mode (models/masking.py) and the checkpoint plumbing
that carries it: training/train_mask.py save_checkpoint(extra=...),
eval/results_stage2.py load_masking, training/train_flow.py load_frozen_masking.
See docs/methodology_and_project_history.md entry 21. Needs only torch.

Run from the repo root:  python test_mask_formulation.py
"""
import ast
import contextlib
import io
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from models.masking import MaskingModule  # noqa: E402
from training.ema import EMA  # noqa: E402

fails = 0


def check(name, ok):
    global fails
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    fails += 0 if ok else 1


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def extract(relpath, name, ns):
    """Pull one top-level function out of a module whose imports need the cluster env."""
    path = os.path.join(ROOT, relpath)
    tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert body, f"{name} not found in {relpath}"
    exec(compile(ast.Module(body=body, type_ignores=[]), path, "exec"), ns)
    return ns[name]


KW = dict(n_mels=80, embed_dim=512, conv_channels=[1, 4, 6, 8, 8], lstm_hidden=416, lstm_dropout=0.1)
torch.manual_seed(0)
x = torch.randn(2, 80, 40) * 6 - 4                      # log-mel-like: mostly negative
d = torch.nn.functional.normalize(torch.randn(2, 512), dim=-1)

mult = quiet(MaskingModule, **KW).eval()
with torch.no_grad():
    y_m, m_m = mult(x, d)
check("default mask_mode is the paper's multiplicative form", mult.mask_mode == "multiplicative")
check("multiplicative output is exactly x * mask", torch.equal(y_m, x * m_m))
check("multiplicative RAISES some negative bins (not pure deletion)", bool((y_m[x < 0] > x[x < 0]).any()))

lg = quiet(MaskingModule, **KW, mask_mode="log_gain")
lg.load_state_dict(mult.state_dict(), strict=True)
lg.eval()
with torch.no_grad():
    y_l, m_l = lg(x, d)
check("same weights load strictly into log_gain and give the same mask", torch.equal(m_l, m_m))
check("log_gain output is x + log(mask)", torch.allclose(y_l, x + torch.log(m_l), atol=1e-4))
check("log_gain never raises ANY bin, positive or negative (true deletion)", bool((y_l <= x).all()))
D, I = lg.compute_di_proportion(x, y_l)
check("log_gain D/I is 100% / 0%", I == 0.0 and D > 99.999)
try:
    quiet(MaskingModule, **KW, mask_mode="bogus")
    check("unknown mask_mode raises ValueError", False)
except ValueError:
    check("unknown mask_mode raises ValueError", True)

st = quiet(MaskingModule, **KW, mask_mode="log_gain").train()
with torch.no_grad():
    st.output_proj.weight.zero_()
    st.output_proj.bias.fill_(-200.0)                   # sigmoid underflows; log(sigmoid) = -inf
out, _ = st(x, d)
((out - x) ** 2).mean().backward()
g = st.output_proj.bias.grad
check("logits of -200: output finite, gradient finite and non-zero",
      bool(torch.isfinite(out).all()) and bool(torch.isfinite(g).all()) and g.abs().sum().item() > 0)

save_checkpoint = extract("training/train_mask.py", "save_checkpoint",
                          {"torch": torch, "nn": nn, "os": os, "Path": Path, "EMA": EMA, "DictConfig": object})
load_masking = extract("eval/results_stage2.py", "load_masking", {"torch": torch, "MaskingModule": MaskingModule})
load_frozen = extract("training/train_flow.py", "load_frozen_masking",
                      {"torch": torch, "MaskingModule": MaskingModule, "Path": Path, "DictConfig": object})

with tempfile.TemporaryDirectory() as tmp:
    cfg = NS(paths=NS(checkpoint_dir=tmp), mel=NS(n_mels=80), speaker_encoder=NS(embed_dim=512),
             masking=NS(conv_channels=[1, 4, 6, 8, 8], lstm_hidden=416, lstm_dropout=0.1))
    ema = quiet(EMA, lg, decay=0.9)
    for k in ema.shadow:                                # make EMA distinguishable from raw params
        ema.shadow[k] = ema.shadow[k] + 0.5
    opt = torch.optim.AdamW(lg.parameters(), lr=1e-4)
    quiet(save_checkpoint, 7, lg, ema, opt, 0.1, cfg, tag="best", stage="s", prefix="new",
          extra={"mask_mode": "log_gain"})
    quiet(save_checkpoint, 7, lg, ema, opt, 0.1, cfg, tag="latest", stage="s", prefix="old")
    new, old = os.path.join(tmp, "s", "new_best.pt"), os.path.join(tmp, "s", "old_latest.pt")
    check("save_checkpoint(extra=...) stores mask_mode", torch.load(new).get("mask_mode") == "log_gain")
    check("save_checkpoint without extra adds no key (payload as before)", "mask_mode" not in torch.load(old))
    m_new, m_old = quiet(load_masking, new, cfg, "cpu"), quiet(load_masking, old, cfg, "cpu")
    check("load_masking restores log_gain; old checkpoint loads multiplicative",
          m_new.mask_mode == "log_gain" and m_old.mask_mode == "multiplicative")
    check("load_masking still overlays EMA params",
          all(torch.equal(p, ema.shadow[n]) for n, p in m_new.named_parameters() if n in ema.shadow))
    f_new, f_old = quiet(load_frozen, cfg, "cpu", new), quiet(load_frozen, cfg, "cpu", old)
    check("load_frozen_masking restores log_gain and freezes; old checkpoint multiplicative",
          f_new.mask_mode == "log_gain" and f_old.mask_mode == "multiplicative"
          and not any(p.requires_grad for p in f_new.parameters()))

print(f"\n{'ALL PASSED' if fails == 0 else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
