"""Unit test for models/flow.py inference_with_onset_splice().

Tests the REAL function (models/flow.py imports only os/sys/math/torch/typing,
so it loads under a plain CPU torch with no checkpoints and no conda env).

The two properties that matter:
  1. pad_frames=0 must be BIT-IDENTICAL to flow_model.inference(...), so every
     existing caller and every number already in docs/ is unaffected until the
     setting is deliberately raised.
  2. with pad_frames>0, everything past the crossfade must still be
     bit-identical to the unpadded run -- that is the whole reason splicing was
     chosen over global padding, which cost 0.24dB SI-SDR across the utterance
     (job 11174) and failed the promotion bar.

Run: python test_onset_splice.py
"""
import sys
import torch

from models.flow import inference_with_onset_splice, ONSET_PAD_VALUE

PASS = 0
FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}")


class FakeFlow:
    """inference() returns tag + frame_index, where tag identifies WHICH call
    this was. That makes every output frame traceable to the run it came from,
    so the splice boundaries can be asserted exactly rather than approximately.
    """

    def __init__(self):
        self.calls = []

    def inference(self, x_enh, d_vector, cfg_scale=1.0, n_steps=1, cfg_warmup_steps=0,
                  solver="euler"):
        self.calls.append({"T": x_enh.shape[-1], "cfg_scale": cfg_scale,
                           "n_steps": n_steps, "cfg_warmup_steps": cfg_warmup_steps,
                           "solver": solver,
                           "first_frame": float(x_enh[0, 0, 0])})
        B, M, T = x_enh.shape
        idx = torch.arange(T, dtype=x_enh.dtype).view(1, 1, T).expand(B, M, T)
        return 1000.0 * len(self.calls) + idx


def make_input(B=2, M=80, T=300):
    # distinctive non-zero content so a stray pad frame would be visible
    return torch.arange(B * M * T, dtype=torch.float32).reshape(B, M, T) * 0.01 + 5.0


print("=" * 70)
print("  inference_with_onset_splice() -- unit test")
print("=" * 70)

# -- 1. pad_frames=0 is the untouched path ------------------------------------
print("\n[1] pad_frames=0 must be bit-identical to plain inference()")
x = make_input()
d = torch.zeros(2, 512)

f_ref = FakeFlow()
ref = f_ref.inference(x, d, cfg_scale=2.5, n_steps=4)

f0 = FakeFlow()
got = inference_with_onset_splice(f0, x, d, cfg_scale=2.5, n_steps=4, pad_frames=0)

check("output equals plain inference()", torch.equal(got, ref))
check("exactly ONE forward pass (no wasted compute)", len(f0.calls) == 1)
check("input length unchanged", f0.calls[0]["T"] == x.shape[-1])
check("input content unchanged", f0.calls[0]["first_frame"] == float(x[0, 0, 0]))

# negative pad treated as disabled, not as a crash or a reversed slice
f_neg = FakeFlow()
got_neg = inference_with_onset_splice(f_neg, x, d, cfg_scale=2.5, n_steps=4, pad_frames=-5)
check("negative pad_frames also disabled", torch.equal(got_neg, ref))
check("negative pad_frames does one pass", len(f_neg.calls) == 1)

# -- 2. padded path: call bookkeeping -----------------------------------------
print("\n[2] pad_frames>0 runs both passes, with the pad on the second only")
PAD, SPLICE, XFADE = 12, 100, 20
T = x.shape[-1]

f = FakeFlow()
out = inference_with_onset_splice(f, x, d, cfg_scale=2.5, n_steps=4,
                                  cfg_warmup_steps=3, pad_frames=PAD,
                                  splice_frames=SPLICE, xfade_frames=XFADE)

check("two forward passes", len(f.calls) == 2)
check("pass 1 is unpadded (base)", f.calls[0]["T"] == T)
check("pass 2 is padded", f.calls[1]["T"] == T + PAD)
check("pad is PREPENDED (pass 2 starts with pad value)",
      f.calls[1]["first_frame"] == ONSET_PAD_VALUE)
check("output shape preserved", out.shape == x.shape)
for i, c in enumerate(f.calls):
    check(f"pass {i+1} forwards cfg_scale", c["cfg_scale"] == 2.5)
    check(f"pass {i+1} forwards n_steps", c["n_steps"] == 4)
    check(f"pass {i+1} forwards cfg_warmup_steps", c["cfg_warmup_steps"] == 3)

# -- 3. exact frame-by-frame reconstruction -----------------------------------
print("\n[3] every output frame comes from the right run")
# base pass was call 1 -> 1000 + i ; padded pass was call 2 -> 2000 + i, then
# sliced by PAD, so padded_slice[i] = 2000 + PAD + i
base_v = torch.arange(T, dtype=torch.float32) + 1000.0
padded_v = torch.arange(T, dtype=torch.float32) + 2000.0 + PAD

onset_ok = torch.equal(out[0, 0, :SPLICE], padded_v[:SPLICE])
check(f"frames [0,{SPLICE}) come from the PADDED run", onset_ok)

tail_ok = torch.equal(out[0, 0, SPLICE + XFADE:], base_v[SPLICE + XFADE:])
check(f"frames [{SPLICE + XFADE},T) are bit-identical to the UNPADDED run", tail_ok)

w = torch.linspace(1.0, 0.0, XFADE)
expect_x = w * padded_v[SPLICE:SPLICE + XFADE] + (1 - w) * base_v[SPLICE:SPLICE + XFADE]
check("crossfade region is the exact linear blend",
      torch.allclose(out[0, 0, SPLICE:SPLICE + XFADE], expect_x, atol=0, rtol=0))
check("crossfade starts fully padded-side", out[0, 0, SPLICE] == padded_v[SPLICE])

# every batch row and mel bin got the same treatment
check("all batch rows / mel bins identical treatment",
      torch.equal(out[0, 0], out[1, 79]))

# -- 4. edge cases ------------------------------------------------------------
print("\n[4] edge cases")

# splice longer than the utterance: everything is the padded run, no crossfade
T_short = 40
xs = make_input(B=1, M=80, T=T_short)
ds = torch.zeros(1, 512)
fs = FakeFlow()
out_s = inference_with_onset_splice(fs, xs, ds, pad_frames=PAD,
                                    splice_frames=SPLICE, xfade_frames=XFADE)
padded_s = torch.arange(T_short, dtype=torch.float32) + 2000.0 + PAD
check("splice_frames > T: whole output from padded run",
      torch.equal(out_s[0, 0], padded_s))
check("splice_frames > T: shape preserved", out_s.shape == xs.shape)

# crossfade would run off the end -> truncated, not an error
T_mid = SPLICE + 5
xm = make_input(B=1, M=80, T=T_mid)
fm = FakeFlow()
out_m = inference_with_onset_splice(fm, xm, ds, pad_frames=PAD,
                                    splice_frames=SPLICE, xfade_frames=XFADE)
check("short crossfade room: shape preserved", out_m.shape == xm.shape)
wm = torch.linspace(1.0, 0.0, 5)
base_m = torch.arange(T_mid, dtype=torch.float32) + 1000.0
padded_m = torch.arange(T_mid, dtype=torch.float32) + 2000.0 + PAD
expect_m = wm * padded_m[SPLICE:] + (1 - wm) * base_m[SPLICE:]
check("short crossfade room: blend uses the frames that exist",
      torch.allclose(out_m[0, 0, SPLICE:], expect_m, atol=0, rtol=0))

# xfade=0 -> hard cut, tail still exact
fh = FakeFlow()
out_h = inference_with_onset_splice(fh, x, d, pad_frames=PAD,
                                    splice_frames=SPLICE, xfade_frames=0)
check("xfade=0: onset from padded run",
      torch.equal(out_h[0, 0, :SPLICE], padded_v[:SPLICE]))
check("xfade=0: tail bit-identical to base",
      torch.equal(out_h[0, 0, SPLICE:], base_v[SPLICE:]))

# splice_frames=0 -> output is entirely the base run
fz = FakeFlow()
out_z = inference_with_onset_splice(fz, x, d, pad_frames=PAD,
                                    splice_frames=0, xfade_frames=0)
check("splice_frames=0: output is entirely the unpadded run",
      torch.equal(out_z[0, 0], base_v))

# -- 5. input is not mutated --------------------------------------------------
print("\n[5] the caller's tensor is left alone")
x_before = x.clone()
fi = FakeFlow()
inference_with_onset_splice(fi, x, d, pad_frames=PAD)
check("input tensor unmodified", torch.equal(x, x_before))

print("\n" + "=" * 70)
print(f"  {PASS}/{PASS + FAIL} checks passed")
print("=" * 70)
sys.exit(1 if FAIL else 0)
