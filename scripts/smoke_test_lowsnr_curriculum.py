"""
Pre-flight check for the low-SNR curriculum, BEFORE burning a ~13h GPU job.

Verifies three things:
  1. Everything imports (catches syntax/wiring errors in the new code).
  2. The DEFAULT path is genuinely unchanged -- curriculum off must draw SNR
     only from the configured [snr_min, snr_max], so every existing caller
     (train_flow.py, train_mask.py, all eval scripts) is unaffected.
  3. The curriculum path actually works -- roughly low_snr_prob of draws land
     in the new low range.

Run (seconds, CPU, no GPU needed):
  python3 scripts/smoke_test_lowsnr_curriculum.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from omegaconf import OmegaConf
from data.librispeech import LibriSpeechTSEDataset

cfg = OmegaConf.load("configs/default_v2.yaml")
LO, HI = cfg.data.snr_min, cfg.data.snr_max
print(f"configured trained SNR range: [{LO}, {HI}] dB\n")

def make_stub(low_snr_prob=0.0, low_snr_range=None):
    """Build just enough object to exercise the SNR draw, without scanning data."""
    d = object.__new__(LibriSpeechTSEDataset)
    d.cfg = cfg
    d.low_snr_prob = low_snr_prob
    d.low_snr_range = low_snr_range
    return d

N = 20000
fail = 0

print("=" * 64)
print("1. DEFAULT path (curriculum off) -- must be unchanged")
print("=" * 64)
d = make_stub()
vals = [d._draw_snr_value() for _ in range(N)]
out_of_range = [v for v in vals if v < LO or v > HI]
override = d._draw_snr_override()
print(f"  _draw_snr_value: min={min(vals):.2f} max={max(vals):.2f} "
      f"(expect within [{LO}, {HI}])")
print(f"  draws outside the configured range: {len(out_of_range)}  (expect 0)")
print(f"  _draw_snr_override() -> {override}  (expect None, so mixer samples internally)")
if out_of_range or override is not None:
    print("  FAIL -- default behavior changed!")
    fail += 1
else:
    print("  PASS")

print()
print("=" * 64)
print("2. RNG-stream check -- default path must consume the same draws")
print("=" * 64)
# If the default path consumed a different number of random draws than the
# original single random.uniform() call, every downstream seeded run would
# shift. _draw_snr_override must consume NOTHING when the curriculum is off.
random.seed(1234)
before = [random.random() for _ in range(3)]
random.seed(1234)
d = make_stub()
d._draw_snr_override()          # must not touch the RNG at all
after = [random.random() for _ in range(3)]
print(f"  RNG sequence with/without _draw_snr_override(): "
      f"{'IDENTICAL' if before == after else 'SHIFTED'}  (expect IDENTICAL)")
if before != after:
    print("  FAIL -- default path shifts the random stream!")
    fail += 1
else:
    print("  PASS")

print()
print("=" * 64)
print("3. CURRICULUM path -- 50% low [-10, 1), 50% trained [1, 10]")
print("=" * 64)
d = make_stub(low_snr_prob=0.5, low_snr_range=(-10.0, 1.0))
vals = [d._draw_snr_value() for _ in range(N)]
n_low = sum(1 for v in vals if v < LO)
n_trained = sum(1 for v in vals if LO <= v <= HI)
frac_low = n_low / N
print(f"  min={min(vals):.2f}  max={max(vals):.2f}")
print(f"  below {LO}dB (new regime) : {n_low}/{N} = {frac_low:.1%}  (expect ~50%)")
print(f"  within [{LO}, {HI}] dB    : {n_trained}/{N} = {n_trained/N:.1%}  (expect ~50%)")
print(f"  anything outside [-10, {HI}]: "
      f"{sum(1 for v in vals if v < -10.0 or v > HI)}  (expect 0)")
ok = (0.47 <= frac_low <= 0.53
      and n_low + n_trained == N
      and min(vals) >= -10.0 and max(vals) <= HI)
print("  PASS" if ok else "  FAIL -- curriculum split is off")
if not ok:
    fail += 1

print()
print("=" * 64)
print("4. Constructor + dataloader wiring accepts the new kwargs")
print("=" * 64)
try:
    from data.librispeech import build_dataloaders_multispeaker
    import inspect
    sig = inspect.signature(build_dataloaders_multispeaker).parameters
    ds_sig = inspect.signature(LibriSpeechTSEDataset.__init__).parameters
    need_ds = {"low_snr_prob", "low_snr_range", "interferer_count_probs"}
    need_bl = {"low_snr_prob", "low_snr_range", "interferer_count_probs"}
    missing = (need_ds - set(ds_sig)) | (need_bl - set(sig))
    print(f"  dataset kwargs    : {sorted(need_ds & set(ds_sig))}")
    print(f"  dataloader kwargs : {sorted(need_bl & set(sig))}")
    print("  PASS" if not missing else f"  FAIL -- missing: {sorted(missing)}")
    if missing:
        fail += 1
except Exception as e:
    print(f"  FAIL -- {type(e).__name__}: {e}")
    fail += 1

print()
print("=" * 64)
print(f"RESULT: {'ALL CHECKS PASSED -- safe to submit the fine-tune' if fail == 0 else f'{fail} CHECK(S) FAILED -- do NOT submit yet'}")
print("=" * 64)
sys.exit(1 if fail else 0)
