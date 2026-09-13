"""
Split an eval_libri2mix.py JSONL by mixture SNR and summarize each slice.

eval_libri2mix.py's own print_summary() reports only the raw n=6000 aggregate,
which mixes two regimes the model behaves completely differently in:

  - snr_db >= 1.0  : IN-DISTRIBUTION. Matches this project's trained mixer
                     range (configs/default_v2.yaml: snr_min=1.0, snr_max=10.0).
  - snr_db <  1.0  : OUT-OF-DISTRIBUTION. The target is quieter than the
                     interferer -- a regime the original training never
                     covered at all, and 60.6% of Libri2Mix's mix_clean set.
                     See docs/results_and_limitations.md Sec 5.5.2.

Reporting only the aggregate makes a checkpoint look far worse (or, after a
low-SNR fine-tune, far better) than it is in either regime individually. Every
SNR-split table in the docs comes from this computation, so it lives here as a
real tool rather than an ad-hoc snippet.

Usage:
  python3 eval/snr_split_summary.py outputs/results/eval_libri2mix_min.jsonl
  python3 eval/snr_split_summary.py <a.jsonl> <b.jsonl>   # side-by-side compare
"""
import os
import sys
import json
import statistics as stats

CATASTROPHIC_PCT = -30.0
BOUNDARY = 1.0   # cfg.data.snr_min -- the trained floor


def load(path):
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "snr_db" in r:
                recs.append(r)
    return recs


def summarize(recs):
    """Return the metric dict for one slice (or None if empty)."""
    if not recs:
        return None
    mel = [r["mel_s2_vs_s1_pct"] for r in recs]
    out = {
        "n": len(recs),
        "mel_median": stats.median(mel),
        "catastrophic_pct": 100.0 * sum(1 for v in mel if v < CATASTROPHIC_PCT) / len(mel),
    }
    g1 = [r["sisdr_gain_vs_s1"] for r in recs if "sisdr_gain_vs_s1" in r]
    gm = [r["sisdr_gain_vs_mix"] for r in recs if "sisdr_gain_vs_mix" in r]
    out["sisdr_vs_s1"] = stats.median(g1) if g1 else None
    out["sisdr_vs_mix"] = stats.median(gm) if gm else None
    return out


BUCKETS = [
    ("<-5dB",   lambda s: s < -5),
    ("[-5,-2)", lambda s: -5 <= s < -2),
    ("[-2,0)",  lambda s: -2 <= s < 0),
    ("[0,1)",   lambda s: 0 <= s < 1),
    ("[1,3)",   lambda s: 1 <= s < 3),
    ("[3,5)",   lambda s: 3 <= s < 5),
    ("[5,7)",   lambda s: 5 <= s < 7),
    ("[7,+)",   lambda s: s >= 7),
]


def fmt(v, unit="", plus=False):
    if v is None:
        return "n/a"
    sign = "+" if (plus and v >= 0) else ""
    return f"{sign}{v:.2f}{unit}" if unit == " dB" else f"{sign}{v:.1f}{unit}"


def report(path):
    recs = load(path)
    if not recs:
        print(f"  no records with snr_db in {path}")
        return
    name = os.path.basename(path)
    in_d = [r for r in recs if r["snr_db"] >= BOUNDARY]
    ood = [r for r in recs if r["snr_db"] < BOUNDARY]

    print(f"\n{'=' * 78}")
    print(f"  {name}   (n={len(recs)})")
    print("=" * 78)
    for label, slice_recs in [
        (f"IN-DISTRIBUTION  (snr_db >= {BOUNDARY}dB)", in_d),
        (f"OUT-OF-DIST      (snr_db <  {BOUNDARY}dB)", ood),
    ]:
        s = summarize(slice_recs)
        if s is None:
            continue
        print(f"\n  {label}   n={s['n']} ({100 * s['n'] / len(recs):.1f}% of total)")
        print(f"    median mel S2-vs-S1     : {fmt(s['mel_median'], '%')}")
        print(f"    catastrophic (< {CATASTROPHIC_PCT:.0f}%)   : {fmt(s['catastrophic_pct'], '%')}")
        print(f"    median SI-SDR vs Stage 1: {fmt(s['sisdr_vs_s1'], ' dB', plus=True)}")
        print(f"    median SI-SDR vs mixture: {fmt(s['sisdr_vs_mix'], ' dB', plus=True)}")

    print(f"\n  Per-bucket catastrophic-rate trend:")
    for label, pred in BUCKETS:
        g = [r for r in recs if pred(r["snr_db"])]
        if not g:
            continue
        s = summarize(g)
        print(f"    {label:9s} n={s['n']:5d}  catastrophic={s['catastrophic_pct']:5.1f}%  "
              f"mel={fmt(s['mel_median'], '%'):>8s}  "
              f"SI-SDR vs S1={fmt(s['sisdr_vs_s1'], ' dB', plus=True):>9s}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for p in sys.argv[1:]:
        if not os.path.exists(p):
            print(f"ERROR: not found: {p}")
            sys.exit(1)
        report(p)
    print()
