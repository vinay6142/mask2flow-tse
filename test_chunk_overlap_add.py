"""
Smoke test for inference/infer.py's chunking + overlap-add helpers,
`_chunk_starts` and `_overlap_add`, added to support real-world input of
arbitrary length (see the module docstring in inference/infer.py for the
original silent-truncation/silence-padding bug these replaced).

Pure-function test: no checkpoints, no GPU, no audio I/O, no real
mixture/reference files needed -- only torch. infer.py's other top-level
imports (soundfile, omegaconf, data.*, eval.results_stage2,
models.speaker_encoder, inference.vocoder) are stubbed out via sys.modules
before import since none of that machinery is exercised by these two
helpers and it is not installed in every environment this test might run
in -- this still imports and tests the REAL `_chunk_starts`/`_overlap_add`
code from inference/infer.py, not a copy.

Run: python test_chunk_overlap_add.py
"""
import sys
import types
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def _stub_module(name, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    return mod


_stub_module("soundfile", write=lambda *a, **k: None)
_stub_module("omegaconf", OmegaConf=object)
_stub_module("data")
_stub_module("data.augment", load_audio=lambda *a, **k: None)
_stub_module("data.mel", MelSpectrogramExtractor=object,
              pad_or_trim=lambda *a, **k: None, segment_waveform=lambda *a, **k: None)
_stub_module("eval")
_stub_module("eval.results_stage2", load_masking=lambda *a, **k: None, load_flow=lambda *a, **k: None)
_stub_module("models")
_stub_module("models.speaker_encoder", SpeakerEncoder=object)
_stub_module("inference.vocoder",
              mel_to_audio_griffinlim=lambda *a, **k: None,
              load_hifigan_generator=lambda *a, **k: None,
              mel_to_audio_hifigan=lambda *a, **k: None)

import torch  # noqa: E402
from inference.infer import _chunk_starts, _overlap_add  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  [PASS] {name}")
    else:
        FAIL.append(name)
        print(f"  [FAIL] {name}   {detail}")


def verify_full_coverage(starts, total_frames, chunk_frames, label):
    check(f"{label}: starts[0] == 0", starts[0] == 0)
    check(f"{label}: non-decreasing", starts == sorted(starts))
    check(f"{label}: last chunk reaches exact end",
          starts[-1] + chunk_frames == total_frames,
          f"got end={starts[-1] + chunk_frames} vs total={total_frames}")
    no_gaps = all(starts[i + 1] <= starts[i] + chunk_frames for i in range(len(starts) - 1))
    check(f"{label}: no coverage gaps between consecutive chunks", no_gaps)
    in_range = all(0 <= s <= total_frames - chunk_frames for s in starts)
    check(f"{label}: every start is in valid range", in_range)


print("=== _chunk_starts: shape/coverage correctness ===")

check("short clip -> single chunk [0]",
      _chunk_starts(total_frames=400, chunk_frames=1000, overlap_frames=200) == [0])

check("exact-length clip -> single chunk [0]",
      _chunk_starts(total_frames=1000, chunk_frames=1000, overlap_frames=200) == [0])

# Realistic long clip at the project's actual sizes (sr=16000, hop=160,
# segment_length=10s -> chunk_frames=1000; overlap_sec=2.0 -> overlap_frames=200).
starts = _chunk_starts(total_frames=2500, chunk_frames=1000, overlap_frames=200)
verify_full_coverage(starts, 2500, 1000, "25s clip (1000f chunk / 200f overlap)")
check("25s clip -> 3 chunks", len(starts) == 3, f"got {starts}")
check("25s clip -> starts are exactly [0, 800, 1500] (last-chunk shift)",
      starts == [0, 800, 1500], f"got {starts}")

# Just barely over one chunk -- regression case for the range()/+1 boundary.
starts = _chunk_starts(total_frames=1001, chunk_frames=1000, overlap_frames=200)
verify_full_coverage(starts, 1001, 1000, "1001-frame edge case")

# Evenly divisible length -- exercises the branch where the shift-append is skipped
# (starts[-1] already equals last_start).
starts = _chunk_starts(total_frames=1800, chunk_frames=1000, overlap_frames=200)
verify_full_coverage(starts, 1800, 1000, "evenly-divisible 1800-frame case")
check("evenly-divisible case has no duplicate starts", len(starts) == len(set(starts)))

# Zero overlap -- hop == chunk_frames, still must not leave gaps.
starts = _chunk_starts(total_frames=3300, chunk_frames=1000, overlap_frames=0)
verify_full_coverage(starts, 3300, 1000, "zero-overlap case")

# Near-total overlap -- hop clamped to 1 internally, should not hang or misbehave.
starts = _chunk_starts(total_frames=1200, chunk_frames=1000, overlap_frames=999)
verify_full_coverage(starts, 1200, 1000, "near-total-overlap case")


print("\n=== _overlap_add: reconstruction / cross-fade correctness ===")
device = torch.device("cpu")
n_mels = 80

# A. Single chunk covering the full length exactly: no ramps should ever apply
#    (start==0 and end==total_frames both fail the ramp conditions), so output
#    must equal the input exactly.
total_frames, chunk_frames, overlap_frames = 1000, 1000, 200
chunk = torch.rand(n_mels, chunk_frames)
out = _overlap_add([chunk], [0], chunk_frames, total_frames, overlap_frames, device)
check("single full-length chunk reproduced exactly",
      torch.allclose(out, chunk, atol=1e-6),
      f"max abs diff {(out - chunk).abs().max().item():.2e}")

# B. Two overlapping constant-valued chunks: hand-derive the expected linear
#    cross-fade in the overlap region and check it EXACTLY, not just bounds --
#    this is the direct test of "boundary cross-fade quality".
total_frames, chunk_frames, overlap_frames = 1800, 1000, 200
starts = _chunk_starts(total_frames, chunk_frames, overlap_frames)
assert starts == [0, 800], starts
val_a, val_b = 1.0, 3.0
chunk_a = torch.full((n_mels, chunk_frames), val_a)
chunk_b = torch.full((n_mels, chunk_frames), val_b)
out = _overlap_add([chunk_a, chunk_b], starts, chunk_frames, total_frames, overlap_frames, device)

check("pre-overlap region [0:800) holds chunk A value exactly",
      torch.allclose(out[:, :800], torch.full((n_mels, 800), val_a), atol=1e-5))
check("post-overlap region [1000:1800) holds chunk B value exactly",
      torch.allclose(out[:, 1000:], torch.full((n_mels, 800), val_b), atol=1e-5))

ramp = torch.linspace(0.0, 1.0, overlap_frames)
expected_overlap = val_a * (1 - ramp) + val_b * ramp
actual_overlap = out[0, 800:1000]
check("overlap region [800:1000) is an exact linear cross-fade A to B",
      torch.allclose(actual_overlap, expected_overlap, atol=1e-5),
      f"max abs diff {(actual_overlap - expected_overlap).abs().max().item():.2e}")

# C. Boundary smoothness: the whole point of overlap-add is no click at the
#    seam. Verify its worst frame-to-frame jump is no bigger than naive hard
#    concatenation's -- a direct, quantitative check of that claim.
hard_concat = torch.cat([chunk_a[:, :800], chunk_b], dim=1)


def max_frame_jump(mel):
    return (mel[:, 1:] - mel[:, :-1]).abs().max().item()


jump_hard, jump_oa = max_frame_jump(hard_concat), max_frame_jump(out)
check("overlap-add worst frame-to-frame jump <= naive hard-concat worst jump",
      jump_oa <= jump_hard + 1e-6,
      f"overlap-add={jump_oa:.4f} vs hard-concat={jump_hard:.4f}")

# D. Realistic 3-chunk case with the last-chunk shift -- the actual scenario
#    flagged in the docstring as an edge case (local overlap width ends up
#    wider than `overlap_frames` between chunks 1 and 2, since start=1500 is
#    pulled left of the regular 800-hop grid). Verify:
#    (1) each chunk untouched interior still reproduces its value exactly,
#    (2) the weighted-sum+normalize keeps every value within [min, max] of
#        the inputs even where local overlap width != overlap_frames.
total_frames, chunk_frames, overlap_frames = 2500, 1000, 200
starts = _chunk_starts(total_frames, chunk_frames, overlap_frames)
assert starts == [0, 800, 1500], starts
vals = [1.0, 2.0, 3.0]
chunks = [torch.full((n_mels, chunk_frames), v) for v in vals]
out = _overlap_add(chunks, starts, chunk_frames, total_frames, overlap_frames, device)

check("3-chunk output shape correct", tuple(out.shape) == (n_mels, total_frames))
check("3-chunk output has no NaN/Inf", torch.isfinite(out).all().item())
check("chunk0 pure region [0:800) matches its value exactly",
      torch.allclose(out[:, :800], torch.full((n_mels, 800), 1.0), atol=1e-4))
check("chunk1 pure region [1000:1500) matches its value exactly",
      torch.allclose(out[:, 1000:1500], torch.full((n_mels, 500), 2.0), atol=1e-4))
check("chunk2 pure region [1800:2500) matches its value exactly",
      torch.allclose(out[:, 1800:], torch.full((n_mels, 700), 3.0), atol=1e-4))
check("3-chunk reconstruction stays within [min(inputs), max(inputs)] everywhere "
      "(convex-combination invariant, even in the uneven 300f local-overlap zone)",
      out.min().item() >= min(vals) - 1e-4 and out.max().item() <= max(vals) + 1e-4,
      f"got range [{out.min().item():.4f}, {out.max().item():.4f}] vs inputs {vals}")

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:")
    for name in FAIL:
        print(f"  - {name}")
    sys.exit(1)
