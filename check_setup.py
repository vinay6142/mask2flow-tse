"""
Mask2Flow-TSE — Setup Verification Script
Checks that everything is installed and working correctly.

Run from project root:
    python check_setup.py

All checks should show ✅ before starting Week 2.
"""

import sys
import os

print("=" * 60)
print("  Mask2Flow-TSE — Setup Verification")
print("=" * 60)

errors   = []
warnings = []

# ── 1. Python version ─────────────────────────────────────────
print(f"\n[1/8] Python version")
v = sys.version_info
if v.major == 3 and v.minor >= 9:
    print(f"  Python {v.major}.{v.minor}.{v.micro} ✅")
else:
    print(f"  Python {v.major}.{v.minor}.{v.micro} ⚠️  (recommend 3.9+)")
    warnings.append("Python < 3.9")

# ── 2. PyTorch + CUDA ─────────────────────────────────────────
print(f"\n[2/8] PyTorch + CUDA")
try:
    import torch
    print(f"  PyTorch  : {torch.__version__} ✅")

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            vram  = props.total_memory / 1e9
            print(f"  GPU {i}    : {props.name} "
                  f"({vram:.1f} GB VRAM) ✅")
        print(f"  CUDA     : {torch.version.cuda} ✅")
        print(f"  cuDNN    : {torch.backends.cudnn.version()} ✅")
    else:
        print(f"  CUDA     : NOT available ⚠️")
        warnings.append("No CUDA — training will be slow")

    # quick GPU tensor test
    if torch.cuda.is_available():
        x = torch.randn(1000, 1000, device="cuda")
        y = x @ x.T
        print(f"  GPU matmul test: {y.shape} ✅")
        del x, y
        torch.cuda.empty_cache()

except ImportError as e:
    print(f"  PyTorch not installed ❌  ({e})")
    errors.append("PyTorch missing")

# ── 3. torchaudio ─────────────────────────────────────────────
print(f"\n[3/8] torchaudio")
try:
    import torchaudio
    print(f"  torchaudio : {torchaudio.__version__} ✅")

    # test mel spectrogram
    import torch
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024,
        hop_length=160, n_mels=80
    )
    wav = torch.randn(1, 16000)
    out = mel(wav)
    print(f"  Mel test   : {tuple(out.shape)} ✅")

except ImportError as e:
    print(f"  torchaudio not installed ❌  ({e})")
    errors.append("torchaudio missing")

# ── 4. Core packages ──────────────────────────────────────────
print(f"\n[4/8] Core packages")
packages = {
    "numpy"       : "numpy",
    "scipy"       : "scipy",
    "omegaconf"   : "omegaconf",
    "yaml"        : "pyyaml",
    "tqdm"        : "tqdm",
    "matplotlib"  : "matplotlib",
    "soundfile"   : "soundfile",
    "einops"      : "einops",
    "pandas"      : "pandas",
}

for import_name, pkg_name in packages.items():
    try:
        mod = __import__(import_name)
        ver = getattr(mod, "__version__", "?")
        print(f"  {import_name:<15}: {ver} ✅")
    except ImportError:
        print(f"  {import_name:<15}: not installed ❌")
        errors.append(f"{pkg_name} missing")

# ── 5. Transformers (for WavLM) ───────────────────────────────
print(f"\n[5/8] Transformers (for WavLM speaker encoder)")
try:
    import transformers
    print(f"  transformers : {transformers.__version__} ✅")

    # test that WavLM config is loadable
    from transformers import AutoConfig
    print(f"  AutoConfig   : ok ✅")

except ImportError as e:
    print(f"  transformers not installed ❌  ({e})")
    errors.append("transformers missing — needed for WavLM in Week 2")

# ── 6. Whisper (for evaluation) ───────────────────────────────
print(f"\n[6/8] Whisper (for WER evaluation)")
try:
    import whisper
    print(f"  whisper : installed ✅")
    models = whisper.available_models()
    print(f"  available : {models[:4]}...")
except ImportError:
    print(f"  whisper : not installed ⚠️  (needed for Week 5 eval)")
    warnings.append("whisper missing — needed for evaluation only")

# ── 7. Project data pipeline ──────────────────────────────────
print(f"\n[7/8] Project data pipeline")
try:
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("configs/default.yaml")
    print(f"  config loaded ✅")
    print(f"    n_mels    : {cfg.mel.n_mels}")
    print(f"    ffn_mult  : {cfg.flow.ffn_mult}  "
          f"{'✅ correct' if cfg.flow.ffn_mult == 2 else '❌ should be 2'}")

    from data.mel import MelSpectrogramExtractor, MelNormalizer
    import torch
    mel   = MelSpectrogramExtractor()
    norm  = MelNormalizer(mode="l2")
    wav   = torch.randn(2, 16000 * 3)
    mels  = mel(wav)
    mnorm = norm.normalize(mels)
    print(f"  mel.py    : {tuple(mels.shape)} ✅")

    from data.augment import MixtureCreator
    creator = MixtureCreator()
    tgt     = torch.randn(16000 * 5)
    intr    = torch.randn(16000 * 5)
    mix, t, cond = creator.create_mixture(tgt, intr, "additive")
    print(f"  augment.py: mixture {tuple(mix.shape)} ✅")

    from data.fake_data import build_fake_dataloaders
    tl, vl = build_fake_dataloaders(cfg, train_size=20, val_size=10)
    batch  = next(iter(tl))
    print(f"  fake_data : batch keys={list(batch.keys())} ✅")

except Exception as e:
    print(f"  Pipeline error ❌: {e}")
    import traceback
    traceback.print_exc()
    errors.append(f"Pipeline: {e}")

# ── 8. Data files ─────────────────────────────────────────────
print(f"\n[8/8] Downloaded data")
from pathlib import Path

data_checks = {
    "data/raw/LibriSpeech" : "LibriSpeech (needed for training)",
    "data/raw/wham_noise"  : "WHAM! noise (optional, for noisy eval)",
    "data/raw/RIRS"        : "RIRs (optional, for reverb condition)",
}

for path, description in data_checks.items():
    p = Path(path)
    if p.exists():
        n_files = len(list(p.rglob("*.flac"))) + \
                  len(list(p.rglob("*.wav")))
        print(f"  {description}")
        print(f"    {path}: {n_files} files ✅")
    else:
        opt = "optional" in description
        sym = "⚠️ " if opt else "❌"
        print(f"  {description}")
        print(f"    {path}: not found {sym}")
        if not opt:
            warnings.append(f"{path} not downloaded — run: "
                            f"python data/download.py")

# ── Summary ───────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")

if not errors and not warnings:
    print(f"\n  All checks passed! ✅")
    print(f"  Ready for Week 2 — model implementation\n")
elif not errors:
    print(f"\n  Core setup OK ✅ with {len(warnings)} warning(s):")
    for w in warnings:
        print(f"    ⚠️  {w}")
    print(f"\n  Safe to continue to Week 2\n")
else:
    print(f"\n  {len(errors)} error(s) must be fixed:")
    for e in errors:
        print(f"    ❌  {e}")
    if warnings:
        print(f"\n  {len(warnings)} warning(s):")
        for w in warnings:
            print(f"    ⚠️  {w}")
    print(f"\n  Fix errors before continuing\n")