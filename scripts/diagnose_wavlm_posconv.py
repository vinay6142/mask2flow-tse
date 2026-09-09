"""
Diagnostic (read-only, makes no changes) for the WavLM pos_conv_embed
weight-norm loading issue.

Background: models/speaker_encoder.py's `WavLMModel.from_pretrained(...)`
call prints a warning that
`encoder.pos_conv_embed.conv.parametrizations.weight.original0`/`original1`
were "not initialized from the model checkpoint" and are "newly
initialized" -- i.e. left at a random draw instead of the pretrained
value, every time SpeakerEncoder() is constructed. This is a known class
of issue: newer torch moved `weight_norm` from directly-stored
`weight_g`/`weight_v` parameters to a `torch.nn.utils.parametrize`-based
implementation (`original0`/`original1`), and if the checkpoint was saved
under the old naming, `from_pretrained`'s state-dict key matching can miss
it depending on the installed transformers/torch version.

This script only INSPECTS -- it does not modify any checkpoint or write
any file -- so it's safe to run directly on the login node, no GPU/sbatch
needed. It answers three questions:
  1. What torch/transformers versions are actually installed?
  2. Does the warning reproduce in isolation (confirms it's WavLM itself,
     not something in the rest of the pipeline)?
  3. What are the ACTUAL key names in the raw checkpoint file for the
     pos_conv_embed submodule? (needed to write a correct manual fix,
     rather than guessing)

Run:  python3 scripts/diagnose_wavlm_posconv.py
"""

import torch
import transformers

print("=" * 70)
print("1. Versions")
print("=" * 70)
print(f"  torch version        : {torch.__version__}")
print(f"  transformers version : {transformers.__version__}")

MODEL_NAME = "microsoft/wavlm-base-plus-sv"

print()
print("=" * 70)
print("2. Reproduce the warning in isolation + check determinism")
print("=" * 70)
from transformers import WavLMModel

torch.manual_seed(1)
m1 = WavLMModel.from_pretrained(MODEL_NAME)
torch.manual_seed(2)
m2 = WavLMModel.from_pretrained(MODEL_NAME)

pc1 = m1.encoder.pos_conv_embed.conv
pc2 = m2.encoder.pos_conv_embed.conv

has_parametrize = hasattr(pc1, "parametrizations")
print(f"  pos_conv_embed.conv uses new-style parametrize : {has_parametrize}")

if has_parametrize:
    o1_0 = pc1.parametrizations.weight.original0
    o2_0 = pc2.parametrizations.weight.original0
    o1_1 = pc1.parametrizations.weight.original1
    o2_1 = pc2.parametrizations.weight.original1
    print(f"  original0 shape : {tuple(o1_0.shape)}")
    print(f"  original1 shape : {tuple(o1_1.shape)}")
    ident0 = torch.allclose(o1_0, o2_0)
    ident1 = torch.allclose(o1_1, o2_1)
    print(f"  Two independently-constructed models, different seeds:")
    print(f"    original0 identical across seeds : {ident0}  (expect False if bug is real)")
    print(f"    original1 identical across seeds : {ident1}  (expect False if bug is real)")
else:
    print("  No `.parametrizations` attribute found -- either the warning didn't")
    print("  reproduce on this install, or this transformers/torch version stores")
    print("  weight_norm differently (check pc1._parameters / pc1.__dict__ below).")
    print(f"  pc1 named_parameters: {[n for n, _ in pc1.named_parameters()]}")

# whole-model sanity: confirm the REST of WavLM loaded identically (i.e. this
# really is isolated to pos_conv_embed, not a wider loading problem)
sd1, sd2 = m1.state_dict(), m2.state_dict()
diffs = [k for k in sd1 if not torch.equal(sd1[k], sd2[k])]
print(f"\n  Total params tensors differing between the two seeded loads: {len(diffs)} / {len(sd1)}")
print(f"  Differing keys: {diffs}")

print()
print("=" * 70)
print("3. Raw checkpoint file -- actual on-disk key names")
print("=" * 70)
from huggingface_hub import hf_hub_download

raw_sd = None
for filename, loader_desc in [
    ("model.safetensors", "safetensors"),
    ("pytorch_model.bin", "torch.load"),
]:
    try:
        path = hf_hub_download(MODEL_NAME, filename=filename)
        if filename.endswith(".safetensors"):
            from safetensors.torch import load_file
            raw_sd = load_file(path)
        else:
            raw_sd = torch.load(path, map_location="cpu")
        print(f"  Loaded raw checkpoint via {loader_desc}: {path}")
        break
    except Exception as e:
        print(f"  {filename} not found/loadable ({e.__class__.__name__}: {e})")

if raw_sd is not None:
    matches = [k for k in raw_sd.keys() if "pos_conv_embed" in k]
    print(f"\n  Raw checkpoint keys containing 'pos_conv_embed' ({len(matches)} found):")
    for k in matches:
        print(f"    {k}  shape={tuple(raw_sd[k].shape)}  dtype={raw_sd[k].dtype}")
    if not matches:
        print("    (none -- 'pos_conv_embed' substring not present under ANY key;")
        print("     try printing a broader sample of raw_sd.keys() to find the real name)")
        sample = list(raw_sd.keys())[:15]
        print(f"    First 15 raw keys for reference: {sample}")
else:
    print("\n  Could not fetch the raw checkpoint file directly -- report the error above.")

print()
print("=" * 70)
print("DONE. Paste this entire output back for the actual fix to be written.")
print("=" * 70)
