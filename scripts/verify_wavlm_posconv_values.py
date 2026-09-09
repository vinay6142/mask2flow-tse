"""
Follow-up to diagnose_wavlm_posconv.py. Read-only, no changes.

That script found original0/original1 are IDENTICAL across two different
seeds -- suspicious if they were truly randomly reinitialized. This script
settles it directly: compare the live model's pos_conv_embed values against
the RAW checkpoint's weight_g/weight_v (the pre-parametrize legacy names),
to check whether the "newly initialized" warning is a real problem or a
stale/cosmetic false-positive in this transformers version.

Run:  python3 scripts/verify_wavlm_posconv_values.py
"""
import torch
from transformers import WavLMModel
from huggingface_hub import hf_hub_download

MODEL_NAME = "microsoft/wavlm-base-plus-sv"

model = WavLMModel.from_pretrained(MODEL_NAME)
pc = model.encoder.pos_conv_embed.conv

live_original0 = pc.parametrizations.weight.original0.detach()
live_original1 = pc.parametrizations.weight.original1.detach()
live_effective_weight = pc.weight.detach()   # computed property: weight_norm(v, g, dim)

path = hf_hub_download(MODEL_NAME, filename="pytorch_model.bin")
raw_sd = torch.load(path, map_location="cpu")
raw_g = raw_sd["wavlm.encoder.pos_conv_embed.conv.weight_g"]
raw_v = raw_sd["wavlm.encoder.pos_conv_embed.conv.weight_v"]

print("=" * 70)
print("Direct tensor comparison: live model vs. raw checkpoint")
print("=" * 70)

def report(name, a, b):
    close = torch.allclose(a, b, atol=1e-6)
    max_diff = (a - b).abs().max().item()
    print(f"  {name}:")
    print(f"    allclose (atol=1e-6) : {close}")
    print(f"    max abs diff          : {max_diff:.8f}")
    print(f"    a.norm()={a.norm().item():.6f}  b.norm()={b.norm().item():.6f}")

report("live original0  vs  raw weight_g", live_original0, raw_g)
report("live original1  vs  raw weight_v", live_original1, raw_v)

# Also reconstruct the effective weight from the RAW g/v directly (same
# formula torch's weight_norm parametrization uses) and compare against
# what the live model actually computes/uses at inference time -- this is
# the number that actually matters (the real conv weight used in forward()).
raw_effective_weight = torch._weight_norm(raw_v, raw_g, 0)
report("live effective conv weight  vs  reconstructed-from-raw-checkpoint", live_effective_weight, raw_effective_weight)

print()
print("=" * 70)
if torch.allclose(live_effective_weight, raw_effective_weight, atol=1e-6):
    print("VERDICT: the effective weight matches the checkpoint exactly.")
    print("The 'newly initialized' warning is a FALSE POSITIVE in this")
    print("transformers version -- pos_conv_embed IS loading the real")
    print("pretrained weights correctly. Nothing to fix.")
else:
    print("VERDICT: the effective weight does NOT match the checkpoint.")
    print("The warning is real -- pos_conv_embed is genuinely using")
    print("uninitialized/wrong values. A manual fix is needed.")
print("=" * 70)
