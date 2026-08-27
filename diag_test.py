import torch
from omegaconf import OmegaConf
from models.speaker_encoder import SpeakerEncoder
from models.masking import MaskingModule
from data.fake_data import build_fake_dataloaders

cfg = OmegaConf.load("configs/default.yaml")
device = torch.device("cpu")

encoder = SpeakerEncoder(
    cfg.speaker_encoder.model_name,
    cfg.speaker_encoder.embed_dim,
    freeze=True,
).to(device)
encoder.eval()

model = MaskingModule(
    n_mels=cfg.mel.n_mels,
    embed_dim=cfg.speaker_encoder.embed_dim,
    conv_channels=cfg.masking.conv_channels,
    lstm_hidden=cfg.masking.lstm_hidden,
    lstm_dropout=cfg.masking.lstm_dropout,
).to(device)
model.train()

train_loader, _ = build_fake_dataloaders(cfg, train_size=20, val_size=10)

# grab two DIFFERENT batches and check if the data itself varies
batch1 = next(iter(train_loader))
batch2 = next(iter(train_loader))

print("--- Data sanity ---")
print("mixture_mel batch1 mean/std:",
      batch1["mixture_mel"].mean().item(), batch1["mixture_mel"].std().item())
print("mixture_mel batch2 mean/std:",
      batch2["mixture_mel"].mean().item(), batch2["mixture_mel"].std().item())
print("target_mel   batch1 mean/std:",
      batch1["target_mel"].mean().item(), batch1["target_mel"].std().item())
print("Are batch1/batch2 mixtures identical?",
      torch.allclose(batch1["mixture_mel"], batch2["mixture_mel"]))
print("Mixture vs target identical?",
      torch.allclose(batch1["mixture_mel"], batch1["target_mel"]))

x_mel   = batch1["mixture_mel"].to(device)
y_mel   = batch1["target_mel"].to(device)
ref_wav = batch1["reference_wav"].to(device)

with torch.no_grad():
    d_vec = encoder(ref_wav)

x_enh, mask = model(x_mel, d_vec)
loss = model.compute_loss(x_enh, y_mel)

print()
print("--- Graph sanity ---")
print("Loss:", loss.item())
print("x_enh requires_grad:", x_enh.requires_grad)
print("x_enh grad_fn:", x_enh.grad_fn)
print("mask requires_grad:", mask.requires_grad)

loss.backward()

total_grad = 0.0
n_zero = 0
n_total = 0
for name, p in model.named_parameters():
    n_total += 1
    if p.grad is None:
        print(f"  NO GRAD: {name}")
        n_zero += 1
    else:
        g = p.grad.abs().sum().item()
        total_grad += g
        if g == 0:
            print(f"  ZERO GRAD: {name}")
            n_zero += 1

print()
print(f"Total grad magnitude: {total_grad:.6f}")
print(f"Params with zero/no grad: {n_zero}/{n_total}")

# Take one real optimizer step and see if loss changes on the SAME batch
opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
opt.step()
opt.zero_grad()

x_enh2, mask2 = model(x_mel, d_vec)
loss2 = model.compute_loss(x_enh2, y_mel)

print()
print("--- One-step update check (same batch) ---")
print("Loss before step:", loss.item())
print("Loss after 1 step:", loss2.item())
print("Mask changed?", not torch.allclose(mask, mask2))