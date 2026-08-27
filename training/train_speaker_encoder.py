"""
Train SpeakerEncoder's projection layer (768->512) via speaker classification.

This is the PERMANENT fix for the never-trained, freshly-random-every-run
projection bug: WavLM stays frozen exactly as the paper specifies; only the
projection (+ a classification head, discarded after training) is trained,
with a real objective forcing the 512-dim embedding to be linearly
speaker-separable.

Standalone stage — does not touch Stage 1 or Stage 2. Run this FIRST, before
re-evaluating whether Stage 1/2 need retraining against the new projection.

Run (fresh start):
  python3 -u training/train_speaker_encoder.py --config configs/default_v2.yaml

Run (resume):
  python3 -u training/train_speaker_encoder.py --config configs/default_v2.yaml --resume
"""
import os
import sys
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.speaker_encoder import SpeakerEncoder
from data.speaker_dataset import SpeakerClassificationDataset


def save_checkpoint(path, step, encoder, classifier, optimizer, best_val, tag):
    """Atomic write (tmp-then-rename), matching this project's established
    fix for checkpoint corruption (bug #5 in the main training scripts)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save({
        "step": step,
        "projection": encoder.projection.state_dict(),
        "classifier": classifier.state_dict(),
        "optimizer": optimizer.state_dict(),
        "val_acc": best_val,
    }, tmp_path)
    os.replace(tmp_path, path)
    print(f"[Checkpoint] Saved {tag} @ step {step} (val_acc={best_val:.4f}) -> {path}")


def load_checkpoint(path, encoder, classifier, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    encoder.projection.load_state_dict(ckpt["projection"])
    classifier.load_state_dict(ckpt["classifier"])
    optimizer.load_state_dict(ckpt["optimizer"])
    step = ckpt["step"]
    best_val = ckpt.get("val_acc", 0.0)
    print(f"[Resume] Loaded checkpoint from step {step}, best_val_acc={best_val:.4f}")
    return step, best_val


@torch.no_grad()
def validate(encoder, classifier, val_loader, device, max_batches=30):
    encoder.eval(); classifier.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        wav = batch["waveform"].to(device)
        labels = batch["label"].to(device)
        d_vec = encoder(wav)
        logits = classifier(d_vec)
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item() * wav.shape[0]
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_n += wav.shape[0]
    encoder.train(); classifier.train()
    return total_loss / max(total_n, 1), total_correct / max(total_n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default_v2.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint_dir", default="checkpoints_speaker_encoder")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_steps", type=int, default=30000)
    parser.add_argument("--val_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SpeakerEncoderTrain] Device: {device}")
    if device.type != "cuda":
        print("[Warn] Not running on GPU — always launch via sbatch, never an "
              "interactive shell (matches this project's established rule).")

    train_ds = SpeakerClassificationDataset(
        librispeech_root=cfg.data.librispeech_path,
        splits=list(cfg.data.train_splits),
        sample_rate=cfg.audio.sample_rate,
        segment_length=getattr(cfg.audio, "reference_length", 3.0),
        is_train=True,
    )
    val_ds = SpeakerClassificationDataset(
        librispeech_root=cfg.data.librispeech_path,
        splits=["test-clean"],   # dev-clean isn't downloaded (confirmed earlier) — reuse
                                 # the same test-clean substitution already used for the
                                 # vocoder's validation split, for the same reason
        sample_rate=cfg.audio.sample_rate,
        segment_length=getattr(cfg.audio, "reference_length", 3.0),
        is_train=False,
    )
    # IMPORTANT: train/val speaker sets must be disjoint from each other's
    # label space for this to be a meaningful check — since train-clean-100
    # and test-clean contain DIFFERENT speakers in LibriSpeech, num_speakers
    # differs; the classifier head is train-set-specific, so validation here
    # measures loss on the train speaker taxonomy only if the split overlaps.
    # LibriSpeech's train/test speakers do NOT overlap, so this val split is
    # actually testing generalization of the embedding space to unseen
    # speakers via a fresh classifier — a stronger check. If you'd prefer a
    # simpler in-distribution validation, hold out ~5% of train-clean-100
    # speakers instead.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, drop_last=False)

    encoder = SpeakerEncoder(
        model_name=cfg.speaker_encoder.model_name,
        embed_dim=cfg.speaker_encoder.embed_dim,
        sample_rate=cfg.audio.sample_rate,
        freeze=True,   # freezes WavLM only — projection stays trainable, as designed
    ).to(device)
    encoder.train()

    classifier = nn.Linear(cfg.speaker_encoder.embed_dim, train_ds.num_speakers).to(device)

    optimizer = torch.optim.AdamW(
        list(encoder.projection.parameters()) + list(classifier.parameters()),
        lr=args.lr,
    )

    step, best_val_acc = 0, 0.0
    ckpt_latest = os.path.join(args.checkpoint_dir, "projection_latest.pt")
    ckpt_best = os.path.join(args.checkpoint_dir, "projection_best.pt")

    if args.resume and os.path.exists(ckpt_latest):
        step, best_val_acc = load_checkpoint(ckpt_latest, encoder, classifier, optimizer, device)
    elif args.resume:
        print("[Resume] --resume passed but no checkpoint found — starting fresh")

    print(f"[SpeakerEncoderTrain] Starting from step {step}, target {args.max_steps} steps")
    data_iter = iter(train_loader)

    while step < args.max_steps:
        t0 = time.time()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        wav = batch["waveform"].to(device)
        labels = batch["label"].to(device)

        d_vec = encoder(wav)
        logits = classifier(d_vec)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1
        dt = time.time() - t0

        if step % 50 == 0:
            acc = (logits.argmax(dim=-1) == labels).float().mean().item()
            print(f"[Step {step}/{args.max_steps}] loss={loss.item():.4f} "
                  f"batch_acc={acc:.3f} ({dt:.2f}s/it)")

        if step % args.val_every == 0:
            val_loss, val_acc = validate(encoder, classifier, val_loader, device)
            print(f"[Val] step={step} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(ckpt_best, step, encoder, classifier, optimizer, best_val_acc, "best")

        if step % args.save_every == 0:
            save_checkpoint(ckpt_latest, step, encoder, classifier, optimizer, best_val_acc, "latest")

    save_checkpoint(ckpt_latest, step, encoder, classifier, optimizer, best_val_acc, "final")
    print("[SpeakerEncoderTrain] Training complete.")
    print(f"[SpeakerEncoderTrain] Best val_acc: {best_val_acc:.4f}")
    print(f"[SpeakerEncoderTrain] Use checkpoints_speaker_encoder/projection_best.pt "
          f"going forward via SpeakerEncoder(..., projection_ckpt=...)")


if __name__ == "__main__":
    main()