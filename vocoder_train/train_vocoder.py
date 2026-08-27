"""
Train the HiFi-GAN vocoder on this project's own LibriSpeech audio,
using the exact same mel spec Stage 2 outputs (16kHz, hop=160, win=400).

Follows this project's established conventions:
  - atomic checkpoint writes (tmp-then-rename) — see bug #5 in project notes
  - best_val restored from disk on resume, not reset to inf — see bug #9
  - resume-chaining across SLURM walltime kills
  - unbuffered-safe (run with `python3 -u`)

Run (fresh start):
  python3 -u train_vocoder.py --config configs/vocoder.yaml

Run (resume):
  python3 -u train_vocoder.py --config configs/vocoder.yaml --resume
"""
import os
import sys
import argparse
import time

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hifigan import (
    Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator,
    feature_loss, discriminator_loss, generator_loss,
)
from dataset import build_vocoder_dataloaders
from data.mel import MelSpectrogramExtractor


def save_checkpoint(path, step, generator, mpd, msd, opt_g, opt_d, best_val, tag):
    """Atomic write: tmp-then-rename, matching this project's established fix
    for the checkpoint-corruption bug (#5) found in the main training scripts."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save({
        "step": step,
        "generator": generator.state_dict(),
        "mpd": mpd.state_dict(),
        "msd": msd.state_dict(),
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict(),
        "val_loss": best_val,
    }, tmp_path)
    os.replace(tmp_path, path)  # atomic on POSIX filesystems
    print(f"[Checkpoint] Saved {tag} @ step {step} -> {path}")


def load_checkpoint(path, generator, mpd, msd, opt_g, opt_d, device):
    ckpt = torch.load(path, map_location=device)
    generator.load_state_dict(ckpt["generator"])
    mpd.load_state_dict(ckpt["mpd"])
    msd.load_state_dict(ckpt["msd"])
    opt_g.load_state_dict(ckpt["opt_g"])
    opt_d.load_state_dict(ckpt["opt_d"])
    step = ckpt["step"]
    best_val = ckpt.get("val_loss", float("inf"))
    print(f"[Resume] Loaded checkpoint from step {step}")
    print(f"[Resume] Restored best_val={best_val}")
    return step, best_val


@torch.no_grad()
def validate(generator, val_loader, mel_extractor, device, max_batches=20):
    generator.eval()
    total_mel_loss = 0.0
    n = 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        mel = batch["mel"].to(device)
        wav = batch["waveform"].to(device).unsqueeze(1)  # (B, 1, T)

        wav_hat = generator(mel)
        # Trim to the shorter of the two in case of any off-by-a-few-sample
        # mismatch between the real crop and the generator's exact output.
        min_len = min(wav.shape[-1], wav_hat.shape[-1])
        wav, wav_hat = wav[..., :min_len], wav_hat[..., :min_len]

        mel_hat = mel_extractor(wav_hat.squeeze(1))
        mel_real = mel_extractor(wav.squeeze(1))
        min_frames = min(mel_hat.shape[-1], mel_real.shape[-1])
        mel_loss = F.l1_loss(mel_hat[..., :min_frames], mel_real[..., :min_frames])

        total_mel_loss += mel_loss.item()
        n += 1
    generator.train()
    return total_mel_loss / max(n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vocoder.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint_dir", default="checkpoints_vocoder")
    parser.add_argument("--log_dir", default="logs_vocoder")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--val_every", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=2000)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    vc = cfg.vocoder  # vocoder-specific hyperparams block

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Vocoder] Device: {device}")
    if device.type != "cuda":
        print("[Warn] Not running on GPU — training will be extremely slow. "
              "Always launch via sbatch, never an interactive shell (matches "
              "this project's established rule from the main training scripts).")

    batch_size = args.batch_size or vc.batch_size
    max_steps = args.max_steps or vc.max_steps

    generator = Generator(
        n_mels=cfg.mel.n_mels,
        upsample_rates=tuple(vc.upsample_rates),
        upsample_kernel_sizes=tuple(vc.upsample_kernel_sizes),
        upsample_initial_channel=vc.upsample_initial_channel,
        resblock_kernel_sizes=tuple(vc.resblock_kernel_sizes),
        resblock_dilation_sizes=tuple(tuple(d) for d in vc.resblock_dilation_sizes),
    ).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    mel_extractor = MelSpectrogramExtractor(
        sample_rate=cfg.audio.sample_rate, n_mels=cfg.mel.n_mels, n_fft=cfg.mel.n_fft,
        hop_length=cfg.mel.hop_length, win_length=cfg.mel.win_length,
        f_min=cfg.mel.f_min, f_max=cfg.mel.f_max, log_offset=cfg.mel.log_offset,
    ).to(device)

    opt_g = torch.optim.AdamW(generator.parameters(), lr=vc.lr, betas=(0.8, 0.99))
    opt_d = torch.optim.AdamW(
        list(mpd.parameters()) + list(msd.parameters()), lr=vc.lr, betas=(0.8, 0.99)
    )

    train_loader, val_loader = build_vocoder_dataloaders(
        cfg, segment_size=vc.segment_size, batch_size=batch_size,
        num_workers=vc.get("num_workers", 4),
    )

    step = 0
    best_val = float("inf")
    ckpt_latest = os.path.join(args.checkpoint_dir, "vocoder_latest.pt")
    ckpt_best = os.path.join(args.checkpoint_dir, "vocoder_best.pt")

    if args.resume and os.path.exists(ckpt_latest):
        step, best_val = load_checkpoint(ckpt_latest, generator, mpd, msd, opt_g, opt_d, device)
    elif args.resume:
        print("[Resume] --resume passed but no checkpoint found — starting fresh")

    writer = SummaryWriter(args.log_dir)
    generator.train(); mpd.train(); msd.train()

    print(f"[Vocoder] Starting from step {step}, target {max_steps} steps")
    data_iter = iter(train_loader)

    while step < max_steps:
        t0 = time.time()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        mel = batch["mel"].to(device)
        wav = batch["waveform"].to(device).unsqueeze(1)  # (B, 1, T)

        wav_hat = generator(mel)
        min_len = min(wav.shape[-1], wav_hat.shape[-1])
        wav_trim, wav_hat_trim = wav[..., :min_len], wav_hat[..., :min_len]

        # ---- Discriminator step ----
        opt_d.zero_grad()
        y_dp_r, y_dp_g, _, _ = mpd(wav_trim, wav_hat_trim.detach())
        loss_disc_p = discriminator_loss(y_dp_r, y_dp_g)
        y_ds_r, y_ds_g, _, _ = msd(wav_trim, wav_hat_trim.detach())
        loss_disc_s = discriminator_loss(y_ds_r, y_ds_g)
        loss_d = loss_disc_p + loss_disc_s
        loss_d.backward()
        opt_d.step()

        # ---- Generator step ----
        opt_g.zero_grad()
        mel_hat = mel_extractor(wav_hat_trim.squeeze(1))
        min_frames = min(mel_hat.shape[-1], mel.shape[-1])
        loss_mel = F.l1_loss(mel_hat[..., :min_frames], mel[..., :min_frames]) * vc.get("mel_loss_weight", 45)

        y_dp_r, y_dp_g, fmap_p_r, fmap_p_g = mpd(wav_trim, wav_hat_trim)
        y_ds_r, y_ds_g, fmap_s_r, fmap_s_g = msd(wav_trim, wav_hat_trim)
        loss_fm = feature_loss(fmap_p_r, fmap_p_g) + feature_loss(fmap_s_r, fmap_s_g)
        loss_gen = generator_loss(y_dp_g) + generator_loss(y_ds_g)

        loss_g = loss_gen + loss_fm + loss_mel
        loss_g.backward()
        opt_g.step()

        step += 1
        dt = time.time() - t0

        if step % 50 == 0:
            print(f"[Step {step}/{max_steps}] loss_g={loss_g.item():.4f} "
                  f"loss_d={loss_d.item():.4f} loss_mel={loss_mel.item():.4f} "
                  f"({dt:.2f}s/it)")
            writer.add_scalar("train/loss_g", loss_g.item(), step)
            writer.add_scalar("train/loss_d", loss_d.item(), step)
            writer.add_scalar("train/loss_mel", loss_mel.item(), step)

        if step % args.val_every == 0:
            val_mel_loss = validate(generator, val_loader, mel_extractor, device)
            print(f"[Val] step={step} mel_l1={val_mel_loss:.4f}")
            writer.add_scalar("val/mel_l1", val_mel_loss, step)
            if val_mel_loss < best_val:
                best_val = val_mel_loss
                save_checkpoint(ckpt_best, step, generator, mpd, msd, opt_g, opt_d, best_val, "best")

        if step % args.save_every == 0:
            save_checkpoint(ckpt_latest, step, generator, mpd, msd, opt_g, opt_d, best_val, "latest")
            # also keep a permanent step-numbered archive, matching the main
            # project's convention (see checkpoints_v2/flow/flow_best_step*.pt)
            save_checkpoint(
                os.path.join(args.checkpoint_dir, f"vocoder_step{step}.pt"),
                step, generator, mpd, msd, opt_g, opt_d, best_val, f"step{step}"
            )

    save_checkpoint(ckpt_latest, step, generator, mpd, msd, opt_g, opt_d, best_val, "final")
    print("[Vocoder] Training complete.")


if __name__ == "__main__":
    main()