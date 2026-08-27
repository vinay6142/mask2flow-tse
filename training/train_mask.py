"""
Stage 1 Training Loop — Masking Module.

Paper Section 5.2.4:
    "First, the masking module is trained with L_mask
     until convergence. Then, the masking module is frozen,
     and the flow matching module is trained."

Training details:
    optimizer : AdamW, lr=2e-4, warmup=10k steps
    loss      : MSE(X_enh, Y_clean)  [Eq. 11]
    batch     : 10 samples, grad_accumulation=2
    EMA decay : 0.9999
    max_steps : 200,000
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from tqdm import tqdm

from models.speaker_encoder import SpeakerEncoder
from models.masking import MaskingModule
from training.ema import EMA
from data.mel import make_frame_mask


# ── Learning rate scheduler ───────────────────────────────────

def get_lr(
    step:             int,
    warmup_steps:     int,
    base_lr:          float,
    max_steps:        int   = None,
    decay_start_step: int   = None,
    min_lr_ratio:     float = 0.01,
) -> float:
    """
    Linear warmup, then optional cosine decay.

    If max_steps is None, behaves exactly as before (constant LR after
    warmup) — fully backward compatible with existing calls.

    If max_steps is set, LR decays via cosine annealing from base_lr down
    to base_lr * min_lr_ratio, starting at decay_start_step (defaults to
    warmup_steps if not given) and reaching the floor at max_steps.

    decay_start_step exists specifically for resuming an already-trained
    checkpoint into a fine-tuning decay phase: pass the step you're
    resuming FROM, so decay is computed relative to that point rather
    than relative to step 0 of the original run (which would otherwise
    already be almost fully decayed).
    """
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)

    if max_steps is None:
        return base_lr

    decay_start = decay_start_step if decay_start_step is not None else warmup_steps
    if step <= decay_start:
        return base_lr

    progress = (step - decay_start) / max(1, max_steps - decay_start)
    progress = min(progress, 1.0)
    min_lr   = base_lr * min_lr_ratio
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ── Checkpoint utilities ──────────────────────────────────────

def save_checkpoint(
    step:      int,
    model:     nn.Module,
    ema:       EMA,
    optimizer: torch.optim.Optimizer,
    val_loss:  float,
    cfg:       DictConfig,
    tag:       str = "latest",
    stage:     str = "masking",
    prefix:    str = "mask",
):
    ckpt_dir = Path(cfg.paths.checkpoint_dir) / stage
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    path     = ckpt_dir / f"{prefix}_{tag}.pt"
    tmp_path = ckpt_dir / f"{prefix}_{tag}.pt.tmp"

    payload = {
        "step"      : step,
        "model"     : model.state_dict(),
        "ema"       : ema.state_dict(),
        "optimizer" : optimizer.state_dict(),
        "val_loss"  : val_loss,
    }
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)
    print(f"  [Checkpoint] Saved → {path}  (step {step})")

    # Permanent, step-numbered snapshot for "best" and "final" — these are
    # the checkpoints we actually compare/report on, and fixed filenames
    # previously caused a real, unrecoverable loss (an original pre-decay
    # checkpoint was silently overwritten mid-project). Never overwritten.
    if tag in ("best", "final"):
        archive_path = ckpt_dir / f"{prefix}_{tag}_step{step}.pt"
        if not archive_path.exists():
            torch.save(payload, archive_path)

def load_checkpoint(
    path:      str,
    model:     nn.Module,
    ema:       EMA,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
) -> int:
    """Load checkpoint and return step to resume from."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    ema.load_state_dict(ckpt["ema"])
    optimizer.load_state_dict(ckpt["optimizer"])
    step = ckpt["step"]
    print(f"[Resume] Loaded checkpoint from step {step}")
    return step


# ── Validation ────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model:   MaskingModule,
    encoder: SpeakerEncoder,
    ema:     EMA,
    loader:  DataLoader,
    device:  torch.device,
    max_batches: int = 50,
) -> dict:
    """
    Run validation with EMA weights.
    Returns dict of metrics.
    """
    model.eval()

    total_loss  = 0.0
    total_d_pct = 0.0
    n_batches   = 0

    with ema.apply():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break

            x_mel   = batch["mixture_mel"].to(device)
            y_mel   = batch["target_mel"].to(device)
            ref_wav = batch["reference_wav"].to(device)

            # speaker embedding
            d_vec  = encoder(ref_wav)

            # forward pass
            x_enh, mask = model(x_mel, d_vec)

            # loss (masked to ignore silence-padded frames on short utterances)
            frame_mask = make_frame_mask(batch["target_valid_frames"], y_mel.shape[-1], device)
            loss = model.compute_loss(x_enh, y_mel, frame_mask=frame_mask)

            # D/I proportion
            D_pct, _ = model.compute_di_proportion(x_mel, x_enh)

            total_loss  += loss.item()
            total_d_pct += D_pct
            n_batches   += 1

    model.train()

    return {
        "val_loss"  : total_loss  / max(1, n_batches),
        "val_d_pct" : total_d_pct / max(1, n_batches),
    }


# ── Main training function ────────────────────────────────────

def train_masking(
    cfg:         DictConfig,
    train_loader: DataLoader,
    val_loader:  DataLoader,
    device:      torch.device,
    resume_from: str = None,
):
    """
    Full Stage 1 training loop.

    Args:
        cfg         : full OmegaConf config
        train_loader: training DataLoader
        val_loader  : validation DataLoader
        device      : torch device
        resume_from : path to checkpoint to resume from
    """

    print("\n" + "=" * 60)
    print("  Stage 1: Masking Module Training")
    print("=" * 60)

    # ── Build models ──────────────────────────────────────────
    encoder = SpeakerEncoder(
        model_name  = cfg.speaker_encoder.model_name,
        embed_dim   = cfg.speaker_encoder.embed_dim,
        freeze      = True,
        projection_ckpt = "checkpoints_speaker_encoder/projection_latest.pt",
    ).to(device)
    for p in encoder.projection.parameters():
        p.requires_grad = False
    encoder.eval()   # always in eval mode (frozen)

    model = MaskingModule(
        n_mels        = cfg.mel.n_mels,
        embed_dim     = cfg.speaker_encoder.embed_dim,
        conv_channels = cfg.masking.conv_channels,
        lstm_hidden   = cfg.masking.lstm_hidden,
        lstm_dropout  = cfg.masking.lstm_dropout,
    ).to(device)
    model.train()

    # ── EMA ───────────────────────────────────────────────────
    ema = EMA(model, decay=cfg.train_masking.ema_decay)

    # ── Optimizer ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg.train_masking.lr,
        weight_decay = 1e-2,
        betas        = (0.9, 0.999),
    )

    # ── Resume from checkpoint ────────────────────────────────
    start_step = 0
    best_val   = float("inf")

    if resume_from and Path(resume_from).exists():
        start_step = load_checkpoint(
            resume_from, model, ema, optimizer, device
        )

    # ── Training loop ─────────────────────────────────────────
    cfg_train    = cfg.train_masking
    max_steps    = cfg_train.max_steps
    warmup_steps = cfg_train.warmup_steps
    accum_steps  = cfg_train.grad_accumulation
    val_every    = cfg_train.val_every
    save_every   = cfg_train.save_every
    clip_norm    = cfg_train.clip_grad_norm

    print(f"\n  max_steps    : {max_steps:,}")
    print(f"  warmup_steps : {warmup_steps:,}")
    print(f"  batch_size   : {cfg_train.batch_size}")
    print(f"  grad_accum   : {accum_steps}  "
          f"(effective batch = {cfg_train.batch_size * accum_steps})")
    print(f"  device       : {device}")
    print(f"  Starting from step {start_step}\n")

    # TensorBoard
    try:
        from torch.utils.tensorboard import SummaryWriter
        log_dir = Path(cfg.paths.log_dir) / "masking"
        writer  = SummaryWriter(log_dir)
        use_tb  = True
        print(f"  TensorBoard logs → {log_dir}")
    except ImportError:
        writer = None
        use_tb = False

    step        = start_step
    accum_loss  = 0.0
    t_start     = time.time()

    # infinite data iterator
    def infinite_loader(loader):
        while True:
            for batch in loader:
                yield batch

    data_iter = infinite_loader(train_loader)

    optimizer.zero_grad()

    pbar = tqdm(
        total   = max_steps,
        initial = start_step,
        desc    = "Stage1",
        dynamic_ncols = True,
    )

    while step < max_steps:

        # ── Gradient accumulation loop ────────────────────────
        for accum_idx in range(accum_steps):
            batch = next(data_iter)

            x_mel   = batch["mixture_mel"].to(device)
            y_mel   = batch["target_mel"].to(device)
            ref_wav = batch["reference_wav"].to(device)

            # speaker embedding (no grad)
            with torch.no_grad():
                d_vec = encoder(ref_wav)

            # forward pass
            x_enh, mask = model(x_mel, d_vec)

            # loss (masked to ignore silence-padded frames; divide by
            # accum_steps for correct gradient scaling)
            frame_mask = make_frame_mask(batch["target_valid_frames"], y_mel.shape[-1], device)
            loss = model.compute_loss(x_enh, y_mel, frame_mask=frame_mask) / accum_steps
            loss.backward()
            accum_loss += loss.item()

        # ── Optimizer step ────────────────────────────────────
        # update learning rate (warmup, then optional cosine decay)
        lr = get_lr(
            step, warmup_steps, cfg_train.lr,
            max_steps        = max_steps,
            decay_start_step = getattr(cfg_train, "decay_start_step", None),
            min_lr_ratio     = getattr(cfg_train, "min_lr_ratio", 0.01),
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), clip_norm
        )

        optimizer.step()
        optimizer.zero_grad()

        # EMA update
        ema.update()

        step      += 1
        train_loss = accum_loss
        accum_loss = 0.0

        # ── Logging ───────────────────────────────────────────
        pbar.update(1)
        pbar.set_postfix({
            "loss": f"{train_loss:.4f}",
            "lr"  : f"{lr:.2e}",
            "gnorm": f"{grad_norm:.2f}",
        })

        if use_tb and step % 100 == 0:
            writer.add_scalar("train/loss",      train_loss, step)
            writer.add_scalar("train/lr",        lr,         step)
            writer.add_scalar("train/grad_norm", grad_norm,  step)

        # ── Validation ────────────────────────────────────────
        if step % val_every == 0:
            metrics = validate(
                model, encoder, ema,
                val_loader, device
            )
            val_loss = metrics["val_loss"]

            elapsed = (time.time() - t_start) / 60
            print(f"\n  [Step {step:6d}] "
                  f"train={train_loss:.4f}  "
                  f"val={val_loss:.4f}  "
                  f"D%={metrics['val_d_pct']:.1f}  "
                  f"lr={lr:.2e}  "
                  f"elapsed={elapsed:.1f}min")

            if use_tb:
                writer.add_scalar("val/loss",  val_loss,             step)
                writer.add_scalar("val/d_pct", metrics["val_d_pct"], step)

            # save best
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    step, model, ema, optimizer,
                    val_loss, cfg, tag="best"
                )
                print(f"  New best val loss: {best_val:.4f} ✅")

        # ── Periodic checkpoint ───────────────────────────────
        if step % save_every == 0:
            save_checkpoint(
                step, model, ema, optimizer,
                train_loss, cfg, tag="latest"
            )

    pbar.close()

    # ── Final save ────────────────────────────────────────────
    save_checkpoint(
        step, model, ema, optimizer,
        train_loss, cfg, tag="final"
    )

    if use_tb:
        writer.close()

    total_time = (time.time() - t_start) / 3600
    print(f"\n{'='*60}")
    print(f"  Stage 1 training complete!")
    print(f"  Total time  : {total_time:.2f} hours")
    print(f"  Best val    : {best_val:.4f}")
    print(f"  Checkpoints : {cfg.paths.checkpoint_dir}/masking/")
    print(f"{'='*60}\n")

    return model, ema


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="configs/default.yaml")
    parser.add_argument("--resume",   default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--fake",     action="store_true",
                        help="Use fake data (for testing without LibriSpeech)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override max_steps from config")
    parser.add_argument("--val_every", type=int, default=None,
                    help="Override val_every from config")
    parser.add_argument("--warmup_steps", type=int, default=None,
                    help="Override warmup_steps from config (use a small "
                         "value for fast sanity/overfit tests)")
    parser.add_argument("--decay_start_step", type=int, default=None,
                    help="Step to start cosine LR decay from (for resuming "
                         "an already-trained checkpoint into a decay phase)")

    args = parser.parse_args()

    cfg    = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.max_steps:
        cfg.train_masking.max_steps = args.max_steps
    if args.val_every:
        cfg.train_masking.val_every = args.val_every
    if args.warmup_steps is not None:
        cfg.train_masking.warmup_steps = args.warmup_steps
    if args.decay_start_step is not None:
        cfg.train_masking.decay_start_step = args.decay_start_step

    # ── Data ──────────────────────────────────────────────────
    if args.fake:
        print("[Data] Using FAKE data for testing")
        from data.fake_data import build_fake_dataloaders
        train_loader, val_loader = build_fake_dataloaders(
            cfg,
            train_size = 200,
            val_size   = 50,
        )
    else:
        print("[Data] Using LibriSpeech")
        from data.librispeech import build_dataloaders
        train_loader, val_loader = build_dataloaders(cfg)

    train_masking(
        cfg          = cfg,
        train_loader = train_loader,
        val_loader   = val_loader,
        device       = device,
        resume_from  = args.resume,
    )