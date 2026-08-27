"""
Stage 2 Training Loop — Flow Matching Module.

Paper Section 5.2.4:
    "Then, the masking module is frozen, and the flow matching
     module is trained with L_flow."

Training details:
    optimizer : AdamW, lr=2e-4, warmup=10k steps
    loss      : MSE(v_θ(X_t,t,d), Y-X_enh)  [Eq. 16]
    batch     : 10 samples, grad_accumulation=2
    EMA decay : 0.9999
    max_steps : 300,000

Key: masking module is loaded from Stage 1 checkpoint and FROZEN.
     Only the flow matching module is trained here.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from tqdm import tqdm

from models.speaker_encoder import SpeakerEncoder
from models.masking import MaskingModule
from models.flow import FlowMatchingModule
from training.ema import EMA
from training.train_mask import get_lr, save_checkpoint, load_checkpoint
from data.mel import make_frame_mask, masked_mse


# ── Load frozen masking module ────────────────────────────────

def load_frozen_masking(
    cfg:        DictConfig,
    device:     torch.device,
    ckpt_path:  str = None,
) -> MaskingModule:
    """
    Load Stage 1 masking module and freeze it completely.
    Used as frozen prefix during Stage 2 training.
    """
    model = MaskingModule(
        n_mels        = cfg.mel.n_mels,
        embed_dim     = cfg.speaker_encoder.embed_dim,
        conv_channels = cfg.masking.conv_channels,
        lstm_hidden   = cfg.masking.lstm_hidden,
        lstm_dropout  = cfg.masking.lstm_dropout,
    ).to(device)

    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device)

        # ALWAYS load the raw state_dict first — sets every buffer
        # (BatchNorm running_mean/running_var) to its real trained value.
        # EMA never tracks buffers; skipping this left buffers at
        # fresh-init defaults, silently training Stage 2 against a
        # Stage 1 with wrong activation statistics.
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"])

        # Then overlay EMA-smoothed values onto parameters only.
        if "ema" in ckpt:
            ema_shadow = ckpt["ema"]["shadow"]
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in ema_shadow:
                        param.data.copy_(ema_shadow[name].to(device))
            print(f"[Masking] Loaded EMA params + raw buffers from {ckpt_path}")
        else:
            print(f"[Masking] Loaded raw weights from {ckpt_path}")
    else:
        print(f"[Masking] WARNING: No checkpoint found at {ckpt_path}")
        print(f"[Masking] Using random weights (train Stage 1 first!)")

    # freeze completely
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Masking] Frozen ({n_params/1e6:.1f}M params)")
    return model


# ── Validation ────────────────────────────────────────────────

@torch.no_grad()
def validate_flow(
    flow_model:  FlowMatchingModule,
    mask_model:  MaskingModule,
    encoder:     SpeakerEncoder,
    ema:         EMA,
    loader:      DataLoader,
    device:      torch.device,
    cfg_scale:   float = 1.5,
    max_batches: int   = 50,
) -> dict:
    """Validate flow model using EMA weights."""
    flow_model.eval()

    total_loss     = 0.0
    total_mask_mse = 0.0
    total_flow_mse = 0.0
    n_batches      = 0

    with ema.apply():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break

            x_mel   = batch["mixture_mel"].to(device)
            y_mel   = batch["target_mel"].to(device)
            ref_wav = batch["reference_wav"].to(device)

            # speaker embedding
            d_vec = encoder(ref_wav)

            # Stage 1: frozen masking
            x_enh, _ = mask_model(x_mel, d_vec)

            # frame mask: ignore silence-padded frames on short utterances
            frame_mask = make_frame_mask(batch["target_valid_frames"], y_mel.shape[-1], device)

            # Stage 2: flow loss
            loss, info = flow_model.compute_loss(x_enh, y_mel, d_vec, frame_mask=frame_mask)

            # inference quality (single step)
            y_hat = flow_model.inference(
                x_enh, d_vec, cfg_scale=cfg_scale, n_steps=1
            )

            # MSE metrics (masked)
            mask_mse = masked_mse(x_enh, y_mel, frame_mask).item()
            flow_mse = masked_mse(y_hat, y_mel, frame_mask).item()

            total_loss     += loss.item()
            total_mask_mse += mask_mse
            total_flow_mse += flow_mse
            n_batches      += 1

    flow_model.train()
    n = max(1, n_batches)

    return {
        "val_loss"    : total_loss     / n,
        "mask_mse"    : total_mask_mse / n,
        "flow_mse"    : total_flow_mse / n,
        "improvement" : (total_mask_mse - total_flow_mse) / n,
    }


# ── Main training function ────────────────────────────────────

def train_flow(
    cfg:              DictConfig,
    train_loader:     DataLoader,
    val_loader:       DataLoader,
    device:           torch.device,
    mask_ckpt:        str  = None,
    resume_from:      str  = None,
):
    """
    Full Stage 2 training loop.

    Args:
        cfg          : full OmegaConf config
        train_loader : training DataLoader
        val_loader   : validation DataLoader
        device       : torch device
        mask_ckpt    : path to Stage 1 (masking) checkpoint
        resume_from  : path to Stage 2 checkpoint to resume from
    """

    print("\n" + "=" * 60)
    print("  Stage 2: Flow Matching Module Training")
    print("=" * 60)

    # ── Build models ──────────────────────────────────────────
    # Speaker encoder (frozen)
    encoder = SpeakerEncoder(
        model_name = cfg.speaker_encoder.model_name,
        embed_dim  = cfg.speaker_encoder.embed_dim,
        freeze     = True,
        projection_ckpt = "checkpoints_speaker_encoder/projection_latest.pt",
    ).to(device)
    for p in encoder.projection.parameters():
        p.requires_grad = False
    encoder.eval()

    # Masking module (loaded from Stage 1, frozen)
    mask_model = load_frozen_masking(cfg, device, mask_ckpt)

    # Flow matching module (trainable)
    flow_model = FlowMatchingModule(
        n_mels      = cfg.mel.n_mels,
        hidden_dim  = cfg.flow.hidden_dim,
        n_heads     = cfg.flow.n_heads,
        n_blocks    = cfg.flow.n_blocks,
        ffn_mult    = cfg.flow.ffn_mult,
        embed_dim   = cfg.speaker_encoder.embed_dim,
        dropout     = cfg.flow.dropout,
        cfg_dropout = cfg.flow.cfg_dropout,
    ).to(device)
    flow_model.train()

    # ── EMA ───────────────────────────────────────────────────
    ema = EMA(flow_model, decay=cfg.train_flow.ema_decay)

    # ── Optimizer (only flow model params) ────────────────────
    optimizer = torch.optim.AdamW(
        flow_model.parameters(),
        lr           = cfg.train_flow.lr,
        weight_decay = 1e-2,
        betas        = (0.9, 0.999),
    )

    # ── Resume ────────────────────────────────────────────────
    start_step = 0
    best_val   = float("inf")

    if resume_from and Path(resume_from).exists():
        start_step = load_checkpoint(
            resume_from, flow_model, ema, optimizer, device
        )
        # best_val is not part of the checkpoint schema, and gets lost across
        # resumes otherwise — restore it from the existing best-checkpoint
        # file's own recorded val_loss, so a resume can't silently overwrite
        # a genuinely better earlier checkpoint with a worse later one.
        best_ckpt_path = Path(cfg.paths.checkpoint_dir) / "flow" / "flow_best.pt"
        if best_ckpt_path.exists():
            prior_best = torch.load(best_ckpt_path, map_location=device)
            best_val = prior_best.get("val_loss", float("inf"))
            print(f"[Resume] Restored best_val={best_val:.4f} from {best_ckpt_path}")

    # ── Training config ───────────────────────────────────────
    cfg_train    = cfg.train_flow
    max_steps    = cfg_train.max_steps
    warmup_steps = cfg_train.warmup_steps
    accum_steps  = cfg_train.grad_accumulation
    val_every    = cfg_train.val_every
    save_every   = cfg_train.save_every
    clip_norm    = cfg_train.clip_grad_norm

    print(f"\n  max_steps    : {max_steps:,}")
    print(f"  warmup_steps : {warmup_steps:,}")
    print(f"  batch_size   : {cfg_train.batch_size}")
    print(f"  grad_accum   : {accum_steps}")
    print(f"  cfg_scale    : {cfg.inference.cfg_scale}  (at inference)")
    print(f"  device       : {device}")
    print(f"  Starting from step {start_step}\n")

    # TensorBoard
    try:
        from torch.utils.tensorboard import SummaryWriter
        log_dir = Path(cfg.paths.log_dir) / "flow"
        writer  = SummaryWriter(log_dir)
        use_tb  = True
    except ImportError:
        writer = None
        use_tb = False

    step       = start_step
    accum_loss = 0.0
    t_start    = time.time()

    def infinite_loader(loader):
        while True:
            for batch in loader:
                yield batch

    data_iter = infinite_loader(train_loader)
    optimizer.zero_grad()

    pbar = tqdm(
        total   = max_steps,
        initial = start_step,
        desc    = "Stage2",
        dynamic_ncols = True,
    )

    while step < max_steps:

        for accum_idx in range(accum_steps):
            batch = next(data_iter)

            x_mel   = batch["mixture_mel"].to(device)
            y_mel   = batch["target_mel"].to(device)
            ref_wav = batch["reference_wav"].to(device)

            # frozen speaker encoder
            with torch.no_grad():
                d_vec = encoder(ref_wav)

            # frozen masking stage
            with torch.no_grad():
                x_enh, _ = mask_model(x_mel, d_vec)

            # flow matching loss (masked to ignore silence-padded frames)
            frame_mask = make_frame_mask(batch["target_valid_frames"], y_mel.shape[-1], device)
            loss, info = flow_model.compute_loss(x_enh, y_mel, d_vec, frame_mask=frame_mask)
            loss       = loss / accum_steps
            loss.backward()
            accum_loss += loss.item()

        # ── Optimizer step ────────────────────────────────────
        lr = get_lr(
            step, warmup_steps, cfg_train.lr,
            max_steps        = max_steps,
            decay_start_step = getattr(cfg_train, "decay_start_step", None),
            min_lr_ratio     = getattr(cfg_train, "min_lr_ratio", 0.01),
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        grad_norm = torch.nn.utils.clip_grad_norm_(
            flow_model.parameters(), clip_norm
        )

        optimizer.step()
        optimizer.zero_grad()
        ema.update()

        step       += 1
        train_loss  = accum_loss
        accum_loss  = 0.0

        pbar.update(1)
        pbar.set_postfix({
            "loss" : f"{train_loss:.4f}",
            "lr"   : f"{lr:.2e}",
            "gnorm": f"{grad_norm:.2f}",
        })

        if use_tb and step % 100 == 0:
            writer.add_scalar("train/flow_loss",  train_loss, step)
            writer.add_scalar("train/lr",         lr,         step)
            writer.add_scalar("train/grad_norm",  grad_norm,  step)

        # ── Validation ────────────────────────────────────────
        if step % val_every == 0:
            metrics = validate_flow(
                flow_model, mask_model, encoder,
                ema, val_loader, device,
                cfg_scale = cfg.inference.cfg_scale,
            )

            elapsed = (time.time() - t_start) / 60
            print(f"\n  [Step {step:6d}] "
                  f"train={train_loss:.4f}  "
                  f"val={metrics['val_loss']:.4f}  "
                  f"mask_mse={metrics['mask_mse']:.4f}  "
                  f"flow_mse={metrics['flow_mse']:.4f}  "
                  f"improvement={metrics['improvement']:.4f}  "
                  f"elapsed={elapsed:.1f}min")

            if use_tb:
                for k, v in metrics.items():
                    writer.add_scalar(f"val/{k}", v, step)

            if metrics["val_loss"] < best_val:
                best_val = metrics["val_loss"]
                save_checkpoint(
                    step, flow_model, ema, optimizer,
                    metrics["val_loss"], cfg, tag="best",
                    stage="flow", prefix="flow"
                )
                print(f"  New best val loss: {best_val:.4f} ✅")

        if step % save_every == 0:
            save_checkpoint(
                step, flow_model, ema, optimizer,
                train_loss, cfg, tag="latest",
                stage="flow", prefix="flow"
            )

    pbar.close()
    save_checkpoint(
        step, flow_model, ema, optimizer,
        train_loss, cfg, tag="final",
        stage="flow", prefix="flow"
    )

    if use_tb:
        writer.close()

    total_time = (time.time() - t_start) / 3600
    print(f"\n{'='*60}")
    print(f"  Stage 2 training complete!")
    print(f"  Total time  : {total_time:.2f} hours")
    print(f"  Best val    : {best_val:.4f}")
    print(f"{'='*60}\n")

    return flow_model, ema


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="configs/default.yaml")
    parser.add_argument("--mask_ckpt", default=None,
                        help="Path to Stage 1 checkpoint")
    parser.add_argument("--resume",    default=None,
                        help="Path to Stage 2 checkpoint to resume from")
    parser.add_argument("--fake",      action="store_true")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--val_every", type=int, default=None,
                        help="Override val_every from config")
    parser.add_argument("--warmup_steps", type=int, default=None,
                        help="Override warmup_steps from config (use a "
                             "small value for fast sanity/overfit tests)")
    parser.add_argument("--decay_start_step", type=int, default=None,
                        help="Step to start cosine LR decay from")
    args = parser.parse_args()

    cfg    = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.max_steps:
        cfg.train_flow.max_steps = args.max_steps
    if args.val_every:
        cfg.train_flow.val_every = args.val_every
    if args.warmup_steps is not None:
        cfg.train_flow.warmup_steps = args.warmup_steps
    if args.decay_start_step is not None:
        cfg.train_flow.decay_start_step = args.decay_start_step

    if args.fake:
        print("[Data] Using FAKE data for testing")
        from data.fake_data import build_fake_dataloaders
        train_loader, val_loader = build_fake_dataloaders(
            cfg, train_size=200, val_size=50
        )
    else:
        from data.librispeech import build_dataloaders
        train_loader, val_loader = build_dataloaders(cfg)

    train_flow(
        cfg          = cfg,
        train_loader = train_loader,
        val_loader   = val_loader,
        device       = device,
        mask_ckpt    = args.mask_ckpt,
        resume_from  = args.resume,
    )