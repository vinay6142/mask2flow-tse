"""
Stage 2 targeted fine-tune -- hard t~0 timestep reweighting.

Root cause (see eval/KNOWN_LIMITATIONS.md): at t=0 the rectified-flow
interpolation x_t = x_enh exactly, so the model must predict the ENTIRE
correction in a single shot from Stage 1's output + speaker embedding
alone, with zero progress signal. Uniform(0,1) t-sampling during the
original 300k-step run gives this hardest regime the same training
weight as every easier, higher-t regime -- this fine-tune does not change
architecture or data, it ONLY oversamples low-t batches so the model gets
disproportionately more gradient signal exactly where the diagnostic
(diagnose_catastrophic.py part [C]) found it failing: cos_sim(pred, true
velocity) collapsing to 0.08-0.15 for the worst offenders at t=0, vs
0.77-0.92 for healthy baseline samples.

t sampling: with probability --t_hard_prob (default 0.5), t is drawn from
Uniform(0, --t_hard_max) (default 0.25 -- matches Euler step 0's window
at the n_steps=4 inference default) instead of Uniform(0,1). The other
half of each batch still uses plain Uniform(0,1), so the model does not
forget how to handle mid/late-trajectory timesteps while it gets extra
practice on the hard regime.

Starts from checkpoints_v2/flow/flow_best.pt (step 298000, the existing
best checkpoint) with a FRESH optimizer and a much lower LR (default
4e-5 vs the original 2e-4) -- standard fine-tune practice, avoids
reusing 298k-step-old Adam momentum estimates under a different LR and
t-distribution. Checkpoints are written to a SEPARATE directory
(checkpoints_v2/flow_finetune_hardt0/), never overwriting flow_best.pt,
so the original checkpoint is always there to fall back to.

Validation here is the same cheap mel-domain check used during the
original training (uniform-t loss + single-step inference MSE) -- it is
NOT the real target metric. Track actual progress against the 87%
target by periodically running eval/verify_eer.py against the saved
flow_ft_latest.pt / flow_ft_stepNNNN.pt checkpoints; this script only
picks its own "best" checkpoint by the cheap proxy (val_loss), same
convention as the original train_flow.py.

Run (SLURM GPU job recommended -- ~25k steps):
  python3 training/finetune_flow_hard_t0.py \
      --config configs/default_v2.yaml \
      --mask_ckpt checkpoints_v2/masking/mask_best.pt \
      --flow_ckpt checkpoints_v2/flow/flow_best.pt \
      --max_steps 25000
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from pathlib import Path
from tqdm import tqdm

from models.speaker_encoder import SpeakerEncoder
from models.flow import FlowMatchingModule
from training.ema import EMA
from training.train_mask import get_lr, save_checkpoint, load_checkpoint
from training.train_flow import load_frozen_masking, validate_flow
from data.mel import make_frame_mask


def compute_loss_biased_t(flow_model, x_enh, y_target, d_vector, frame_mask,
                           t_hard_prob, t_hard_max):
    """
    Same rectified-flow loss as FlowMatchingModule.compute_loss, except t
    is drawn from a mixture that oversamples the hard near-0 regime
    instead of plain Uniform(0,1). Kept as a standalone function (not a
    change to models/flow.py) so this experiment stays fully isolated
    from the core module every other script depends on.
    """
    B = x_enh.shape[0]
    device = x_enh.device

    hard_mask = torch.rand(B, device=device) < t_hard_prob
    t_uniform = torch.rand(B, device=device)
    t_hard    = torch.rand(B, device=device) * t_hard_max
    t = torch.where(hard_mask, t_hard, t_uniform)

    t_exp = t.view(B, 1, 1)
    x_t = (1 - t_exp) * x_enh + t_exp * y_target
    target_vel = y_target - x_enh

    pred_vel = flow_model.forward(x_t, t, d_vector)

    if frame_mask is None:
        loss = F.mse_loss(pred_vel, target_vel)
    else:
        fm = frame_mask
        if fm.dim() == 2:
            fm = fm.unsqueeze(1)
        diff2 = (pred_vel - target_vel) ** 2 * fm
        denom = fm.sum() * pred_vel.shape[1] + 1e-8
        loss = diff2.sum() / denom

    info = {
        "loss": loss.item(),
        "t_mean": t.mean().item(),
        "frac_hard": hard_mask.float().mean().item(),
    }
    return loss, info


def load_flow_for_finetune(ckpt_path, cfg, device):
    """
    Like eval/results_stage2.py's load_flow, but keeps requires_grad=True
    and leaves the model in train() mode -- that loader is inference-only
    (freezes everything), not reusable here.
    """
    model = FlowMatchingModule(
        n_mels      = cfg.mel.n_mels,
        hidden_dim  = cfg.flow.hidden_dim,
        n_heads     = cfg.flow.n_heads,
        n_blocks    = cfg.flow.n_blocks,
        ffn_mult    = cfg.flow.ffn_mult,
        embed_dim   = cfg.speaker_encoder.embed_dim,
        dropout     = cfg.flow.dropout,
        cfg_dropout = cfg.flow.cfg_dropout,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    if "ema" in ckpt:
        ema_shadow = ckpt["ema"]["shadow"]
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in ema_shadow:
                    param.data.copy_(ema_shadow[name].to(device))
        step = ckpt.get("step", "?")
        val_loss = ckpt.get("val_loss", float("inf"))
        print(f"[Finetune] Initialized from {ckpt_path} "
              f"(EMA weights, step {step}, val_loss={val_loss:.4f})")
    else:
        print(f"[Finetune] Initialized from {ckpt_path} (raw weights only)")

    model.train()
    return model


def finetune_flow(cfg, train_loader, val_loader, device,
                   mask_ckpt, flow_ckpt,
                   max_steps=25000, lr=4e-5, ema_decay=0.999,
                   t_hard_prob=0.5, t_hard_max=0.25,
                   warmup_steps=500, accum_steps=2, clip_norm=1.0,
                   val_every=1000, save_every=2500,
                   stage="flow_finetune_hardt0", prefix="flow_ft",
                   resume_from=None):
    """
    Checkpoints land at <cfg.paths.checkpoint_dir>/<stage>/<prefix>_*.pt
    (default: checkpoints_v2/flow_finetune_hardt0/flow_ft_*.pt) -- a
    separate directory from checkpoints_v2/flow/, so flow_best.pt is
    never touched and stays available as a fallback.
    """
    print("\n" + "=" * 60)
    print("  Stage 2 fine-tune -- hard t~0 timestep reweighting")
    print(f"  t_hard_prob={t_hard_prob}  t_hard_max={t_hard_max}  lr={lr}")
    print(f"  checkpoints -> {cfg.paths.checkpoint_dir}/{stage}/{prefix}_*.pt")
    print("=" * 60)
    encoder = SpeakerEncoder(
        model_name = cfg.speaker_encoder.model_name,
        embed_dim  = cfg.speaker_encoder.embed_dim,
        freeze     = True,
        projection_ckpt = "checkpoints_speaker_encoder/projection_latest.pt",
    ).to(device)
    for p in encoder.projection.parameters():
        p.requires_grad = False
    encoder.eval()

    mask_model = load_frozen_masking(cfg, device, mask_ckpt)
    flow_model = load_flow_for_finetune(flow_ckpt, cfg, device)

    ema = EMA(flow_model, decay=ema_decay)
    optimizer = torch.optim.AdamW(
        flow_model.parameters(), lr=lr, weight_decay=1e-2, betas=(0.9, 0.999)
    )

    start_step = 0
    best_val = float("inf")
    if resume_from and Path(resume_from).exists():
        start_step = load_checkpoint(resume_from, flow_model, ema, optimizer, device)

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(Path(cfg.paths.log_dir) / stage)
        use_tb = True
    except ImportError:
        writer = None
        use_tb = False

    def infinite_loader(loader):
        while True:
            for batch in loader:
                yield batch

    data_iter = infinite_loader(train_loader)
    step = start_step
    optimizer.zero_grad()
    t_start = time.time()

    pbar = tqdm(total=max_steps, initial=start_step, desc="FinetuneHardT0", dynamic_ncols=True)

    while step < max_steps:
        accum_loss = 0.0
        last_info = {}
        for _ in range(accum_steps):
            batch = next(data_iter)
            x_mel   = batch["mixture_mel"].to(device)
            y_mel   = batch["target_mel"].to(device)
            ref_wav = batch["reference_wav"].to(device)

            with torch.no_grad():
                d_vec = encoder(ref_wav)
                x_enh, _ = mask_model(x_mel, d_vec)

            frame_mask = make_frame_mask(batch["target_valid_frames"], y_mel.shape[-1], device)
            loss, info = compute_loss_biased_t(
                flow_model, x_enh, y_mel, d_vec, frame_mask, t_hard_prob, t_hard_max
            )
            loss = loss / accum_steps
            loss.backward()
            accum_loss += loss.item()
            last_info = info

        lr_now = get_lr(step, warmup_steps, lr, max_steps=None)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        grad_norm = torch.nn.utils.clip_grad_norm_(flow_model.parameters(), clip_norm)
        optimizer.step()
        optimizer.zero_grad()
        ema.update()

        step += 1
        pbar.update(1)
        pbar.set_postfix({
            "loss": f"{accum_loss:.4f}", "lr": f"{lr_now:.2e}",
            "t_mean": f"{last_info.get('t_mean', 0):.2f}",
            "gnorm": f"{grad_norm:.2f}",
        })

        if use_tb and step % 100 == 0:
            writer.add_scalar("train/loss", accum_loss, step)
            writer.add_scalar("train/lr", lr_now, step)
            writer.add_scalar("train/t_mean", last_info.get("t_mean", 0), step)

        if step % val_every == 0:
            metrics = validate_flow(
                flow_model, mask_model, encoder, ema, val_loader, device,
                cfg_scale=cfg.inference.cfg_scale,
            )
            elapsed = (time.time() - t_start) / 60
            print(f"\n  [Step {step:6d}] val_loss={metrics['val_loss']:.4f}  "
                  f"mask_mse={metrics['mask_mse']:.4f}  flow_mse={metrics['flow_mse']:.4f}  "
                  f"improvement={metrics['improvement']:.4f}  elapsed={elapsed:.1f}min")
            if use_tb:
                for k, v in metrics.items():
                    writer.add_scalar(f"val/{k}", v, step)
            if metrics["val_loss"] < best_val:
                best_val = metrics["val_loss"]
                save_checkpoint(step, flow_model, ema, optimizer, best_val, cfg,
                                 tag="best", stage=stage, prefix=prefix)
                print(f"  New best val_loss: {best_val:.4f} (checkpoint saved) --"
                      f" remember this is the cheap mel-domain proxy, not the real "
                      f"target metric. Confirm with eval/verify_eer.py before trusting it.")

        if step % save_every == 0:
            save_checkpoint(step, flow_model, ema, optimizer, accum_loss, cfg,
                             tag="latest", stage=stage, prefix=prefix)

    pbar.close()
    save_checkpoint(step, flow_model, ema, optimizer, accum_loss, cfg,
                     tag="final", stage=stage, prefix=prefix)
    if use_tb:
        writer.close()

    total_time = (time.time() - t_start) / 3600
    print(f"\n{'='*60}")
    print(f"  Fine-tune complete. Total time: {total_time:.2f}h  Best val_loss: {best_val:.4f}")
    print(f"  Checkpoints: {cfg.paths.checkpoint_dir}/{stage}/")
    print(f"  Next: run eval/verify_eer.py against flow_ft_best.pt / flow_ft_latest.pt")
    print(f"  (pass --flow_ckpt checkpoints_v2/{stage}/{prefix}_best.pt) to check the")
    print(f"  real target metric before deciding whether to extend or stop.")
    print(f"{'='*60}\n")
    return flow_model, ema


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="configs/default_v2.yaml")
    parser.add_argument("--mask_ckpt", default="checkpoints_v2/masking/mask_best.pt")
    parser.add_argument("--flow_ckpt", default="checkpoints_v2/flow/flow_best.pt",
                         help="Starting point -- the existing best Stage 2 checkpoint.")
    parser.add_argument("--resume",    default=None,
                         help="Path to a flow_ft_*.pt checkpoint to resume THIS "
                              "fine-tune from (e.g. after a SLURM timeout), NOT "
                              "the same thing as --flow_ckpt.")
    parser.add_argument("--max_steps",   type=int,   default=25000)
    parser.add_argument("--lr",          type=float, default=4e-5)
    parser.add_argument("--ema_decay",   type=float, default=0.999)
    parser.add_argument("--t_hard_prob", type=float, default=0.5,
                         help="Fraction of each batch sampled from the hard "
                              "low-t range instead of Uniform(0,1).")
    parser.add_argument("--t_hard_max",  type=float, default=0.25,
                         help="Upper bound of the hard t range -- matches Euler "
                              "step 0's window at the n_steps=4 inference default.")
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--accum_steps",  type=int, default=2)
    parser.add_argument("--val_every",    type=int, default=1000)
    parser.add_argument("--save_every",   type=int, default=2500)
    parser.add_argument("--stage",  default="flow_finetune_hardt0")
    parser.add_argument("--prefix", default="flow_ft")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[Finetune] WARNING: no GPU detected -- a 77M-parameter DiT for "
              f"{args.max_steps} steps on CPU would take an impractically long "
              "time. Submit this as a GPU SLURM job instead of running it here.")

    from data.librispeech import build_dataloaders
    train_loader, val_loader = build_dataloaders(cfg)

    finetune_flow(
        cfg, train_loader, val_loader, device,
        mask_ckpt=args.mask_ckpt, flow_ckpt=args.flow_ckpt,
        max_steps=args.max_steps, lr=args.lr, ema_decay=args.ema_decay,
        t_hard_prob=args.t_hard_prob, t_hard_max=args.t_hard_max,
        warmup_steps=args.warmup_steps, accum_steps=args.accum_steps,
        val_every=args.val_every, save_every=args.save_every,
        stage=args.stage, prefix=args.prefix,
        resume_from=args.resume,
    )
