"""
Stage 2 targeted fine-tune -- hard multi-speaker curriculum.

Root cause (see docs/results_and_limitations.md Sec 5.5.3 / the
mask2flow-tse-multi-speaker-test project memory): training data is built
EXCLUSIVELY from 2-total-speaker mixtures (1 target + exactly 1 interferer --
data/augment.py's MixtureCreator, used unconditionally by every existing
training script). Measured on real LibriSpeech test-clean mixtures (n=300
per condition, eval/eval_multi_speaker.py):

    Speakers   Stage 2 accuracy (1-EER)   Stage 1 alone vs mixture
    2 (trained)        86.3%                      8.7%
    3                  76.7%                      n/a (not attributed)
    4                  72.6%                      10.9%

Stage-attribution (mel_s1_imp_pct / mel_s2_vs_s1_pct already in the
collected eval JSONL, no extra run needed) showed Stage 1 (masking) is
UNAFFECTED by speaker count -- if anything marginally better at 4 speakers,
0/300 samples where it makes things worse than the raw mixture. The ENTIRE
degradation is in Stage 2 (flow matching): it adds 78.3% improvement over
Stage 1 at 2 speakers but only 53.7% at 4. A cfg_scale/n_steps sweep at the
4-speaker condition (4 variants, n=100 each) landed all within ~1pp of each
other -- inference-time tuning does NOT close this gap, matching the
precedent from the (separately diagnosed, differently-caused) t≈0 gap.

This fine-tune does NOT change architecture. It changes what Stage 2 sees
during continued training: instead of every batch being a fixed 2-speaker
mixture, LibriSpeechTSEDataset (via the NEW interferer_count_probs param --
see its docstring in data/librispeech.py) now samples a variable interferer
count per example, using the eval-proven mix_multi_at_snr() generalization
of the original mix_at_snr(). The curriculum is WEIGHTED TOWARD 2-speaker
(default --interferer_count_probs 0.5,0.3,0.2 = 50% 2spk / 30% 3spk / 20%
4spk), not exclusively 4-speaker, specifically so the model keeps seeing
plenty of the original trained condition and doesn't regress the validated
86.3%/78.3% 2-speaker numbers while gaining exposure to harder mixtures.

Unlike finetune_flow_hard_t0.py, this does NOT bias the t-sampling -- t
stays plain Uniform(0,1) via FlowMatchingModule's own compute_loss(). The
two fine-tunes target different, independently-diagnosed gaps (t≈0
single-shot prediction error vs. speaker-count-coverage) and are not meant
to be conflated; this script could in principle be chained after the
hard-t0 fine-tune's output checkpoint, or run independently from
flow_best.pt -- --flow_ckpt controls the starting point either way.

VALIDATION here uses build_dataloaders_multispeaker()'s val_loader, which
stays on the ORIGINAL fixed 2-speaker distribution (see that function's
docstring) -- val_loss is comparable to every other training run's curve,
but is still only a cheap mel-domain proxy, NOT the real target metric.
Track actual progress with eval/eval_multi_speaker.py's accuracy table
(2/3/4 speakers) against saved checkpoints, same discipline as
finetune_flow_hard_t0.py's own guidance about eval/verify_eer.py.

Starts from --flow_ckpt (default checkpoints_v2/flow/flow_best.pt, the
current hard-t0-fine-tuned checkpoint) with a FRESH optimizer and a low LR
(default 4e-5, matching hard-t0's choice) -- standard fine-tune practice.
Checkpoints land in a SEPARATE directory
(checkpoints_v2/flow_finetune_multispeaker/), never overwriting
flow_best.pt, so the current best checkpoint is always there to fall back
to regardless of how this fine-tune turns out.

Run (SLURM GPU job recommended -- ~25k steps, similar budget to hard-t0's
~11-12h on a P100; this is a STARTING POINT, not a guarantee -- re-check
eval_multi_speaker.py's accuracy table at intermediate checkpoints and
extend or stop based on the real metric, exactly as hard-t0 did):
  python3 training/finetune_flow_hard_multispeaker.py \
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
from omegaconf import OmegaConf
from pathlib import Path
from tqdm import tqdm

from models.speaker_encoder import SpeakerEncoder
from models.flow import FlowMatchingModule
from training.ema import EMA
from training.train_mask import get_lr, save_checkpoint, load_checkpoint
from training.train_flow import load_frozen_masking, validate_flow
from data.mel import make_frame_mask


def load_flow_for_finetune(ckpt_path, cfg, device):
    """
    Identical to finetune_flow_hard_t0.py's loader of the same name (kept as
    a separate copy, not a shared import, so the two fine-tune scripts stay
    fully independent experiments -- see that script's own docstring for
    why this is deliberate elsewhere in this project). Loads EMA weights if
    present, leaves the model in train() mode with requires_grad=True
    (unlike eval/results_stage2.py's inference-only load_flow()).
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


def finetune_flow_multispeaker(
    cfg, train_loader, val_loader, device,
    mask_ckpt, flow_ckpt,
    max_steps=25000, lr=4e-5, ema_decay=0.999,
    warmup_steps=500, accum_steps=2, clip_norm=1.0,
    val_every=1000, save_every=2500,
    stage="flow_finetune_multispeaker", prefix="flow_ft_ms",
    resume_from=None,
):
    """
    Checkpoints land at <cfg.paths.checkpoint_dir>/<stage>/<prefix>_*.pt
    (default: checkpoints_v2/flow_finetune_multispeaker/flow_ft_ms_*.pt) --
    a separate directory from checkpoints_v2/flow/, so flow_best.pt is
    never touched and stays available as a fallback.
    """
    print("\n" + "=" * 60)
    print("  Stage 2 fine-tune -- hard multi-speaker curriculum")
    print(f"  lr={lr}")
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

    pbar = tqdm(total=max_steps, initial=start_step, desc="FinetuneMultiSpeaker", dynamic_ncols=True)

    while step < max_steps:
        accum_loss = 0.0
        for _ in range(accum_steps):
            batch = next(data_iter)
            x_mel   = batch["mixture_mel"].to(device)
            y_mel   = batch["target_mel"].to(device)
            ref_wav = batch["reference_wav"].to(device)

            with torch.no_grad():
                d_vec = encoder(ref_wav)
                x_enh, _ = mask_model(x_mel, d_vec)

            frame_mask = make_frame_mask(batch["target_valid_frames"], y_mel.shape[-1], device)
            loss, info = flow_model.compute_loss(x_enh, y_mel, d_vec, frame_mask=frame_mask)
            loss = loss / accum_steps
            loss.backward()
            accum_loss += loss.item()

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
            "loss": f"{accum_loss:.4f}", "lr": f"{lr_now:.2e}", "gnorm": f"{grad_norm:.2f}",
        })

        if use_tb and step % 100 == 0:
            writer.add_scalar("train/loss", accum_loss, step)
            writer.add_scalar("train/lr", lr_now, step)

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
                      f" this val_loader is 2-speaker-only (see build_dataloaders_multispeaker's "
                      f"docstring), so this proxy is comparable to past runs but is NOT the real "
                      f"target metric for THIS fine-tune. Confirm with eval/eval_multi_speaker.py's "
                      f"2/3/4-speaker accuracy table before trusting it.")

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
    print(f"  Next: run eval/eval_multi_speaker.py (n_interferers=1, 2, 3) against")
    print(f"  checkpoints_v2/{stage}/{prefix}_final.pt (pass --flow_ckpt) to check the")
    print(f"  real target metric -- 2-speaker accuracy must not have regressed below")
    print(f"  ~86%, and 4-speaker accuracy is the number to compare against the 72.6%")
    print(f"  pre-fine-tune baseline / 75-80% target.")
    print(f"{'='*60}\n")
    return flow_model, ema


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="configs/default_v2.yaml")
    parser.add_argument("--mask_ckpt", default="checkpoints_v2/masking/mask_best.pt")
    parser.add_argument("--flow_ckpt", default="checkpoints_v2/flow/flow_best.pt",
                         help="Starting point -- the existing best Stage 2 checkpoint "
                              "(currently the hard-t0 fine-tune).")
    parser.add_argument("--resume",    default=None,
                         help="Path to a flow_ft_ms_*.pt checkpoint to resume THIS "
                              "fine-tune from (e.g. after a SLURM timeout), NOT "
                              "the same thing as --flow_ckpt.")
    parser.add_argument("--interferer_count_probs", default="0.5,0.3,0.2",
                         help="Comma-separated probabilities for 1,2,3,... simultaneous "
                              "interferers (i.e. 2,3,4,... total speakers). Must sum to 1.0. "
                              "Default weights toward the original 2-speaker condition so "
                              "existing performance doesn't regress while the model gains "
                              "exposure to harder mixtures.")
    parser.add_argument("--max_steps",   type=int,   default=25000)
    parser.add_argument("--lr",          type=float, default=4e-5)
    parser.add_argument("--ema_decay",   type=float, default=0.999)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--accum_steps",  type=int, default=2)
    parser.add_argument("--val_every",    type=int, default=1000)
    parser.add_argument("--save_every",   type=int, default=2500)
    parser.add_argument("--stage",  default="flow_finetune_multispeaker")
    parser.add_argument("--prefix", default="flow_ft_ms")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    interferer_count_probs = [float(x) for x in args.interferer_count_probs.split(",")]
    assert abs(sum(interferer_count_probs) - 1.0) < 1e-5, \
        f"--interferer_count_probs must sum to 1.0, got {interferer_count_probs} (sum={sum(interferer_count_probs)})"

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[Finetune] WARNING: no GPU detected -- a 77M-parameter DiT for "
              f"{args.max_steps} steps on CPU would take an impractically long "
              "time. Submit this as a GPU SLURM job instead of running it here.")

    print(f"[Finetune] interferer_count_probs={interferer_count_probs} -> "
          + ", ".join(f"P({i+2} total speakers)={p}" for i, p in enumerate(interferer_count_probs)))

    from data.librispeech import build_dataloaders_multispeaker
    train_loader, val_loader = build_dataloaders_multispeaker(
        cfg, interferer_count_probs, num_workers=args.num_workers
    )

    finetune_flow_multispeaker(
        cfg, train_loader, val_loader, device,
        mask_ckpt=args.mask_ckpt, flow_ckpt=args.flow_ckpt,
        max_steps=args.max_steps, lr=args.lr, ema_decay=args.ema_decay,
        warmup_steps=args.warmup_steps, accum_steps=args.accum_steps,
        val_every=args.val_every, save_every=args.save_every,
        stage=args.stage, prefix=args.prefix,
        resume_from=args.resume,
    )
