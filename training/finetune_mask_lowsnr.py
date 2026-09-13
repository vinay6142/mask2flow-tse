"""
Stage-1 fine-tune for low SNR, in either mask formulation: the two arms of the
A/B in docs/methodology_and_project_history.md entry 21.

Entry 20's oracles showed low-SNR extraction is capped by BOTH Stage 1's trained
network and its mask formulation (flow_best.pt, [-10,1) dB):
    trained Stage 1                           61.3% acc   -15.86 dB vs mixture
    best mask within the paper's formulation  73.0%        +7.98 dB
    true energy deletion                      89.9%       +20.21 dB
The oracles use the clean target, so those are ceilings. This trains real
networks toward each, with everything but the formulation held fixed:
    --mask_mode multiplicative   X * M, the paper's Eq. 9   (ceiling 73.0%)
    --mask_mode log_gain         X + log(M), true deletion  (ceiling 89.9%)
Both arms warm-start from the same mask_best.pt (the modes share parameters;
only the final application differs), with the same curriculum, LR, steps, seed.

Curriculum: 50% of examples from [-10,1) dB, 50% from the trained [1,10] dB, with
the 2/3/4-speaker mix kept on -- as in the Stage-2 low-SNR run. lr 1e-4 sits
between the Stage-2 fine-tunes' 4e-5 and Stage 1's original 2e-4: the log_gain
arm has to re-learn gain magnitudes, and a rate chosen for a gentle nudge would
handicap it; both arms must share it for the comparison to be fair.

Stage 2 is untouched. Evaluate with checkpoints_v2/flow/flow_best.pt, NOT the
low-SNR Stage-2 fine-tune: that one gains only +4.5 dB from a better mask versus
+16.2 dB for flow_best, because it learned to compensate for Stage 1.

Validation is the usual 2-speaker, trained-SNR set. Both arms minimise the same
MSE, so val_loss is comparable between them, but it only covers the easy regime:
judge by the evaluation battery. D% is logged too -- 100% by construction for
log_gain, not for multiplicative.

Run:  sbatch scripts/run_finetune_mask_lowsnr_gpu.sh <multiplicative|log_gain>
"""
import os
import sys
import time
import random
import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.speaker_encoder import SpeakerEncoder
from models.masking import MaskingModule
from training.ema import EMA
from training.train_mask import get_lr, save_checkpoint, load_checkpoint, validate
from data.mel import make_frame_mask
from data.librispeech import build_dataloaders_multispeaker


def load_for_finetune(path, cfg, device, mask_mode):
    """Raw state_dict first (sets BatchNorm buffers), then EMA over parameters --
    the same order as load_frozen_masking -- but left trainable, in mask_mode."""
    model = MaskingModule(
        n_mels=cfg.mel.n_mels, embed_dim=cfg.speaker_encoder.embed_dim,
        conv_channels=cfg.masking.conv_channels, lstm_hidden=cfg.masking.lstm_hidden,
        lstm_dropout=cfg.masking.lstm_dropout, mask_mode=mask_mode,
    ).to(device)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    shadow = ckpt.get("ema", {}).get("shadow", {})
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in shadow:
                p.data.copy_(shadow[name].to(device))
    print(f"[FinetuneMask] warm start {path} (step {ckpt.get('step', '?')}, trained as "
          f"{ckpt.get('mask_mode', 'multiplicative')}) -> training as {mask_mode}")
    return model.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default_v2.yaml")
    ap.add_argument("--mask_ckpt", default="checkpoints_v2/masking/mask_best.pt",
                    help="Warm-start weights; both arms start here.")
    ap.add_argument("--mask_mode", required=True, choices=MaskingModule.MASK_MODES)
    ap.add_argument("--resume", default=None, help="A latest.pt from THIS fine-tune (same mask_mode).")
    ap.add_argument("--low_snr_prob", type=float, default=0.5)
    ap.add_argument("--low_snr_range", default="-10,1",
                    help="low,high dB. Pass as --low_snr_range=-10,1: space-separated, "
                         "argparse reads the leading '-' as an option name.")
    ap.add_argument("--interferer_count_probs", default="0.5,0.3,0.2")
    ap.add_argument("--max_steps", type=int, default=25000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup_steps", type=int, default=1000)
    ap.add_argument("--accum_steps", type=int, default=2)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--save_every", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=8)
    a = ap.parse_args()

    probs = [float(v) for v in a.interferer_count_probs.split(",")]
    lo, hi = (float(v) for v in a.low_snr_range.split(","))
    assert lo < hi, f"--low_snr_range needs low < high, got {a.low_snr_range}"
    assert abs(sum(probs) - 1.0) < 1e-5, f"--interferer_count_probs must sum to 1, got {probs}"
    random.seed(a.seed)
    torch.manual_seed(a.seed)          # same construction + data order in both arms

    cfg = OmegaConf.load(a.config)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage, prefix = f"mask_finetune_lowsnr_{a.mask_mode}", f"mask_ft_{a.mask_mode}"
    extra = {"mask_mode": a.mask_mode}
    print(f"[FinetuneMask] mask_mode={a.mask_mode}  lr={a.lr}  steps={a.max_steps}  "
          f"{a.low_snr_prob:.0%} of examples from [{lo}, {hi}) dB  speaker mix={probs}  "
          f"-> {cfg.paths.checkpoint_dir}/{stage}/", flush=True)

    encoder = SpeakerEncoder(model_name=cfg.speaker_encoder.model_name,
                             embed_dim=cfg.speaker_encoder.embed_dim, freeze=True,
                             projection_ckpt="checkpoints_speaker_encoder/projection_latest.pt").to(dev)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    model = load_for_finetune(a.mask_ckpt, cfg, dev, a.mask_mode)
    ema = EMA(model, decay=a.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2, betas=(0.9, 0.999))

    step, best, run_loss = 0, float("inf"), float("nan")
    if a.resume and Path(a.resume).exists():
        saved = torch.load(a.resume, map_location="cpu").get("mask_mode", "multiplicative")
        if saved != a.mask_mode:
            raise SystemExit(f"--resume {a.resume} was trained as {saved}, not {a.mask_mode}")
        step = load_checkpoint(a.resume, model, ema, opt, dev)

    train_loader, val_loader = build_dataloaders_multispeaker(
        cfg, probs, num_workers=a.num_workers, low_snr_prob=a.low_snr_prob, low_snr_range=(lo, hi))
    clip = getattr(cfg.train_masking, "clip_grad_norm", 1.0)

    def batches():
        while True:
            yield from train_loader
    data = batches()

    t0 = time.time()
    opt.zero_grad()
    bar = tqdm(total=a.max_steps, initial=step, desc=f"FinetuneMask[{a.mask_mode}]", dynamic_ncols=True)
    while step < a.max_steps:
        run_loss = 0.0
        for _ in range(a.accum_steps):
            b = next(data)
            x, y = b["mixture_mel"].to(dev), b["target_mel"].to(dev)
            with torch.no_grad():
                d = encoder(b["reference_wav"].to(dev))
            x_enh, _ = model(x, d)
            fm = make_frame_mask(b["target_valid_frames"], y.shape[-1], dev)
            loss = model.compute_loss(x_enh, y, frame_mask=fm) / a.accum_steps
            loss.backward()
            run_loss += loss.item()
        lr = get_lr(step, a.warmup_steps, a.lr)
        for group in opt.param_groups:
            group["lr"] = lr
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        opt.zero_grad()
        ema.update()
        step += 1
        bar.update(1)
        bar.set_postfix(loss=f"{run_loss:.4f}", lr=f"{lr:.2e}", gnorm=f"{grad_norm:.2f}")

        if step % a.val_every == 0:
            m = validate(model, encoder, ema, val_loader, dev)
            print(f"\n  [Step {step:6d}] train={run_loss:.4f}  val={m['val_loss']:.4f}  "
                  f"D%={m['val_d_pct']:.1f}  elapsed={(time.time() - t0) / 60:.1f}min", flush=True)
            if m["val_loss"] < best:
                best = m["val_loss"]
                save_checkpoint(step, model, ema, opt, best, cfg, tag="best",
                                stage=stage, prefix=prefix, extra=extra)
        if step % a.save_every == 0:
            save_checkpoint(step, model, ema, opt, run_loss, cfg, tag="latest",
                            stage=stage, prefix=prefix, extra=extra)
    bar.close()

    save_checkpoint(step, model, ema, opt, run_loss, cfg, tag="final", stage=stage, prefix=prefix, extra=extra)
    print(f"\n[FinetuneMask] done: {a.mask_mode}, {(time.time() - t0) / 3600:.2f}h, best val {best:.4f}\n"
          f"  Evaluate with flow_best.pt as Stage 2:\n"
          f"  sbatch scripts/run_eval_candidate_gpu.sh checkpoints_v2/flow/flow_best.pt mask_{a.mask_mode} "
          f"{cfg.paths.checkpoint_dir}/{stage}/{prefix}_final.pt", flush=True)


if __name__ == "__main__":
    main()
