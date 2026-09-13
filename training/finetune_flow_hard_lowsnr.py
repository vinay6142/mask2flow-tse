"""
Stage-2 fine-tune closing the SNR-coverage gap.

WHY THIS EXISTS
---------------
This project's mixer (data/augment.py's MixtureCreator, configured by
configs/default_v2.yaml) has always drawn mixture SNR from [snr_min=1.0,
snr_max=10.0] dB. The target speaker is therefore ALWAYS at least as loud as
the interferer in every training example the model has ever seen -- the model
has never once been asked to extract the QUIETER of two speakers.

That gap is measurable, and large. On Libri2Mix (whose mix_clean condition has
no SNR floor, so 60.6% of its samples sit below 1dB) the current checkpoint
degrades smoothly and severely as SNR drops -- see
docs/results_and_limitations.md Sec 5.5.2 for the characterization and Sec 4
timeline entry 14 for the stage attribution:

    SNR bucket   Stage 2 vs Stage 1 (median mel)   fraction where S2 HURTS
    <-5dB                    -30.6%                       94.8%
    [-5,-2)                  -32.2%                       78.8%
    [-2,0)                    -7.8%                       53.8%
    [0,1)                    +53.6%                       29.3%
    [1,3)   <- trained       +67.2%                       15.0%
    [5,10]+ <- trained       +75.7%                        1.4%

Below -5dB Stage 2 makes 94.8% of samples WORSE than Stage 1 alone. That is a
total breakdown in that regime, not an outlier tail, and it is exactly the
signature of a training-distribution gap rather than an architectural flaw --
the same diagnosis, and the same fix strategy, as the two fine-tunes that
already worked here (finetune_flow_hard_t0.py,
finetune_flow_hard_multispeaker.py).

WHAT THIS DOES
--------------
Stage-2-only fine-tune (Stage 1 frozen, matching both prior fine-tunes),
continuing from the current best checkpoint, with an SNR curriculum layered on
the TRAIN split: --low_snr_prob of examples draw their SNR from --low_snr_range
(default 50% from [-10, 1) dB, covering essentially all of Libri2Mix's range),
the rest keep drawing from the configured [1, 10] dB exactly as before. The
50/50 split deliberately mirrors the multi-speaker curriculum's weighting
toward preserving the already-validated condition, rather than chasing the new
regime at the old one's expense.

The speaker-count curriculum stays ON at the same time
(--interferer_count_probs, default 0.5/0.3/0.2). This matters: the checkpoint
this starts from IS the hard-multispeaker result (3-speaker 81.9%, 4-speaker
76.7%), so fine-tuning it on single-interferer mixtures alone would risk
handing those gains straight back. Both curricula run together so the model
widens its SNR range without narrowing its speaker-count range.

Implementation note: this script deliberately does NOT duplicate the training
loop. It reuses finetune_flow_multispeaker() from
training/finetune_flow_hard_multispeaker.py verbatim -- that loop is
curriculum-agnostic (it consumes whatever the dataloaders yield), already
proven across a completed 25k-step run, and takes its checkpoint directory
from --stage/--prefix, which are overridden below so this run writes to its
own directory and never touches flow_best.pt.

A KNOWN RISK, STATED UP FRONT
-----------------------------
Unlike the speaker-count gap -- where Stage 1 was unaffected and a Stage-2-only
fix was therefore clearly sufficient -- Stage 1 ALSO degrades at low SNR (its
median mel improvement decays 8.1%->1.1%, and its SI-SDR contribution goes from
+0.57dB to -7.83dB, because a deletion-only mask forced to remove most of the
spectrum takes target energy with it). Stage 2 is generative and can in
principle reconstruct what Stage 1 over-deleted, which is why Stage-2-only is
the right first attempt -- but Stage 1 is a plausible CEILING here. If the
post-fine-tune numbers plateau well short of the in-distribution result, that
is the signal that Stage 1 needs retraining too, not a surprise.

VALIDATION uses the same 2-speaker, [1,10]dB val_loader as every other run in
this project, on purpose: the val_loss proxy stays comparable across runs
rather than shifting under a harder task mix. Do NOT read it as the target
metric -- as in both prior fine-tunes, it is expected to look WORSE while the
real metric improves. Track eval/eval_libri2mix.py's SNR-split table instead.

Run:
  sbatch scripts/run_finetune_flow_lowsnr_gpu.sh

  # or directly -- note --low_snr_range MUST use the equals form, since
  # argparse reads the leading '-' of a space-separated value as an option name:
  python3 -u training/finetune_flow_hard_lowsnr.py \
      --flow_ckpt checkpoints_v2/flow/flow_best.pt \
      --low_snr_prob 0.5 --low_snr_range=-10,1 \
      --interferer_count_probs 0.5,0.3,0.2 \
      --max_steps 25000
"""
import os
import sys

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.librispeech import build_dataloaders_multispeaker
from training.finetune_flow_hard_multispeaker import finetune_flow_multispeaker


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="configs/default_v2.yaml")
    parser.add_argument("--mask_ckpt", default="checkpoints_v2/masking/mask_best.pt")
    parser.add_argument("--flow_ckpt", default="checkpoints_v2/flow/flow_best.pt",
                         help="Starting point -- the existing best Stage 2 checkpoint "
                              "(currently the hard-multispeaker fine-tune).")
    parser.add_argument("--resume",    default=None,
                         help="Path to a flow_ft_lowsnr_*.pt checkpoint to resume THIS "
                              "fine-tune from (e.g. after a SLURM timeout), NOT the "
                              "same thing as --flow_ckpt.")
    parser.add_argument("--low_snr_prob", type=float, default=0.5,
                         help="Fraction of TRAIN examples drawing SNR from --low_snr_range "
                              "instead of the config's [snr_min, snr_max]. Default 0.5 "
                              "mirrors the multi-speaker curriculum's weighting toward "
                              "preserving the already-validated condition.")
    parser.add_argument("--low_snr_range", default="-10,1",
                         help="Comma-separated low,high dB for the extended range. "
                              "Default -10,1 covers essentially all of Libri2Mix "
                              "(which spans -11.6 to +11.6dB) below the trained floor. "
                              "IMPORTANT: pass this with an EQUALS sign "
                              "(--low_snr_range=-10,1). Space-separated "
                              "('--low_snr_range -10,1') fails, because argparse reads a "
                              "leading '-' as the start of another option name -- it only "
                              "exempts plain negative numbers like -10, and the comma "
                              "breaks that match.")
    parser.add_argument("--interferer_count_probs", default="0.5,0.3,0.2",
                         help="Kept ON alongside the SNR curriculum so the multi-speaker "
                              "ability of the starting checkpoint isn't given back. Set "
                              "to 1.0 to disable (pure 2-speaker), but see this "
                              "script's docstring before doing so.")
    parser.add_argument("--max_steps",   type=int,   default=25000)
    parser.add_argument("--lr",          type=float, default=4e-5)
    parser.add_argument("--ema_decay",   type=float, default=0.999)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--accum_steps",  type=int, default=2)
    parser.add_argument("--val_every",    type=int, default=1000)
    parser.add_argument("--save_every",   type=int, default=2500)
    parser.add_argument("--stage",  default="flow_finetune_lowsnr")
    parser.add_argument("--prefix", default="flow_ft_lowsnr")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    interferer_count_probs = [float(x) for x in args.interferer_count_probs.split(",")]
    assert abs(sum(interferer_count_probs) - 1.0) < 1e-5, \
        f"--interferer_count_probs must sum to 1.0, got {interferer_count_probs}"

    low_snr_range = tuple(float(x) for x in args.low_snr_range.split(","))
    assert len(low_snr_range) == 2 and low_snr_range[0] < low_snr_range[1], \
        f"--low_snr_range must be low,high with low < high, got {args.low_snr_range}"
    assert 0.0 <= args.low_snr_prob <= 1.0, \
        f"--low_snr_prob must be in [0, 1], got {args.low_snr_prob}"

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[Finetune] WARNING: no GPU detected -- a 77M-parameter DiT for "
              f"{args.max_steps} steps on CPU would take an impractically long "
              "time. Submit this as a GPU SLURM job instead of running it here.")

    trained_lo, trained_hi = cfg.data.snr_min, cfg.data.snr_max
    print(f"[Finetune] SNR curriculum: {args.low_snr_prob:.0%} of train examples from "
          f"[{low_snr_range[0]}, {low_snr_range[1]}) dB (NEW -- never trained on before), "
          f"{1 - args.low_snr_prob:.0%} from [{trained_lo}, {trained_hi}] dB (trained range)")
    print("[Finetune] speaker-count curriculum stays ON: "
          + ", ".join(f"P({i+2} total speakers)={p}"
                      for i, p in enumerate(interferer_count_probs)))
    print("[Finetune] val_loader is 2-speaker / trained-SNR only on purpose -- expect "
          "val_loss to look WORSE while the real metric improves (same as both prior "
          "fine-tunes). Judge this run by eval/eval_libri2mix.py's SNR-split table.")

    train_loader, val_loader = build_dataloaders_multispeaker(
        cfg, interferer_count_probs, num_workers=args.num_workers,
        low_snr_prob=args.low_snr_prob, low_snr_range=low_snr_range,
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
