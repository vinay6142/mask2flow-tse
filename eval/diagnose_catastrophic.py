"""
Mask2Flow-TSE — Diagnostic for the catastrophic Stage-2 outlier failures
(samples where Stage 2 flow-matching makes MSE dramatically WORSE than
Stage 1 alone — e.g. -2465%, -1346%, -430% in the eval_newproj_10555 run).

Two independent checks, run against the exact seed=42/batch=60 eval batch
used by eval/results_stage2.py:

  [A] Speaker-encoder generalization, measured PROPERLY.
      training/train_speaker_encoder.py's val_acc is NOT a valid metric:
      the classifier head's output layer has train_ds.num_speakers classes
      (speaker identities from train-clean-100), but validate() compares
      its argmax against val_ds labels, which index a COMPLETELY DIFFERENT
      and disjoint label space (test-clean speaker identities, arbitrarily
      re-numbered 0..N by SpeakerClassificationDataset's own enumeration).
      There is no reason index 7 in the train taxonomy should mean anything
      next to index 7 in the val taxonomy — the reported val_acc=0.0083 is
      indistinguishable from chance (~1/num_train_speakers) BY CONSTRUCTION
      and says nothing about whether the projection generalizes to unseen
      speakers. This script instead measures the standard, meaningful
      check: same-speaker vs different-speaker cosine similarity on
      held-out test-clean speakers, no classifier head involved.

  [B] Per-Euler-step instrumentation of the full pipeline on the flagged
      catastrophic samples (auto-detected: S2-vs-S1 MSE regression beyond
      a threshold), to see WHERE the blow-up happens:
        - is the sample's d-vector degenerate (near null_spk, or an
          outlier vs the rest of the batch)?
        - does ||v_cond - v_uncond|| (the CFG delta being amplified by
          cfg_scale=1.5) spike for this sample specifically?
        - which Euler step does ||x|| actually diverge on?
      This distinguishes "bad speaker conditioning" from "CFG/numerical
      extrapolation instability independent of conditioning quality".

Run:
  python3 eval/diagnose_catastrophic.py \
      --mask_ckpt checkpoints_v2/masking/mask_best.pt \
      --flow_ckpt checkpoints_v2/flow/flow_best.pt \
      --projection_ckpt checkpoints_speaker_encoder/projection_latest.pt \
      --batch_size 60 --seed 42 --n_steps 4
"""
import os
import sys
import argparse
import random

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.mel import make_frame_mask, masked_mse, segment_waveform
from data.augment import load_audio
from models.speaker_encoder import SpeakerEncoder

from eval.results_stage2 import load_masking, load_flow, get_real_batch


# ─────────────────────────────────────────────────────────────────────
# [A] Proper unseen-speaker verification (no classifier head involved)
# ─────────────────────────────────────────────────────────────────────

def build_speaker_utt_index(librispeech_root, split, min_utts=4, max_speakers=20):
    from pathlib import Path
    root = Path(librispeech_root) / split
    spk_utts = {}
    for spk_dir in sorted(root.iterdir()):
        if not spk_dir.is_dir():
            continue
        paths = [str(p) for p in spk_dir.rglob("*.flac")]
        if len(paths) >= min_utts:
            spk_utts[spk_dir.name] = paths
    speakers = sorted(spk_utts.keys())
    rng = random.Random(42)
    rng.shuffle(speakers)
    speakers = speakers[:max_speakers]
    return {s: spk_utts[s] for s in speakers}


@torch.no_grad()
def verify_speaker_generalization(encoder, cfg, device, utts_per_speaker=4, max_speakers=20):
    print("\n" + "=" * 86)
    print("  [A] SPEAKER-ENCODER GENERALIZATION — proper unseen-speaker check")
    print("=" * 86)
    print("  (replaces train_speaker_encoder.py's val_acc, which compares a "
          "train-taxonomy classifier's\n   argmax against an unrelated "
          "val-taxonomy label space — chance-level BY CONSTRUCTION, not a "
          "real signal)")

    seg_len = getattr(cfg.audio, "reference_length", 3.0)
    spk_utts = build_speaker_utt_index(
        cfg.data.librispeech_path, "test-clean",
        min_utts=utts_per_speaker, max_speakers=max_speakers,
    )
    print(f"\n  Using {len(spk_utts)} held-out test-clean speakers, "
          f"{utts_per_speaker} utterances each (segment={seg_len}s)")

    embeddings = {}  # spk -> (utts_per_speaker, 512)
    for spk, paths in spk_utts.items():
        rng = random.Random(hash(spk) % (2**31))
        chosen = rng.sample(paths, utts_per_speaker)
        wavs = []
        for p in chosen:
            wav = load_audio(p, cfg.audio.sample_rate)
            wav = segment_waveform(wav, cfg.audio.sample_rate, seg_len, random_start=False)
            wavs.append(wav)
        batch = torch.stack(wavs).to(device)
        embeddings[spk] = encoder(batch)  # (utts_per_speaker, 512)

    intra, inter = [], []
    speakers = list(embeddings.keys())
    for spk in speakers:
        e = embeddings[spk]
        sim = e @ e.T
        n = e.shape[0]
        iu = torch.triu_indices(n, n, offset=1)
        intra.extend(sim[iu[0], iu[1]].tolist())
    for i in range(len(speakers)):
        for j in range(i + 1, len(speakers)):
            sim = embeddings[speakers[i]] @ embeddings[speakers[j]].T
            inter.extend(sim.flatten().tolist())

    import statistics as st
    print(f"\n  Intra-speaker cosine sim : mean={st.mean(intra):.4f}  "
          f"median={st.median(intra):.4f}  min={min(intra):.4f}  (n={len(intra)} pairs)")
    print(f"  Inter-speaker cosine sim : mean={st.mean(inter):.4f}  "
          f"median={st.median(inter):.4f}  max={max(inter):.4f}  (n={len(inter)} pairs)")
    gap = st.mean(intra) - st.mean(inter)
    print(f"  Separation gap (intra - inter, mean) : {gap:+.4f}")

    # simple threshold-free separation quality: AUC of intra > inter
    import itertools
    correct = sum(1 for a, b in itertools.product(intra, inter) if a > b)
    total = len(intra) * len(inter)
    auc = correct / total if total else float("nan")
    print(f"  Pairwise separation AUC (P[intra_sim > inter_sim]) : {auc:.4f}"
          f"   (0.5=random/degenerate, 1.0=perfect)")

    if auc < 0.75:
        print("\n  ⚠️  VERDICT: projection does NOT meaningfully separate unseen "
              "speakers (AUC<0.75).\n      This IS consistent with degenerate/"
              "uninformative d-vectors feeding Stage 1+2.")
    else:
        print("\n  ✅ VERDICT: projection separates unseen speakers well above chance.\n"
              "      Speaker-embedding degeneracy is likely NOT the (sole) cause of "
              "the catastrophic samples;\n      see part [B] for the CFG/numerical "
              "explanation instead.")
    return auc


# ─────────────────────────────────────────────────────────────────────
# [B] Per-step instrumentation of the full pipeline, flagged samples
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def instrumented_inference(flow_model, x_enh, d_vector, cfg_scale, n_steps, watch_idx,
                            cfg_warmup_steps=0):
    """Re-implements FlowMatchingModule.inference()'s Euler loop but logs
    per-step norms for the sample indices in `watch_idx`. cfg_warmup_steps
    mirrors the fix in models/flow.py: run the first N steps at
    cfg_scale=1.0 (no guidance amplification) before switching to the
    full cfg_scale, to confirm it flattens the t=0 cfg_delta_norm spike."""
    B = x_enh.shape[0]
    x = x_enh.clone()
    dt = 1.0 / n_steps

    logs = {i: [] for i in watch_idx}
    for step in range(n_steps):
        t = torch.full((B,), step * dt, device=x.device)
        step_cfg_scale = 1.0 if step < cfg_warmup_steps else cfg_scale
        if step_cfg_scale > 1.0:
            v_cond = flow_model.forward(x, t, d_vector)
            v_uncond = flow_model.forward(x, t, d_vector, force_cfg_drop=True)
            v = v_uncond + step_cfg_scale * (v_cond - v_uncond)
        else:
            v_cond = flow_model.forward(x, t, d_vector)
            v_uncond = v_cond
            v = v_cond

        for i in watch_idx:
            logs[i].append({
                "step": step,
                "x_norm": x[i].norm().item(),
                "v_cond_norm": v_cond[i].norm().item(),
                "v_uncond_norm": v_uncond[i].norm().item(),
                "cfg_delta_norm": (v_cond[i] - v_uncond[i]).norm().item(),
                "v_applied_norm": v[i].norm().item(),
            })

        x = x + v * dt

    for i in watch_idx:
        logs[i].append({
            "step": n_steps, "x_norm": x[i].norm().item(),
            "v_cond_norm": None, "v_uncond_norm": None,
            "cfg_delta_norm": None, "v_applied_norm": None,
        })
    return x, logs


@torch.no_grad()
def diagnose_flagged_samples(mask_model, flow_model, mixture, target, ref_wav,
                              d_vec, valid_frames, cfg_scale, n_steps,
                              regression_threshold_pct=-30.0, max_flag=8,
                              cfg_warmup_steps=0):
    print("\n" + "=" * 86)
    print("  [B] PER-STEP INSTRUMENTATION — catastrophic samples"
          + (f"  (cfg_warmup_steps={cfg_warmup_steps})" if cfg_warmup_steps else ""))
    print("=" * 86)

    stage1_out, _ = mask_model(mixture, d_vec)

    frame_mask = make_frame_mask(valid_frames, target.shape[-1], target.device)
    B = mixture.shape[0]

    # cheap pass with the real (non-instrumented) call, to score+flag samples
    stage2_out = flow_model.inference(stage1_out, d_vec, cfg_scale=cfg_scale, n_steps=n_steps,
                                       cfg_warmup_steps=cfg_warmup_steps)

    s2_vs_s1 = []
    for b in range(B):
        fm_b = frame_mask[[b]]
        s1_mse = masked_mse(stage1_out[[b]], target[[b]], fm_b).item()
        s2_mse = masked_mse(stage2_out[[b]], target[[b]], fm_b).item()
        s2_vs_s1.append((s1_mse - s2_mse) / max(s1_mse, 1e-8) * 100)

    # Always report the 8 samples known-catastrophic at cfg_warmup_steps=0
    # (from eval_newproj_10555), so a warmup>0 rerun shows the before/after
    # directly instead of just "nothing flagged" if warmup already fixed them.
    known_catastrophic = [2, 31, 19, 26, 28, 45, 29, 39]  # 0-based: samples 3,32,20,27,29,46,30,40
    print(f"\n  S2-vs-S1 MSE regression on the samples flagged catastrophic at "
          f"cfg_warmup_steps=0:")
    for i in known_catastrophic:
        if i < B:
            flag = "  <-- still flagged" if s2_vs_s1[i] < regression_threshold_pct else "  (fixed)"
            print(f"    Sample {i + 1:<3} S2-vs-S1 = {s2_vs_s1[i]:>+9.1f}%{flag}")

    flagged = [i for i, v in enumerate(s2_vs_s1) if v < regression_threshold_pct]
    flagged = sorted(flagged, key=lambda i: s2_vs_s1[i])[:max_flag]
    print(f"\n  Flagged {len(flagged)} sample(s) with S2-vs-S1 MSE regression "
          f"< {regression_threshold_pct:.0f}%: "
          f"{[(i + 1, round(s2_vs_s1[i], 1)) for i in flagged]}")
    if not flagged:
        print("  Nothing crossed the threshold — nothing to instrument.")
        return

    # d-vector geometry: is the flagged sample's d-vector an outlier vs the batch?
    sim_matrix = d_vec @ d_vec.T  # (B, B)
    null_sim = F.cosine_similarity(
        d_vec, flow_model.null_spk.expand(B, -1), dim=-1
    )

    for i in flagged:
        others = [j for j in range(B) if j != i]
        nn_sim = sim_matrix[i, others].max().item()
        mean_sim = sim_matrix[i, others].mean().item()
        print(f"\n  --- Sample {i + 1} (S2 vs S1 = {s2_vs_s1[i]:+.1f}%) ---")
        print(f"    d-vector: nearest-neighbor cos-sim in batch = {nn_sim:.4f}, "
              f"mean cos-sim to rest of batch = {mean_sim:.4f}, "
              f"cos-sim to null_spk (CFG unconditional) = {null_sim[i].item():.4f}")

    # ── [C] Velocity DIRECTION check ────────────────────────────────
    # cfg_warmup_steps=1 barely moved the worst offenders (confirmed
    # empirically: x_norm trajectories were nearly identical with/without
    # CFG at step 0), even though it zeroed the CFG delta there. That
    # rules out "CFG amplification inflates magnitude" as the (sole)
    # mechanism. What's left: v_cond itself, with NO guidance involved,
    # already points somewhere that produces huge per-element error vs
    # the true required velocity (target - stage1_out), despite its NORM
    # looking unremarkable. Check the actual cosine similarity between
    # the model's t=0 prediction and the true velocity directly — this
    # distinguishes "undertrained/OOD region of input space" (low or
    # negative cosine sim to true velocity) from "data corruption for
    # these specific samples" (true velocity itself has some anomaly:
    # huge norm, NaN, or stage1_out/target have extreme values).
    print("\n" + "=" * 86)
    print("  [C] VELOCITY DIRECTION CHECK — predicted vs true velocity at t=0")
    print("=" * 86)
    print("  (cfg_warmup barely moved the worst offenders — x_norm grew almost "
          "identically with/without\n   CFG at step 0 — so this checks whether "
          "v_cond itself, not CFG, is pointing the wrong way)")

    t0 = torch.zeros(B, device=stage1_out.device)
    v_cond_t0 = flow_model.forward(stage1_out, t0, d_vec)
    true_vel = target - stage1_out

    baseline_idx = [i for i in range(B) if i not in flagged][:4]
    print(f"\n  {'Sample':<10} {'true_vel_norm':>14} {'v_cond_norm':>13} "
          f"{'cos_sim(pred,true)':>20} {'tgt_max_abs':>12} {'s1out_max_abs':>14}")
    for i in flagged + baseline_idx:
        tv = true_vel[i][:, :int(valid_frames[i].item())]
        vc = v_cond_t0[i][:, :int(valid_frames[i].item())]
        cos = F.cosine_similarity(vc.flatten(), tv.flatten(), dim=0).item()
        tag = " <-- flagged" if i in flagged else "  (baseline)"
        print(f"  Sample {i+1:<3} {tv.norm().item():>14.2f} {vc.norm().item():>13.2f} "
              f"{cos:>20.4f} {target[i].abs().max().item():>12.2f} "
              f"{stage1_out[i].abs().max().item():>14.2f}{tag}")

    print("\n  How to read this:")
    print("    - cos_sim near 0 or negative for flagged samples (vs clearly "
          "positive for baseline samples)\n      -> the model's prediction is "
          "uncorrelated with (or opposed to) what's actually needed for these\n"
          "      specific inputs: an undertrained/OOD region of input space, "
          "not a numerical/CFG artifact.")
    print("    - tgt_max_abs or s1out_max_abs far outside the rest of the batch's "
          "range for flagged samples\n      -> points to a data anomaly (e.g. "
          "clipping, an unusually loud/quiet source) specific to those\n      "
          "utterances/mixing draws, worth checking in data/librispeech.py's "
          "mixing for this seed/index.")

    watch_idx = flagged
    _, logs = instrumented_inference(
        flow_model, stage1_out, d_vec, cfg_scale, n_steps, watch_idx,
        cfg_warmup_steps=cfg_warmup_steps,
    )

    for i in flagged:
        print(f"\n  Sample {i + 1} per-step trace (target x_norm should stay "
              f"roughly flat/bounded — a real spectrogram's norm doesn't grow "
              f"unboundedly):")
        print(f"  {'step':>4} {'x_norm':>10} {'v_cond_norm':>12} "
              f"{'v_uncond_norm':>13} {'cfg_delta_norm':>14} {'v_applied_norm':>14}")
        for row in logs[i]:
            def fmt(v):
                return f"{v:>10.2f}" if v is not None else f"{'--':>10}"
            print(f"  {row['step']:>4} {fmt(row['x_norm'])} "
                  f"{fmt(row['v_cond_norm']):>12} {fmt(row['v_uncond_norm']):>13} "
                  f"{fmt(row['cfg_delta_norm']):>14} {fmt(row['v_applied_norm']):>14}")

    print("\n  How to read this:")
    print("    - If x_norm is already large/erratic at step 0 (i.e. Stage 1's "
          "output itself), the root\n      cause is upstream in Stage 1, not "
          "flow matching.")
    print("    - If cfg_delta_norm is far larger for flagged samples than a "
          "typical sample, CFG amplification\n      (cfg_scale=1.5x on an "
          "already-uncertain v_cond/v_uncond disagreement) is implicated — "
          "try\n      --cfg_scale 1.0 on just these samples as a confirming test.")
    print("    - If x_norm stays reasonable through early steps then jumps "
          "sharply on the last step,\n      it's a late-step extrapolation "
          "blow-up, not a gradual drift — consistent with too few Euler "
          "steps\n      (n_steps=4) for this particular trajectory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default_v2.yaml")
    parser.add_argument("--mask_ckpt", required=True)
    parser.add_argument("--flow_ckpt", required=True)
    parser.add_argument("--projection_ckpt", default="checkpoints_speaker_encoder/projection_latest.pt")
    parser.add_argument("--batch_size", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg_scale", type=float, default=None)
    parser.add_argument("--n_steps", type=int, default=None)
    parser.add_argument("--cfg_warmup_steps", type=int, default=0,
                         help="Run the first N Euler steps at cfg_scale=1.0 "
                              "before switching to the full cfg_scale — the "
                              "candidate fix for the t=0 CFG-delta spike. "
                              "Rerun with --cfg_warmup_steps 1 to confirm.")
    parser.add_argument("--regression_threshold_pct", type=float, default=-30.0)
    parser.add_argument("--skip_speaker_check", action="store_true")
    parser.add_argument("--verify_speakers", type=int, default=20,
                         help="Number of held-out test-clean speakers to use for part [A]")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    cfg_scale = args.cfg_scale if args.cfg_scale is not None else cfg.inference.cfg_scale
    n_steps = args.n_steps if args.n_steps is not None else cfg.inference.flow_steps

    print(f"[Diagnose] Device: {device}  cfg_scale={cfg_scale}  n_steps={n_steps}")

    encoder = SpeakerEncoder(
        model_name=cfg.speaker_encoder.model_name,
        embed_dim=cfg.speaker_encoder.embed_dim,
        freeze=True,
        projection_ckpt=args.projection_ckpt,
    ).to(device)
    encoder.eval()

    mask_model = load_masking(args.mask_ckpt, cfg, device)
    flow_model = load_flow(args.flow_ckpt, cfg, device)

    if not args.skip_speaker_check:
        verify_speaker_generalization(encoder, cfg, device, max_speakers=args.verify_speakers)

    mixture, target, ref_wav, condition, snr_db, valid_frames = get_real_batch(
        cfg, device, B=args.batch_size, seed=args.seed
    )
    with torch.no_grad():
        d_vec = encoder(ref_wav)

    diagnose_flagged_samples(
        mask_model, flow_model, mixture, target, ref_wav, d_vec, valid_frames,
        cfg_scale, n_steps, regression_threshold_pct=args.regression_threshold_pct,
        cfg_warmup_steps=args.cfg_warmup_steps,
    )
