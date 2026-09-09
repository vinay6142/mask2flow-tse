"""
Corpus-wide speaker-verification EER/AUC for Mask2Flow-TSE extraction output.

full_eval.py earlier in-batch verification metric (genuine vs. hardest/mean
impostor drawn from the SAME 20-sample batch) turned out to be unreliable --
it swung from 65% (hardest impostor) to 98.3% (mean impostor, but so did the
do-nothing mixture baseline, at 90%, meaning that version does not
discriminate extraction quality at all). Batch-local impostor selection is
inherently noisy with only ~40 test-clean speakers and a small batch.

This script fixes it properly: collects embeddings for every evaluated
sample first (Stage 2 extraction, ground-truth target, raw mixture, and
each sample own drawn reference), then builds genuine/impostor trials
ACROSS THE WHOLE evaluated set (not one batch) and reports Equal Error
Rate (EER) and AUC -- the standard speaker-verification protocol, and the
same one already used elsewhere in this project (see
verify_speaker_generalization() in diagnose_catastrophic.py) to validate
the raw encoder AUC=0.93-0.97. Applying it here for the first time to the
actual EXTRACTED audio, not just the raw encoder.

  genuine trial : (probe[i] embedding, i's own reference embedding)
  impostor trial: (probe[i] embedding, j's reference embedding)
                  for every j with a DIFFERENT target_speaker than i
                  (same-speaker-different-utterance pairs, i != j but
                  speaker[i]==speaker[j], are excluded from both pools --
                  ambiguous, not a clean genuine-vs-impostor trial)

Reports EER for three probe sources (mixture / Stage 2 extraction / oracle
ground-truth target) against the SAME reference matrix and impostor pool,
so the numbers are directly comparable -- analogous to full_eval.py mel-
MSE "vs mixture / vs Stage1" structure. "Accuracy" = 1 - EER (%).

Run:
  python3 eval/verify_eer.py \
      --mask_ckpt checkpoints_v2/masking/mask_best.pt \
      --flow_ckpt checkpoints_v2/flow/flow_best.pt \
      --projection_ckpt checkpoints_speaker_encoder/projection_latest.pt \
      --seed 42 --n_steps 4 --max_samples 400
"""
import os
import sys
import time
import argparse

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.vocoder import load_hifigan_generator, mel_to_audio_hifigan
from models.speaker_encoder import SpeakerEncoder
from data.augment import trim_trailing_silence
from eval.results_stage2 import load_masking, load_flow
from eval.full_eval import build_loader


@torch.no_grad()
def collect_embeddings(loader, mask_model, flow_model, encoder, vocoder,
                        cfg_scale, n_steps, device, sr):
    """
    Runs the full pipeline over every batch in loader and returns, for
    every sample: its reference embedding, mixture/Stage2/target probe
    embeddings, and its target_speaker id -- everything the EER computation
    needs, collected across the WHOLE dataset rather than per-batch.
    """
    ref_embs, mix_embs, s2_embs, tgt_embs, speakers = [], [], [], [], []

    t_start = time.time()
    n_done = 0
    for batch in loader:
        mixture      = batch["mixture_mel"].to(device)
        target       = batch["target_mel"].to(device)
        ref_wav      = batch["reference_wav"].to(device)
        valid_frames = batch["target_valid_frames"].to(device)
        speaker      = batch["target_speaker"]

        B = mixture.shape[0]

        # Trim trailing silence off each reference clip before embedding --
        # the shared LibriSpeechTSEDataset's segment_waveform() zero-pads any
        # reference utterance shorter than reference_length (3.0s), diluting
        # the mean-pooled WavLM embedding. Same fix already applied in
        # eval_multi_speaker.py / eval_libri2mix.py / results_stage2.py; done
        # per-sample (not batched) since trimming produces variable lengths.
        # This is the SAME d_vec used both as extraction conditioning below
        # and as the stored reference embedding for the accuracy/EER metric,
        # so both get the fix consistently for free.
        d_vec = torch.stack([
            encoder(trim_trailing_silence(ref_wav[b], sr).unsqueeze(0))[0]
            for b in range(B)
        ])
        stage1_out, _ = mask_model(mixture, d_vec)
        stage2_out = flow_model.inference(
            stage1_out, d_vec, cfg_scale=cfg_scale, n_steps=n_steps,
        )

        for b in range(B):
            vf = max(int(valid_frames[b].item()), 1)
            mix_wav = torch.from_numpy(mel_to_audio_hifigan(mixture[b, :, :vf], vocoder)).float().to(device)
            s2_wav  = torch.from_numpy(mel_to_audio_hifigan(stage2_out[b, :, :vf], vocoder)).float().to(device)
            tgt_wav = torch.from_numpy(mel_to_audio_hifigan(target[b, :, :vf], vocoder)).float().to(device)

            ref_embs.append(d_vec[b].cpu())
            mix_embs.append(encoder(mix_wav.unsqueeze(0))[0].cpu())
            s2_embs.append(encoder(s2_wav.unsqueeze(0))[0].cpu())
            tgt_embs.append(encoder(tgt_wav.unsqueeze(0))[0].cpu())
            speakers.append(speaker[b])

        n_done += B
        elapsed = time.time() - t_start
        print(f"[VerifyEER] {n_done} samples embedded ({elapsed/60:.1f} min elapsed)")

    return (torch.stack(ref_embs), torch.stack(mix_embs),
            torch.stack(s2_embs), torch.stack(tgt_embs), speakers)


def genuine_impostor_scores(sim_matrix, speakers):
    """
    sim_matrix: (N, N) cosine similarities, probe[i] vs reference[j].
    speakers  : list[str] length N, target speaker id per sample.
    Returns (genuine, impostor) 1D score tensors:
      genuine  = sim_matrix[i, i]                    for every i
      impostor = sim_matrix[i, j], i != j, speaker[i] != speaker[j]
    Same-speaker off-diagonal pairs (i != j, same speaker) are excluded
    from both pools -- not a clean genuine or impostor trial.
    """
    N = sim_matrix.shape[0]
    uniq = {s: idx for idx, s in enumerate(sorted(set(speakers)))}
    codes = torch.tensor([uniq[s] for s in speakers])
    same_speaker = codes.unsqueeze(1) == codes.unsqueeze(0)   # (N, N)
    diag_mask    = torch.eye(N, dtype=torch.bool)

    genuine  = sim_matrix[diag_mask]
    impostor = sim_matrix[(~same_speaker) & (~diag_mask)]
    return genuine, impostor


def compute_eer_auc(genuine, impostor):
    """
    genuine, impostor: 1D tensors of similarity scores.
    Returns (eer, eer_threshold, auc). Standard cumulative-count sweep,
    O((Ng+Ni) log(Ng+Ni)) via a single sort -- no sklearn dependency.
    """
    scores = torch.cat([genuine, impostor])
    labels = torch.cat([torch.ones_like(genuine), torch.zeros_like(impostor)])
    order  = torch.argsort(scores, descending=True)
    labels_sorted = labels[order]
    scores_sorted = scores[order]

    n_gen = genuine.numel()
    n_imp = impostor.numel()

    tp_cum = torch.cumsum(labels_sorted, dim=0)
    fp_cum = torch.cumsum(1.0 - labels_sorted, dim=0)

    frr = 1.0 - tp_cum / n_gen
    far = fp_cum / n_imp

    diff = (far - frr).abs()
    eer_idx = torch.argmin(diff).item()
    eer = ((far[eer_idx] + frr[eer_idx]) / 2).item()
    eer_threshold = scores_sorted[eer_idx].item()

    ranks = torch.argsort(torch.argsort(scores)).float() + 1.0
    sum_ranks_genuine = ranks[labels == 1].sum()
    auc = ((sum_ranks_genuine - n_gen * (n_gen + 1) / 2) / (n_gen * n_imp)).item()

    return eer, eer_threshold, auc


def report(name, probe_emb, ref_emb, speakers):
    sim = probe_emb @ ref_emb.T
    genuine, impostor = genuine_impostor_scores(sim, speakers)
    eer, thresh, auc = compute_eer_auc(genuine, impostor)
    print(f"    {name:<38}: accuracy(1-EER)={100*(1-eer):5.1f}%   "
          f"EER={100*eer:5.1f}%  AUC={auc:.4f}  "
          f"(n_genuine={genuine.numel()}, n_impostor={impostor.numel()})")
    return eer, auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default_v2.yaml")
    parser.add_argument("--mask_ckpt", required=True)
    parser.add_argument("--flow_ckpt", required=True)
    parser.add_argument("--projection_ckpt", default="checkpoints_speaker_encoder/projection_latest.pt")
    parser.add_argument("--vocoder_ckpt", default="checkpoints_vocoder/vocoder_best.pt")
    parser.add_argument("--vocoder_config", default="configs/vocoder.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--max_samples", type=int, default=400,
                         help="Cap total samples (default 400 -- impostor pairs "
                              "grow as N^2, so the full 2620 gives ~43x more "
                              "impostor pairs than 400; start smaller and scale "
                              "up once the pipeline is confirmed working).")
    parser.add_argument("--cfg_scale", type=float, default=None)
    parser.add_argument("--n_steps", type=int, default=None)
    parser.add_argument("--save_embeddings", default=None,
                         help="Optional path to save collected embeddings (.pt) "
                              "for reuse without rerunning the model.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    cfg_scale = args.cfg_scale if args.cfg_scale is not None else cfg.inference.cfg_scale
    n_steps = args.n_steps if args.n_steps is not None else cfg.inference.flow_steps

    print(f"[VerifyEER] Device: {device}  cfg_scale={cfg_scale}  n_steps={n_steps}  "
          f"max_samples={args.max_samples}")

    encoder = SpeakerEncoder(
        model_name=cfg.speaker_encoder.model_name,
        embed_dim=cfg.speaker_encoder.embed_dim,
        freeze=True,
        projection_ckpt=args.projection_ckpt,
    ).to(device)
    encoder.eval()

    mask_model = load_masking(args.mask_ckpt, cfg, device)
    flow_model = load_flow(args.flow_ckpt, cfg, device)

    vc_cfg = OmegaConf.load(args.vocoder_config)
    vocoder = load_hifigan_generator(args.vocoder_ckpt, vc_cfg, device)

    loader, total_n = build_loader(cfg, args.seed, args.batch_size, args.max_samples)

    ref_emb, mix_emb, s2_emb, tgt_emb, speakers = collect_embeddings(
        loader, mask_model, flow_model, encoder, vocoder, cfg_scale, n_steps, device,
        cfg.audio.sample_rate,
    )

    n_uniq_speakers = len(set(speakers))
    print(f"\n[VerifyEER] Collected {len(speakers)} samples, "
          f"{n_uniq_speakers} unique speakers.")

    if args.save_embeddings:
        torch.save({
            "ref_emb": ref_emb, "mix_emb": mix_emb,
            "s2_emb": s2_emb, "tgt_emb": tgt_emb,
            "speakers": speakers,
        }, args.save_embeddings)
        print(f"[VerifyEER] Saved embeddings -> {args.save_embeddings}")

    print("\n" + "=" * 100)
    print(f"  MASK2FLOW-TSE -- CORPUS-WIDE SPEAKER-VERIFICATION EER/AUC  (n={len(speakers)} samples)")
    print("=" * 100)
    print(f"\n  [Verification accuracy]  (genuine = probe[i] vs. its own reference; "
          f"impostor = probe[i] vs. every OTHER speaker reference in the set)")
    report("Mixture (do-nothing baseline)",    mix_emb, ref_emb, speakers)
    report("Stage 2 extraction (this system)", s2_emb,  ref_emb, speakers)
    report("Ground-truth target (ceiling)",    tgt_emb, ref_emb, speakers)
    print("=" * 100)
