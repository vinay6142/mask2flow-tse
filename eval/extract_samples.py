"""
Extract and save mixture/Stage1/Stage2/target/reference audio for specific
dataset indices, so they can be transferred off the cluster and listened to.

Reuses the exact same deterministic dataset construction as eval/full_eval.py
(seed+idx keyed __getitem__, same LibriSpeechTSEDataset(seed=...) convention),
so --indices 713 here is the SAME sample as idx=713 in full_eval.py's output —
no need to re-run the full 800/2620-sample pass just to inspect a few.

Usage:
  python3 eval/extract_samples.py \
      --mask_ckpt checkpoints_v2/masking/mask_best.pt \
      --flow_ckpt checkpoints_v2/flow/flow_best.pt \
      --projection_ckpt checkpoints_speaker_encoder/projection_latest.pt \
      --indices 713,757,582,445,634,243,507,781,685,784 \
      --output_dir outputs/results/listening

Then, from YOUR LOCAL machine (not the cluster), transfer the folder:
  scp -r <user>@<cluster-host>:<path-to-repo>/outputs/results/listening .
(or use VS Code's Remote-SSH file explorer to drag it out / click-to-preview)
"""
import os
import sys
import argparse

import torch
import soundfile as sf
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.librispeech import LibriSpeechTSEDataset
from inference.vocoder import load_hifigan_generator, mel_to_audio_hifigan
from models.speaker_encoder import SpeakerEncoder

from eval.results_stage2 import load_masking, load_flow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default_v2.yaml")
    parser.add_argument("--mask_ckpt", required=True)
    parser.add_argument("--flow_ckpt", required=True)
    parser.add_argument("--projection_ckpt", default="checkpoints_speaker_encoder/projection_latest.pt")
    parser.add_argument("--vocoder_ckpt", default="checkpoints_vocoder/vocoder_best.pt")
    parser.add_argument("--vocoder_config", default="configs/vocoder.yaml")
    parser.add_argument("--seed", type=int, default=42,
                         help="Must match the seed used for the eval run that produced "
                              "these indices (full_eval.py's default is 42).")
    parser.add_argument("--cfg_scale", type=float, default=None)
    parser.add_argument("--n_steps", type=int, default=None)
    parser.add_argument("--cfg_warmup_steps", type=int, default=0)
    parser.add_argument("--indices", required=True,
                         help="Comma-separated dataset indices to extract, e.g. 713,757,582")
    parser.add_argument("--output_dir", default="outputs/results/listening")
    args = parser.parse_args()

    indices = [int(x) for x in args.indices.split(",") if x.strip() != ""]
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg_scale = args.cfg_scale if args.cfg_scale is not None else cfg.inference.cfg_scale
    n_steps   = args.n_steps   if args.n_steps   is not None else cfg.inference.flow_steps

    print(f"[Extract] Device: {device}  cfg_scale={cfg_scale}  n_steps={n_steps}")
    print(f"[Extract] Indices to extract: {indices}")

    encoder = SpeakerEncoder(
        model_name=cfg.speaker_encoder.model_name,
        embed_dim=cfg.speaker_encoder.embed_dim,
        freeze=True,
        projection_ckpt=args.projection_ckpt,
    ).to(device)
    encoder.eval()

    mask_model = load_masking(args.mask_ckpt, cfg, device)
    flow_model = load_flow(args.flow_ckpt, cfg, device)

    vc_cfg  = OmegaConf.load(args.vocoder_config)
    vocoder = load_hifigan_generator(args.vocoder_ckpt, vc_cfg, device)

    # NOTE: seed=... makes __getitem__ deterministic per-index (seed+idx keyed,
    # per data/librispeech.py) — same seed as the eval run means dataset[idx]
    # here is exactly the same target/interferer/mixture/SNR draw as idx=idx
    # in that run's output, regardless of batch_size or DataLoader shuffling.
    dataset = LibriSpeechTSEDataset(
        librispeech_root=cfg.data.librispeech_path,
        splits=["test-clean"],
        cfg=cfg,
        is_train=False,
        seed=args.seed,
    )
    print(f"[Extract] Dataset has {len(dataset)} samples total (seed={args.seed})")

    sr = cfg.audio.sample_rate

    with torch.no_grad():
        for idx in indices:
            if idx >= len(dataset):
                print(f"[Extract] WARNING: idx={idx} out of range (dataset has "
                      f"{len(dataset)} samples) — skipping")
                continue

            sample = dataset[idx]
            mixture      = sample["mixture_mel"].unsqueeze(0).to(device)
            target       = sample["target_mel"].unsqueeze(0).to(device)
            ref_wav      = sample["reference_wav"].unsqueeze(0).to(device)
            valid_frames = sample["target_valid_frames"].unsqueeze(0).to(device)

            d_vec = encoder(ref_wav)
            stage1_out, _ = mask_model(mixture, d_vec)
            stage2_out = flow_model.inference(
                stage1_out, d_vec, cfg_scale=cfg_scale, n_steps=n_steps,
                cfg_warmup_steps=args.cfg_warmup_steps,
            )

            vf = max(int(valid_frames[0].item()), 1)

            mix_wav = mel_to_audio_hifigan(mixture[0, :, :vf], vocoder)
            s1_wav  = mel_to_audio_hifigan(stage1_out[0, :, :vf], vocoder)
            s2_wav  = mel_to_audio_hifigan(stage2_out[0, :, :vf], vocoder)
            tgt_wav = mel_to_audio_hifigan(target[0, :, :vf], vocoder)
            ref_wav_np = ref_wav[0].detach().cpu().numpy()

            prefix = os.path.join(args.output_dir, f"idx{idx}")
            sf.write(f"{prefix}_1_mixture.wav",   mix_wav,     sr)
            sf.write(f"{prefix}_2_stage1.wav",    s1_wav,      sr)
            sf.write(f"{prefix}_3_stage2.wav",    s2_wav,      sr)
            sf.write(f"{prefix}_4_target.wav",    tgt_wav,     sr)
            sf.write(f"{prefix}_5_reference.wav", ref_wav_np,  sr)

            snr = sample["snr_db"]
            snr_str = f"{snr:.2f}dB" if snr == snr else "N/A"  # snr==snr is False for NaN
            print(f"[Extract] idx={idx:<5} snr={snr_str:<8} cond={sample['condition']:<10} "
                  f"-> saved 5 files as {prefix}_*.wav")

    print(f"\n[Extract] Done. Files saved under {args.output_dir}/")
    print(f"Each idx has 5 files: 1_mixture (input, both speakers), "
          f"2_stage1 (after masking), 3_stage2 (final output — the one to judge), "
          f"4_target (ground truth, what it should sound like), "
          f"5_reference (the enrollment clip used to pick the target speaker).")
    print(f"\nTransfer to your local machine (run this on YOUR machine, not the cluster):")
    print(f"  scp -r <your-username>@<cluster-hostname>:{os.path.abspath(args.output_dir)} .")


if __name__ == "__main__":
    main()