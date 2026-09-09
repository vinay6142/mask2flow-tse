"""
Real-world inference: mixture.wav + reference.wav -> extracted_speaker.wav

No synthetic mixing, no ground truth -- takes a user-supplied mixture
recording directly. Replicates the exact preprocessing chain used in
data/librispeech.py __getitem__ (mel extraction with NO mixture
normalization, reference via raw waveform for the speaker encoder),
so results stay consistent with everything already validated in eval/.

Chunked overlap-add (NEW): the model was trained on fixed
cfg.audio.segment_length (10s) windows, and the ORIGINAL version of this
script silently pad_or_trim()-ed every input to exactly that length --
audio longer than 10s got truncated (everything past 10s discarded, no
warning), and audio shorter than 10s got padded with silence that was
never trimmed back out of the saved output (a 3s input produced a 10s
output file, the last ~7s being vocoded silence-padding). Neither
behavior is acceptable for real-world clips of arbitrary length, which
is the actual deployment target of this project. Fixed by:
  - short clips: still a single model pass, but the OUTPUT mel is
    trimmed back to the real input length before vocoding.
  - long clips: split into overlapping segment_length-second chunks,
    run the model on each independently, and reassemble with a linear
    cross-fade in the overlap region (standard overlap-add) so there is
    no discontinuity / click at chunk boundaries.

Also fixed: a pre-existing NameError bug -- the old version referenced
a bare global args inside extract_target_speaker() for vocoder
selection, so calling this function directly (not via __main__) would
crash. vocoder / vocoder_ckpt / vocoder_config are now real parameters.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import soundfile as sf
from omegaconf import OmegaConf

from data.augment import load_audio
from data.mel import MelSpectrogramExtractor, pad_or_trim, segment_waveform
from eval.results_stage2 import load_masking, load_flow
from models.speaker_encoder import SpeakerEncoder
from inference.vocoder import mel_to_audio_griffinlim, load_hifigan_generator, mel_to_audio_hifigan


def _chunk_starts(total_frames, chunk_frames, overlap_frames):
    """
    Frame-index start offsets covering [0, total_frames) with
    chunk_frames-wide, overlap_frames-overlapping windows. The final
    chunk is shifted left (not shortened) so every chunk stays exactly
    chunk_frames wide -- simplest way to keep every chunk within the
    length the model was trained on, without special-casing a short
    last chunk.
    """
    if total_frames <= chunk_frames:
        return [0]
    hop = max(chunk_frames - overlap_frames, 1)
    starts = list(range(0, total_frames - chunk_frames + 1, hop))
    last_start = total_frames - chunk_frames
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _overlap_add(chunks, starts, chunk_frames, total_frames, overlap_frames, device):
    """
    Reassemble overlapping (n_mels, chunk_frames) mel chunks into one
    (n_mels, total_frames) mel using linear cross-fade in overlap regions
    -- avoids the audible discontinuity a hard concatenation would cause
    at every chunk boundary. Weighted-sum + normalize (rather than
    relying on the ramps summing to exactly 1) so it stays correct even
    at edge cases (very short last-chunk shift, chunk_frames close to
    2*overlap_frames, etc).
    """
    n_mels = chunks[0].shape[0]
    out    = torch.zeros(n_mels, total_frames, device=device)
    weight = torch.zeros(1,      total_frames, device=device)

    for chunk, start in zip(chunks, starts):
        end = start + chunk_frames
        w = torch.ones(chunk_frames, device=device)
        if start > 0:
            ramp_len = min(overlap_frames, chunk_frames)
            w[:ramp_len] = torch.linspace(0.0, 1.0, ramp_len, device=device)
        if end < total_frames:
            ramp_len = min(overlap_frames, chunk_frames)
            w[-ramp_len:] = torch.minimum(w[-ramp_len:], torch.linspace(1.0, 0.0, ramp_len, device=device))

        out[:, start:end]    += chunk * w.unsqueeze(0)
        weight[:, start:end] += w.unsqueeze(0)

    return out / weight.clamp(min=1e-8)


def extract_target_speaker(
    mixture_path, reference_path, cfg, device,
    mask_ckpt="checkpoints_v2/masking/mask_best.pt",
    flow_ckpt="checkpoints_v2/flow/flow_best.pt",
    out_path="outputs/extracted.wav",
    vocoder="hifigan",
    vocoder_ckpt="checkpoints_vocoder/vocoder_best.pt",
    vocoder_config="configs/vocoder.yaml",
    overlap_sec=2.0,
):
    sr = cfg.audio.sample_rate

    mel_extractor = MelSpectrogramExtractor(
        sample_rate=sr, n_mels=cfg.mel.n_mels, n_fft=cfg.mel.n_fft,
        hop_length=cfg.mel.hop_length, win_length=cfg.mel.win_length,
        f_min=cfg.mel.f_min, f_max=cfg.mel.f_max, log_offset=cfg.mel.log_offset,
    ).to(device)

    encoder    = SpeakerEncoder(cfg.speaker_encoder.model_name, cfg.speaker_encoder.embed_dim, freeze=True).to(device).eval()
    mask_model = load_masking(mask_ckpt, cfg, device)
    flow_model = load_flow(flow_ckpt, cfg, device)

    mixture_wav   = load_audio(mixture_path, sr).to(device)
    reference_wav = load_audio(reference_path, sr).to(device)

    ref_seg_len   = getattr(cfg.audio, "reference_length", 3.0)
    reference_wav = segment_waveform(reference_wav, sr, ref_seg_len, random_start=False)

    with torch.no_grad():
        d_vec = encoder(reference_wav.unsqueeze(0))

    mixture_mel_full = mel_extractor(mixture_wav)
    n_mels, total_frames = mixture_mel_full.shape

    dummy = torch.zeros(int(cfg.audio.segment_length * sr), device=device)
    chunk_frames = mel_extractor(dummy).shape[-1]
    overlap_frames = min(int(overlap_sec * sr / cfg.mel.hop_length), chunk_frames // 2)

    starts = _chunk_starts(total_frames, chunk_frames, overlap_frames)
    if len(starts) > 1:
        print(f"[infer] {total_frames} frames ({total_frames*cfg.mel.hop_length/sr:.1f}s) > "
              f"model {chunk_frames}-frame ({cfg.audio.segment_length:.0f}s) training window -- "
              f"processing as {len(starts)} overlapping chunks ({overlap_sec:.1f}s overlap).")

    stage2_chunks = []
    with torch.no_grad():
        for start in starts:
            end = start + chunk_frames
            chunk = mixture_mel_full[:, start:end]
            if chunk.shape[-1] < chunk_frames:
                chunk = pad_or_trim(chunk, chunk_frames, pad_value=-11.5)
            chunk = chunk.unsqueeze(0)

            stage1_out, _ = mask_model(chunk, d_vec)
            stage2_out = flow_model.inference(
                stage1_out, d_vec,
                cfg_scale=cfg.inference.cfg_scale,
                n_steps=cfg.inference.flow_steps,
            )
            stage2_chunks.append(stage2_out[0])

    if len(starts) == 1:
        final_mel = stage2_chunks[0][:, :total_frames]
    else:
        final_mel = _overlap_add(
            stage2_chunks, starts, chunk_frames, total_frames, overlap_frames, device
        )

    if vocoder == "hifigan":
        vc_cfg = OmegaConf.load(vocoder_config)
        generator = load_hifigan_generator(vocoder_ckpt, vc_cfg, device)
        waveform = mel_to_audio_hifigan(final_mel, generator)
    else:
        waveform = mel_to_audio_griffinlim(final_mel, cfg)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sf.write(out_path, waveform, sr)
    print(f"Saved extracted speech -> {out_path}  ({total_frames*cfg.mel.hop_length/sr:.1f}s)")
    return final_mel


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixture",   required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--config",    default="configs/default_v2.yaml")
    parser.add_argument("--mask_ckpt", default="checkpoints_v2/masking/mask_best.pt")
    parser.add_argument("--flow_ckpt", default="checkpoints_v2/flow/flow_best.pt")
    parser.add_argument("--out",       default="outputs/extracted.wav")
    parser.add_argument("--vocoder", choices=["griffinlim", "hifigan"], default="hifigan")
    parser.add_argument("--vocoder_ckpt", default="checkpoints_vocoder/vocoder_best.pt")
    parser.add_argument("--vocoder_config", default="configs/vocoder.yaml")
    parser.add_argument("--overlap_sec", type=float, default=2.0,
                         help="Cross-fade overlap in seconds between consecutive "
                              "chunks, used only when input audio is longer than "
                              "segment_length (default 10s).")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extract_target_speaker(
        args.mixture, args.reference, cfg, device,
        args.mask_ckpt, args.flow_ckpt, args.out,
        vocoder=args.vocoder, vocoder_ckpt=args.vocoder_ckpt,
        vocoder_config=args.vocoder_config, overlap_sec=args.overlap_sec,
    )
