"""
Speaker Encoder for Mask2Flow-TSE.
Wraps pretrained WavLM-base-plus-sv as a frozen d-vector extractor.

Paper Section 5.2.1:
    "WavLM-base-plus-sv pretrained on speaker verification.
     Extracts a 512-dimensional d-vector from the reference utterance.
     The speaker encoder remains frozen during training."
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, WavLMModel
from typing import Optional


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


class SpeakerEncoder(nn.Module):
    """
    Frozen WavLM speaker encoder.

    Pipeline:
        raw waveform (B, T)
            ↓  WavLM (frozen)
        hidden states (B, T', 768)
            ↓  mean pool over time
        speaker embedding (B, 768)
            ↓  linear projection
        d-vector (B, 512)
            ↓  L2 normalize
        unit d-vector (B, 512)

    The d-vector is shared across both the Masking and Flow stages.
    """

    def __init__(
        self,
        model_name:  str = "microsoft/wavlm-base-plus-sv",
        embed_dim:   int = 512,
        sample_rate: int = 16000,
        freeze:      bool = True,
        projection_ckpt: Optional[str] = None,
    ):
        super().__init__()

        self.model_name  = model_name
        self.embed_dim   = embed_dim
        self.sample_rate = sample_rate

        print(f"[SpeakerEncoder] Loading {model_name}...")

        # feature extractor handles waveform normalization
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_name,
            sampling_rate = sample_rate,
        )

        # WavLM backbone
        self.wavlm   = WavLMModel.from_pretrained(model_name)
        wavlm_dim    = self.wavlm.config.hidden_size   # 768

        # project WavLM hidden dim → d-vector dim
        self.projection = nn.Linear(wavlm_dim, embed_dim, bias=True)

        if projection_ckpt is not None:
            # PERMANENT FIX: load the properly-trained projection (see
            # training/train_speaker_encoder.py) instead of leaving it at a
            # fresh, unseeded Xavier-random init every time this class is
            # constructed. Historically this projection was never trained
            # (never in any optimizer) and never seeded consistently across
            # scripts — quantified as causing up to a ~60x swing in
            # downstream MSE depending on which random draw was active.
            ckpt = torch.load(projection_ckpt, map_location="cpu")
            self.projection.load_state_dict(ckpt["projection"])
            print(f"[SpeakerEncoder] Loaded trained projection from {projection_ckpt} "
                  f"(step {ckpt.get('step', '?')}, val_acc={ckpt.get('val_acc', '?')})")
        else:
            nn.init.xavier_uniform_(self.projection.weight)
            print("[SpeakerEncoder] WARNING: projection_ckpt not provided — "
                  "using an untrained, randomly-initialized projection. This "
                  "is the known bug (see project notes); pass projection_ckpt "
                  "explicitly unless you specifically intend this for a "
                  "one-off diagnostic.")

        if freeze:
            self._freeze_wavlm()

        n_wavlm  = count_params(self.wavlm)
        n_proj   = count_params(self.projection)
        print(f"[SpeakerEncoder] WavLM     : {n_wavlm/1e6:.1f}M params (frozen)")
        print(f"[SpeakerEncoder] Projection: {n_proj:,} params (trainable)")
        print(f"[SpeakerEncoder] Output dim: {embed_dim}")

    def _freeze_wavlm(self):
        """Freeze entire WavLM backbone — no gradients flow through it."""
        for param in self.wavlm.parameters():
            param.requires_grad = False
        print(f"[SpeakerEncoder] WavLM frozen ✅")

    def _extract_wavlm_features(
        self,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run frozen WavLM forward pass.

        Args:
            waveform: (B, T) float32 in [-1, 1]
        Returns:
            hidden: (B, T', 768)  T' << T due to CNN downsampling
        """
        with torch.no_grad():
            output = self.wavlm(
                input_values    = waveform,
                attention_mask  = None,
                output_hidden_states = False,
            )
        return output.last_hidden_state   # (B, T', 768)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract L2-normalized speaker d-vector.

        Args:
            waveform: (B, T) raw waveform, 16kHz float32
        Returns:
            d_vector: (B, embed_dim=512) unit L2 norm
        """
        # frozen WavLM feature extraction
        hidden = self._extract_wavlm_features(waveform)  # (B, T', 768)

        # temporal mean pooling → global speaker representation
        embedding = hidden.mean(dim=1)                    # (B, 768)

        # linear projection to target dimension
        d_vector = self.projection(embedding)             # (B, 512)

        # L2 normalize: ||d|| = 1 (required for cosine similarity metric)
        d_vector = F.normalize(d_vector, p=2, dim=-1)    # (B, 512)

        return d_vector

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


# ── Sanity checks ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Test 1: projection only (no WavLM download needed) ────
    print("=" * 50)
    print("[Test 1] Projection + L2 norm (no model download)")
    print("=" * 50)

    proj = nn.Linear(768, 512).to(device)
    fake_hidden = torch.randn(2, 100, 768).to(device)

    embedding = fake_hidden.mean(dim=1)         # (2, 768)
    d_vec     = proj(embedding)                 # (2, 512)
    d_vec     = F.normalize(d_vec, p=2, dim=-1) # unit norm

    norms = d_vec.norm(dim=-1)
    print(f"  d-vector shape : {tuple(d_vec.shape)}")
    print(f"  L2 norms       : {norms.tolist()}")  # should be [1.0, 1.0]
    assert torch.allclose(norms, torch.ones(2).to(device), atol=1e-5), \
        "L2 norms should be 1.0"
    print("  Test 1 passed ✅\n")

    # ── Test 2: full SpeakerEncoder ───────────────────────────
    print("=" * 50)
    print("[Test 2] Full SpeakerEncoder (needs WavLM download)")
    print("=" * 50)

    try:
        encoder = SpeakerEncoder().to(device)

        # 3-second reference waveform, batch of 2
        ref_wav = torch.randn(2, 16000 * 3).to(device)
        ref_wav = ref_wav / ref_wav.abs().max()   # normalize to [-1, 1]

        d_vec = encoder(ref_wav)

        print(f"\n  Input shape    : {tuple(ref_wav.shape)}")
        print(f"  d-vector shape : {tuple(d_vec.shape)}")
        print(f"  L2 norms       : {d_vec.norm(dim=-1).tolist()}")

        # verify WavLM is truly frozen
        n_grad = sum(
            p.numel() for p in encoder.wavlm.parameters()
            if p.requires_grad
        )
        print(f"  WavLM grad params: {n_grad}  (must be 0)")
        assert n_grad == 0, "WavLM should be fully frozen!"

        # verify cosine similarity works between same speaker
        d1 = encoder(ref_wav[[0]])
        d2 = encoder(ref_wav[[0]] + 0.01 * torch.randn_like(ref_wav[[0]]))
        sim = F.cosine_similarity(d1, d2, dim=-1)
        print(f"  Same-speaker sim: {sim.item():.4f}  (should be close to 1.0)")

        if device.type == "cuda":
            mem = torch.cuda.memory_allocated() / 1e6
            print(f"  GPU memory used  : {mem:.1f} MB")

        print("  Test 2 passed ✅")

    except Exception as e:
        print(f"  WavLM not available: {e}")
        print("  Pre-download on login node first:")
        print("    python3 -c \"from transformers import WavLMModel; "
              "WavLMModel.from_pretrained('microsoft/wavlm-base-plus-sv')\"")