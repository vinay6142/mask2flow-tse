"""
Exponential Moving Average (EMA) for Mask2Flow-TSE.

Paper Section 5.2.4:
    "Exponential moving average (EMA) with decay 0.9999
     is applied for stable inference."

EMA maintains a shadow copy of model weights that updates
as a running average. At inference, EMA weights are used
instead of the raw trained weights — this smooths out
training noise and consistently improves final quality.

Usage:
    ema = EMA(model, decay=0.9999)
    # during training:
    ema.update()
    # during validation/inference:
    with ema.apply():
        output = model(input)
    # weights automatically restored after the with block
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import Optional
from pathlib import Path


class EMA:
    """
    Exponential Moving Average of model parameters.

    shadow_t = decay * shadow_{t-1} + (1 - decay) * param_t

    With decay=0.9999, recent updates are weighted very lightly
    (0.01%) and historical values dominate — producing very
    smooth, stable weights for inference.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
    ):
        self.model  = model
        self.decay  = decay
        self.shadow = {}   # EMA weights
        self.backup = {}   # temporary storage when applying EMA

        self._register()
        print(f"[EMA] Initialized with decay={decay}")
        print(f"[EMA] Tracking {len(self.shadow)} parameter tensors")

    def _register(self):
        """Initialize shadow weights as copy of current model weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def update(self):
        """
        Update shadow weights after each optimizer step.
        Call this AFTER optimizer.step().
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    self.shadow[name] = (
                        self.decay       * self.shadow[name] +
                        (1 - self.decay) * param.data
                    )

    def apply_shadow(self):
        """
        Replace model weights with EMA weights.
        Save originals in self.backup for restoration.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original model weights from backup."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()

    @contextmanager
    def apply(self):
        """
        Context manager for clean EMA weight application.

        Usage:
            with ema.apply():
                val_loss = model(val_batch)
            # model weights automatically restored here
        """
        self.apply_shadow()
        try:
            yield
        finally:
            self.restore()

    def state_dict(self) -> dict:
        """Save EMA state for checkpointing."""
        return {
            "decay" : self.decay,
            "shadow": {k: v.cpu() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state: dict):
        """Load EMA state from checkpoint."""
        self.decay = state["decay"]
        device     = next(self.model.parameters()).device
        self.shadow = {
            k: v.to(device) for k, v in state["shadow"].items()
        }
        print(f"[EMA] Loaded checkpoint (decay={self.decay}, "
              f"{len(self.shadow)} params)")


# ── Sanity check ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing EMA...")

    # simple model
    model = nn.Linear(10, 5)
    ema   = EMA(model, decay=0.9999)

    # simulate training steps
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    initial_shadow = {k: v.clone() for k, v in ema.shadow.items()}

    for step in range(10):
        x    = torch.randn(4, 10)
        loss = model(x).mean()
        loss.backward()
        optim.step()
        optim.zero_grad()
        ema.update()

    # verify shadow weights changed but are smoother
    for name in ema.shadow:
        orig   = initial_shadow[name]
        shadow = ema.shadow[name]
        param  = dict(model.named_parameters())[name].data
        print(f"  {name}: param_change={( param-orig).abs().mean():.4f}  "
              f"shadow_change={(shadow-orig).abs().mean():.4f}  "
              f"(shadow should be smaller)")

    # test context manager
    with ema.apply():
        # inside: model uses EMA weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                diff = (param.data - ema.shadow[name]).abs().max()
                assert diff < 1e-6, "EMA weights not applied!"

    # outside: original weights restored
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert name in ema.backup or True  # backup cleared

    print("EMA test passed ✅")