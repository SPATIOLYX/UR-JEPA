"""
Packaging: Spatiolyx LLC, https://spatiolyx.ai
Date: May 30, 2026

SIGReg baseline for head-to-head comparison with URJEPA.

Vendored from LeJEPA's MINIMAL.md (galilai-group/lejepa) under MIT, with
two small changes vs the original:

  1. ``device="cuda"`` is removed from the random-projection matrix; we
     create it on the input's device so the module is testable on CPU
     and DDP-portable.
  2. ``forward`` accepts ``(V, N, D)`` *or* ``(N, D)`` matching the
     :class:`ur_jepa.URJEPA` interface, so ``pretrain.py`` can
     swap regularizers without touching the training loop.

Reference: R. Balestriero, Y. LeCun, *LeJEPA: Provable and Scalable
Self-Supervised Learning Without the Heuristics*, arXiv:2511.08544, 2025.
"""

import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularizer (LeJEPA, MINIMAL.md)."""

    def __init__(self, knots: int = 17, num_slices: int = 256):
        super().__init__()
        self.num_slices = int(num_slices)
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        # Accept (V, N, D) or (N, D) and pool views, mirroring URJEPA.
        if proj.ndim == 2:
            proj = proj.unsqueeze(0)  # (1, N, D)
        elif proj.ndim != 3:
            raise ValueError(
                f"proj must be (V, N, D) or (N, D); got {tuple(proj.shape)}"
            )

        D = proj.size(-1)
        A = torch.randn(D, self.num_slices, device=proj.device, dtype=proj.dtype)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t      # (V, N, num_slices, knots)
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()
