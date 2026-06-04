"""
Author: Spatiolyx LLC, https://spatiolyx.ai
Date: May 30, 2026

Shared utilities for UR-JEPA regularizers.

Provides:
    - distributed all-reduce helpers analogous to the ones in LeJEPA,
    - a helper to build a dyadic radius ladder from the current batch,
    - a pairwise squared-distance primitive shared by both loss variants,
    - a thin base class with world-size / distributed plumbing.
"""

import math
import torch
from torch import distributed as dist


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def all_reduce(x, op: str = "SUM"):
    """Functional all-reduce that becomes a no-op when not distributed."""
    if is_dist_avail_and_initialized():
        from torch.distributed.nn import all_reduce as _ar
        from torch.distributed import ReduceOp

        return _ar(x, getattr(ReduceOp, op.upper()))
    return x


def world_size():
    if is_dist_avail_and_initialized():
        return dist.get_world_size()
    return 1


def pairwise_sq_distances(anchors: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """
    Squared Euclidean distances between two sets of points.

    Shape:
        anchors: (A, D)
        points:  (N, D)
        return:  (A, N)

    Implemented as ``||a||^2 + ||z||^2 - 2 a z^T`` (the textbook broadcasted
    formulation) to avoid materializing the ``(A, N, D)`` tensor.
    """
    a2 = anchors.pow(2).sum(-1, keepdim=True)          # (A, 1)
    z2 = points.pow(2).sum(-1, keepdim=True).T         # (1, N)
    return (a2 + z2 - 2.0 * anchors @ points.T).clamp_min_(0.0)


def dyadic_scales(
    Z: torch.Tensor,
    n: int,
    n_scales: int = 5,
    r_max: float | None = None,
    r_min: float | None = None,
    quantile: float = 0.5,
) -> torch.Tensor:
    """
    Build a dyadic ladder of radii ``r_k = r_max * 2^{-k}`` covering the
    embedding scale.

    Args:
        Z: ``(N, D)`` current batch of embeddings.
        n: target intrinsic dimension (controls ``r_min`` default).
        n_scales: number ``K`` of dyadic levels (output length is ``K+1``).
        r_max: if ``None``, uses the ``quantile``-th pairwise-to-centroid
            distance of ``Z``.
        r_min: if ``None``, uses ``r_max * N^{-1/n}`` (the smallest scale at
            which an empirical ``n``-dimensional measure can be resolved).
        quantile: quantile used to set ``r_max`` when it is ``None``.

    Returns:
        ``(K+1,)`` tensor of radii in decreasing order.
    """
    with torch.no_grad():
        if r_max is None:
            centred = Z - Z.mean(0, keepdim=True)
            r_max = centred.norm(dim=-1).quantile(quantile).clamp_min(1e-6).item()
        if r_min is None:
            r_min = r_max * float(Z.size(0)) ** (-1.0 / max(n, 1))
        n_scales = max(int(n_scales), 1)
        log_range = math.log(r_max / max(r_min, 1e-12))
        # evenly spaced in log2 space, K+1 points
        ks = torch.linspace(0.0, log_range / math.log(2.0), n_scales + 1)
        return r_max * torch.pow(2.0, -ks)


class URTest(torch.nn.Module):
    """
    Base class for uniform-rectifiability regularizers. Provides a
    consistent handle on the target intrinsic dimension ``n`` and the
    scale configuration, plus helpers for distributed reductions.

    Subclasses implement ``forward(Z, anchors_idx=None)``.
    """

    def __init__(
        self,
        n: int,
        n_scales: int = 5,
        r_max: float | None = None,
        r_min: float | None = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self.n = int(n)
        self.n_scales = int(n_scales)
        self.r_max = r_max
        self.r_min = r_min
        self.eps = float(eps)

    # --- distributed helpers (mirroring LeJEPA conventions) ---
    @property
    def world_size(self) -> int:
        return world_size()

    def all_reduce_sum(self, x):
        return all_reduce(x, op="SUM")

    # --- scale ladder built from the incoming batch ---
    def _scales(self, Z: torch.Tensor) -> torch.Tensor:
        return dyadic_scales(
            Z,
            n=self.n,
            n_scales=self.n_scales,
            r_max=self.r_max,
            r_min=self.r_min,
        ).to(Z.device, dtype=Z.dtype)

    # --- anchor resolution (indices or None → all points) ---
    def _anchors(self, Z: torch.Tensor, anchors_idx) -> torch.Tensor:
        if anchors_idx is None:
            return Z
        if isinstance(anchors_idx, int):
            idx = torch.randperm(Z.size(0), device=Z.device)[:anchors_idx]
            return Z.index_select(0, idx)
        return Z.index_select(0, anchors_idx)
