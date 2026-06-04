"""
Author: Spatiolyx LLC, https://spatiolyx.ai
Date: May 30, 2026

URJEPA: drop-in replacement for LeJEPA's SIGReg regularization block
that enforces uniform n-rectifiability instead of full-D Gaussianity.

Given a batch of embeddings ``proj`` shaped ``(V, N, D)`` (V views,
N items, D channels) or ``(N, D)``, URJEPA computes

    loss = lambda_ur * L_UR(Z) + lambda_ad * L_AD(Z)

where ``Z`` is the pooled-over-views batch of embeddings. ``L_UR`` is
either :class:`BetaNumber` (default, recommended for large ``D``) or
:class:`CGLT` (Gaussian-kernel density-difference loss).  ``L_AD`` is
the :class:`ADRegularity` anchor, enabled for either variant when
``lambda_ad > 0``. For the β-number variant, AD is the canonical fix
for its well-known structural collapse mode (point-mass cloud trivially
minimizes β² = 0). For CGLT, AD is a refinement that improves the peak
(CGLT already prevents collapse via ``Δ log θ = n·log 2`` at point-mass).

The predictive / invariance term of JEPA remains the responsibility of
the training script (see ``scripts/pretrain.py``), exactly as in LeJEPA
where ``SIGReg`` is combined with ``inv_loss`` in the main loop.
"""

import torch

from .beta_number import BetaNumber
from .cglt import CGLT, CGLTDeriv, CGLTDerivRaw, ADRegularity


class URJEPA(torch.nn.Module):
    r"""
    Combined uniform-rectifiability regularizer.

    Args:
        n (int): target intrinsic dimension.
        variant (str, optional): ``"beta"`` for the β-number loss
            (default) or ``"cglt"`` for the Gaussian-kernel CGLT loss +
            AD anchor.
        n_scales (int, optional): number of dyadic levels. Default ``5``.
        r_max, r_min (float, optional): scale-ladder endpoints.
        anchors (int or None, optional): if an int, subsample this many
            anchors uniformly from each batch; if ``None``, use every
            point.  ``None`` matches LeJEPA's convention.
        lambda_ad (float, optional): weight on the AD anchor. When
            ``> 0`` the AD anchor is included for either variant
            (β or CGLT). Default ``0.0`` (no AD) -- callers must opt
            in explicitly; in practice ``pretrain.py`` reads
            ``cfg.lambda_ad`` per-experiment (typically 0.1 for CGLT).
        eps (float, optional): numerical safety constant.

    Shape:
        - Input ``proj``: ``(V, N, D)`` or ``(N, D)``.
        - Output:         scalar loss tensor.

    Example (LeJEPA-style training step):

        >>> backbone = build_encoder()
        >>> ur_loss  = URJEPA(n=8, variant="beta", anchors=128)
        >>> emb, proj = backbone(views)                # proj: (V, N, D)
        >>> inv_loss = (proj.mean(0) - proj).square().mean()
        >>> ur = ur_loss(proj)
        >>> total = ur * cfg.lamb + inv_loss * (1 - cfg.lamb)
    """

    def __init__(
        self,
        n: int,
        variant: str = "beta",
        n_scales: int = 5,
        r_max: float | None = None,
        r_min: float | None = None,
        anchors: int | None = None,
        lambda_ad: float = 0.0,
        eps: float = 1e-8,
        **variant_kwargs,
    ):
        super().__init__()
        if variant not in ("beta", "cglt", "cglt_deriv", "cglt_deriv_raw"):
            raise ValueError(
                f"variant must be 'beta', 'cglt', 'cglt_deriv', or "
                f"'cglt_deriv_raw', got {variant}"
            )
        self.variant = variant
        self.anchors = anchors
        self.lambda_ad = float(lambda_ad)

        common = dict(
            n=n, n_scales=n_scales, r_max=r_max, r_min=r_min, eps=eps
        )
        if variant == "beta":
            self.ur = BetaNumber(**common, **variant_kwargs)
        elif variant == "cglt":
            self.ur = CGLT(**common, **variant_kwargs)
        elif variant == "cglt_deriv":
            self.ur = CGLTDeriv(**common, **variant_kwargs)
        else:  # cglt_deriv_raw
            self.ur = CGLTDerivRaw(**common, **variant_kwargs)
        # AD anchor: enabled for any variant when lambda_ad > 0.  The
        # Eq.~(1.5) characterization (cglt_deriv) presupposes n-AD
        # regularity exactly as Eq.~(1.4) does (cglt), so the same
        # anchor applies.
        self.ad = ADRegularity(**common) if self.lambda_ad > 0 else None

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        # pool views: accept (V, N, D) or (N, D)
        if proj.ndim == 3:
            Z = proj.flatten(0, 1)
        elif proj.ndim == 2:
            Z = proj
        else:
            raise ValueError(f"proj must be (V, N, D) or (N, D); got {tuple(proj.shape)}")

        loss = self.ur(Z, anchors_idx=self.anchors)
        if self.ad is not None and self.lambda_ad != 0.0:
            loss = loss + self.lambda_ad * self.ad(Z, anchors_idx=self.anchors)
        return loss
