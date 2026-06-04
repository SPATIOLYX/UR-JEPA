"""
Author: Spatiolyx LLC, https://spatiolyx.ai
Date: May 30, 2026

Gaussian-kernel Chousionis--Garnett--Le--Tolsa (CGLT) density-difference
regularizer and Ahlfors--David (AD) anchor for UR-JEPA.

Implements the Carleson loss built from the *log* dyadic increment of the
scale-normalized smoothed density

    θ_r(x) = (1 / (N · r^n)) Σ_j exp(-||z_j - x||^2 / (2 r^2)),

    Δ_r(x) = log θ_{2r}(x) - log θ_r(x),

    L_CGLT(θ) = log 2 · mean_{x∈A} Σ_k |Δ_{r_k}(x)|^2 .

Working with ``log θ`` rather than ``θ`` makes the loss exactly
scale-invariant: on a uniformly n-rectifiable, n-AD regular measure
θ_r(x) ≍ const in ``r`` so every log-increment vanishes.  By Theorem 1.1
of Chousionis, Garnett, Le, Tolsa (2014), a Carleson bound on this
dyadic square-function characterizes UR of an n-AD regular measure; we
pair ``CGLT`` with the ``ADRegularity`` anchor to pin θ_r to a common
level across anchors and so recover the AD side of the hypothesis.

CAVEAT (AD-regularity): the CGLT characterization, like David--Semmes /
Pajot, *assumes* n-AD-regularity -- which a finite embedding measure does
not satisfy. The ``ADRegularity`` anchor here *imposes* that hypothesis
rather than verifying it. Read the CGLT losses as the AD-regular branch
of the theory; the general-measure (non-AD) footing is via Jones' L^2
quantities and is exercised on the β family (:class:`BetaNumber`).

The module is API-compatible with :class:`BetaNumber` and with LeJEPA's
``SIGReg`` -- both expose a ``forward(Z, anchors_idx=None) -> scalar``.
"""

import math
import torch

from .base import URTest, pairwise_sq_distances


class CGLT(URTest):
    r"""
    Gaussian-smoothed CGLT square-function regularizer.

    Args:
        n (int): target intrinsic dimension.
        n_scales (int, optional): number of dyadic levels. Default ``5``.
        r_max, r_min (float, optional): scale-ladder endpoints.  If ``None``
            they are inferred from each batch.
        eps (float, optional): small constant for numerical safety.

    Shape:
        - Input ``Z``:             ``(N, D)``.
        - Input ``anchors_idx``:   ``None`` | ``int`` | ``LongTensor``.
        - Output:                  scalar loss tensor.

    Notes:
        The ``(2π)^{n/2}`` constant in θ_r(x) is chosen so that on an
        ideal n-AD regular set of locally unit density one has
        ``E θ_r(x) ≈ 1``; this makes the AD anchor (:class:`ADRegularity`)
        a direct attempt to pin ``log θ_r(x)`` to ``0``.
    """

    def __init__(
        self,
        n: int,
        n_scales: int = 5,
        r_max: float | None = None,
        r_min: float | None = None,
        eps: float = 1e-8,
    ):
        super().__init__(n=n, n_scales=n_scales, r_max=r_max, r_min=r_min, eps=eps)

    # ------------------------------------------------------------------
    # log θ_r(x) at each scale
    # ------------------------------------------------------------------
    def _log_theta_at_scales(
        self,
        d2: torch.Tensor,       # (Na, N)
        r_ladder: torch.Tensor, # (K+1,)
        N: int,
    ) -> torch.Tensor:
        """Return ``(Na, K+1)`` tensor of ``log θ_r(x)`` at each anchor × scale.

        Computed in a numerically stable way via ``logsumexp``:

            log θ_r(x) = logsumexp_j(-||z_j-x||^2 / (2 r^2))
                         - n · log r  -  log N .
        """
        log_N = math.log(max(N, 1))
        log_thetas = []
        for r in r_ladder:
            log_w_sum = torch.logsumexp((-0.5 / (r * r)) * d2, dim=-1)   # (Na,)
            lt = log_w_sum - self.n * torch.log(r.clamp_min(self.eps)) - log_N
            log_thetas.append(lt)
        return torch.stack(log_thetas, dim=-1)                            # (Na, K+1)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, Z: torch.Tensor, anchors_idx=None) -> torch.Tensor:
        if Z.ndim != 2:
            raise ValueError(f"Z must be (N, D); got shape {tuple(Z.shape)}")
        N = Z.size(0)
        A = self._anchors(Z, anchors_idx)                   # (Na, D)
        r_ladder = self._scales(Z)                          # (K+1,)  decreasing
        d2 = pairwise_sq_distances(A, Z)                    # (Na, N)

        log_theta = self._log_theta_at_scales(d2, r_ladder, N)  # (Na, K+1)
        # r_ladder is decreasing, so (log θ_r - log θ_{2r}) at level k
        # is ``log_theta[:, k] - log_theta[:, k-1]``.
        delta = log_theta[:, 1:] - log_theta[:, :-1]            # (Na, K)

        loss = delta.pow(2).mean(0).sum()                       # Σ over dyadic levels
        # Riemann-sum step in log-radius; equals log 2 only when the
        # ladder is exactly dyadic. dyadic_scales builds a log-uniform
        # ladder with K = n_scales gaps, so the per-level weight is
        # log(r_max/r_min) / n_scales.
        log_step = torch.log(r_ladder[0] / r_ladder[-1].clamp_min(self.eps)) / max(self.n_scales, 1)
        return loss * log_step


class CGLTDeriv(URTest):
    r"""
    Gaussian-smoothed UR regularizer using the *scale-derivative*
    characterization of uniform n-rectifiability
    (Chousionis--Garnett--Le--Tolsa 2014, Theorem 1.2 (c) / Eq.~(1.5)),
    in log form for numerical stability.

    The CGLT paper proves that, for an n-AD-regular measure mu, uniform
    n-rectifiability is equivalent to a Carleson bound on the *kernel
    t-derivative*

        ∂_φ(x, t) := t · ∂_t φ_t(x),
        Δ̃_{μ,φ}(x, t) := ∫ ∂_φ(y - x, t) dμ(y).

    For φ(x) = exp(-|x|²/2), a direct computation gives

        ∂_φ(x, t) = φ_t(x) · (|x|² / t² - n),

    and the per-anchor *log* t-derivative of the smoothed density is

        t · ∂_t log θ_t(x)
            = ⟨ ||z_j - x||² / t² - n ⟩_{w(x,t)},

    where ⟨·⟩_w is the Gaussian-kernel-weighted average
    w_j(x, t) ∝ exp(-||z_j - x||² / (2 t²)).  On an n-AD-regular,
    n-rectifiable cloud the kernel-weighted second moment of distance
    scales as n t², so the per-anchor quantity vanishes pointwise.  We
    work with the log derivative rather than the raw integral Δ̃ for
    the same reason :class:`CGLT` uses log Δ rather than Δ: the log
    version is scale-invariant in the kernel normalization and does
    not require a target density level.

    The resulting Carleson loss

        L_CGLT^∂(θ) = log(r_max / r_min)/n_scales
                       · Σ_k  mean_{x ∈ A} |t · ∂_t log θ_{r_k}(x)|²

    is a Riemann sum for ∫ |t · ∂_t log θ_t|² dt/t.

    Unlike :class:`CGLT`, this variant computes one per-anchor
    quantity per scale rather than a difference between adjacent
    scales, so its gradient signal is not cancelled by cross-scale
    subtractions and it does not need to hold two adjacent
    log-densities in memory simultaneously.  Pair with
    :class:`ADRegularity` when ``lambda_ad > 0`` to recover the
    n-AD-regularity side of Theorem~1.2's hypothesis.

    Args:
        n (int): target intrinsic dimension.
        n_scales, r_max, r_min, eps: same semantics as :class:`CGLT`.

    Shape:
        - Input ``Z``:             ``(N, D)``.
        - Input ``anchors_idx``:   ``None`` | ``int`` | ``LongTensor``.
        - Output:                  scalar loss tensor.
    """

    def __init__(
        self,
        n: int,
        n_scales: int = 5,
        r_max: float | None = None,
        r_min: float | None = None,
        eps: float = 1e-8,
    ):
        super().__init__(n=n, n_scales=n_scales, r_max=r_max, r_min=r_min, eps=eps)

    def forward(self, Z: torch.Tensor, anchors_idx=None) -> torch.Tensor:
        if Z.ndim != 2:
            raise ValueError(f"Z must be (N, D); got shape {tuple(Z.shape)}")
        A = self._anchors(Z, anchors_idx)             # (Na, D)
        r_ladder = self._scales(Z)                    # (K+1,) decreasing
        d2 = pairwise_sq_distances(A, Z)              # (Na, N)

        n = float(self.n)
        loss = Z.new_zeros(())
        for k in range(r_ladder.numel()):
            r = r_ladder[k].clamp_min(self.eps)
            # Kernel weights w_j ∝ exp(-d²_j / (2 r²)).  Subtract the
            # row-wise max in the exponent so the largest weight in
            # each row is 1; this prevents underflow when r is small
            # relative to the typical pairwise distance.
            exponent = (-0.5 / (r * r)) * d2          # (Na, N), <= 0
            row_max = exponent.max(dim=-1, keepdim=True).values  # (Na, 1)
            w = torch.exp(exponent - row_max)         # (Na, N) in [0, 1]
            v = d2 / (r * r) - n                      # (Na, N)
            num = (w * v).sum(dim=-1)                 # (Na,)
            den = w.sum(dim=-1).clamp_min(self.eps)   # (Na,)
            t_dlog = num / den                        # (Na,)
            loss = loss + t_dlog.pow(2).mean()

        # Riemann step in log r.  Matches :class:`CGLT`'s log_step.
        log_step = torch.log(r_ladder[0] / r_ladder[-1].clamp_min(self.eps)) / max(self.n_scales, 1)
        return loss * log_step


class CGLTDerivRaw(URTest):
    r"""
    Scale-derivative variant of UR-CGLT (Eq.~(1.5) of
    Chousionis--Garnett--Le--Tolsa 2014), with the kernel
    normalization rescaled by ``t_max`` so the loss magnitude stays
    in the same range as :class:`CGLT` and :class:`CGLTDeriv`.

    The literal Eq.~(1.5) integrand on the empirical measure
    ``μ_N = (1/N) Σ_j δ_{z_j}`` is

        Δ̃_{μ_N,φ}(x, t)
            = (1 / (N t^n)) Σ_j exp(-||z_j - x||² / (2 t²))
                              · (||z_j - x||² / t² - n).

    The ``1/(N t^n)`` prefactor blows up by ``t_max^n / t_min^n ≈
    (N^{1/n})^n = N`` between the largest and smallest scales of the
    ladder, which makes the literal Δ̃ magnitude tiny (∼10⁻⁹) for
    typical recipes and forces a ``reg_scale ≈ 10^{10}`` retune. We
    instead use the *dimensionless* relative scale ``t' := t / t_max``
    in the prefactor while keeping the actual ``t`` in the kernel
    exponential:

        Δ̃'_{μ_N,φ}(x, t)
            := (1 / (N · (t/t_max)^n)) Σ_j exp(-||z_j - x||² / (2 t²))
                              · (||z_j - x||² / t² - n).

    Equivalently, ``Δ̃'(x, t) = t_max^n · Δ̃_{literal}(x, t)``, i.e.\
    the n-normalized kernel ``φ_t = t^{-n} φ(x/t)`` is replaced by
    the ``t_max``-normalized kernel ``φ_t^{rel} = (t/t_max)^{-n}
    φ(x/t)``, whose normalization is *constant across the batch's
    ladder*. The discriminative property is preserved:

      * On the n-AD-regular UR class, Δ̃' = 0 (same as literal).
      * On a true point-mass cloud, ``t_max → 0`` and the prefactor
        ``1 / (N · (t/t_max)^n)`` diverges; ``|Δ̃'|^2`` blows up
        as ``(n N / t'^n)^2`` (very strong anti-collapse signal).
      * On a non-degenerate cloud at any absolute scale (since
        ``t/t_max`` depends only on the dyadic ladder index, not on
        the cloud's diameter), magnitudes stay in the O(10) range
        regardless of how the cloud has been rescaled.

    Pair with :class:`ADRegularity` (``lambda_ad > 0``) to recover
    the AD-regularity side of Theorem~1.2 of \cite{CGLT2014}.

    Args:
        n, n_scales, r_max, r_min, eps: as in :class:`CGLT`.

    Shape:
        - Input ``Z``:             ``(N, D)``.
        - Input ``anchors_idx``:   ``None`` | ``int`` | ``LongTensor``.
        - Output:                  scalar loss tensor.
    """

    def __init__(
        self,
        n: int,
        n_scales: int = 5,
        r_max: float | None = None,
        r_min: float | None = None,
        eps: float = 1e-8,
    ):
        super().__init__(n=n, n_scales=n_scales, r_max=r_max, r_min=r_min, eps=eps)

    def forward(self, Z: torch.Tensor, anchors_idx=None) -> torch.Tensor:
        if Z.ndim != 2:
            raise ValueError(f"Z must be (N, D); got shape {tuple(Z.shape)}")
        N = Z.size(0)
        A = self._anchors(Z, anchors_idx)             # (Na, D)
        r_ladder = self._scales(Z)                    # (K+1,) decreasing
        d2 = pairwise_sq_distances(A, Z)              # (Na, N)

        t_max = r_ladder[0].clamp_min(self.eps)       # largest scale in batch
        n_eff = float(self.n)
        loss = Z.new_zeros(())
        for k in range(r_ladder.numel()):
            r = r_ladder[k].clamp_min(self.eps)
            t_prime = r / t_max                        # dimensionless, in [t_min/t_max, 1]
            w = torch.exp((-0.5 / (r * r)) * d2)       # (Na, N) in [0, 1]
            v = d2 / (r * r) - n_eff                   # (Na, N)
            # Δ̃' = (1 / (N · t'^n)) · Σ_j w_j · v_j
            partial = (w * v).sum(dim=-1)              # (Na,)
            partial = partial / (N * t_prime.pow(n_eff) + self.eps)
            loss = loss + partial.pow(2).mean()

        log_step = torch.log(r_ladder[0] / r_ladder[-1].clamp_min(self.eps)) / max(self.n_scales, 1)
        return loss * log_step


class ADRegularity(URTest):
    r"""
    Ahlfors--David regularity anchor for :class:`CGLT`.

    Penalizes the variance of ``log θ_r(x)`` across anchors and scales,
    where θ_r is the same Gaussian-smoothed density used in :class:`CGLT`.
    This is a kernel-normalization-invariant surrogate for the AD
    regularity condition ``μ(B(x,r)) ≍ r^n``; it drives the empirical
    density towards a scale-invariant constant without requiring the
    user to supply that constant.

    Args:
        n (int): target intrinsic dimension (must match the :class:`CGLT`
            that this anchor is paired with).
        n_scales, r_max, r_min, eps: same semantics as :class:`CGLT`.

    Shape:
        - Input ``Z``:             ``(N, D)``.
        - Input ``anchors_idx``:   ``None`` | ``int`` | ``LongTensor``.
        - Output:                  scalar loss tensor.
    """

    def __init__(
        self,
        n: int,
        n_scales: int = 5,
        r_max: float | None = None,
        r_min: float | None = None,
        eps: float = 1e-8,
    ):
        super().__init__(n=n, n_scales=n_scales, r_max=r_max, r_min=r_min, eps=eps)

    def forward(self, Z: torch.Tensor, anchors_idx=None) -> torch.Tensor:
        if Z.ndim != 2:
            raise ValueError(f"Z must be (N, D); got shape {tuple(Z.shape)}")
        N = Z.size(0)
        A = self._anchors(Z, anchors_idx)
        r_ladder = self._scales(Z)
        d2 = pairwise_sq_distances(A, Z)

        log_N = math.log(max(N, 1))
        logthetas = []
        for r in r_ladder:
            log_w_sum = torch.logsumexp((-0.5 / (r * r)) * d2, dim=-1)
            lt = log_w_sum - self.n * torch.log(r.clamp_min(self.eps)) - log_N
            logthetas.append(lt)
        lt = torch.stack(logthetas, dim=-1)                 # (Na, K+1)

        # For AD regularity, θ_r(x) should not depend on the anchor.  Penalize
        # the variance of ``log θ_r`` across anchors at each scale and average
        # over scales.  This is a free-floating, c_n-invariant surrogate for
        # the AD condition μ(B(x,r)) ≍ r^n.
        return lt.var(unbiased=False, dim=0).mean()
