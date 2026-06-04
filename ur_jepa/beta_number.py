"""
Author: Spatiolyx LLC, https://spatiolyx.ai
Date: May 30, 2026

Local-PCA β-number regularizer for UR-JEPA.

Implements the Carleson β-loss
    L_beta(θ) = (log 2 / (|A| R^n)) * Σ_{x∈A} Σ_k r_k^n · β̂_2(x, r_k)^2
where
    β̂_2(x, r)^2 = (1 / r^{n+2}) · (Σ_{j>n} σ_j^2(S_r(x))) / Σ_j w_r(z_j - x)
with ``S_r(x)`` the Gaussian-weighted centered scatter matrix of the
embeddings around ``x``:
    w_r(u) = exp(-||u||^2 / (2 r^2)),
    z̄_r(x) = Σ_j w_r(z_j-x) z_j / Σ_j w_r(z_j-x),
    S_r(x) = Σ_j w_r(z_j-x) (z_j-z̄_r)(z_j-z̄_r)^T / Σ_j w_r.

By Pajot / David--Semmes the Carleson bound on β_2^2 characterizes
uniform n-rectifiability of μ. We turn it into a differentiable
self-supervised regularizer: minimizing this loss drives the empirical
embedding measure towards a uniformly n-rectifiable measure of the user-
specified intrinsic dimension ``n``.

The style (torch.nn.Module, ``register_buffer`` for constants, forward
that takes the embeddings directly) mirrors ``lejepa.multivariate.SIGReg``
so that this module is drop-in compatible with LeJEPA-style training
scripts.
"""

import math
import torch

from .base import URTest, pairwise_sq_distances


class BetaNumber(URTest):
    r"""
    Jones β-number UR regularizer evaluated through local PCA.

    This is the recommended default for ambient dimensions large enough
    that direct density estimation (see :class:`CGLT`) becomes noisy.

    Args:
        n (int): target intrinsic dimension.
        n_scales (int, optional): number of dyadic levels in the scale
            ladder. Default ``5``.
        r_max, r_min (float, optional): absolute endpoints of the scale
            ladder. If ``None`` they are inferred from each batch.
        eps (float, optional): ridge added to the scatter matrix for
            numerical stability before eigendecomposition. Default ``1e-8``.
        topk_method (str, optional): one of ``"eigvalsh"`` (exact,
            ``O(D^3)`` per anchor) or ``"lowrank"`` (top-n via
            ``torch.lobpcg``). Default ``"eigvalsh"``.
        eigval_threshold (float, optional): when ``> 0``, select tangent
            eigenvectors adaptively per anchor/scale instead of taking the
            top ``n``: keep every direction with
            ``lambda_i > eigval_threshold * trace(S) / D``. This is the
            "variance-share relative to uniform" rule (``tau = 1`` =
            "above mean variance"). ``residual = trace - sum of selected``
            then varies per anchor (a per-point local intrinsic dim
            estimate). The ``r^{n+2}`` normalization still uses the
            global ``cfg.n`` -- the threshold is a tangent/normal
            selector, not a redefinition of the rectifiability target.
            Default ``0.0`` (off; fall back to top-n). Requires
            ``topk_method="eigvalsh"``.
        gamma_logtrace (float, optional): weight on a per-anchor
            ``-log trace(S_r(x))`` anti-collapse penalty added to the
            Carleson β² loss. Default ``0.0`` (off, exact β² loss). When
            ``> 0`` this rescues the β² loss from its structural collapse
            mode (point-mass cloud, ``trace(S) → 0``, trivially gives
            ``β² = 0``) without an external AD anchor. ``log trace(S)``
            is averaged over anchors and scales before applying the weight.
        log_beta_eps (float, optional): when ``> 0``, the per-anchor
            per-scale β² is replaced by ``log(β² + log_beta_eps)`` BEFORE
            the anchor mean and scale weighting. Default ``0.0`` (off,
            exact β² loss). Compresses the dynamic range of β² across
            scales (β² can span several orders of magnitude on a single
            batch). Note: log(β²) does NOT change the collapse mode -- the
            trivial minimum at β² = 0 becomes log(ε) (a finite floor),
            still attractive. Anti-collapse complement (``lambda_ad`` or
            ``gamma_logtrace``) is still required for non-trivial
            optimization. Setting ``log_beta_eps`` independently of those
            knobs lets you measure the dynamic-range compression effect
            without changing the collapse-mode handling.

    Shape:
        - Input ``Z``:             ``(N, D)``.
        - Input ``anchors_idx``:   ``None`` (use all of ``Z`` as anchors),
          an ``int`` (random subset size), or a 1-D ``LongTensor`` of
          indices.
        - Output:                  scalar loss tensor.

    Example:
        >>> Z = torch.randn(256, 128, requires_grad=True)
        >>> loss_fn = BetaNumber(n=8, n_scales=5)
        >>> loss = loss_fn(Z)
        >>> loss.backward()
    """

    def __init__(
        self,
        n: int,
        n_scales: int = 5,
        r_max: float | None = None,
        r_min: float | None = None,
        eps: float = 1e-8,
        topk_method: str = "eigvalsh",
        gamma_logtrace: float = 0.0,
        eigval_threshold: float = 0.0,
        log_beta_eps: float = 0.0,
    ):
        super().__init__(n=n, n_scales=n_scales, r_max=r_max, r_min=r_min, eps=eps)
        if topk_method not in ("eigvalsh", "lowrank"):
            raise ValueError(
                f"topk_method must be 'eigvalsh' or 'lowrank', got {topk_method}"
            )
        if eigval_threshold > 0 and topk_method != "eigvalsh":
            raise ValueError(
                "eigval_threshold > 0 requires topk_method='eigvalsh' "
                "(the lowrank path only computes the top-n spectrum, "
                "which is incompatible with an adaptive selection rule)"
            )
        self.topk_method = topk_method
        self.gamma_logtrace = float(gamma_logtrace)
        self.eigval_threshold = float(eigval_threshold)
        self.log_beta_eps = float(log_beta_eps)

    # ------------------------------------------------------------------
    # per-scale β̂_2^2 estimator
    # ------------------------------------------------------------------
    def _beta2_at_scale(
        self,
        A: torch.Tensor,      # (Na, D) anchors
        Z: torch.Tensor,      # (N,  D) all embeddings
        d2: torch.Tensor,     # (Na, N) squared distances anchors--points
        r: torch.Tensor,      # scalar tensor on Z.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(beta2, log_tr)`` per anchor: ``β̂_2(x, r)^2`` and ``log trace(S_r(x))``."""
        N, D = Z.shape
        n = self.n

        # Gaussian weights (Na, N). Stable against over/underflow by
        # subtracting the per-row minimum exponent (monotone shift that
        # cancels in mean/centered sums downstream is not exact here, so
        # we just clamp the exponent).
        w = torch.exp((-0.5 / (r * r)) * d2)

        # mass per anchor (Na, 1) and weighted mean (Na, D)
        m = w.sum(-1, keepdim=True).clamp_min(self.eps)
        zbar = (w @ Z) / m

        # Weighted scatter  S = Y^T diag(w) Y / m
        # Compute without materializing (Na, N, D): use einsum trick
        # S_{a p q} = Σ_j w_{aj} (z_{jp} - zbar_{ap}) (z_{jq} - zbar_{aq}) / m_a
        #           = (Σ_j w_{aj} z_{jp} z_{jq}) / m_a  -  zbar_{ap} zbar_{aq}
        # The second term follows from the Koenig formula for weighted covariance.
        wz = w.unsqueeze(-1) * Z.unsqueeze(0)               # (Na, N, D)
        WZ = torch.einsum("anp,nq->apq", wz, Z) / m.unsqueeze(-1)  # (Na, D, D)
        S = WZ - zbar.unsqueeze(-1) * zbar.unsqueeze(-2)    # (Na, D, D)

        # symmetrize and ridge-regularize
        S = 0.5 * (S + S.transpose(-1, -2))
        S = S + self.eps * torch.eye(D, device=Z.device, dtype=Z.dtype)

        tr = torch.diagonal(S, dim1=-2, dim2=-1).sum(-1)    # (Na,)

        if self.topk_method == "eigvalsh":
            # ascending eigvals; take last n and sum (or adaptive selection)
            evals = torch.linalg.eigvalsh(S)                # (Na, D)
            if self.eigval_threshold > 0.0:
                # Adaptive: keep λ_i with λ_i > τ · trace(S) / D.
                # The boolean mask has zero gradient (discrete), so
                # gradients flow only through `evals` itself -- standard
                # "selection is detached" Rayleigh-style update.
                cutoff = self.eigval_threshold * tr.unsqueeze(-1) / D   # (Na, 1)
                mask = (evals > cutoff).to(evals.dtype)                 # (Na, D)
                top_n = (evals * mask).sum(-1)                          # (Na,)
            else:
                top_n = evals[..., -n:].sum(-1)             # (Na,)
        else:  # lowrank: only the top-n via Lanczos
            # lobpcg needs positive-definite; S is PSD after the ridge.
            # Detach evecs and reconstruct top-n eigvals as
            # trace(U^T S U) = Σ_n Σ_{d,e} U[a,d,n] S[a,d,e] U[a,e,n].
            # The detach is necessary because lobpcg's backward is
            # numerically brittle (Cholesky on near-degenerate spectra);
            # treating U as constant gives gradients via S only, which
            # is the standard Rayleigh-quotient trick for eigendecomp.
            with torch.no_grad():
                _, evecs = torch.lobpcg(S, k=n, largest=True)
            top_n = torch.einsum("adn,ade,aen->a", evecs, S, evecs)

        residual = (tr - top_n).clamp_min(0.0)              # Σ_{j>n} σ_j^2

        # β̂_2^2 = residual / r^{n+2} / 1   (mass already normalized in S)
        beta2 = residual / (r.pow(n + 2) + self.eps)
        # log trace, used as an anti-collapse penalty when gamma_logtrace > 0.
        # The ridge in tr (self.eps * D) gives a finite log even at total
        # collapse, but we additionally clamp for safety.
        log_tr = torch.log(tr.clamp_min(self.eps))           # (Na,)
        return beta2, log_tr

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, Z: torch.Tensor, anchors_idx=None) -> torch.Tensor:
        if Z.ndim != 2:
            raise ValueError(f"Z must be (N, D); got shape {tuple(Z.shape)}")
        A = self._anchors(Z, anchors_idx)                   # (Na, D)
        r_ladder = self._scales(Z)                          # (K+1,)
        d2 = pairwise_sq_distances(A, Z)                    # (Na, N)

        loss = Z.new_zeros(())
        log_tr_acc = Z.new_zeros(())
        n_levels = r_ladder.numel() - 1
        # Carleson sum.  The Riemann sum of β^2 dμ dr/r, restricted to an
        # n-dim support, picks up a factor r^n per dyadic level.
        for k in range(n_levels):
            r = r_ladder[k + 1]                             # use finer radius
            beta2, log_tr = self._beta2_at_scale(A, Z, d2, r)  # (Na,), (Na,)
            if self.log_beta_eps > 0.0:
                # log(β² + ε) per anchor before the anchor mean and
                # scale weighting. Compresses the dynamic range of β²
                # across scales; does NOT change the collapse mode
                # (anti-collapse complement still required).
                beta_term = torch.log(beta2 + self.log_beta_eps)
            else:
                beta_term = beta2
            loss = loss + beta_term.mean() * r.pow(self.n)
            log_tr_acc = log_tr_acc + log_tr.mean()

        # Riemann-sum step in log-radius: log(r_{k-1}/r_k). This equals
        # log 2 only when the ladder is exactly dyadic; in general it is
        # log(r_max/r_min) / n_scales, which is what dyadic_scales builds.
        r_max = r_ladder[0]
        r_min = r_ladder[-1]
        log_step = torch.log(r_max / r_min.clamp_min(self.eps)) / max(self.n_scales, 1)
        loss = loss * log_step / (r_max.pow(self.n) + self.eps)

        # Anti-collapse: subtract γ · mean log trace(S) (averaged over
        # anchors and scales). At collapse trace → 0, log → -∞, so this
        # term blows up and is what stops the cloud from imploding.
        if self.gamma_logtrace > 0.0 and n_levels > 0:
            loss = loss - self.gamma_logtrace * (log_tr_acc / n_levels)
        return loss
