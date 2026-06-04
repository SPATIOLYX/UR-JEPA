"""
Author: Spatiolyx LLC, https://spatiolyx.ai
Date: May 30, 2026

Geometry-aware visualization of UR-JEPA vs LeJEPA projector outputs.

Three invocation modes:

1. Two-method (default, back-compat with the original SIGReg vs UR-CGLT
   layout documented in report/viz_geometry_panels.md):

       python scripts/viz_geometry.py \\
           +ckpt_sigreg=<...sigreg/ckpt-final.pt> \\
           +ckpt_urcglt=<...ur_cglt/ckpt-final.pt> \\
           +dataset=imagenet100 +out_dir=<...>

   Produces the original 2x2 diagnostic figure (spectrum, per-dim W,
   Q-Q at worst-W dim, histogram at worst-W dim).

2. Multi-method (>=2 ckpts, e.g., the Galaxy10 6-way panel):

       python scripts/viz_geometry.py \\
           +ckpts.sigreg=<.../sigreg/ckpt-final.pt> \\
           +ckpts.ur_cglt=<.../ur_cglt/ckpt-final.pt> \\
           ... \\
           +labels.sigreg='SIGReg' \\
           +labels.ur_cglt='UR-CGLT (integral)' \\
           +dataset=galaxy10 +out_dir=<...>

   `cfg.ckpts` is an ordered DictConfig {key: ckpt_path}. `cfg.labels`
   (optional) maps the same keys to LaTeX-friendly display labels.
   Iteration order is preserved (Hydra/OmegaConf DictConfig). Produces
   a 2x2 overlay across all methods.

3. Relabel-only (skip forward pass; re-render an existing .npz with
   new labels):

       python scripts/viz_geometry.py \\
           +relabel_from=<.../viz_geometry.npz> \\
           +labels.m0='LeJEPA($\\mathcal{L}^{\\mathrm{SIGReg}}$)' \\
           +labels.m1='UR-JEPA($\\mathcal{L}^{\\mathrm{CGLT}}$)' \\
           +out_dir=<...> [+update_npz=true]

   Useful when the forward pass has already been run but the figure
   needs a different label set (e.g., the paper's LaTeX naming).
   Labels can also be supplied as a plain-text file
   (one label per line, ``m0..m{N-1}`` order):

       +labels_file=<path/to/labels.txt>

   The optional ``+update_npz=true`` flag rewrites the ``labels``
   array inside the source .npz so future tools see the new labels.

In modes (1) and (2) the script writes:
    <out_dir>/viz_geometry.png  -- 2x2 diagnostic figure
    <out_dir>/viz_geometry.npz  -- raw projector outputs + diagnostics

Mode (3) only writes the figure (and optionally rewrites the source
.npz when ``+update_npz=true``).

Expectation:
    SIGReg's sliced-characteristic-function regularizer pushes the
    projected distribution toward isotropic Gaussian (flat spectrum at
    ~1, per-dim W ~ 1, Q-Q on the diagonal). UR-CGLT and the
    geometric-loss family (CGLT-deriv, CGLT-deriv-raw, beta+log-trace,
    beta+log-trace+eigthresh) do not constrain isotropy globally; the
    projector outputs can exhibit anisotropy and a covariance "cliff"
    while still satisfying uniform n-rectifiability locally.
"""

import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
import hydra
from omegaconf import DictConfig
import scipy.stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tqdm

from pretrain import ViTEncoder, HFDataset, PRETRAIN_DATASETS


def load_pretrained_net(ckpt_path: str, device: str):
    """Reconstruct the ViTEncoder (backbone + 3-layer MLP projector) from a
    pretrain.py checkpoint. Strips DDP prefixes."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    state = ckpt["net"]
    state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
             for k, v in state.items()}

    net = ViTEncoder(
        proj_dim=int(cfg.get("proj_dim", 128)),
        projector_norm=cfg.get("projector_norm", "bn"),
        backbone=cfg.get("backbone", "vit_s8_128"),
        drop_path_rate=float(cfg.get("drop_path_rate", 0.1)),
    )
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys ({len(missing)}): "
              f"{missing[:3]}{'...' if len(missing) > 3 else ''}")
    if unexpected:
        print(f"[warn] unexpected keys ({len(unexpected)}): "
              f"{unexpected[:3]}{'...' if len(unexpected) > 3 else ''}")
    net.to(device).eval()
    for p in net.parameters():
        p.requires_grad = False
    return net, cfg


@torch.inference_mode()
def collect_projector_outputs(net: ViTEncoder, loader, device: str,
                              max_samples: int) -> np.ndarray:
    """Run `net` on `loader` (V=1 eval transform) and stack projector
    outputs up to `max_samples`. Returns (N, D) float32."""
    outs = []
    n = 0
    pbar = tqdm.tqdm(loader, desc="collect", leave=False)
    use_amp = device == "cuda"
    for x, _ in pbar:
        x = x.to(device, non_blocking=True)
        B, V = x.shape[:2]
        assert V == 1, f"expected V=1 eval split, got V={V}"
        if use_amp:
            with autocast("cuda", dtype=torch.bfloat16):
                emb = net.backbone(x.flatten(0, 1))
                proj = net.proj(emb)
        else:
            emb = net.backbone(x.flatten(0, 1))
            proj = net.proj(emb)
        outs.append(proj.float().cpu().numpy())
        n += B
        pbar.set_postfix(n=n)
        if n >= max_samples:
            break
    Z = np.concatenate(outs, axis=0)[:max_samples]
    return Z


def compute_diagnostics(Z: np.ndarray, seed: int = 0) -> dict:
    """Eigenvalue spectrum, per-dim mean/std, per-dim Shapiro-Wilk W.

    Z: (N, D) projector outputs.
    """
    N, D = Z.shape
    Zc = Z - Z.mean(axis=0, keepdims=True)
    cov = (Zc.T @ Zc) / (N - 1)
    eigvals = np.linalg.eigvalsh(cov)[::-1]

    rng = np.random.default_rng(seed)
    # scipy.stats.shapiro uses an exact algorithm only up to N=5000; subsample.
    n_test = min(N, 5000)
    sw_W = np.empty(D, dtype=np.float64)
    sw_p = np.empty(D, dtype=np.float64)
    for d in range(D):
        col = Z[:, d]
        if N > n_test:
            col = rng.choice(col, size=n_test, replace=False)
        col = (col - col.mean()) / (col.std() + 1e-12)
        try:
            res = scipy.stats.shapiro(col)
            sw_W[d] = float(res.statistic)
            sw_p[d] = float(res.pvalue)
        except Exception as e:
            print(f"[warn] shapiro failed on dim {d}: {e}")
            sw_W[d] = np.nan
            sw_p[d] = np.nan

    return {
        "eigvals": eigvals,
        "mean": Z.mean(axis=0),
        "std": Z.std(axis=0),
        "shapiro_W": sw_W,
        "shapiro_p": sw_p,
    }


def _standardize_col(z: np.ndarray) -> np.ndarray:
    return (z - z.mean()) / (z.std() + 1e-12)


def plot_diagnostics(diag_a, diag_b, label_a: str, label_b: str,
                     Z_a: np.ndarray, Z_b: np.ndarray, out_path: Path):
    """2x2 figure: spectrum, marginal-Gaussianity, Q-Q (worst-W dim),
    histogram (same dim)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    col_a, col_b = "tab:blue", "tab:orange"

    # (0,0) eigenvalue spectrum, log-y overlay
    ax = axes[0, 0]
    ka = np.arange(1, len(diag_a["eigvals"]) + 1)
    kb = np.arange(1, len(diag_b["eigvals"]) + 1)
    ax.semilogy(ka, diag_a["eigvals"], "-o", color=col_a, label=label_a)
    ax.semilogy(kb, diag_b["eigvals"], "-s", color=col_b, label=label_b)
    ax.set_xlabel("eigenvalue index (sorted descending)")
    ax.set_ylabel("eigenvalue")
    ax.set_title("Projector covariance eigenvalue spectrum")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # (0,1) Shapiro-Wilk W per dim, scatter
    D = len(diag_a["shapiro_W"])
    ax = axes[0, 1]
    ax.scatter(np.arange(D), diag_a["shapiro_W"], color=col_a, alpha=0.75,
               s=36, label=label_a)
    ax.scatter(np.arange(D), diag_b["shapiro_W"], color=col_b, alpha=0.75,
               s=36, marker="s", label=label_b)
    ax.axhline(1.0, color="k", linestyle=":", alpha=0.5,
               label="Gaussian ($W=1$)")
    ax.set_ylim(min(0.0, np.nanmin(diag_a["shapiro_W"]) - 0.05,
                    np.nanmin(diag_b["shapiro_W"]) - 0.05),
                1.05)
    ax.set_xlabel("projector dimension")
    ax.set_ylabel("Shapiro-Wilk $W$")
    ax.set_title("Per-dim marginal Gaussianity")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    # Pick the lowest-W dim per method for the Q-Q and histogram panels.
    dim_a = int(np.nanargmin(diag_a["shapiro_W"]))
    dim_b = int(np.nanargmin(diag_b["shapiro_W"]))

    # (1,0) Q-Q plot, lowest-W dim per method, overlaid
    ax = axes[1, 0]
    z_a = _standardize_col(Z_a[:, dim_a])
    z_b = _standardize_col(Z_b[:, dim_b])
    osm_a, osr_a = scipy.stats.probplot(z_a, dist="norm", fit=False)
    osm_b, osr_b = scipy.stats.probplot(z_b, dist="norm", fit=False)
    ax.scatter(osm_a, osr_a, color=col_a, alpha=0.45, s=12,
               label=f"{label_a} (dim {dim_a}, $W$={diag_a['shapiro_W'][dim_a]:.3f})")
    ax.scatter(osm_b, osr_b, color=col_b, alpha=0.45, s=12,
               label=f"{label_b} (dim {dim_b}, $W$={diag_b['shapiro_W'][dim_b]:.3f})")
    lim = 4.5
    ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.5,
            label="Gaussian (diagonal)")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("standard normal quantile")
    ax.set_ylabel("ordered sample (standardized)")
    ax.set_title("Q-Q plot, lowest-$W$ dim per method")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    # (1,1) histogram + N(0,1) overlay, same dims
    ax = axes[1, 1]
    bins = np.linspace(-4.0, 4.0, 60)
    ax.hist(z_a, bins=bins, density=True, alpha=0.45, color=col_a,
            label=f"{label_a} (dim {dim_a})")
    ax.hist(z_b, bins=bins, density=True, alpha=0.45, color=col_b,
            label=f"{label_b} (dim {dim_b})")
    xs = np.linspace(-4.0, 4.0, 200)
    ax.plot(xs, scipy.stats.norm.pdf(xs), "k-", alpha=0.7,
            label=r"$\mathcal{N}(0,1)$")
    ax.set_xlabel("standardized value")
    ax.set_ylabel("density")
    ax.set_title("Marginal histogram, same dims (standardized)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_diagnostics_multi(results, out_path: Path):
    """N-method overlay of the same 2x2 panels.

    `results` is a list of (label, Z, diag) tuples; up to ~10 methods
    render cleanly with the tab10 colormap. The Q-Q and histogram
    panels use each method's own lowest-W dim (so dim indices differ
    across methods - shown in the legend).
    """
    n = len(results)
    cmap = matplotlib.colormaps.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(n)]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) eigenvalue spectrum, log-y overlay
    ax = axes[0, 0]
    for i, (label, _Z, diag) in enumerate(results):
        k = np.arange(1, len(diag["eigvals"]) + 1)
        ax.semilogy(k, diag["eigvals"], "-" + markers[i % len(markers)],
                    color=colors[i], label=label, markersize=5, alpha=0.9)
    ax.set_xlabel("eigenvalue index (sorted descending)")
    ax.set_ylabel("eigenvalue")
    ax.set_title("Projector covariance eigenvalue spectrum")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    # (0,1) Shapiro-Wilk W per dim, scatter (x-jittered per method)
    ax = axes[0, 1]
    D = len(results[0][2]["shapiro_W"])
    jitter = np.linspace(-0.3, 0.3, n) if n > 1 else np.zeros(n)
    min_W = 1.0
    for i, (label, _Z, diag) in enumerate(results):
        xs = np.arange(D) + jitter[i]
        ax.scatter(xs, diag["shapiro_W"], color=colors[i], alpha=0.75,
                   s=28, marker=markers[i % len(markers)], label=label)
        min_W = min(min_W, float(np.nanmin(diag["shapiro_W"])))
    ax.axhline(1.0, color="k", linestyle=":", alpha=0.5,
               label="Gaussian ($W=1$)")
    ax.set_ylim(min(0.0, min_W - 0.05), 1.05)
    ax.set_xlabel("projector dimension")
    ax.set_ylabel("Shapiro-Wilk $W$")
    ax.set_title("Per-dim marginal Gaussianity")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    # (1,0) Q-Q at each method's lowest-W dim
    ax = axes[1, 0]
    lim = 4.5
    for i, (label, Z, diag) in enumerate(results):
        dim = int(np.nanargmin(diag["shapiro_W"]))
        z = _standardize_col(Z[:, dim])
        osm, osr = scipy.stats.probplot(z, dist="norm", fit=False)
        ax.scatter(osm, osr, color=colors[i], alpha=0.35, s=10,
                   marker=markers[i % len(markers)],
                   label=f"{label} (dim {dim}, $W$={diag['shapiro_W'][dim]:.3f})")
    ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.5,
            label="Gaussian (diagonal)")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("standard normal quantile")
    ax.set_ylabel("ordered sample (standardized)")
    ax.set_title("Q-Q plot, lowest-$W$ dim per method")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=7)

    # (1,1) histogram + N(0,1) overlay at lowest-W dim
    ax = axes[1, 1]
    bins = np.linspace(-4.0, 4.0, 60)
    for i, (label, Z, diag) in enumerate(results):
        dim = int(np.nanargmin(diag["shapiro_W"]))
        z = _standardize_col(Z[:, dim])
        # Use step histograms so 6 overlaid methods stay readable.
        ax.hist(z, bins=bins, density=True, histtype="step",
                color=colors[i], linewidth=1.5, alpha=0.85,
                label=f"{label} (dim {dim})")
    xs = np.linspace(-4.0, 4.0, 200)
    ax.plot(xs, scipy.stats.norm.pdf(xs), "k-", alpha=0.7, linewidth=1.5,
            label=r"$\mathcal{N}(0,1)$")
    ax.set_xlabel("standardized value")
    ax.set_ylabel("density")
    ax.set_title("Marginal histogram, same dims (standardized)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _summary_line(name, diag):
    ev = diag["eigvals"]
    sw = diag["shapiro_W"]
    ratio = ev[0] / max(ev[-1], 1e-12)
    print(f"  {name}: eig[0]={ev[0]:.4f}  eig[-1]={ev[-1]:.4f}  "
          f"eig[0]/eig[-1]={ratio:.2f}  "
          f"W: mean={np.nanmean(sw):.4f}  min={np.nanmin(sw):.4f}  "
          f"max={np.nanmax(sw):.4f}")


def _safe_key(label: str) -> str:
    """Slugify a label for use as a key in the .npz dump."""
    out = []
    for c in label:
        if c.isalnum() or c in ("_", "-"):
            out.append(c)
        elif c == " ":
            out.append("_")
    s = "".join(out).strip("_")
    return s or "method"


def load_results_from_npz(npz_path: Path):
    """Recover the ``(label, Z, diag)`` list saved by :func:`main`.

    Supports both serialization schemas this script has used:

    - Multi-method (>=2 methods): per-method keys ``m<i>_<slug>_{Z,
      eigvals, mean, std, shapiro_W, shapiro_p}`` plus a top-level
      ``labels`` array, where ``i`` is the method index.
    - Two-method legacy: ``Z_a / Z_b``, ``eigvals_a / eigvals_b``, ...,
      ``label_a / label_b``.
    """
    d = np.load(npz_path, allow_pickle=True)

    # --- Multi-method schema -----------------------------------------------
    if "labels" in d.files:
        old_labels = list(d["labels"])
        n = len(old_labels)
        method_prefix = {}
        pat = re.compile(
            r"^(m(\d+)_.+?)_(Z|eigvals|mean|std|shapiro_W|shapiro_p)$"
        )
        for k in d.files:
            m = pat.match(k)
            if not m:
                continue
            method_prefix.setdefault(int(m.group(2)), m.group(1))
        results = []
        for i in range(n):
            prefix = method_prefix[i]
            results.append((
                str(old_labels[i]),
                d[f"{prefix}_Z"],
                {
                    "eigvals": d[f"{prefix}_eigvals"],
                    "mean": d[f"{prefix}_mean"],
                    "std": d[f"{prefix}_std"],
                    "shapiro_W": d[f"{prefix}_shapiro_W"],
                    "shapiro_p": d[f"{prefix}_shapiro_p"],
                },
            ))
        return results

    # --- Two-method legacy schema ------------------------------------------
    if "Z_a" in d.files and "Z_b" in d.files:
        results = []
        for suf in ("a", "b"):
            results.append((
                str(d[f"label_{suf}"]),
                d[f"Z_{suf}"],
                {
                    "eigvals": d[f"eigvals_{suf}"],
                    "mean": d[f"mean_{suf}"],
                    "std": d[f"std_{suf}"],
                    "shapiro_W": d[f"shapiro_W_{suf}"],
                    "shapiro_p": d[f"shapiro_p_{suf}"],
                },
            ))
        return results

    raise ValueError(
        f"{npz_path}: neither multi-method nor legacy two-method schema "
        f"recognized (keys: {list(d.files)[:6]}{'...' if len(d.files) > 6 else ''})"
    )


def _resolve_relabel_overrides(cfg: DictConfig, n_methods: int,
                               old_labels: list[str]) -> list[str]:
    """Build the new label list for relabel mode from cfg.

    Sources (highest precedence first):
    - ``+labels_file=<path>`` -- plain-text file, one label per line
    - ``+labels.m<i>=<str>`` -- per-method positional Hydra override
    - ``+labels.<old_label>=<str>`` -- override by the existing label
    - fallback to the old label
    """
    if cfg.get("labels_file"):
        path = Path(cfg.labels_file)
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        if len(lines) != n_methods:
            raise ValueError(
                f"{path} has {len(lines)} non-empty lines but npz has "
                f"{n_methods} methods"
            )
        return [ln.rstrip("\n") for ln in lines]

    labels_cfg = cfg.get("labels", {}) or {}
    new_labels = []
    for i, old in enumerate(old_labels):
        key_i = f"m{i}"
        if key_i in labels_cfg:
            new_labels.append(str(labels_cfg[key_i]))
        elif old in labels_cfg:
            new_labels.append(str(labels_cfg[old]))
        else:
            new_labels.append(old)
    return new_labels


@hydra.main(version_base=None)
def main(cfg: DictConfig):
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Mode 3: relabel-only. Skip dataset / loader / forward pass and just
    # re-render an existing .npz with new display labels.
    # -----------------------------------------------------------------------
    if cfg.get("relabel_from"):
        npz_in = Path(cfg.relabel_from)
        results_old = load_results_from_npz(npz_in)
        old_labels = [lbl for lbl, _, _ in results_old]
        new_labels = _resolve_relabel_overrides(cfg, len(results_old), old_labels)
        results = [(lbl, Z, diag)
                   for lbl, (_, Z, diag) in zip(new_labels, results_old)]

        fig_path = out_dir / "viz_geometry.png"
        if len(results) == 2:
            (la, Za, da), (lb, Zb, db) = results
            plot_diagnostics(da, db, la, lb, Za, Zb, fig_path)
        else:
            plot_diagnostics_multi(results, fig_path)
        print(f"figure -> {fig_path}")

        print("=== summary (new labels)")
        for label, _, diag in results:
            _summary_line(label, diag)

        if cfg.get("update_npz", False):
            d = np.load(npz_in, allow_pickle=True)
            save = {k: d[k] for k in d.files}
            if "labels" in save:
                save["labels"] = np.array(new_labels)
            else:
                # Legacy two-method schema.
                save["label_a"] = np.array(new_labels[0])
                save["label_b"] = np.array(new_labels[1])
            np.savez(npz_in, **save)
            print(f"updated labels in -> {npz_in}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dname = cfg.get("dataset", "imagenet100")
    if dname not in PRETRAIN_DATASETS:
        raise ValueError(
            f"Unknown dataset {dname!r}. Choose from {list(PRETRAIN_DATASETS)}"
        )
    spec = PRETRAIN_DATASETS[dname]
    val_split = cfg.get("split", spec.get("val_split"))
    if val_split is None:
        raise ValueError(
            f"Dataset {dname!r} has no val_split; pass +split=<name> "
            f"explicitly."
        )

    img_size = int(cfg.get("img_size", 128))
    test_ds = HFDataset(dname, val_split, V=1, img_size=img_size)
    loader = DataLoader(
        test_ds,
        batch_size=int(cfg.get("bs", 128)),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=True,
        drop_last=False,
    )

    max_samples = int(cfg.get("max_samples", 5000))

    # ---- Build method list -----------------------------------------------
    # Two modes:
    #   - multi-method: +ckpts.<key>=<path> [+labels.<key>=<display>]
    #   - two-method (back-compat): +ckpt_sigreg=... +ckpt_urcglt=...
    method_list = []  # list of (label, ckpt_path)
    if "ckpts" in cfg and cfg.ckpts is not None:
        labels_cfg = cfg.get("labels", {}) or {}
        for key, path in cfg.ckpts.items():
            label = str(labels_cfg[key]) if key in labels_cfg else str(key)
            method_list.append((label, str(path)))
    else:
        label_a = str(cfg.get("label_sigreg",
                              r"LeJEPA($\mathcal{L}^{\mathrm{SIGReg}}$)"))
        label_b = str(cfg.get("label_urcglt",
                              r"UR-JEPA($\mathcal{L}^{\mathrm{CGLT}}$)"))
        method_list = [
            (label_a, str(cfg.ckpt_sigreg)),
            (label_b, str(cfg.ckpt_urcglt)),
        ]

    if len(method_list) < 2:
        raise ValueError(
            f"Need at least 2 methods to visualize; got {len(method_list)}"
        )

    # ---- Run each method through the validation loader -------------------
    results = []  # list of (label, Z, diag)
    for label, ckpt in method_list:
        print(f"=== {label}")
        print(f"    ckpt={ckpt}")
        net, _ = load_pretrained_net(ckpt, device)
        Z = collect_projector_outputs(net, loader, device, max_samples)
        print(f"    collected Z shape={Z.shape}")
        diag = compute_diagnostics(Z)
        del net
        if device == "cuda":
            torch.cuda.empty_cache()
        results.append((label, Z, diag))

    # ---- Plot ------------------------------------------------------------
    fig_path = out_dir / "viz_geometry.png"
    if len(results) == 2:
        (la, Za, da), (lb, Zb, db) = results
        plot_diagnostics(da, db, la, lb, Za, Zb, fig_path)
    else:
        plot_diagnostics_multi(results, fig_path)
    print(f"figure -> {fig_path}")

    # ---- Dump raw arrays + diagnostics -----------------------------------
    npz_path = out_dir / "viz_geometry.npz"
    if len(results) == 2:
        # Preserve legacy keys so downstream notebooks keep working.
        (la, Za, da), (lb, Zb, db) = results
        np.savez(
            npz_path,
            Z_a=Za, Z_b=Zb,
            eigvals_a=da["eigvals"], eigvals_b=db["eigvals"],
            mean_a=da["mean"], mean_b=db["mean"],
            std_a=da["std"], std_b=db["std"],
            shapiro_W_a=da["shapiro_W"], shapiro_W_b=db["shapiro_W"],
            shapiro_p_a=da["shapiro_p"], shapiro_p_b=db["shapiro_p"],
            label_a=la, label_b=lb,
        )
    else:
        save_dict = {"labels": np.array([la for la, _, _ in results])}
        for i, (label, Z, diag) in enumerate(results):
            key = f"m{i}_{_safe_key(label)}"
            save_dict[f"{key}_Z"] = Z
            save_dict[f"{key}_eigvals"] = diag["eigvals"]
            save_dict[f"{key}_mean"] = diag["mean"]
            save_dict[f"{key}_std"] = diag["std"]
            save_dict[f"{key}_shapiro_W"] = diag["shapiro_W"]
            save_dict[f"{key}_shapiro_p"] = diag["shapiro_p"]
        np.savez(npz_path, **save_dict)
    print(f"raw data -> {npz_path}")

    print("=== summary")
    for label, _, diag in results:
        _summary_line(label, diag)


if __name__ == "__main__":
    main()
