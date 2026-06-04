"""
Author: Spatiolyx LLC, https://spatiolyx.ai
Date: May 30, 2026

Plot per-epoch trajectories (losses + test accuracy) from one or more
``results.json`` files written by ``scripts/pretrain.py`` (or
``scripts/eval_downstream.py``).

Two modes:

(A) **Single run, multi-panel figure**: pass one ``results.json``;
    produces a 2x2 grid (test_acc + reg_loss + inv_loss + probe_loss)
    in a single PNG.
(B) **Overlay across runs**: pass multiple paths, or use ``--root <dir>``
    to auto-discover all ``*/results.json`` two levels deep. Each run
    is one labeled line; a single shared legend identifies them.

Examples:

    # Single run (auto-named output)
    python scripts/plot_trajectory.py runs/inet100_lejepa/ur_cglt_s0/results.json

    # Two-method overlay
    python scripts/plot_trajectory.py \\
        runs/inet100_lejepa/sigreg_s0/results.json \\
        runs/inet100_lejepa/ur_cglt_s0/results.json \\
        --out trajectory_inet100.png

    # Auto-discover all runs under a root
    python scripts/plot_trajectory.py --root runs/inet100_lejepa --out inet100.png

    # Plot just test accuracy (single panel)
    python scripts/plot_trajectory.py --root runs/inet100_lejepa --metric test_acc

    # Smooth losses with a rolling mean (default 1 = no smoothing)
    python scripts/plot_trajectory.py --root runs/inet100_lejepa --smooth 10
"""

import argparse
import json
import re
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"ERROR: matplotlib is required but not importable ({e})", file=sys.stderr)
    print("       conda activate $SCRATCH/conda-envs/urjepa  &&  pip install matplotlib", file=sys.stderr)
    sys.exit(1)


# Metric configuration: (key in results['epochs'], pretty label, log-y?)
METRICS = {
    "test_acc":   ("Test acc (linear probe)", False),
    "reg_loss":   ("Regularizer loss",         True),
    "inv_loss":   ("Invariance loss",          True),
    "probe_loss": ("Online probe loss",        True),
}

SEED_SUFFIX_RE = re.compile(r"_s(\d+)$")

# Display labels for each regularizer choice. Maps the cfg.regularizer
# value to the paper-style display name -- method family (LeJEPA /
# UR-JEPA) followed by the associated loss in math mode. Falls back to
# the cfg string for any value not listed. Keep in sync with
# pretrain.py's build_regularizer() registry and the paper's notation.
#
# LaTeX bits below use matplotlib's mathtext, which supports
# \mathcal{L}, \text{}, and \beta natively. The UR<en-dash>JEPA uses
# the Unicode en-dash (U+2013) to render the paper's "UR--JEPA"
# without requiring usetex.
REGULARIZER_DISPLAY = {
    "sigreg":             r"LeJEPA ($\mathcal{L}^{\text{SIGReg}}$)",
    "ur_cglt":            "UR–JEPA " + r"($\mathcal{L}^{\text{CGLT}}$)",
    "ur_cglt_deriv":      "UR–JEPA " + r"($\mathcal{L}^{\text{CGLT-deriv}}$)",
    "ur_cglt_deriv_raw":  "UR–JEPA " + r"($\mathcal{L}^{\text{CGLT-raw}}$)",
    "ur_beta":            "UR–JEPA " + r"($\mathcal{L}^{\beta}$)",
    "combined":           r"$\mathcal{L}^{\text{SIGReg}} + \mathcal{L}^{\text{CGLT}}$",
}


def beta_variant_display(cfg: dict) -> str:
    """Disambiguate ur_beta variants by which anti-collapse complement is on.

    cfg.regularizer == "ur_beta" for all variants; the difference is:
    * gamma_logtrace > 0  -> -gamma * mean log trace(S) intrinsic anti-collapse
    * lambda_ad > 0       -> external Ahlfors-David anchor
    * eigval_threshold > 0 -> adaptive tangent selection (typically paired
                              with log-trace)
    """
    gamma_lt = float(cfg.get("gamma_logtrace") or 0)
    lambda_ad = float(cfg.get("lambda_ad") or 0)
    eig_thresh = float(cfg.get("eigval_threshold") or 0)
    base = r"\mathcal{L}^{\beta}"
    suffixes = []
    if gamma_lt > 0:
        suffixes.append(r"\text{lt}")
    if lambda_ad > 0:
        suffixes.append(r"\text{AD}")
    if eig_thresh > 0:
        suffixes.append(r"\tau")
    if suffixes:
        loss = f"${base}_{{{','.join(suffixes)}}}$"
    else:
        loss = f"${base}$"
    return f"UR–JEPA ({loss})"


def label_for(path: Path, cfg: dict, include_seed: bool = False) -> str:
    """Generate a human-readable label for a run, naming the associated loss.

    Uses the pretty regularizer display name from REGULARIZER_DISPLAY
    so the legend reads like the paper (e.g., "SIGReg", "UR-CGLT (integral)")
    rather than the cfg string ("sigreg", "ur_cglt"). When ``include_seed``
    is True, appends ``(s<N>)`` for cross-seed comparisons. ``ur_beta``
    is disambiguated by which complement is active (see beta_variant_display).
    """
    reg = cfg.get("regularizer")
    if reg == "ur_beta":
        pretty = beta_variant_display(cfg)
    else:
        pretty = REGULARIZER_DISPLAY.get(reg, reg) if reg else None
    if not pretty:
        return path.parent.name
    if include_seed:
        seed = cfg.get("seed")
        if seed is None:
            m = SEED_SUFFIX_RE.search(path.parent.name)
            if m:
                seed = int(m.group(1))
        if seed is not None:
            return f"{pretty} (s{seed})"
    return pretty


def smooth(values: list[float], window: int) -> list[float]:
    """Centered rolling mean. window=1 -> identity."""
    if window <= 1 or not values:
        return values
    out = []
    half = window // 2
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def load_run(path: Path) -> tuple[dict, list[dict]]:
    """Return (config, epochs_list). Raises on malformed JSON."""
    data = json.loads(path.read_text())
    cfg = data.get("config", {})
    epochs = data.get("epochs", [])
    return cfg, epochs


def collect_runs(args) -> list[tuple[str, list[dict]]]:
    """Resolve --root and positional paths into a list of (label, epochs).

    If multiple runs share the same regularizer, auto-enables seed in
    the label so they remain distinguishable; otherwise legend shows
    only the pretty loss name. Override either way via --show-seed
    / --no-show-seed.
    """
    paths: list[Path] = []
    if args.root:
        root = Path(args.root)
        paths.extend(sorted(root.rglob("results.json")))
    for p in args.paths:
        paths.append(Path(p))

    # First pass: load and detect duplicated regularizers.
    loaded: list[tuple[Path, dict, list[dict]]] = []
    for p in paths:
        if not p.exists():
            print(f"<!-- skip: {p} does not exist -->", file=sys.stderr)
            continue
        try:
            cfg, epochs = load_run(p)
        except json.JSONDecodeError as e:
            print(f"<!-- skip: {p} ({e}) -->", file=sys.stderr)
            continue
        if not epochs:
            print(f"<!-- skip: {p} has no epochs yet -->", file=sys.stderr)
            continue
        loaded.append((p, cfg, epochs))

    # Decide whether to include seed in labels. Auto-enable seed only
    # when distinct seed-less labels would collide (e.g., 3 sigreg seeds
    # all label as "LeJEPA (...)"). Check on the *rendered* labels so
    # ur_beta variants that differ by complement (log-trace vs eigthresh)
    # are correctly treated as already-distinct.
    if args.show_seed is None:
        seedless_labels = [label_for(p, cfg, include_seed=False)
                           for p, cfg, _ in loaded]
        include_seed = len(seedless_labels) != len(set(seedless_labels))
    else:
        include_seed = args.show_seed

    return [(label_for(p, cfg, include_seed=include_seed), epochs)
            for p, cfg, epochs in loaded]


def make_default_outpath(runs: list[tuple[str, list[dict]]]) -> Path:
    """Synthesize an output PNG path from the input runs."""
    if not runs:
        return Path("trajectory.png")
    if len(runs) == 1:
        # Single run: name after the parent dir of its results.json
        return Path(f"trajectory_{runs[0][0].replace(' ', '_')}.png")
    return Path(f"trajectory_{len(runs)}runs.png")


def _layout(n_metrics: int) -> tuple[int, int, tuple[float, float]]:
    """Pick a (rows, cols, figsize) for N metric panels."""
    if n_metrics == 1:
        return (1, 1, (8, 5))
    if n_metrics == 2:
        return (1, 2, (13, 5))
    return (2, 2, (12, 9))   # 3 or 4 metrics -> 2x2 grid


def plot_metrics(runs, metric_keys, smooth_window, out_path, title):
    """Plot one or more metrics, one panel per metric."""
    rows, cols, figsize = _layout(len(metric_keys))
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else list(axes)
    for ax, metric_key in zip(axes_flat, metric_keys):
        label, log_y = METRICS[metric_key]
        for run_label, epochs in runs:
            xs = [ep["epoch"] for ep in epochs]
            ys = [ep.get(metric_key, float("nan")) for ep in epochs]
            ys_smoothed = smooth(ys, smooth_window)
            ax.plot(xs, ys_smoothed, label=run_label, linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.set_title(label)
    # Hide unused axes in the 2x2 grid (e.g., 3 metrics out of 4 slots)
    for ax in axes_flat[len(metric_keys):]:
        ax.set_visible(False)
    # Shared legend on the first panel
    axes_flat[0].legend(fontsize=9, loc="best")
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="Paths to results.json files")
    ap.add_argument("--root", type=str, default=None,
                    help="Root dir to recursively find */results.json")
    ap.add_argument("--metric", nargs="+", choices=list(METRICS.keys()), default=None,
                    help="One or more metrics to plot (default: 2x2 grid of all 4). "
                         "1 metric -> single panel; 2 metrics -> side-by-side; "
                         "3-4 metrics -> 2x2 grid.")
    ap.add_argument("--smooth", type=int, default=1,
                    help="Window size for centered rolling-mean smoothing (default 1 = no smoothing)")
    ap.add_argument("--out", type=str, default=None,
                    help="Output PNG path (default auto-named from inputs)")
    ap.add_argument("--title", type=str, default=None,
                    help="Optional figure title")
    ap.add_argument("--show-seed", dest="show_seed", action="store_true", default=None,
                    help="Include seed in legend labels (auto-on when same regularizer appears twice)")
    ap.add_argument("--no-show-seed", dest="show_seed", action="store_false",
                    help="Omit seed from labels even when multiple seeds present")
    args = ap.parse_args()

    runs = collect_runs(args)
    if not runs:
        print("No usable results.json files found.", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out) if args.out else make_default_outpath(runs)

    print(f"Plotting {len(runs)} run(s):")
    for label, epochs in runs:
        print(f"  - {label}: {len(epochs)} epochs")

    metric_keys = args.metric if args.metric else list(METRICS.keys())
    plot_metrics(runs, metric_keys, args.smooth, out_path, args.title)


if __name__ == "__main__":
    main()
