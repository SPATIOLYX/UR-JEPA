"""
Author: Spatiolyx LLC, https://spatiolyx.ai
Date: May 30, 2026

Cross-dataset transfer matrix from downstream linear-probe results.

Sibling to ``scripts/aggregate_downstream.py``. That one produces a
per-dataset summary (one root, one target dataset). This one walks a
two-level directory tree where the first level is the target dataset
and the second level is the source-method × eval-seed:

    <root>/<target_dataset>/<source_method>_s<eval_seed>/results.json

and produces a transfer matrix:

* Rows: source methods (with seeds aggregated as mean ± std)
* Columns: target datasets
* Cells: best linear-probe accuracy

Layout convention matches ``run_downstream_inet100.sbatch``:

    runs/inet100_downstream/
    ├── aircraft/sigreg_s0/results.json
    ├── aircraft/ur_cglt_s0/results.json
    ├── cifar100/sigreg_s0/results.json
    ├── cifar100/ur_cglt_s0/results.json
    ├── dtd/...
    └── flowers/...

Usage:
    python scripts/aggregate_transfer_matrix.py runs/inet100_downstream
    python scripts/aggregate_transfer_matrix.py runs/inet100_downstream --csv summary.csv
    python scripts/aggregate_transfer_matrix.py runs/inet100_downstream --metric final
"""

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


SEED_SUFFIX_RE = re.compile(r"_s(\d+)$")


def parse_method_seed(dir_name: str) -> tuple[str, int]:
    """Parse 'sigreg_s0' -> ('sigreg', 0); 'ur_cglt_d32_n7_s2' -> ('ur_cglt_d32_n7', 2)."""
    m = SEED_SUFFIX_RE.search(dir_name)
    if not m:
        return (dir_name, -1)
    return (dir_name[: m.start()], int(m.group(1)))


def safe_best_acc(data: dict) -> float:
    if data.get("best_acc") is not None:
        return float(data["best_acc"])
    epochs = data.get("epochs", [])
    if not epochs:
        return float("nan")
    return max((e.get("test_acc", float("nan")) for e in epochs), default=float("nan"))


def safe_final_acc(data: dict) -> float:
    if data.get("final_acc") is not None:
        return float(data["final_acc"])
    epochs = data.get("epochs", [])
    if not epochs:
        return float("nan")
    return float(epochs[-1].get("test_acc", float("nan")))


def fmt_cell(values: list[float]) -> str:
    """'mean ± std (n)' for n>=2, just the value for n=1, '—' for empty."""
    values = [v for v in values if isinstance(v, float) and v == v]
    if not values:
        return "—"
    if len(values) == 1:
        return f"{values[0]:.4f}"
    mean = statistics.fmean(values)
    std = statistics.stdev(values)
    return f"{mean:.4f} ± {std:.4f} ({len(values)})"


def fmt_delta(ur_vals: list[float], sig_vals: list[float]) -> str:
    """Per-cell mean Δ = ur − sig. '—' if either side has no values."""
    ur_vals = [v for v in ur_vals if isinstance(v, float) and v == v]
    sig_vals = [v for v in sig_vals if isinstance(v, float) and v == v]
    if not ur_vals or not sig_vals:
        return "—"
    d = statistics.fmean(ur_vals) - statistics.fmean(sig_vals)
    return f"{d:+.4f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path,
                    help="Output root (e.g. runs/inet100_downstream)")
    ap.add_argument("--csv", type=Path, default=None,
                    help="Optional CSV output for the transfer matrix")
    ap.add_argument("--metric", choices=["best", "final"], default="best",
                    help="Which accuracy to put in the transfer matrix (default: best)")
    args = ap.parse_args()

    metric_fn = safe_best_acc if args.metric == "best" else safe_final_acc

    # cells[(method, dataset)] = list of best/final accuracies across seeds
    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    # per_run: list of (method, seed, dataset, best, final, n_epochs) for the per-run table
    per_run: list[tuple[str, int, str, float, float, int]] = []
    datasets_seen: set[str] = set()
    methods_seen: set[str] = set()

    if not args.root.exists():
        print(f"# {args.root} does not exist")
        return

    # Walk: <root>/<dataset>/<method_seed>/results.json
    for dataset_dir in sorted(p for p in args.root.iterdir() if p.is_dir()):
        dataset = dataset_dir.name
        for method_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            results_file = method_dir / "results.json"
            if not results_file.exists():
                continue
            try:
                data = json.loads(results_file.read_text())
            except json.JSONDecodeError:
                print(f"<!-- skip: could not parse {results_file} -->")
                continue
            method, seed = parse_method_seed(method_dir.name)
            best = safe_best_acc(data)
            final = safe_final_acc(data)
            n_epochs = len(data.get("epochs", []))
            cells[(method, dataset)].append(metric_fn(data))
            per_run.append((method, seed, dataset, best, final, n_epochs))
            datasets_seen.add(dataset)
            methods_seen.add(method)

    if not per_run:
        print(f"# No results.json files found under {args.root}")
        return

    datasets = sorted(datasets_seen)
    # Put sigreg first if present, then ur_*, then everything else.
    def method_sort_key(m: str):
        if m == "sigreg":
            return (0, m)
        if m.startswith("ur_"):
            return (1, m)
        return (2, m)
    methods = sorted(methods_seen, key=method_sort_key)

    print(f"# Downstream transfer matrix — {args.root}\n")

    # ----- Per-run table -----
    print("## Per-run\n")
    print("| method | seed | dataset | epochs | final_acc | best_acc |")
    print("|---|---|---|---|---|---|")
    for method, seed, dataset, best, final, n_epochs in sorted(per_run):
        print(f"| {method} | {seed} | {dataset} | {n_epochs} | {final:.4f} | {best:.4f} |")

    # ----- Transfer matrix (method × dataset → metric mean ± std) -----
    print(f"\n## Transfer matrix ({args.metric}_acc, mean ± std across source seeds)\n")
    header = "| method (source pretrain) | " + " | ".join(datasets) + " | mean |"
    print(header)
    print("|" + "---|" * (len(datasets) + 2))
    for m in methods:
        row_cells = [fmt_cell(cells[(m, d)]) for d in datasets]
        # Row mean: across-dataset mean of the per-method per-dataset means
        per_dataset_means = []
        for d in datasets:
            vals = [v for v in cells[(m, d)] if v == v]
            if vals:
                per_dataset_means.append(statistics.fmean(vals))
        row_mean = f"{statistics.fmean(per_dataset_means):.4f}" if per_dataset_means else "—"
        print(f"| {m} | " + " | ".join(row_cells) + f" | {row_mean} |")

    # ----- Per-cell pairwise Δ vs sigreg (if both sides present) -----
    if "sigreg" in methods_seen and any(m.startswith("ur_") for m in methods_seen):
        ur_methods = [m for m in methods if m.startswith("ur_")]
        print(f"\n## Pairwise Δ vs sigreg ({args.metric}_acc, positive = UR variant wins)\n")
        header = "| ur variant | " + " | ".join(datasets) + " | mean Δ |"
        print(header)
        print("|" + "---|" * (len(datasets) + 2))
        for m in ur_methods:
            deltas_per_dataset = []
            row_cells = []
            for d in datasets:
                sig_vals = [v for v in cells[("sigreg", d)] if v == v]
                ur_vals = [v for v in cells[(m, d)] if v == v]
                row_cells.append(fmt_delta(ur_vals, sig_vals))
                if sig_vals and ur_vals:
                    deltas_per_dataset.append(statistics.fmean(ur_vals) - statistics.fmean(sig_vals))
            mean_delta = f"{statistics.fmean(deltas_per_dataset):+.4f}" if deltas_per_dataset else "—"
            print(f"| {m} | " + " | ".join(row_cells) + f" | {mean_delta} |")

    # ----- Optional CSV export -----
    if args.csv:
        import csv
        with args.csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["method"] + datasets + ["row_mean"])
            for m in methods:
                row: list[str] = [m]
                vals_for_mean: list[float] = []
                for d in datasets:
                    vals = [v for v in cells[(m, d)] if v == v]
                    if vals:
                        row.append(f"{statistics.fmean(vals):.4f}")
                        vals_for_mean.append(statistics.fmean(vals))
                    else:
                        row.append("")
                row.append(f"{statistics.fmean(vals_for_mean):.4f}" if vals_for_mean else "")
                w.writerow(row)
            w.writerow([])
            w.writerow(["dataset_n_seeds_summary"] + datasets)
            for m in methods:
                row = [m] + [str(len(cells[(m, d)])) for d in datasets]
                w.writerow(row)
        print(f"\nCSV written to {args.csv}")


if __name__ == "__main__":
    main()
