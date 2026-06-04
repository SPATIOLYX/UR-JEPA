"""
Author: Spatiolyx LLC, https://spatiolyx.ai
Date: May 30, 2026

Aggregate UR-JEPA run results into a comparison table.

Walks an output root, reads each ``results.json``, and prints a markdown
table grouped by the full hyperparameter tuple
``(regularizer, lamb, projector_norm, reg_scale, lambda_ad, n, accum_steps)``
with mean ± std of final + best probe accuracy across seeds. Optionally
writes the same table as CSV.

Usage:
    python scripts/aggregate_results.py $SCRATCH/projects/UR-JEPA/runs/tier_b
    python scripts/aggregate_results.py $SCRATCH/projects/UR-JEPA/runs/tier_b --csv summary.csv
"""

import argparse
import json
import math
import statistics
from pathlib import Path
from collections import defaultdict


# Order matters: this is the column order in tables and CSV.
# Each entry: (cfg_key, table_label, default_if_missing)
KEY_FIELDS = [
    ("regularizer",     "regularizer", "?"),
    ("lamb",            "lamb",        None),
    ("reg_scale",       "scale",       1.0),
    ("lambda_ad",       "λ_AD",        None),   # ur_cglt/ur_beta; None for sigreg
    ("gamma_logtrace",  "γ_lt",        None),   # ur_beta intrinsic anti-collapse; None otherwise
    ("eigval_threshold","τ_eig",       None),   # ur_beta adaptive tangent threshold; None otherwise
    ("projector_norm",  "norm",        "bn"),
    ("proj_dim",        "D",           None),   # projector output dim (16/32/512/...)
    ("n",               "n",           None),   # ur_*-only; None for sigreg
    ("n_scales",        "K",           None),   # ur_*-only; None for sigreg
    ("accum_steps",     "accum",       1),
    ("alpha_sigreg",    "α_sig",       None),   # combined-only; None for solo regs
    ("alpha_ur",        "α_ur",        None),   # combined-only; None for solo regs
]


def make_key(cfg: dict) -> tuple:
    """Extract the grouping key from a run config, applying defaults."""
    return tuple(cfg.get(k, default) for k, _, default in KEY_FIELDS)


def key_sort(k: tuple) -> tuple:
    """Stable, NaN/None-safe ordering across heterogeneous keys."""
    out = []
    for value in k:
        if value is None:
            out.append(-1)
        elif isinstance(value, (int, float)):
            out.append(float(value))
        else:
            out.append(str(value))
    return tuple(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="Output root (parent of <run>/ dirs)")
    ap.add_argument("--csv", type=Path, default=None, help="Optional CSV output path")
    args = ap.parse_args()

    groups = defaultdict(list)
    for results_file in sorted(args.root.rglob("results.json")):
        try:
            data = json.loads(results_file.read_text())
        except json.JSONDecodeError:
            print(f"[skip] could not parse {results_file}")
            continue

        cfg = data.get("config", {})
        key = make_key(cfg)
        seed = data.get("seed", cfg.get("seed", -1))

        epochs = data.get("epochs", [])
        final_acc = data.get("final_acc", epochs[-1]["test_acc"] if epochs else float("nan"))
        best_acc = data.get("best_acc", max((e["test_acc"] for e in epochs), default=float("nan")))
        wall = data.get("wall_seconds", float("nan"))

        groups[key].append({
            "run": results_file.parent.name,
            "seed": seed,
            "final_acc": final_acc,
            "best_acc": best_acc,
            "wall_h": wall / 3600 if wall == wall else float("nan"),
            "epochs_done": len(epochs),
        })

    if not groups:
        print(f"No results.json files found under {args.root}")
        return

    labels = [label for _, label, _ in KEY_FIELDS]
    cfg_keys = [k for k, _, _ in KEY_FIELDS]

    # Per-run detail
    print(f"# Results — {args.root}\n")
    print("## Per-run\n")
    header_cols = labels + ["seed", "epochs", "wall (h)", "final acc", "best acc"]
    print("| " + " | ".join(header_cols) + " |")
    print("|" + "|".join(["---"] * len(header_cols)) + "|")
    rows_csv = [tuple(cfg_keys) + ("seed", "epochs", "wall_h", "final_acc", "best_acc")]
    for k in sorted(groups, key=key_sort):
        for r in sorted(groups[k], key=lambda x: x["seed"]):
            cells = list(k) + [
                r["seed"], r["epochs_done"], f"{r['wall_h']:.2f}",
                f"{r['final_acc']:.4f}", f"{r['best_acc']:.4f}",
            ]
            print("| " + " | ".join(str(c) for c in cells) + " |")
            rows_csv.append(tuple(k) + (
                r["seed"], r["epochs_done"], f"{r['wall_h']:.3f}",
                f"{r['final_acc']:.4f}", f"{r['best_acc']:.4f}",
            ))

    # Aggregated across seeds, per condition
    print("\n## Across seeds (mean ± std)\n")
    header_cols = labels + ["n_seeds", "final acc", "best acc"]
    print("| " + " | ".join(header_cols) + " |")
    print("|" + "|".join(["---"] * len(header_cols)) + "|")
    rows_csv.append(())
    rows_csv.append(tuple(cfg_keys) + ("n_seeds", "final_mean", "final_std", "best_mean", "best_std"))
    for k in sorted(groups, key=key_sort):
        runs = groups[k]
        n_seeds = len(runs)
        finals = [r["final_acc"] for r in runs if not math.isnan(r["final_acc"])]
        bests = [r["best_acc"] for r in runs if not math.isnan(r["best_acc"])]
        fm = statistics.mean(finals) if finals else float("nan")
        fs = statistics.stdev(finals) if len(finals) > 1 else 0.0
        bm = statistics.mean(bests) if bests else float("nan")
        bs = statistics.stdev(bests) if len(bests) > 1 else 0.0
        cells = list(k) + [n_seeds, f"{fm:.4f} ± {fs:.4f}", f"{bm:.4f} ± {bs:.4f}"]
        print("| " + " | ".join(str(c) for c in cells) + " |")
        rows_csv.append(tuple(k) + (
            n_seeds, f"{fm:.4f}", f"{fs:.4f}", f"{bm:.4f}", f"{bs:.4f}",
        ))

    # Sweep summary: for each regularizer, find the (lamb, reg_scale,
    # lambda_ad, n, accum) combination with the highest best_acc.
    by_reg_norm = defaultdict(list)
    for k, runs in groups.items():
        reg = k[0]
        pnorm = k[4]
        lamb = k[1]
        if lamb is None:
            continue
        best_acc = max((r["best_acc"] for r in runs), default=float("nan"))
        by_reg_norm[(reg, pnorm)].append((k, best_acc))
    if any(len(v) > 1 for v in by_reg_norm.values()):
        print("\n## Best configuration per (regularizer, norm) by best_acc\n")
        tuned_labels = [labels[i] for i in (1, 2, 3, 5, 6)]  # lamb, scale, λ_AD, n, accum
        header_cols = ["regularizer", "norm"] + [f"best {lab}" for lab in tuned_labels] + ["best acc"]
        print("| " + " | ".join(header_cols) + " |")
        print("|" + "|".join(["---"] * len(header_cols)) + "|")
        for (reg, pnorm) in sorted(by_reg_norm):
            k_best, acc = max(by_reg_norm[(reg, pnorm)],
                              key=lambda x: (x[1] if x[1] == x[1] else -1))
            tuned_vals = [k_best[i] for i in (1, 2, 3, 5, 6)]
            cells = [reg, pnorm] + tuned_vals + [f"{acc:.4f}"]
            print("| " + " | ".join(str(c) for c in cells) + " |")

    if args.csv:
        import csv
        with args.csv.open("w", newline="") as f:
            w = csv.writer(f)
            for row in rows_csv:
                w.writerow(row)
        print(f"\nCSV written to {args.csv}")


if __name__ == "__main__":
    main()
