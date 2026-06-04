"""
Author: Spatiolyx LLC, https://spatiolyx.ai
Date: May 30, 2026

Aggregate downstream linear-probe results into a per-dataset summary.

Reads ``results.json`` from each ``<root>/<method_tag>_s<seed>/``
directory written by ``eval_downstream.py``, groups by ``method_tag``,
and produces:

1. **Per-run table** -- one row per (method_tag, seed) with epochs,
   wall time, final_acc, best_acc. For transparency / spotting
   outliers before they hide in the mean.
2. **Mean ± std table** -- one row per method_tag, computed across
   the seeds that ran. Headline numbers for the report.
3. **Paired-t tests** for the canonical UR-CGLT vs SIGReg comparisons
   that share matched seeds (D=16 matched-n=8, D=16 at UR-CGLT optimum
   n=7, D=32 matched). Output: mean Delta, std Delta, t-statistic on
   (n_seeds-1) dof. Significance lookup is left to the reader (use
   scipy.stats.t.sf if you need a p-value).

Folder name convention (set by EVAL_PLAN.md and the run_downstream_*
sbatches):

    <method_tag>_s<seed>/   e.g.  ur_cglt_d32_n7_s0/

Usage:
    python scripts/aggregate_downstream.py \\
        $SCRATCH/projects/UR-JEPA/runs/downstream_aircraft

Optional CSV output:
    python scripts/aggregate_downstream.py <root> --csv out.csv
"""

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


# `<method_tag>_s<seed>` with seed = nonneg int.
DIR_RE = re.compile(r"^(?P<tag>.+)_s(?P<seed>\d+)$")

# Canonical sort order for method_tags. Anything not listed sorts after,
# alphabetically. Keep this in sync with EVAL_PLAN.md "Sbatch structure"
# section.
TAG_SORT_ORDER = [
    "sigreg_d16",
    "sigreg_d32",
    "ur_cglt_d16_n8",
    "ur_cglt_d16_n7",
    "ur_cglt_d32_n7",
    "combined_d16_a11",
]

# Canonical paired comparisons. Each is (label, baseline_tag, candidate_tag).
# Reported delta = candidate - baseline (positive means candidate beats
# baseline). Emitted only if both sides have >= 2 seeds in common.
PAIRED_COMPARISONS = [
    ("D=16, matched-n=8", "sigreg_d16", "ur_cglt_d16_n8"),
    ("D=16, UR-CGLT optimum n=7", "sigreg_d16", "ur_cglt_d16_n7"),
    ("D=32, matched n=7", "sigreg_d32", "ur_cglt_d32_n7"),
    ("D=16, Combined vs SIGReg", "sigreg_d16", "combined_d16_a11"),
    ("D=16, Combined vs UR-CGLT n=8", "ur_cglt_d16_n8", "combined_d16_a11"),
]


def parse_dir_name(name: str):
    m = DIR_RE.match(name)
    if not m:
        return None, None
    return m.group("tag"), int(m.group("seed"))


def tag_sort_key(tag: str):
    if tag in TAG_SORT_ORDER:
        return (0, TAG_SORT_ORDER.index(tag))
    return (1, tag)


def mean_std(values):
    """Returns (mean, std) using sample stdev (n-1). std=0 if n<2."""
    if not values:
        return float("nan"), float("nan")
    m = statistics.fmean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return m, s


def paired_t(deltas):
    """One-sample t against mean=0 on the differences. Returns
    (mean, std, t, df). df = len(deltas) - 1; t = mean / SE."""
    if len(deltas) < 2:
        return statistics.fmean(deltas) if deltas else float("nan"), float("nan"), float("nan"), 0
    m = statistics.fmean(deltas)
    s = statistics.stdev(deltas)
    se = s / math.sqrt(len(deltas))
    t = m / se if se > 0 else float("inf") if m != 0 else 0.0
    df = len(deltas) - 1
    return m, s, t, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="Output root, e.g., runs/downstream_aircraft")
    ap.add_argument("--csv", type=Path, default=None, help="Optional CSV output")
    args = ap.parse_args()

    # Discover (method_tag, seed) -> results dict.
    groups = defaultdict(dict)            # tag -> {seed: results}
    dataset_name = None
    for results_file in sorted(args.root.rglob("results.json")):
        run_dir = results_file.parent
        tag, seed = parse_dir_name(run_dir.name)
        if tag is None:
            print(f"<!-- skipping {run_dir.name}: not in <tag>_s<seed> form -->")
            continue
        try:
            res = json.loads(results_file.read_text())
        except json.JSONDecodeError as e:
            print(f"<!-- skipping {results_file}: {e} -->")
            continue
        if dataset_name is None:
            dataset_name = res.get("dataset")
        if seed in groups[tag]:
            print(f"<!-- WARN: duplicate (tag={tag}, seed={seed}); using latest -->")
        groups[tag][seed] = res

    if not groups:
        print(f"# No runs found under {args.root}")
        return

    sorted_tags = sorted(groups.keys(), key=tag_sort_key)

    # ---- Header --------------------------------------------------------
    print(f"# Downstream eval -- {dataset_name or '?'} -- {args.root}\n")

    # ---- Per-run -------------------------------------------------------
    print("## Per-run\n")
    print("| method_tag | seed | epochs | wall (h) | final_acc | best_acc |")
    print("|---|---|---|---|---|---|")
    for tag in sorted_tags:
        for seed in sorted(groups[tag].keys()):
            res = groups[tag][seed]
            epochs = len(res.get("epochs", []))
            wall_s = res.get("wall_seconds", float("nan"))
            wall_h = wall_s / 3600.0 if isinstance(wall_s, (int, float)) else float("nan")
            final = res.get("final_acc", float("nan"))
            best = res.get("best_acc", float("nan"))
            wall_str = f"{wall_h:.2f}" if wall_h == wall_h else "nan"
            print(f"| {tag} | {seed} | {epochs} | {wall_str} | {final:.4f} | {best:.4f} |")
    print()

    # ---- Mean ± std --------------------------------------------------
    print("## Across seeds (mean ± std)\n")
    print("| method_tag | n_seeds | final_acc | best_acc |")
    print("|---|---|---|---|")
    for tag in sorted_tags:
        bests = [r.get("best_acc", float("nan")) for r in groups[tag].values()]
        finals = [r.get("final_acc", float("nan")) for r in groups[tag].values()]
        bests = [b for b in bests if isinstance(b, (int, float)) and b == b]
        finals = [f for f in finals if isinstance(f, (int, float)) and f == f]
        if not bests:
            continue
        fm, fs = mean_std(finals)
        bm, bs = mean_std(bests)
        print(f"| {tag} | {len(bests)} | {fm:.4f} ± {fs:.4f} | {bm:.4f} ± {bs:.4f} |")
    print()

    # ---- Paired comparisons -------------------------------------------
    any_paired = False
    paired_rows = []
    for label, sig_tag, ur_tag in PAIRED_COMPARISONS:
        if sig_tag not in groups or ur_tag not in groups:
            continue
        shared_seeds = sorted(set(groups[sig_tag].keys()) & set(groups[ur_tag].keys()))
        if len(shared_seeds) < 2:
            continue
        deltas = [
            groups[ur_tag][s].get("best_acc", float("nan"))
            - groups[sig_tag][s].get("best_acc", float("nan"))
            for s in shared_seeds
        ]
        deltas = [d for d in deltas if d == d]
        if len(deltas) < 2:
            continue
        m, s, t, df = paired_t(deltas)
        paired_rows.append((label, sig_tag, ur_tag, len(deltas), m, s, t, df))
        any_paired = True

    if any_paired:
        print("## Paired comparisons (matched seeds, best_acc; delta = candidate - baseline)\n")
        print("| comparison | baseline | candidate | n_seeds | mean Delta | std Delta | t | df |")
        print("|---|---|---|---|---|---|---|---|")
        for label, sig_tag, ur_tag, n, m, s, t, df in paired_rows:
            sign = "+" if m >= 0 else ""
            print(f"| {label} | {sig_tag} | {ur_tag} | {n} | {sign}{m:.4f} | {s:.4f} | {t:+.2f} | {df} |")
        print()
        print("Lookup (one-sided): df=2 -> |t|>=2.92 for p<0.05, |t|>=6.96 for p<0.01.")
        print("                    df=3 -> |t|>=2.35 for p<0.05, |t|>=4.54 for p<0.01.")
        print()

    # ---- Optional CSV --------------------------------------------------
    if args.csv:
        import csv
        with args.csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dataset", "method_tag", "seed", "epochs",
                        "wall_seconds", "final_acc", "best_acc"])
            for tag in sorted_tags:
                for seed in sorted(groups[tag].keys()):
                    res = groups[tag][seed]
                    w.writerow([
                        dataset_name, tag, seed,
                        len(res.get("epochs", [])),
                        res.get("wall_seconds", ""),
                        res.get("final_acc", ""),
                        res.get("best_acc", ""),
                    ])
        print(f"<!-- wrote CSV to {args.csv} -->")


if __name__ == "__main__":
    main()
