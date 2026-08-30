#!/usr/bin/env python3

"""Reproduce the paper's TOPSIS learning-curve figures (Figs. 4-6).

For each TOPSIS pool CSV: box plots of the date-tagged TOPSIS distribution
per (method, interval), median trend lines for the DGMR paradigms, and a
dotted horizontal reference line at the STEPS median.

Output: figures.tar.gz containing one PNG per pool plus a summary CSV of
per-(method, interval) medians.
"""

import argparse
import csv
import logging
import os
import tarfile
import tempfile
from collections import defaultdict

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_pool(path):
    """Return (pool_name, {(method, interval_or_None): [scores]})."""
    groups = defaultdict(list)
    pool = os.path.basename(path).split("_")[0]
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            pool = row["pool"]
            interval = int(row["interval"]) if row["interval"] else None
            groups[(row["method"], interval)].append(float(row["topsis"]))
    return pool, groups


def plot_pool(pool, groups, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = sorted({m for (m, L) in groups if L is not None})
    intervals = sorted({L for (m, L) in groups if L is not None})
    steps_scores = groups.get(("steps", None), [])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    n_m = max(len(methods), 1)
    width = 0.8 / n_m
    colors = plt.cm.tab10(np.linspace(0, 1, n_m))

    for mi, method in enumerate(methods):
        positions, data, medians = [], [], []
        for li, L in enumerate(intervals):
            scores = groups.get((method, L), [])
            if not scores:
                continue
            pos = li + (mi - (n_m - 1) / 2) * width
            positions.append(pos)
            data.append(scores)
            medians.append(np.median(scores))
        if not data:
            continue
        bp = ax.boxplot(data, positions=positions, widths=width * 0.9,
                        patch_artist=True, showfliers=False,
                        medianprops={"color": "black"})
        for box in bp["boxes"]:
            box.set_facecolor(colors[mi])
            box.set_alpha(0.6)
        ax.plot(positions, medians, "-o", color=colors[mi], markersize=3,
                label=method)

    if steps_scores:
        med = float(np.median(steps_scores))
        ax.axhline(med, linestyle=":", color="gray",
                   label=f"STEPS {med:.3f}")

    ax.set_xticks(range(len(intervals)))
    ax.set_xticklabels([str(L) for L in intervals])
    ax.set_xlabel("Span (months)")
    ax.set_ylabel("TOPSIS score")
    ax.set_ylim(0, 1)
    ax.set_title(f"Pool {pool.upper()}: TOPSIS learning curves")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Build TOPSIS learning-curve figures")
    parser.add_argument("--topsis", action="append", required=True,
                        help="Pool TOPSIS CSV (repeatable)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        summary_rows = []
        for path in args.topsis:
            pool, groups = load_pool(path)
            out_png = os.path.join(tmp, f"{pool}_learning_curves.png")
            try:
                plot_pool(pool, groups, out_png)
                logger.info("Wrote %s", out_png)
            except Exception as exc:  # noqa: BLE001
                logger.error("Plot failed for pool %s: %s", pool, exc)
            for (method, interval), scores in sorted(groups.items()):
                summary_rows.append([
                    pool, method, interval if interval is not None else "",
                    float(np.median(scores)), float(np.min(scores)),
                    float(np.max(scores)), len(scores),
                ])

        summary_csv = os.path.join(tmp, "topsis_summary.csv")
        with open(summary_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pool", "method", "interval", "median",
                             "min", "max", "n"])
            writer.writerows(summary_rows)

        with tarfile.open(args.output, "w:gz") as tar:
            for name in sorted(os.listdir(tmp)):
                tar.add(os.path.join(tmp, name), arcname=name)

    logger.info("Figures archive -> %s", args.output)


if __name__ == "__main__":
    main()
