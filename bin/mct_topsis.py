#!/usr/bin/env python3

"""Objective-side-balanced TOPSIS over one candidate pool (paper Eq. 5-6).

Candidates: one per trained model, keyed (method, interval, date_tag) — the
paper's "date-tagged instances" are models, not events (AUTHOR_FEEDBACK.md). Each
candidate's criterion is its per-event score (already lead-averaged by
mct_verify) averaged over the benchmark events. Normalization and ideals are
fitted over THIS pool only (SPEC.md constraint 14): scores are never
comparable across pools.

Criteria (paper Table I):
  Benefit (higher better):  POD PSNR ACC CSI HSS GSS MCC F1 SEDI
  Cost (lower better):      RAPSD FAR FA CRPS Executing_Time
  Target-deviation (->1):   HK BIAS   (converted to |x-1|, cost side)

Weighting: each objective side gets total weight 0.5, split equally among
its active criteria. Vector normalization. No clipping, no epsilon
stabilization (legacy-faithful; SPEC constraint 14).

Output CSV: pool, method, interval, date_tag, n_events, topsis.
"""

import argparse
import csv
import logging
import sys
from collections import defaultdict

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BENEFIT = ["POD", "PSNR", "ACC", "CSI", "HSS", "GSS", "MCC", "F1", "SEDI"]
COST = ["RAPSD", "FAR", "FA", "CRPS", "Executing_Time"]
TARGET_ONE = ["HK", "BIAS"]


def main():
    parser = argparse.ArgumentParser(
        description="TOPSIS aggregation over one candidate pool")
    parser.add_argument("--pool", required=True)
    parser.add_argument("--metrics", action="append", required=True,
                        help="Per-(method, L) metrics CSV (repeatable)")
    parser.add_argument("--benchmark", required=True,
                        help="benchmark_sequences.npz: every candidate must "
                             "be scored on exactly its events")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with np.load(args.benchmark, allow_pickle=False) as data:
        expected = frozenset(str(e) for e in data["event_id"])

    # per_event[(method, interval, date_tag)][metric][event_id] = value
    per_event = defaultdict(lambda: defaultdict(dict))
    for path in args.metrics:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    value = float(row["value"])
                except ValueError:
                    continue
                per_event[(row["method"], row["interval"],
                           row["date_tag"])][
                    row["metric"]][row["event_id"]] = value

    # Average each metric over events. A candidate missing an event would
    # be scored on an easier or harder subset, and a pool missing one
    # everywhere would be ranked on less than the benchmark, so every
    # candidate's event set must be the benchmark's, exactly.
    event_sets = {k: frozenset().union(*(set(v) for v in m.values()))
                  for k, m in per_event.items()}
    wrong = {f"{k[0]} L={k[1] or '-'} {k[2]}": sorted(expected ^ v)
             for k, v in sorted(event_sets.items()) if v != expected}
    if wrong:
        logger.error("Pool %s: candidates not scored on exactly the %d "
                     "benchmark events (symmetric difference): %s",
                     args.pool, len(expected), wrong)
        with open(args.output, "w", newline="") as f:
            csv.writer(f).writerow(["pool", "method", "interval",
                                    "date_tag", "n_events", "topsis"])
        sys.exit(1)
    candidates = {
        k: {metric: float(np.mean(list(vals.values())))
            for metric, vals in m.items()}
        for k, m in per_event.items()
    }
    n_events = {k: len(v) for k, v in event_sets.items()}

    if not candidates:
        logger.error("No candidates in pool %s", args.pool)
        with open(args.output, "w", newline="") as f:
            csv.writer(f).writerow(["pool", "method", "interval",
                                    "date_tag", "n_events", "topsis"])
        sys.exit(1)

    keys = sorted(candidates)
    # Active criteria: present (finite) for every candidate in the pool.
    all_criteria = BENEFIT + COST + TARGET_ONE
    active = [
        c for c in all_criteria
        if all(np.isfinite(candidates[k].get(c, np.nan)) for k in keys)
    ]
    dropped = set(all_criteria) - set(active)
    if dropped:
        logger.warning("Dropped criteria with missing values: %s",
                       sorted(dropped))

    benefit = [c for c in active if c in BENEFIT]
    # HK/BIAS deviations |x-1| join the cost side (paper Eq. 5).
    cost = [c for c in active if c in COST or c in TARGET_ONE]
    if not benefit or not cost:
        logger.error("Pool %s lacks criteria on one objective side "
                     "(benefit=%s cost=%s)", args.pool, benefit, cost)
        sys.exit(1)

    # Build z matrix (M candidates x J criteria).
    z = np.zeros((len(keys), len(benefit) + len(cost)))
    criteria = benefit + cost
    for i, k in enumerate(keys):
        for j, c in enumerate(criteria):
            x = candidates[k][c]
            z[i, j] = abs(x - 1.0) if c in TARGET_ONE else x

    # Vector normalization, side-balanced weights (Eq. 5).
    norms = np.sqrt(np.sum(z ** 2, axis=0))
    r = z / norms  # no epsilon stabilization (legacy-faithful)
    w = np.array([0.5 / len(benefit)] * len(benefit)
                 + [0.5 / len(cost)] * len(cost))
    v = w[None, :] * r

    # Ideals and closeness (Eq. 6). All cost-side after conversion.
    n_b = len(benefit)
    v_pos = np.concatenate([v[:, :n_b].max(axis=0), v[:, n_b:].min(axis=0)])
    v_neg = np.concatenate([v[:, :n_b].min(axis=0), v[:, n_b:].max(axis=0)])
    d_pos = np.sqrt(np.sum((v - v_pos[None, :]) ** 2, axis=1))
    d_neg = np.sqrt(np.sum((v - v_neg[None, :]) ** 2, axis=1))
    closeness = d_neg / (d_pos + d_neg)

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pool", "method", "interval", "date_tag",
                         "n_events", "topsis"])
        for k, score in zip(keys, closeness):
            writer.writerow([args.pool, *k, n_events[k], float(score)])

    for k, score in zip(keys, closeness):
        method, interval, date_tag = k
        logger.info("%s: %s L=%s %s TOPSIS %.4f (%d events)", args.pool,
                    method, interval or "-", date_tag, float(score),
                    n_events[k])


if __name__ == "__main__":
    main()
