#!/usr/bin/env python3

"""Project full-study wall-clock from a measured Fed-Cast timing run.

Reads per-transformation runtime statistics from a completed Pegasus run
(via `pegasus-statistics`) and extrapolates to a target configuration —
e.g. from the E1-lite timing run (2 intervals, 6 rounds, 64 sequences per
client) to the paper's full E1 (6 intervals, 100 rounds, real sequence
counts).

Cost model
----------
Per-job runtime is split into a fixed part (container start, 582 MB model
load/save, staging) and a training part proportional to the number of
sequences processed:

    fl_train_client(n) ~= fixed + per_seq * n

`fixed` is estimated from fl_aggregate/fl_init, which move models of the
same size but do no training. Round wall-clock accounts for GPU
concurrency: with C clients and G usable GPU slots, client jobs run in
ceil(C / G) waves.

Usage:
    tools/timing_extrapolate.py RUN_DIR \
        --measured-rounds 6 --measured-intervals 2 --measured-seq-cap 64 \
        --target-rounds 100 --target-intervals 6 --target-seq 600 \
        --clients 7 --gpu-slots 2
"""

import argparse
import math
import re
import subprocess
import sys


def parse_statistics(run_dir):
    """Return {transformation: (count, mean_runtime_s)} via pegasus-statistics."""
    try:
        out = subprocess.run(
            ["pegasus-statistics", "-s", "all", run_dir],
            capture_output=True, text=True, timeout=600,
        )
    except FileNotFoundError:
        print("error: pegasus-statistics not on PATH", file=sys.stderr)
        sys.exit(1)
    text = out.stdout + out.stderr

    stats = {}
    # Breakdown rows look like:
    #   <transformation> <count> <succeeded> <failed> <min> <max> <mean> <total>
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 8:
            continue
        name = parts[0]
        if not re.match(r"^[a-zA-Z][\w.:-]*$", name):
            continue
        try:
            count = int(parts[1])
            mean = float(parts[-2])
        except ValueError:
            continue
        stats[name] = (count, mean)
    return stats, text


def fmt_hours(seconds):
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    if seconds < 86400:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} days"


def main():
    p = argparse.ArgumentParser(
        description="Extrapolate full-study wall-clock from a timing run")
    p.add_argument("run_dir")
    p.add_argument("--measured-rounds", type=int, required=True)
    p.add_argument("--measured-intervals", type=int, required=True)
    p.add_argument("--measured-seq-cap", type=int, required=True,
                   help="--limit-train-sequences used in the timing run")
    p.add_argument("--target-rounds", type=int, default=100)
    p.add_argument("--target-intervals", type=int, default=6)
    p.add_argument("--target-seq", type=int, default=600,
                   help="Expected train sequences per client at full scale")
    p.add_argument("--clients", type=int, default=7)
    p.add_argument("--gpu-slots", type=int, default=2,
                   help="Concurrent GPU jobs the pool can run")
    p.add_argument("--validate-every", type=int, default=5)
    args = p.parse_args()

    stats, raw = parse_statistics(args.run_dir)
    if not stats:
        print("Could not parse any transformation statistics. Raw output:\n")
        print(raw[:4000])
        sys.exit(1)

    print("=" * 68)
    print("MEASURED (this run)")
    print("=" * 68)
    print(f"{'transformation':<24}{'count':>7}{'mean runtime':>18}")
    for name in sorted(stats):
        count, mean = stats[name]
        print(f"{name:<24}{count:>7}{fmt_hours(mean):>18}")

    def mean_of(*names, default=None):
        for n in names:
            if n in stats:
                return stats[n][1]
        return default

    train_client = mean_of("fl_train_client")
    aggregate = mean_of("fl_aggregate")
    validate = mean_of("fl_validate")
    centralized = mean_of("train_dgmr")

    if train_client is None:
        print("\nNo fl_train_client rows — cannot project federated cost.")
        sys.exit(1)

    # Fixed (model I/O + container + staging) estimated from the
    # aggregation job, which moves models but does not train.
    fixed = aggregate if aggregate is not None else 0.0
    fixed = min(fixed, train_client)          # never exceed the whole job
    per_seq = (train_client - fixed) / max(args.measured_seq_cap, 1)

    print()
    print("=" * 68)
    print("COST MODEL")
    print("=" * 68)
    print(f"fixed per client job (model I/O + staging): {fmt_hours(fixed)}")
    print(f"training cost per sequence per epoch:       "
          f"{per_seq:.2f} s")
    print(f"projected client job at {args.target_seq} seqs:            "
          f"{fmt_hours(fixed + per_seq * args.target_seq)}")

    waves = math.ceil(args.clients / max(args.gpu_slots, 1))
    client_job = fixed + per_seq * args.target_seq
    round_cost = waves * client_job + (aggregate or 0.0)
    val_rounds = args.target_rounds // max(args.validate_every, 1)
    fed_per_interval = (args.target_rounds * round_cost
                        + val_rounds * (validate or 0.0))

    # Centralized: measured job covers a segment of epochs over the pooled
    # (clients * cap) sequences; scale to pooled real counts.
    cen_per_interval = 0.0
    if centralized is not None:
        measured_pooled = args.clients * args.measured_seq_cap
        target_pooled = args.clients * args.target_seq
        # measured job = one segment; total segments in the timing run is
        # implicit in the count, so use per-epoch cost via the round count.
        cen_per_epoch = centralized / max(
            args.measured_rounds / max(stats["train_dgmr"][0]
                                       / args.measured_intervals, 1), 1)
        cen_per_interval = (args.target_rounds * cen_per_epoch
                            * target_pooled / max(measured_pooled, 1))

    print()
    print("=" * 68)
    print(f"PROJECTED — {args.target_intervals} intervals x "
          f"{args.target_rounds} rounds, {args.clients} clients, "
          f"{args.gpu_slots} GPU slots")
    print("=" * 68)
    print(f"federated, per interval:   {fmt_hours(fed_per_interval)} "
          f"({waves} GPU waves/round)")
    print(f"federated, all intervals:  "
          f"{fmt_hours(fed_per_interval * args.target_intervals)}")
    if cen_per_interval:
        print(f"centralized, per interval: {fmt_hours(cen_per_interval)}")
        print(f"centralized, all:          "
              f"{fmt_hours(cen_per_interval * args.target_intervals)}")
    total = (fed_per_interval + cen_per_interval) * args.target_intervals
    print(f"E1 TOTAL (serial chains):  {fmt_hours(total)}")
    print()
    print("Caveats: intervals can overlap in the pool, so the total is an "
          "upper bound if GPU slots are free; conversely GPU contention "
          "between concurrent intervals is not modelled. Sequence scaling "
          "assumes training cost is linear in sequence count.")


if __name__ == "__main__":
    main()
