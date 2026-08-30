#!/usr/bin/env python3

"""Build 16-frame training sequences for one site from monthly cropped NetCDFs.

Implements SPEC.md constraints 4-5:
  - Sequences: 16 contiguous frames at 2-min cadence — 4 input + 12 target.
  - Precipitation-content filter: a sequence is retained if at least
    `--min-rain-fraction` of pixels exceed `--rain-threshold` mm/h in at
    least one frame (calibration knob — SPEC open question 2; tune against
    the paper's published retained-sequence counts).
  - Split rule: test = first three available days of each month;
    validation = fixed-seed sample of the remainder, sized ~equal to test;
    everything else = train.

Outputs:
  - {site}_sequences.npz : arrays `sequences` (N, 16, 300, 300) float16,
    `start_epoch` (N,), `split` (N,) in {0=train, 1=val, 2=test}
  - {site}_manifest.json : split membership, retention stats, SHA-256 of the
    sequence array — the Tier-0 determinism artifact (SPEC Sec. 5).
"""

import argparse
import hashlib
import json
import logging
import sys
import time

import numpy as np
from netCDF4 import Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SEQ_LEN = 16
NOMINAL_CADENCE_S = 120  # 2-minute nominal cadence (paper)
TEST_DAYS = 3            # first N days of each month are test


def load_month(path):
    """Return (times, frames) from one monthly NetCDF; frames NaN-filled."""
    nc = Dataset(path, "r")
    try:
        times = nc.variables["time"][:].astype(np.float64)
        frames = nc.variables["precip_rate"][:].astype(np.float32)
        frames = np.ma.filled(frames, np.nan)
    finally:
        nc.close()
    return times, frames


def main():
    parser = argparse.ArgumentParser(
        description="Build 16-frame sequences and splits for one site")
    parser.add_argument("--site", required=True)
    parser.add_argument("--input", action="append", required=True,
                        help="Monthly cropped NetCDF (repeatable)")
    parser.add_argument("--output-sequences", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--val-seed", type=int, required=True)
    parser.add_argument("--rain-threshold", type=float, default=0.1)
    parser.add_argument("--min-rain-fraction", type=float, default=0.05)
    args = parser.parse_args()

    all_times, all_frames = [], []
    for path in sorted(args.input):
        try:
            times, frames = load_month(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read %s: %s — skipping month",
                           path, exc)
            continue
        if len(times) == 0:
            logger.warning("%s is empty — skipping month", path)
            continue
        all_times.append(times)
        all_frames.append(frames)

    if not all_times:
        logger.error("No usable input months for %s", args.site)
        # Write declared outputs before failing (SPEC constraint 17).
        np.savez_compressed(args.output_sequences,
                            sequences=np.zeros((0,), dtype=np.float16),
                            start_epoch=np.zeros((0,)),
                            split=np.zeros((0,), dtype=np.int8))
        with open(args.output_manifest, "w") as f:
            json.dump({"site": args.site, "error": "no input data"}, f)
        sys.exit(1)

    times = np.concatenate(all_times)
    frames = np.concatenate(all_frames, axis=0)
    order = np.argsort(times)
    times, frames = times[order], frames[order]
    logger.info("%s: %d frames total", args.site, len(times))

    # Effective cadence: inferred from the data so that strided pilot
    # ingests still form sequences. Full-cadence runs infer ~120 s; a
    # deviation from the paper's 2-min cadence is recorded in the manifest.
    if len(times) > 1:
        cadence_s = float(np.median(np.diff(times)))
    else:
        cadence_s = float(NOMINAL_CADENCE_S)
    cadence_tol_s = 0.25 * cadence_s
    if abs(cadence_s - NOMINAL_CADENCE_S) > 1.0:
        logger.warning("%s: effective cadence %.0f s != nominal %d s "
                       "(pilot stride?) — recorded in manifest",
                       args.site, cadence_s, NOMINAL_CADENCE_S)

    # -- Extract contiguous, precipitation-bearing sequences ---------------
    sequences, starts = [], []
    stats = {"candidates": 0, "gap_rejected": 0, "rain_rejected": 0}
    i = 0
    while i + SEQ_LEN <= len(times):
        window_t = times[i:i + SEQ_LEN]
        gaps = np.diff(window_t)
        stats["candidates"] += 1
        if np.any(np.abs(gaps - cadence_s) > cadence_tol_s):
            stats["gap_rejected"] += 1
            i += 1
            continue
        window = frames[i:i + SEQ_LEN]
        wet = np.nanmean(window > args.rain_threshold, axis=(1, 2))
        if np.nanmax(wet) < args.min_rain_fraction:
            stats["rain_rejected"] += 1
            i += 1
            continue
        sequences.append(np.nan_to_num(window).astype(np.float16))
        starts.append(window_t[0])
        # Non-overlapping sequences: jump a full window.
        i += SEQ_LEN

    n = len(sequences)
    logger.info("%s: retained %d sequences (%s)", args.site, n, stats)
    if n == 0:
        logger.error("Zero retained sequences for %s", args.site)

    seq_arr = (np.stack(sequences) if n
               else np.zeros((0, SEQ_LEN, 1, 1), dtype=np.float16))
    start_arr = np.array(starts)

    # -- Splits (SPEC constraint 5) -----------------------------------------
    split = np.zeros(n, dtype=np.int8)  # 0=train
    day_of_month = np.array([
        time.gmtime(t).tm_mday for t in start_arr
    ]) if n else np.zeros(0)
    split[day_of_month <= TEST_DAYS] = 2  # test

    remainder = np.where(split == 0)[0]
    n_val = min(len(remainder), int(np.sum(split == 2)))
    rng = np.random.default_rng(args.val_seed)
    val_idx = rng.choice(remainder, size=n_val, replace=False)
    split[val_idx] = 1  # val

    np.savez_compressed(args.output_sequences, sequences=seq_arr,
                        start_epoch=start_arr, split=split)

    digest = hashlib.sha256(seq_arr.tobytes()).hexdigest()
    manifest = {
        "site": args.site,
        "retained": n,
        "splits": {"train": int(np.sum(split == 0)),
                   "val": int(np.sum(split == 1)),
                   "test": int(np.sum(split == 2))},
        "filter": {"rain_threshold_mmh": args.rain_threshold,
                   "min_rain_fraction": args.min_rain_fraction},
        "effective_cadence_s": cadence_s,
        "val_seed": args.val_seed,
        "retention_stats": stats,
        "sequence_sha256": digest,
        "start_epochs": start_arr.tolist(),
        "split_labels": split.tolist(),
    }
    with open(args.output_manifest, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("%s: train=%d val=%d test=%d sha256=%s...",
                args.site, manifest["splits"]["train"],
                manifest["splits"]["val"], manifest["splits"]["test"],
                digest[:12])


if __name__ == "__main__":
    main()
