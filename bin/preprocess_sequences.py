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


def read_times(path):
    """Return just the time vector of one monthly NetCDF (cheap)."""
    nc = Dataset(path, "r")
    try:
        return np.ma.filled(
            nc.variables["time"][:].astype(np.float64), np.nan)
    finally:
        nc.close()


def iter_chunks(path, chunk_frames):
    """Yield (times, frames) blocks of one monthly NetCDF.

    Streaming keeps peak memory at ~chunk_frames * 300 * 300 * 4 bytes
    instead of a whole month (~7.8 GB at full 2-min cadence).
    """
    nc = Dataset(path, "r")
    try:
        n = nc.variables["time"].shape[0]
        for start in range(0, n, chunk_frames):
            stop = min(start + chunk_frames, n)
            times = np.ma.filled(
                nc.variables["time"][start:stop].astype(np.float64), np.nan)
            frames = np.ma.filled(
                nc.variables["precip_rate"][start:stop].astype(np.float32),
                np.nan)
            yield times, frames
    finally:
        nc.close()


def scan_buffer(times, frames, cadence_s, tol, thr, min_frac,
                out_seqs, out_starts, stats):
    """Extract sequences from a buffer; return first unconsumed index.

    The caller carries the unconsumed tail (< SEQ_LEN frames) into the
    next chunk so sequences spanning chunk/month boundaries survive.
    """
    i, n = 0, len(times)
    while i + SEQ_LEN <= n:
        window_t = times[i:i + SEQ_LEN]
        stats["candidates"] += 1
        if np.any(np.abs(np.diff(window_t) - cadence_s) > tol):
            stats["gap_rejected"] += 1
            i += 1
            continue
        window = frames[i:i + SEQ_LEN]
        wet = np.nanmean(window > thr, axis=(1, 2))
        if np.nanmax(wet) < min_frac:
            stats["rain_rejected"] += 1
            i += 1
            continue
        out_seqs.append(np.nan_to_num(window).astype(np.float16))
        out_starts.append(window_t[0])
        # Non-overlapping sequences: jump a full window.
        i += SEQ_LEN
    return i


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
    parser.add_argument("--chunk-frames", type=int, default=512,
                        help="Frames read per streaming block (default: "
                             "512 ~ 184 MB at 300x300 float32)")
    args = parser.parse_args()

    # -- Pass 1: time vectors only (cheap) — order months, infer cadence ----
    month_times = {}
    for path in args.input:
        try:
            t = read_times(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read %s: %s — skipping month",
                           path, exc)
            continue
        if len(t) == 0:
            logger.warning("%s is empty — skipping month", path)
            continue
        month_times[path] = t

    if not month_times:
        logger.error("No usable input months for %s", args.site)
        # Write declared outputs before failing (SPEC constraint 17).
        np.savez_compressed(args.output_sequences,
                            sequences=np.zeros((0,), dtype=np.float16),
                            start_epoch=np.zeros((0,)),
                            split=np.zeros((0,), dtype=np.int8))
        with open(args.output_manifest, "w") as f:
            json.dump({"site": args.site, "error": "no input data"}, f)
        sys.exit(1)

    ordered = sorted(month_times, key=lambda p: float(month_times[p][0]))
    all_t = np.concatenate([month_times[p] for p in ordered])
    total_frames = len(all_t)
    logger.info("%s: %d frames across %d month(s)", args.site,
                total_frames, len(ordered))

    # Effective cadence: inferred from the data so that strided pilot
    # ingests still form sequences. Full-cadence runs infer ~120 s; a
    # deviation from the paper's 2-min cadence is recorded in the manifest.
    if total_frames > 1:
        cadence_s = float(np.median(np.diff(all_t)))
    else:
        cadence_s = float(NOMINAL_CADENCE_S)
    cadence_tol_s = 0.25 * cadence_s
    if abs(cadence_s - NOMINAL_CADENCE_S) > 1.0:
        logger.warning("%s: effective cadence %.0f s != nominal %d s "
                       "(pilot stride?) — recorded in manifest",
                       args.site, cadence_s, NOMINAL_CADENCE_S)
    del all_t, month_times

    # -- Pass 2: stream frames, extracting sequences as we go ---------------
    sequences, starts = [], []
    stats = {"candidates": 0, "gap_rejected": 0, "rain_rejected": 0}
    carry_t = np.zeros(0, dtype=np.float64)
    carry_f = None
    warned_mem = False
    for path in ordered:
        for c_t, c_f in iter_chunks(path, args.chunk_frames):
            if carry_f is None or len(carry_t) == 0:
                buf_t, buf_f = c_t, c_f
            else:
                buf_t = np.concatenate([carry_t, c_t])
                buf_f = np.concatenate([carry_f, c_f], axis=0)
            used = scan_buffer(buf_t, buf_f, cadence_s, cadence_tol_s,
                               args.rain_threshold,
                               args.min_rain_fraction,
                               sequences, starts, stats)
            carry_t = buf_t[used:].copy()
            carry_f = buf_f[used:].copy()
            del buf_t, buf_f
            mem_gb = len(sequences) * SEQ_LEN * 300 * 300 * 2 / 1e9
            if mem_gb > 6.0 and not warned_mem:
                logger.warning("%s: retained sequences already ~%.1f GB in "
                               "RAM — consider a stricter "
                               "--min-rain-fraction", args.site, mem_gb)
                warned_mem = True
        logger.info("%s: %s done — %d sequences so far",
                    args.site, path, len(sequences))

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
