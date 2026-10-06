#!/usr/bin/env python3

"""Fetch the frozen 12-event benchmark: one 16-frame sample per event.

The benchmark is the paper appendix's fixed event table (AUTHOR_FEEDBACK.md,
committed as benchmark_events.csv), not something derived from the clients'
data: 9 of its 12 centroids lie outside every client window and 3 events
postdate the study interval. Each event therefore gets its own MRMS fetch.

Per event, one forecast initialization (AUTHOR_FEEDBACK.md Sec. 4, our rule
for open question 1): the window start is the first target frame, so the 16
frames run from start - 8 min to start + 22 min at 2-min steps. If the
archive has a gap in that sample, the start slides in 2-min steps, nearest
first, to the closest complete sample whose target frames still overlap the
event window; the shift is recorded per event (`init_shift_s`). This is
needed in practice: MRMS has no 2022-03-07 21:24 file, and every sample
anchored on that event's window contains it (AUTHOR_FEEDBACK.md Q6). Each frame is
cropped to MODEL_SIZE x MODEL_SIZE on the 0.01 degree grid, centered on the
event centroid, and encoded exactly like the client shards (capped,
quantized), so DGMR and STEPS see the same value domain they do in training.

Required-source semantics (SPEC.md constraint 17): the output is always
written; a missing or undecodable frame drops that event (no interpolation,
as in the paper) and the job exits non-zero unless --allow-missing is given,
because a reproduction must evaluate on all 12 events.

Output npz: sequences (N, 16, S, S) encoded like the shards, event_id,
selection, start_epoch (first frame), init_shift_s, lat, lon.
"""

import argparse
import calendar
import csv
import gzip
import logging
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.getcwd())  # fedcast_common.py staged into job cwd
import fedcast_common as fc  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BUCKET = "noaa-mrms-pds"
PRODUCT = "PrecipRate_00.00"
INPUT_FRAMES = 4
SEQ_LEN = 16
CADENCE_S = 120
KEY_TOLERANCE_S = 59    # a key within a minute of the nominal time matches
MAX_SHIFT_S = 30 * 60   # how far the init may slide to avoid an archive gap

MAX_RETRIES = 5
BACKOFF_BASE_S = 5


def s3_call(fn, *args, **kwargs):
    """Call an S3 operation with exponential-backoff retries."""
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - retry any transient error
            last_exc = exc
            wait = BACKOFF_BASE_S * (2 ** attempt)
            logger.warning("S3 error (attempt %d/%d): %s — retrying in %ds",
                           attempt + 1, MAX_RETRIES, exc, wait)
            time.sleep(wait)
    raise last_exc


def parse_utc(value):
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc).timestamp()


def key_epoch(key):
    """Epoch seconds (UTC) from ..._YYYYMMDD-HHMMSS.grib2.gz."""
    stamp = key.rsplit("_", 1)[-1].split(".")[0]
    return calendar.timegm(time.strptime(stamp, "%Y%m%d-%H%M%S"))


def decode_grib2(path):
    """Decode one MRMS GRIB2 file -> (data, lats desc, lons in [-180, 180))."""
    import xarray as xr

    ds = xr.open_dataset(path, engine="cfgrib",
                         backend_kwargs={"indexpath": ""})
    da = ds[list(ds.data_vars)[0]]
    lats = da.latitude.values
    lons = da.longitude.values
    lons = np.where(lons >= 180.0, lons - 360.0, lons)
    data = da.values.astype(np.float32)
    ds.close()
    return data, lats, lons


def crop_centered(data, lats, lons, lat0, lon0, size):
    """size x size window whose center pixel is nearest (lat0, lon0)."""
    i = int(np.argmin(np.abs(lats - lat0)))
    j = int(np.argmin(np.abs(lons - lon0)))
    half = size // 2
    if (i - half < 0 or j - half < 0 or i + half > data.shape[0]
            or j + half > data.shape[1]):
        return None
    return data[i - half:i + half, j - half:j + half]


class DayIndex:
    """Lazily listed MRMS keys per (domain, UTC day)."""

    def __init__(self, s3):
        self.s3 = s3
        self.paginator = s3.get_paginator("list_objects_v2")
        self.cache = {}

    def keys(self, domain, epoch_s):
        day = time.strftime("%Y%m%d", time.gmtime(epoch_s))
        if (domain, day) not in self.cache:
            prefix = f"{domain}/{PRODUCT}/{day}/"
            pages = s3_call(lambda: list(
                self.paginator.paginate(Bucket=BUCKET, Prefix=prefix)))
            found = {}
            for page in pages:
                for obj in page.get("Contents", []):
                    if obj["Key"].endswith(".grib2.gz"):
                        found[key_epoch(obj["Key"])] = obj["Key"]
            self.cache[(domain, day)] = found
        return self.cache[(domain, day)]

    def find(self, domain, epoch_s):
        found = self.keys(domain, epoch_s)
        best = min(found, key=lambda t: abs(t - epoch_s), default=None)
        if best is None or abs(best - epoch_s) > KEY_TOLERANCE_S:
            return None
        return found[best]


def find_sample(index, ev):
    """(t_first, shift_s, keys) of the nearest complete 16-frame sample.

    Candidates are the default anchoring shifted by multiples of the
    cadence, nearest first (later before earlier on ties), kept only while
    the target frames overlap the event window. Key lookups are listings,
    so trying a candidate costs no downloads.
    """
    ws, we = parse_utc(ev["start_utc"]), parse_utc(ev["end_utc"])
    base = ws - INPUT_FRAMES * CADENCE_S
    steps = range(0, MAX_SHIFT_S // CADENCE_S + 1)
    for shift in (d * sign * CADENCE_S for d in steps for sign in (1, -1)
                  if d or sign == 1):
        t_first = base + shift
        t_target0 = t_first + INPUT_FRAMES * CADENCE_S
        t_target1 = t_first + (SEQ_LEN - 1) * CADENCE_S
        if t_target1 < ws or t_target0 > we:
            continue
        keys = [index.find(ev["domain"], t_first + k * CADENCE_S)
                for k in range(SEQ_LEN)]
        if all(keys):
            return t_first, shift, keys
    raise LookupError("no complete 16-frame sample overlaps the window")


def fetch_frame(s3, key, ev, size):
    with tempfile.TemporaryDirectory() as tmp:
        gz_path = os.path.join(tmp, "frame.grib2.gz")
        grib_path = os.path.join(tmp, "frame.grib2")
        s3_call(s3.download_file, BUCKET, key, gz_path)
        with gzip.open(gz_path, "rb") as fin, open(grib_path, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        data, lats, lons = decode_grib2(grib_path)
    return crop_centered(data, lats, lons, ev["lat"], ev["lon"], size)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch the frozen event benchmark samples")
    parser.add_argument("--events", required=True,
                        help="benchmark_events.csv (frozen event table)")
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-missing", action="store_true",
                        help="Exit 0 even if some events could not be "
                             "built (pilot use only)")
    args = parser.parse_args()

    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    with open(args.events, newline="") as f:
        events = [{**row, "lat": float(row["lat"]), "lon": float(row["lon"]),
                   "domain": row.get("domain") or "CONUS"}
                  for row in csv.DictReader(f)]
    logger.info("Benchmark table: %d events", len(events))

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED,
                                          retries={"max_attempts": 0}))
    index = DayIndex(s3)

    seqs, ids, selections, starts, shifts, lats, lons = ([] for _ in
                                                         range(7))
    missing = []
    for ev in events:
        frames = []
        try:
            t_first, shift, keys = find_sample(index, ev)
            if shift:
                logger.warning("%s: default sample has an archive gap; "
                               "init shifted %+d min to the nearest "
                               "complete sample", ev["event_id"],
                               shift // 60)
            for key in keys:
                window = fetch_frame(s3, key, ev, args.crop_size)
                if window is None:
                    raise LookupError("crop leaves the domain")
                frames.append(window)
        except Exception as exc:  # noqa: BLE001 - drop this event, go on
            logger.error("%s: cannot build a complete sample (%s) — "
                         "dropped", ev["event_id"], exc)
            missing.append(ev["event_id"])
            continue
        seqs.append(fc.encode_precip(np.stack(frames)))
        ids.append(ev["event_id"])
        selections.append(ev.get("selection", ""))
        starts.append(t_first)
        shifts.append(shift)
        lats.append(ev["lat"])
        lons.append(ev["lon"])
        logger.info("%s: 16 frames from %s", ev["event_id"],
                    time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                  time.gmtime(t_first)))

    size = args.crop_size
    np.savez_compressed(
        args.output,
        sequences=(np.stack(seqs) if seqs else
                   fc.encode_precip(np.zeros((0, SEQ_LEN, size, size),
                                             dtype=np.float32))),
        event_id=np.array(ids, dtype=str),
        selection=np.array(selections, dtype=str),
        start_epoch=np.array(starts, dtype=np.float64),
        init_shift_s=np.array(shifts, dtype=np.int64),
        lat=np.array(lats), lon=np.array(lons),
    )
    logger.info("Wrote %d/%d event samples -> %s", len(seqs), len(events),
                args.output)
    if missing:
        logger.error("Missing events: %s", ", ".join(missing))
        if not args.allow_missing:
            sys.exit(1)
    if not seqs:
        sys.exit(1)


if __name__ == "__main__":
    main()
