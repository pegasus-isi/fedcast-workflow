#!/usr/bin/env python3

"""Fetch MRMS PrecipRate for one month and crop all sites in one domain.

Crop-on-ingest (SPEC.md open question 8): each MRMS GRIB2 file is downloaded
from s3://noaa-mrms-pds once, decoded, cropped to every requested 3°x3°
radar-centered window, and discarded. Full-domain files are never persisted.

Required-source semantics (SPEC.md constraint 17): transient S3 failures are
retried with exponential backoff. On permanent failure the declared outputs
are still written (possibly empty) BEFORE exiting non-zero, so HTCondor can
stage out and DAGMan sees a clean failure instead of a held job.

Outputs: one NetCDF per site with an unlimited time dimension, variable
`precip_rate` (mm/h), dims (time, y, x) with lat/lon coordinate vectors.
"""

import argparse
import gzip
import logging
import os
import shutil
import sys
import tempfile
import time
from datetime import date, timedelta

import boto3
import numpy as np
from botocore import UNSIGNED
from botocore.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BUCKET = "noaa-mrms-pds"
PRODUCT = "PrecipRate_00.00"
HALF_WINDOW_DEG = 1.5   # 3 degree x 3 degree window
GRID_POINTS = 300       # 300 x 300 at 0.01 degree

MAX_RETRIES = 5
BACKOFF_BASE_S = 5


def parse_site(spec):
    """Parse NAME:LAT:LON:OUTPUT_LFN."""
    name, lat, lon, output = spec.split(":")
    return {"name": name, "lat": float(lat), "lon": float(lon),
            "output": output}


def month_days(month):
    year, mon = (int(x) for x in month.split("-"))
    d = date(year, mon, 1)
    while d.month == mon:
        yield d
        d += timedelta(days=1)


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


def decode_grib2(path):
    """Decode one MRMS PrecipRate GRIB2 file.

    Returns (data 2D float32 array [lat descending, lon ascending],
    lats 1D, lons 1D in [-180, 180]).
    """
    import xarray as xr

    ds = xr.open_dataset(path, engine="cfgrib",
                         backend_kwargs={"indexpath": ""})
    var = list(ds.data_vars)[0]
    da = ds[var]
    lats = da.latitude.values
    lons = da.longitude.values
    # MRMS stores longitudes in [0, 360); normalize to [-180, 180).
    lons = np.where(lons >= 180.0, lons - 360.0, lons)
    data = da.values.astype(np.float32)
    ds.close()
    return data, lats, lons


def crop_window(data, lats, lons, lat0, lon0):
    """Crop a GRID_POINTS x GRID_POINTS window centered at (lat0, lon0)."""
    lat_idx = np.where(
        (lats <= lat0 + HALF_WINDOW_DEG) & (lats > lat0 - HALF_WINDOW_DEG)
    )[0]
    lon_idx = np.where(
        (lons >= lon0 - HALF_WINDOW_DEG) & (lons < lon0 + HALF_WINDOW_DEG)
    )[0]
    lat_idx = lat_idx[:GRID_POINTS]
    lon_idx = lon_idx[:GRID_POINTS]
    window = data[np.ix_(lat_idx, lon_idx)]
    return window, lats[lat_idx], lons[lon_idx]


class SiteWriter:
    """Incrementally append cropped frames to a per-site NetCDF file."""

    def __init__(self, site):
        from netCDF4 import Dataset

        self.site = site
        self.nc = Dataset(site["output"], "w", format="NETCDF4")
        self.nc.createDimension("time", None)
        self.nc.createDimension("y", GRID_POINTS)
        self.nc.createDimension("x", GRID_POINTS)
        self.t_var = self.nc.createVariable("time", "f8", ("time",))
        self.t_var.units = "seconds since 1970-01-01T00:00:00Z"
        self.lat_var = self.nc.createVariable("lat", "f4", ("y",))
        self.lon_var = self.nc.createVariable("lon", "f4", ("x",))
        self.rate = self.nc.createVariable(
            "precip_rate", "f4", ("time", "y", "x"),
            zlib=True, complevel=4,
        )
        self.rate.units = "mm/h"
        self.nc.site = site["name"]
        self.nc.center_lat = site["lat"]
        self.nc.center_lon = site["lon"]
        self.n = 0
        self.coords_set = False

    def append(self, epoch_s, window, lats, lons):
        if window.shape != (GRID_POINTS, GRID_POINTS):
            logger.warning("%s: window shape %s != (%d, %d) — skipping frame",
                           self.site["name"], window.shape,
                           GRID_POINTS, GRID_POINTS)
            return
        if not self.coords_set:
            self.lat_var[:] = lats
            self.lon_var[:] = lons
            self.coords_set = True
        # MRMS uses negative sentinels for missing/no-coverage.
        window = np.where(window < 0, np.nan, window)
        self.t_var[self.n] = epoch_s
        self.rate[self.n, :, :] = window
        self.n += 1

    def close(self):
        self.nc.close()


def main():
    parser = argparse.ArgumentParser(
        description="Fetch and crop MRMS PrecipRate for one (domain, month)")
    parser.add_argument("--domain", required=True,
                        choices=["CONUS", "ALASKA"])
    parser.add_argument("--month", required=True, help="YYYY-MM")
    parser.add_argument("--stride", type=int, default=1,
                        help="Keep every Nth frame (1 = full 2-min cadence)")
    parser.add_argument("--site", action="append", required=True,
                        help="NAME:LAT:LON:OUTPUT_LFN (repeatable)")
    args = parser.parse_args()

    sites = [parse_site(s) for s in args.site]
    logger.info("Domain=%s month=%s stride=%d sites=%s",
                args.domain, args.month, args.stride,
                [s["name"] for s in sites])

    writers = [SiteWriter(s) for s in sites]

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED,
                                          retries={"max_attempts": 0}))
    paginator = s3.get_paginator("list_objects_v2")

    frames = 0
    failed = False
    try:
        for day in month_days(args.month):
            prefix = (f"{args.domain}/{PRODUCT}/"
                      f"{day.strftime('%Y%m%d')}/")
            keys = []
            for page in s3_call(
                lambda p=prefix: list(
                    paginator.paginate(Bucket=BUCKET, Prefix=p)
                )
            ):
                for obj in page.get("Contents", []):
                    if obj["Key"].endswith(".grib2.gz"):
                        keys.append(obj["Key"])
            keys.sort()
            keys = keys[:: args.stride]
            logger.info("%s: %d frames after stride", prefix, len(keys))

            for key in keys:
                with tempfile.TemporaryDirectory() as tmp:
                    gz_path = os.path.join(tmp, "frame.grib2.gz")
                    grib_path = os.path.join(tmp, "frame.grib2")
                    s3_call(s3.download_file, BUCKET, key, gz_path)
                    with gzip.open(gz_path, "rb") as fin, \
                            open(grib_path, "wb") as fout:
                        shutil.copyfileobj(fin, fout)
                    try:
                        data, lats, lons = decode_grib2(grib_path)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Decode failed for %s: %s — skipping",
                                       key, exc)
                        continue
                    # Timestamp from the key:
                    # ..._YYYYMMDD-HHMMSS.grib2.gz
                    stamp = key.rsplit("_", 1)[-1].split(".")[0]
                    epoch_s = time.mktime(
                        time.strptime(stamp, "%Y%m%d-%H%M%S")
                    )
                    for site, writer in zip(sites, writers):
                        window, wlats, wlons = crop_window(
                            data, lats, lons, site["lat"], site["lon"]
                        )
                        writer.append(epoch_s, window, wlats, wlons)
                frames += 1
    except Exception as exc:  # noqa: BLE001
        logger.error("Permanent failure after retries: %s", exc)
        failed = True
    finally:
        for writer in writers:
            logger.info("%s: wrote %d frames -> %s",
                        writer.site["name"], writer.n,
                        writer.site["output"])
            writer.close()

    if failed or frames == 0:
        # Declared outputs exist (possibly empty); fail loud (SPEC c17).
        logger.error("Fetch failed or produced zero frames for %s %s",
                     args.domain, args.month)
        sys.exit(1)

    logger.info("Done: %d source frames processed", frames)


if __name__ == "__main__":
    main()
