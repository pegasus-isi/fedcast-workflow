#!/usr/bin/env python3

"""Fetch one event source for the benchmark: WPC MPD, LSR, or Storm Events.

Best-effort source semantics (SPEC.md constraint 17): transient HTTP failures
are retried with backoff; on permanent failure this wrapper writes an EMPTY
declared output, logs an ERROR, and exits 0. The downstream build_benchmark
job fails only if EVERY source is empty — preserving the multi-source
resilience the benchmark is designed around.

Sources:
  mpd          — WPC Mesoscale Precipitation Discussions via the Iowa
                 Environmental Mesonet (IEM) API.
  lsr          — Local Storm Reports via the IEM GeoJSON service (flood /
                 heavy-rain types), fetched month by month.
  storm_events — NOAA/NCEI Storm Events Database details CSVs (per year;
                 the exact filename embeds a creation date, so the directory
                 listing is scraped for each year's file).

Output JSON schema (unified): list of
  {"source", "event_id", "start_utc", "end_utc", "lat", "lon", "type"}
"""

import argparse
import csv
import gzip
import io
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MAX_RETRIES = 4
BACKOFF_BASE_S = 5
TIMEOUT_S = 120

IEM_MPD_URL = ("https://mesonet.agron.iastate.edu/cgi-bin/request/gis/"
               "wpc_mpd.py")
IEM_LSR_URL = "https://mesonet.agron.iastate.edu/geojson/lsr.php"
NCEI_SE_DIR = ("https://www.ncei.noaa.gov/pub/data/swdi/stormevents/"
               "csvfiles/")

# LSR type texts relevant to heavy-precipitation events.
LSR_TYPES = {"FLASH FLOOD", "FLOOD", "HEAVY RAIN", "DEBRIS FLOW"}
# Storm Events event types relevant to heavy-precipitation events.
SE_TYPES = {"Flash Flood", "Flood", "Heavy Rain", "Debris Flow"}


def http_get(url, params=None):
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=TIMEOUT_S)
            resp.raise_for_status()
            return resp
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            wait = BACKOFF_BASE_S * (2 ** attempt)
            logger.warning("HTTP error (attempt %d/%d) for %s: %s — "
                           "retrying in %ds", attempt + 1, MAX_RETRIES,
                           url, exc, wait)
            time.sleep(wait)
    raise last_exc


def month_bounds(month):
    """Return (start_dt, end_dt_exclusive) for YYYY-MM."""
    year, mon = (int(x) for x in month.split("-"))
    start = datetime(year, mon, 1, tzinfo=timezone.utc)
    if mon == 12:
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end = datetime(year, mon + 1, 1, tzinfo=timezone.utc)
    return start, end


def iter_months(start_month, end_month):
    year, mon = (int(x) for x in start_month.split("-"))
    end_year, end_mon = (int(x) for x in end_month.split("-"))
    while (year, mon) <= (end_year, end_mon):
        yield f"{year:04d}-{mon:02d}"
        mon += 1
        if mon > 12:
            mon = 1
            year += 1


def fetch_mpd(start_month, end_month):
    """WPC MPDs via the IEM GIS request interface (zipped shapefile).

    This is the interface the paper itself cites (its reference [12]).
    Polygon footprints are reduced to their bounding-box centroid here;
    build_benchmark matches centroids against the 3°x3° site windows.
    """
    import io as _io
    import zipfile

    import shapefile  # pyshp

    events = []
    for month in iter_months(start_month, end_month):
        start, end = month_bounds(month)
        resp = http_get(IEM_MPD_URL, params={
            "sts": start.strftime("%Y-%m-%dT%H:%MZ"),
            "ets": end.strftime("%Y-%m-%dT%H:%MZ"),
        })
        try:
            zf = zipfile.ZipFile(_io.BytesIO(resp.content))
            shp_name = next(n for n in zf.namelist()
                            if n.endswith(".shp"))
            base = shp_name[:-4]
            reader = shapefile.Reader(
                shp=_io.BytesIO(zf.read(base + ".shp")),
                dbf=_io.BytesIO(zf.read(base + ".dbf")),
                shx=_io.BytesIO(zf.read(base + ".shx")),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("MPD %s: could not parse shapefile: %s",
                           month, exc)
            continue

        field_names = [f[0].upper() for f in reader.fields[1:]]

        def field(rec, *candidates):
            for cand in candidates:
                if cand in field_names:
                    return rec[field_names.index(cand)]
            return None

        for idx, (shape, rec) in enumerate(
                zip(reader.shapes(), reader.records())):
            bbox = getattr(shape, "bbox", None)
            if not bbox:
                continue
            lon = (bbox[0] + bbox[2]) / 2.0
            lat = (bbox[1] + bbox[3]) / 2.0
            issue = field(rec, "ISSUE", "ISSUED", "UTC_ISSUE")
            expire = field(rec, "EXPIRE", "EXPIRED", "UTC_EXPIRE")
            num = field(rec, "PRODUCT_NU", "NUM", "PRODUCT_ID") or idx
            events.append({
                "source": "mpd",
                "event_id": f"mpd_{num}_{month}",
                "start_utc": str(issue) if issue else None,
                "end_utc": str(expire) if expire else None,
                "lat": lat,
                "lon": lon,
                "type": "MPD",
            })
    return events


def fetch_lsr(start_month, end_month):
    events = []
    for month in iter_months(start_month, end_month):
        start, end = month_bounds(month)
        resp = http_get(IEM_LSR_URL, params={
            "sts": start.strftime("%Y%m%d%H%M"),
            "ets": end.strftime("%Y%m%d%H%M"),
            "wfos": "",  # all offices
        })
        data = resp.json()
        for feat in data.get("features", []):
            props = feat.get("properties", {})
            if str(props.get("typetext", "")).upper() not in LSR_TYPES:
                continue
            geom = feat.get("geometry") or {}
            coords = geom.get("coordinates") or [None, None]
            valid = props.get("valid")
            events.append({
                "source": "lsr",
                "event_id": f"lsr_{len(events)}_{month}",
                "start_utc": valid,
                "end_utc": valid,  # LSRs are point-in-time reports
                "lat": coords[1],
                "lon": coords[0],
                "type": props.get("typetext"),
            })
    return events


def fetch_storm_events(start_month, end_month):
    years = sorted({int(m.split("-")[0])
                    for m in iter_months(start_month, end_month)})
    listing = http_get(NCEI_SE_DIR).text
    events = []
    for year in years:
        pattern = rf'(StormEvents_details-ftp_v1\.0_d{year}_c\d+\.csv\.gz)'
        matches = re.findall(pattern, listing)
        if not matches:
            logger.warning("No Storm Events file found for %d", year)
            continue
        fname = sorted(set(matches))[-1]  # latest creation date
        resp = http_get(NCEI_SE_DIR + fname)
        text = gzip.decompress(resp.content).decode("utf-8",
                                                    errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            if row.get("EVENT_TYPE") not in SE_TYPES:
                continue
            try:
                lat = float(row.get("BEGIN_LAT") or "nan")
                lon = float(row.get("BEGIN_LON") or "nan")
            except ValueError:
                continue
            begin = (f"{row.get('BEGIN_YEARMONTH')}"
                     f"{row.get('BEGIN_DAY'):0>2}"
                     f" {row.get('BEGIN_TIME'):0>4}")
            end = (f"{row.get('END_YEARMONTH')}"
                   f"{row.get('END_DAY'):0>2}"
                   f" {row.get('END_TIME'):0>4}")
            events.append({
                "source": "storm_events",
                "event_id": f"se_{row.get('EVENT_ID')}",
                "start_utc": begin,
                "end_utc": end,
                "lat": lat,
                "lon": lon,
                "type": row.get("EVENT_TYPE"),
            })
    # Keep only events inside the requested month range.
    start, _ = month_bounds(start_month)
    _, end = month_bounds(end_month)
    kept = []
    for ev in events:
        try:
            dt = datetime.strptime(ev["start_utc"], "%Y%m%d %H%M")
            dt = dt.replace(tzinfo=timezone.utc)
        except Exception:  # noqa: BLE001
            continue
        if start <= dt < end:
            kept.append(ev)
    return kept


FETCHERS = {
    "mpd": fetch_mpd,
    "lsr": fetch_lsr,
    "storm_events": fetch_storm_events,
}


def main():
    parser = argparse.ArgumentParser(
        description="Fetch one benchmark event source (best-effort)")
    parser.add_argument("--source", required=True, choices=list(FETCHERS))
    parser.add_argument("--start-month", required=True, help="YYYY-MM")
    parser.add_argument("--end-month", required=True, help="YYYY-MM")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    events = []
    try:
        events = FETCHERS[args.source](args.start_month, args.end_month)
        logger.info("%s: fetched %d events", args.source, len(events))
    except Exception as exc:  # noqa: BLE001
        # Best-effort: degrade gracefully (SPEC constraint 17).
        logger.error("%s: permanent failure after retries: %s — writing "
                     "empty output and exiting 0", args.source, exc)

    with open(args.output, "w") as f:
        json.dump({"source": args.source, "events": events}, f)

    sys.exit(0)


if __name__ == "__main__":
    main()
