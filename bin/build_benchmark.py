#!/usr/bin/env python3

"""Compile the frozen event-driven benchmark set B from fetched sources.

Merges the three best-effort event sources (MPD, LSR, Storm Events), keeps
events whose location falls inside a site's 3°x3° window, and balances the
selection per (site, source) with a fixed seed and a per-site cap
(SPEC.md open question 5 — our documented rule).

Fails ONLY if every source is empty (SPEC.md constraint 17).

Output CSV columns:
  event_id, source, site, start_utc, end_utc, lat, lon, type
"""

import argparse
import csv
import json
import logging
import sys

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

HALF_WINDOW_DEG = 1.5


def main():
    parser = argparse.ArgumentParser(
        description="Compile the frozen benchmark event set")
    parser.add_argument("--events", action="append", required=True,
                        help="Per-source events JSON (repeatable)")
    parser.add_argument("--site", action="append", required=True,
                        help="NAME:LAT:LON (repeatable)")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-events-per-site", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    sites = []
    for spec in args.site:
        name, lat, lon = spec.split(":")
        sites.append({"name": name, "lat": float(lat), "lon": float(lon)})

    all_events = []
    non_empty_sources = 0
    for path in args.events:
        try:
            with open(path) as f:
                payload = json.load(f)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read %s: %s", path, exc)
            continue
        events = payload.get("events", [])
        if events:
            non_empty_sources += 1
        logger.info("%s: %d events", payload.get("source", path),
                    len(events))
        all_events.extend(events)

    if non_empty_sources == 0:
        logger.error("ALL event sources are empty — cannot build benchmark")
        # Write declared output before failing (SPEC constraint 17).
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["event_id", "source", "site", "start_utc",
                             "end_utc", "lat", "lon", "type"])
        sys.exit(1)

    # -- Assign events to sites by footprint --------------------------------
    per_site_source = {}
    for ev in all_events:
        lat, lon = ev.get("lat"), ev.get("lon")
        if lat is None or lon is None:
            continue
        try:
            lat, lon = float(lat), float(lon)
        except (TypeError, ValueError):
            continue
        for site in sites:
            if (abs(lat - site["lat"]) <= HALF_WINDOW_DEG
                    and abs(lon - site["lon"]) <= HALF_WINDOW_DEG):
                key = (site["name"], ev.get("source", "unknown"))
                per_site_source.setdefault(key, []).append(
                    {**ev, "site": site["name"]}
                )

    # -- Balanced selection: cap per site, spread across sources ------------
    rng = np.random.default_rng(args.seed)
    selected = []
    site_names = sorted({s["name"] for s in sites})
    for site_name in site_names:
        source_pools = {
            src: sorted(evs, key=lambda e: str(e.get("event_id")))
            for (sname, src), evs in per_site_source.items()
            if sname == site_name
        }
        for pool in source_pools.values():
            rng.shuffle(pool)
        picked = []
        # Round-robin across sources until the site cap is reached.
        while (len(picked) < args.max_events_per_site
               and any(source_pools.values())):
            for src in sorted(source_pools):
                if source_pools[src] and \
                        len(picked) < args.max_events_per_site:
                    picked.append(source_pools[src].pop())
        logger.info("%s: selected %d events", site_name, len(picked))
        selected.extend(picked)

    if not selected:
        logger.error("No events fell inside any site window")
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["event_id", "source", "site", "start_utc",
                             "end_utc", "lat", "lon", "type"])
        sys.exit(1)

    selected.sort(key=lambda e: (e["site"], str(e.get("start_utc"))))
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["event_id", "source", "site", "start_utc",
                         "end_utc", "lat", "lon", "type"])
        for ev in selected:
            writer.writerow([
                ev.get("event_id"), ev.get("source"), ev["site"],
                ev.get("start_utc"), ev.get("end_utc"),
                ev.get("lat"), ev.get("lon"), ev.get("type"),
            ])

    logger.info("Benchmark frozen: %d events across %d sites -> %s",
                len(selected), len(site_names), args.output)


if __name__ == "__main__":
    main()
