#!/usr/bin/env python3

"""MCT forecast adapter: run inference for one method on the benchmark set.

For each benchmark event, selects held-out TEST sequences at the event's
site whose 24-min target window intersects the event's UTC window (padded by
30 min; LSRs are point-in-time). This is our documented mapping rule —
SPEC.md open question 10.

Methods:
  steps       — PySTEPS STEPS: 20-member ensemble, 6 cascade levels,
                nonparametric noise, Bowler-Pierce-Seed velocity
                perturbations, incremental mask (paper Sec. IV-B.2).
  <anything else> — DGMR from --checkpoint: K stochastic samples
                (paper Sec. IV-B.1, K=6).

Output npz:
  forecasts (N, K, 12, H, W) float16, observations (N, 12, H, W) float16,
  inputs (N, 4, H, W) float16, instance metadata arrays, exec_time_s (N,).
"""

import argparse
import csv
import logging
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

INPUT_FRAMES = 4
FORECAST_STEPS = 12
CADENCE_S = 120
EVENT_PAD_S = 1800  # pad event windows by 30 min (documented rule)

# All methods are evaluated on the same center-cropped 288x288 grid
# (= 9 * 32): DGMR requires spatial dims divisible by 32, and comparing
# methods on different grids would bias the candidate pool.
MODEL_SIZE = 288


def center_crop(arr, size=MODEL_SIZE):
    """Center-crop the last two (H, W) dims to size x size."""
    h, w = arr.shape[-2], arr.shape[-1]
    top = max(0, (h - size) // 2)
    left = max(0, (w - size) // 2)
    return arr[..., top:top + size, left:left + size]


def parse_client(spec):
    name, seq_lfn, manifest_lfn = spec.split(":")
    return {"name": name, "sequences": seq_lfn, "manifest": manifest_lfn}


def parse_event_time(value):
    """Parse the heterogeneous time formats of the three event sources."""
    if not value:
        return None
    value = str(value).strip()
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S", "%Y%m%d %H%M", "%Y-%m-%dT%H:%MZ"):
        try:
            return datetime.strptime(value, fmt).replace(
                tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    return None


def load_events(path):
    events = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            start = parse_event_time(row.get("start_utc"))
            end = parse_event_time(row.get("end_utc")) or start
            if start is None:
                continue
            events.append({**row, "start_s": start - EVENT_PAD_S,
                           "end_s": end + EVENT_PAD_S})
    return events


def match_instances(events, clients, max_per_event):
    """Yield (event, site, sequence, start_epoch) for matching test seqs."""
    site_data = {}
    for c in clients:
        with np.load(c["sequences"]) as data:
            mask = data["split"] == 2  # test only
            site_data[c["name"]] = {
                "sequences": data["sequences"][mask],
                "starts": data["start_epoch"][mask],
            }

    instances = []
    for ev in events:
        sd = site_data.get(ev["site"])
        if sd is None or sd["sequences"].shape[0] == 0:
            continue
        # Target window: frames 4..15 -> [start + 8 min, start + 32 min].
        t0 = sd["starts"] + INPUT_FRAMES * CADENCE_S
        t1 = sd["starts"] + (INPUT_FRAMES + FORECAST_STEPS) * CADENCE_S
        hit = np.where((t1 >= ev["start_s"]) & (t0 <= ev["end_s"]))[0]
        for idx in hit[:max_per_event]:
            instances.append({
                "event_id": ev["event_id"], "site": ev["site"],
                "sequence": sd["sequences"][idx],
                "start_epoch": float(sd["starts"][idx]),
            })
    return instances


def forecast_steps_method(precip_in, n_members):
    """PySTEPS STEPS nowcast for one instance (paper Sec. IV-B.2)."""
    from pysteps import motion, nowcasts
    from pysteps.utils import transformation

    rate = precip_in.astype(np.float64)
    db, meta = transformation.dB_transform(rate, threshold=0.1,
                                           zerovalue=-15.0)
    db[~np.isfinite(db)] = -15.0
    oflow = motion.get_method("LK")(db)
    nowcast = nowcasts.get_method("steps")(
        db, oflow, FORECAST_STEPS,
        n_ens_members=n_members,
        n_cascade_levels=6,
        precip_thr=meta["threshold"],
        kmperpixel=1.0,
        timestep=CADENCE_S / 60.0,
        noise_method="nonparametric",
        vel_pert_method="bps",
        mask_method="incremental",
    )
    out, _ = transformation.dB_transform(nowcast, inverse=True,
                                         threshold=meta["threshold"],
                                         zerovalue=meta["zerovalue"])
    return np.nan_to_num(out)  # (K, 12, H, W)


def forecast_dgmr(model, precip_in, n_members):
    """DGMR stochastic ensemble for one instance (paper Eq. 1)."""
    import torch

    device = next(model.parameters()).device
    x = torch.from_numpy(precip_in.astype(np.float32))[None, :, None].to(
        device)
    members = []
    with torch.no_grad():
        for _ in range(n_members):
            pred = model(x)  # (1, 12, 1, H, W)
            members.append(pred[0, :, 0].cpu().numpy())
    return np.stack(members)  # (K, 12, H, W)


def main():
    parser = argparse.ArgumentParser(
        description="MCT forecast adapter for one method")
    parser.add_argument("--method", required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="DGMR best checkpoint (omit for steps)")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--client", action="append", required=True,
                        help="SITE:sequences_lfn:manifest_lfn (repeatable)")
    parser.add_argument("--ensemble-size", type=int, required=True)
    parser.add_argument("--max-instances-per-event", type=int, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    clients = [parse_client(c) for c in args.client]
    events = load_events(args.benchmark)
    logger.info("Benchmark: %d events", len(events))

    instances = match_instances(events, clients,
                                args.max_instances_per_event)
    logger.info("Matched %d forecast instances", len(instances))
    if not instances:
        logger.error("No benchmark events matched any test sequence")
        np.savez_compressed(args.output,
                            forecasts=np.zeros((0,), dtype=np.float16))
        sys.exit(1)

    model = None
    if args.method != "steps":
        import torch
        from dgmr import DGMR

        model = DGMR(forecast_steps=FORECAST_STEPS,
                     output_shape=MODEL_SIZE)
        payload = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(payload["state_dict"])
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()

    forecasts, observations, inputs = [], [], []
    exec_times, event_ids, sites, start_epochs = [], [], [], []
    for inst in instances:
        seq = center_crop(inst["sequence"].astype(np.float32))
        precip_in, obs = seq[:INPUT_FRAMES], seq[INPUT_FRAMES:]
        t0 = time.time()
        try:
            if args.method == "steps":
                ens = forecast_steps_method(precip_in, args.ensemble_size)
            else:
                ens = forecast_dgmr(model, precip_in, args.ensemble_size)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Forecast failed for %s (%s): %s — skipping",
                           inst["event_id"], inst["site"], exc)
            continue
        exec_times.append(time.time() - t0)
        forecasts.append(np.clip(ens, 0, None).astype(np.float16))
        observations.append(obs.astype(np.float16))
        inputs.append(precip_in.astype(np.float16))
        event_ids.append(inst["event_id"])
        sites.append(inst["site"])
        start_epochs.append(inst["start_epoch"])

    if not forecasts:
        logger.error("All forecasts failed")
        np.savez_compressed(args.output,
                            forecasts=np.zeros((0,), dtype=np.float16))
        sys.exit(1)

    np.savez_compressed(
        args.output,
        forecasts=np.stack(forecasts),
        observations=np.stack(observations),
        inputs=np.stack(inputs),
        exec_time_s=np.array(exec_times),
        event_id=np.array(event_ids),
        site=np.array(sites),
        start_epoch=np.array(start_epochs),
        method=np.array([args.method]),
    )
    logger.info("%s: %d instances -> %s", args.method, len(forecasts),
                args.output)


if __name__ == "__main__":
    main()
