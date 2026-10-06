#!/usr/bin/env python3

"""MCT forecast adapter: run inference for one method on the benchmark set.

One forecast instance per benchmark event: the event's 16-frame sample as
built by fetch_benchmark (4 inputs + 12 targets, one predefined
initialization per event — AUTHOR_FEEDBACK.md). The benchmark is
independent of the clients' data and splits.

Methods:
  steps       — PySTEPS STEPS: 20-member ensemble, 6 cascade levels,
                nonparametric noise, Bowler-Pierce-Seed velocity
                perturbations, incremental mask (paper Sec. IV-B.2).
  <anything else> — DGMR from --checkpoint: K stochastic samples
                (paper Sec. IV-B.1, K=6).

Output npz:
  forecasts (N, K, 12, H, W) float16, observations (N, 12, H, W) float16,
  inputs (N, 4, H, W) float16, event_id/site/start_epoch (N,),
  exec_time_s (N,). `site` carries the event's selection label, since a
  benchmark event belongs to no client site.
"""

import argparse
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.getcwd())  # fedcast_common.py staged into job cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fedcast_common as fc  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

INPUT_FRAMES = 4
FORECAST_STEPS = 12
CADENCE_S = 120

# All methods are evaluated on the same center-cropped model grid:
# DGMR requires spatial dims divisible by 32, and comparing
# methods on different grids would bias the candidate pool.
# FEDCAST_MODEL_SIZE is a PILOT-ONLY override for low-memory smoke tests.
MODEL_SIZE = int(os.environ.get("FEDCAST_MODEL_SIZE", "256"))


def center_crop(arr, size=MODEL_SIZE):
    """Center-crop the last two (H, W) dims to size x size."""
    h, w = arr.shape[-2], arr.shape[-1]
    top = max(0, (h - size) // 2)
    left = max(0, (w - size) // 2)
    return arr[..., top:top + size, left:left + size]


def load_benchmark(path):
    """One instance per event from fetch_benchmark's npz."""
    with np.load(path, allow_pickle=False) as data:
        seqs = data["sequences"]
        return [{"event_id": str(data["event_id"][i]),
                 "site": str(data["selection"][i]),
                 "sequence": fc.decode_sequences(seqs[i]),
                 "start_epoch": float(data["start_epoch"][i])}
                for i in range(seqs.shape[0])]


# STEPS grid parameters. "paper" is what the authors ran: the PySTEPS
# example's 2 km / 5 min, not MRMS's 1 km / 2 min (AUTHOR_FEEDBACK.md Q5).
# "mrms" matches the data; the authors are re-running with it.
STEPS_GRIDS = {"paper": (2.0, 5.0), "mrms": (1.0, CADENCE_S / 60.0)}


def forecast_steps_method(precip_in, n_members, grid="paper"):
    """PySTEPS STEPS nowcast for one instance (paper Sec. IV-B.2)."""
    from pysteps import motion, nowcasts
    from pysteps.utils import transformation

    rate = precip_in.astype(np.float64)
    db, meta = transformation.dB_transform(rate, threshold=0.1,
                                           zerovalue=-15.0)
    db[~np.isfinite(db)] = -15.0
    oflow = motion.get_method("LK")(db)
    kmperpixel, timestep = STEPS_GRIDS[grid]
    nowcast = nowcasts.get_method("steps")(
        db, oflow, FORECAST_STEPS,
        n_ens_members=n_members,
        n_cascade_levels=6,
        precip_thr=meta["threshold"],
        kmperpixel=kmperpixel,
        timestep=timestep,
        seed=24,
        ar_order=2,
        extrap_method="semilagrangian",
        decomp_method="fft",
        bandpass_filter_method="gaussian",
        noise_method="nonparametric",
        vel_pert_method="bps",
        mask_method="incremental",
        probmatching_method="cdf",
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
    parser.add_argument("--benchmark", required=True,
                        help="benchmark_sequences.npz from fetch_benchmark")
    parser.add_argument("--ensemble-size", type=int, required=True)
    parser.add_argument("--steps-grid", choices=sorted(STEPS_GRIDS),
                        default="paper",
                        help="STEPS kmperpixel/timestep: 'paper' = 2 km / "
                             "5 min as published, 'mrms' = 1 km / 2 min")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    instances = load_benchmark(args.benchmark)
    logger.info("Benchmark: %d event instances", len(instances))
    if not instances:
        logger.error("Benchmark holds no event samples")
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
    failed = []
    for inst in instances:
        seq = center_crop(inst["sequence"].astype(np.float32))
        precip_in, obs = seq[:INPUT_FRAMES], seq[INPUT_FRAMES:]
        t0 = time.time()
        try:
            if args.method == "steps":
                ens = forecast_steps_method(precip_in, args.ensemble_size,
                                            args.steps_grid)
            else:
                ens = forecast_dgmr(model, precip_in, args.ensemble_size)
        except Exception as exc:  # noqa: BLE001
            logger.error("Forecast failed for %s (%s): %s",
                         inst["event_id"], inst["site"], exc)
            failed.append(inst["event_id"])
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
    if failed:
        # Every method must be scored on the full benchmark; a dropped
        # event would let TOPSIS rank on a subset.
        logger.error("%s: no forecast for %s", args.method, ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
