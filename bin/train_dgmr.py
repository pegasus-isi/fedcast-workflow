#!/usr/bin/env python3

"""Train one segment of DGMR — centralized or federated (FedAvg).

One invocation runs `--segment-size` epochs (centralized) or FL rounds
(federated), resuming from `--state-in` and emitting `--state-out`, a tarball
containing:
  model_state.pt   — weights after this segment
  best_state.pt    — weights with the lowest generator validation loss so far
  history.json     — per-validation-point losses + best-unit bookkeeping

The final segment additionally extracts best_state.pt to `--best-out`
(SPEC.md constraint 9: checkpoint = lowest generator validation loss).

Training paradigms (SPEC.md constraints 6-10):
  centralized — pooled data from all clients, Lightning fit.
  federated   — synchronous FedAvg: every round, each client trains one
                local epoch from the global weights; the server averages
                state dicts with uniform (E1) or quadratic (E2.1) weights.
                Implemented as an in-process simulation (SPEC non-constraint
                6: any synchronous FedAvg-faithful implementation qualifies).

Interval selection: the training interval L uses the LAST L months of the
archive (SPEC open question 11 — our documented rule).

TODO (SPEC): E2.2 generator-side SAM is not implemented yet — this wrapper
exits with an explicit error if --sam-rho is passed.
TODO (SPEC): validation loss currently uses the grid-cell regularizer term
of Eq. 3 (lambda=20); the discriminator hinge term must be added to fully
match the paper's generator validation objective.
"""

import argparse
import json
import logging
import os
import sys
import tarfile
import tempfile
from datetime import datetime, timezone

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

INPUT_FRAMES = 4
FORECAST_STEPS = 12
GRID_LAMBDA = 20.0  # grid-cell regularizer weight (paper Sec. IV-C)
BATCH_SIZE = 2      # conservative default for 300x300 fields


def parse_client(spec):
    name, seq_lfn, manifest_lfn = spec.split(":")
    return {"name": name, "sequences": seq_lfn, "manifest": manifest_lfn}


def interval_start_epoch(archive_start, archive_months, interval_months):
    """Epoch seconds of the first month inside the LAST L months."""
    year, mon = (int(x) for x in archive_start.split("-"))
    total = year * 12 + (mon - 1) + archive_months - interval_months
    y, m = divmod(total, 12)
    return datetime(y, m + 1, 1, tzinfo=timezone.utc).timestamp()


def load_client_data(client, t_start):
    """Return dict with train/val tensors for one client, interval-filtered."""
    import torch

    with np.load(client["sequences"]) as data:
        seqs = data["sequences"]
        starts = data["start_epoch"]
        split = data["split"]
    keep = starts >= t_start
    seqs, split = seqs[keep], split[keep]

    def to_tensor(mask):
        arr = seqs[mask].astype(np.float32)
        if arr.shape[0] == 0:
            return None, None
        # (N, T, H, W) -> inputs (N, 4, 1, H, W), targets (N, 12, 1, H, W)
        x = torch.from_numpy(arr[:, :INPUT_FRAMES])[:, :, None]
        y = torch.from_numpy(arr[:, INPUT_FRAMES:])[:, :, None]
        return x, y

    train_x, train_y = to_tensor(split == 0)
    val_x, val_y = to_tensor(split == 1)
    n_train = 0 if train_x is None else train_x.shape[0]
    logger.info("%s: %d train / %d val sequences in interval",
                client["name"], n_train,
                0 if val_x is None else val_x.shape[0])
    return {"name": client["name"], "train": (train_x, train_y),
            "val": (val_x, val_y), "n_train": n_train}


def build_model(seed):
    """Instantiate DGMR (openclimatefix skillful_nowcasting)."""
    import torch
    from dgmr import DGMR

    torch.manual_seed(seed)
    np.random.seed(seed)
    return DGMR(forecast_steps=FORECAST_STEPS)


def make_loader(x, y):
    import torch

    ds = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE,
                                       shuffle=True)


def fit_one_epoch(model, loader, epochs=1):
    """Run Lightning fit for a fixed number of epochs on one loader."""
    import pytorch_lightning as pl
    import torch

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(model, loader)


def generator_val_loss(model, clients):
    """Grid-cell-regularizer validation loss over all clients' val sets.

    TODO: add the discriminator hinge term to fully match Eq. 3; the
    grid-cell term (lambda=20, intensity-weighted MAE on the ensemble
    mean of 6 samples) is the dominant, checkpoint-driving component.
    """
    import torch

    model.eval()
    device = next(model.parameters()).device
    losses = []
    with torch.no_grad():
        for c in clients:
            val_x, val_y = c["val"]
            if val_x is None:
                continue
            for i in range(0, val_x.shape[0], BATCH_SIZE):
                x = val_x[i:i + BATCH_SIZE].to(device)
                y = val_y[i:i + BATCH_SIZE].to(device)
                preds = torch.stack(
                    [model(x) for _ in range(6)]
                ).mean(dim=0)
                weight = torch.clamp(y + 1.0, max=24.0)
                grid_loss = (torch.abs(preds - y) * weight).mean()
                losses.append(float(GRID_LAMBDA * grid_loss))
    return float(np.mean(losses)) if losses else float("inf")


def average_state_dicts(states, weights):
    """FedAvg: weighted average of client state dicts."""
    import torch

    total = float(sum(weights))
    avg = {}
    for key in states[0]:
        stacked = torch.stack([
            s[key].float() * (w / total)
            for s, w in zip(states, weights)
        ])
        avg[key] = stacked.sum(dim=0).type(states[0][key].dtype)
    return avg


def load_state(path):
    """Extract state tarball -> (model_state, best_state, history)."""
    import torch

    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(tmp)
        model_state = torch.load(os.path.join(tmp, "model_state.pt"),
                                 map_location="cpu")
        best_state = torch.load(os.path.join(tmp, "best_state.pt"),
                                map_location="cpu")
        with open(os.path.join(tmp, "history.json")) as f:
            history = json.load(f)
    return model_state, best_state, history


def save_state(path, model_state, best_state, history):
    import torch

    with tempfile.TemporaryDirectory() as tmp:
        torch.save(model_state, os.path.join(tmp, "model_state.pt"))
        torch.save(best_state, os.path.join(tmp, "best_state.pt"))
        with open(os.path.join(tmp, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        with tarfile.open(path, "w:gz") as tar:
            for name in ("model_state.pt", "best_state.pt", "history.json"):
                tar.add(os.path.join(tmp, name), arcname=name)


def main():
    parser = argparse.ArgumentParser(
        description="Train one DGMR segment (centralized or federated)")
    parser.add_argument("--mode", required=True,
                        choices=["centralized", "federated"])
    parser.add_argument("--client", action="append", required=True,
                        help="SITE:sequences_lfn:manifest_lfn (repeatable)")
    parser.add_argument("--interval-months", type=int, required=True)
    parser.add_argument("--archive-start", required=True, help="YYYY-MM")
    parser.add_argument("--archive-months", type=int, required=True)
    parser.add_argument("--segment-index", type=int, required=True)
    parser.add_argument("--segment-size", type=int, required=True)
    parser.add_argument("--total-units", type=int, required=True)
    parser.add_argument("--validate-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aggregation", default="uniform",
                        choices=["uniform", "quadratic"])
    parser.add_argument("--sam-rho", type=float, default=None)
    parser.add_argument("--limit-train-sequences", type=int, default=None,
                        help="Cap train/val sequences per client (pilot/"
                             "CPU smoke tests only — NOT for reproduction "
                             "runs)")
    parser.add_argument("--state-in", default=None)
    parser.add_argument("--state-out", required=True)
    parser.add_argument("--best-out", default=None)
    args = parser.parse_args()

    if args.sam_rho is not None:
        # TODO(E2.2): implement generator-side SAM (SPEC Sec. 2 item 15).
        logger.error("E2.2 SAM training is not implemented yet "
                     "(--sam-rho %s)", args.sam_rho)
        sys.exit(1)

    import torch

    clients = [parse_client(c) for c in args.client]
    t_start = interval_start_epoch(args.archive_start, args.archive_months,
                                   args.interval_months)
    data = [load_client_data(c, t_start) for c in clients]
    if args.limit_train_sequences:
        cap = args.limit_train_sequences
        logger.warning("PILOT: capping sequences per client at %d — not "
                       "valid for reproduction runs", cap)
        for d in data:
            for key in ("train", "val"):
                x, y = d[key]
                if x is not None:
                    d[key] = (x[:cap], y[:cap])
            d["n_train"] = 0 if d["train"][0] is None \
                else d["train"][0].shape[0]
    data = [d for d in data if d["n_train"] > 0]
    if not data:
        logger.error("No training data in interval L=%d",
                     args.interval_months)
        save_state(args.state_out, {}, {}, {"error": "no data"})
        sys.exit(1)

    model = build_model(args.seed)
    if args.state_in:
        model_state, best_state, history = load_state(args.state_in)
        model.load_state_dict(model_state)
        logger.info("Resumed from %s (unit %d, best %.4f @ unit %d)",
                    args.state_in, history["unit"],
                    history["best_val"], history["best_unit"])
    else:
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        history = {"unit": 0, "best_val": float("inf"), "best_unit": -1,
                   "mode": args.mode, "aggregation": args.aggregation,
                   "seed": args.seed, "interval_months":
                   args.interval_months, "val_points": []}

    start_unit = history["unit"]
    end_unit = min(start_unit + args.segment_size, args.total_units)
    logger.info("Segment %d: units %d..%d of %d (%s)",
                args.segment_index, start_unit + 1, end_unit,
                args.total_units, args.mode)

    for unit in range(start_unit + 1, end_unit + 1):
        if args.mode == "centralized":
            # Pool all clients' training data (paper Sec. IV-C.1).
            xs = torch.cat([d["train"][0] for d in data])
            ys = torch.cat([d["train"][1] for d in data])
            fit_one_epoch(model, make_loader(xs, ys), epochs=1)
        else:
            # One synchronous FedAvg round (paper Sec. IV-C.2, Eq. 4/8).
            global_state = {k: v.clone()
                            for k, v in model.state_dict().items()}
            client_states, weights = [], []
            n_max = max(d["n_train"] for d in data)
            for d in data:
                model.load_state_dict(global_state)
                fit_one_epoch(model, make_loader(*d["train"]), epochs=1)
                client_states.append({
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                })
                if args.aggregation == "uniform":
                    weights.append(1.0)
                else:  # quadratic (E2.1, Eq. 8)
                    weights.append(
                        max(1.0, float(d["n_train"] ** 2 // n_max))
                    )
            model.load_state_dict(
                average_state_dicts(client_states, weights)
            )

        history["unit"] = unit
        if unit % args.validate_every == 0 or unit == args.total_units:
            val = generator_val_loss(model, data)
            history["val_points"].append({"unit": unit, "val_loss": val})
            logger.info("Unit %d: generator val loss %.6f", unit, val)
            if val < history["best_val"]:
                history["best_val"] = val
                history["best_unit"] = unit
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

    model_state = {k: v.detach().cpu().clone()
                   for k, v in model.state_dict().items()}
    save_state(args.state_out, model_state, best_state, history)
    logger.info("State written to %s", args.state_out)

    if args.best_out:
        torch.save({"state_dict": best_state, "history": history},
                   args.best_out)
        logger.info("Best checkpoint (unit %d, val %.6f) -> %s",
                    history["best_unit"], history["best_val"],
                    args.best_out)


if __name__ == "__main__":
    main()
