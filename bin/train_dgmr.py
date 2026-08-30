#!/usr/bin/env python3

"""Train one segment of CENTRALIZED DGMR (pooled data from all clients).

Federated training is expressed as Pegasus SubWorkflows (one per FL round;
see fl_round.py and the fl_* wrappers) — this wrapper only handles the
centralized paradigm (paper Sec. IV-C.1) and its E2.2 SAM ablation slot.

One invocation runs `--segment-size` epochs, resuming from `--state-in` and
emitting `--state-out`, a tarball containing:
  model_state.pt   — weights after this segment
  best_state.pt    — weights with the lowest generator validation loss so far
  history.json     — per-validation-point losses + best-unit bookkeeping

The final segment additionally extracts best_state.pt to `--best-out`
(SPEC.md constraint 9: checkpoint = lowest generator validation loss).

TODO (SPEC): E2.2 generator-side SAM is not implemented yet — this wrapper
exits with an explicit error if --sam-rho is passed.
"""

import argparse
import json
import logging
import os
import sys
import tarfile
import tempfile

sys.path.insert(0, os.getcwd())  # fedcast_common.py staged into job cwd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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
        description="Train one centralized DGMR segment")
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

    import fedcast_common as fc

    if args.limit_train_sequences:
        logger.warning("PILOT: capping sequences per client at %d — not "
                       "valid for reproduction runs",
                       args.limit_train_sequences)

    clients = [fc.parse_client(c) for c in args.client]
    t_start = fc.interval_start_epoch(args.archive_start,
                                      args.archive_months,
                                      args.interval_months)
    data = [fc.load_client_data(c, t_start,
                                limit=args.limit_train_sequences)
            for c in clients]
    data = [d for d in data if d["n_train"] > 0]
    if not data:
        logger.error("No training data in interval L=%d",
                     args.interval_months)
        save_state(args.state_out, {}, {}, {"error": "no data"})
        sys.exit(1)

    model = fc.build_model(args.seed)
    if args.state_in:
        model_state, best_state, history = load_state(args.state_in)
        model.load_state_dict(model_state)
        logger.info("Resumed from %s (unit %d, best %s @ unit %d)",
                    args.state_in, history["unit"],
                    history["best_val"], history["best_unit"])
    else:
        best_state = fc.cpu_state_dict(model)
        history = {"unit": 0, "best_val": None, "best_unit": -1,
                   "mode": "centralized", "seed": args.seed,
                   "interval_months": args.interval_months,
                   "val_points": []}

    start_unit = history["unit"]
    end_unit = min(start_unit + args.segment_size, args.total_units)
    logger.info("Segment %d: epochs %d..%d of %d (centralized)",
                args.segment_index, start_unit + 1, end_unit,
                args.total_units)

    # Pool all clients' training data (paper Sec. IV-C.1).
    xs = torch.cat([d["train"][0] for d in data])
    ys = torch.cat([d["train"][1] for d in data])

    for unit in range(start_unit + 1, end_unit + 1):
        fc.fit_one_epoch(model, fc.make_loader(xs, ys), epochs=1)
        history["unit"] = unit
        if unit % args.validate_every == 0 or unit == args.total_units:
            val = fc.generator_val_loss(model, data)
            history["val_points"].append({"unit": unit, "val_loss": val})
            logger.info("Epoch %d: generator val loss %.6f", unit, val)
            if history["best_val"] is None or val < history["best_val"]:
                history["best_val"] = val
                history["best_unit"] = unit
                best_state = fc.cpu_state_dict(model)

    save_state(args.state_out, fc.cpu_state_dict(model), best_state,
               history)
    logger.info("State written to %s", args.state_out)

    if args.best_out:
        torch.save({"state_dict": best_state, "history": history},
                   args.best_out)
        logger.info("Best checkpoint (unit %d, val %s) -> %s",
                    history["best_unit"], history["best_val"],
                    args.best_out)


if __name__ == "__main__":
    main()
