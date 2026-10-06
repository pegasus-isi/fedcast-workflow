#!/usr/bin/env python3

"""Validate the global model after an FL round; track the best checkpoint.

Computes the generator validation loss (fedcast_common.generator_val_loss)
over all clients' validation splits, appends it to the chained history, and
updates the chained best-so-far file when the loss improves (SPEC.md
constraint 9: checkpoint = lowest generator validation loss).

Runs inside the FL-round SubWorkflow on validation rounds only (every
--validate-every rounds, plus the final round). With --final-best the best
weights are additionally written in the {"state_dict":..., "history":...}
format consumed by mct_infer.py.

Two data paths, same number:
  --client SITE:seq:manifest   emulated mode — this job loads every
                               client's validation split itself.
  --client-metrics FILE        cross-silo mode — each client already
                               scored the global model at its own silo
                               (fl_validate_client.py) and shipped only
                               these small JSON files; the server never
                               reads client sequences.
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.getcwd())  # fedcast_common.py staged into job cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # direct runs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Validate global model, chain best checkpoint")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--client", action="append", default=[],
                        help="SITE:sequences_lfn:manifest_lfn (repeatable; "
                             "emulated mode)")
    parser.add_argument("--client-metrics", action="append", default=[],
                        help="Per-client validation metrics JSON from "
                             "fl_validate_client.py (repeatable; "
                             "cross-silo mode)")
    parser.add_argument("--interval-months", type=int, required=True)
    parser.add_argument("--archive-start", required=True, help="YYYY-MM")
    parser.add_argument("--archive-months", type=int, required=True)
    parser.add_argument("--limit-train-sequences", type=int, default=None)
    parser.add_argument("--global-model", required=True)
    parser.add_argument("--history-in", required=True)
    parser.add_argument("--best-in", required=True)
    parser.add_argument("--history-out", required=True)
    parser.add_argument("--best-out", required=True)
    parser.add_argument("--final-best", default=None,
                        help="Also write the best checkpoint in "
                             "mct_infer format (final round only)")
    args = parser.parse_args()

    if bool(args.client) == bool(args.client_metrics):
        parser.error("pass either --client (emulated mode) or "
                     "--client-metrics (cross-silo mode), not both/neither")

    import torch

    import fedcast_common as fc

    with open(args.history_in) as f:
        history = json.load(f)
    best = torch.load(args.best_in, map_location="cpu")

    global_state = torch.load(args.global_model, map_location="cpu")

    if args.client_metrics:
        # Cross-silo: clients scored the model at their own silos and
        # sent back batch-loss sums only. No client data is read here.
        per_client = []
        for path in args.client_metrics:
            with open(path) as f:
                per_client.append(json.load(f))
        val = fc.combine_client_val_metrics(per_client)
        history.setdefault("client_val", []).append(
            {"unit": args.round + 1,
             "clients": [{"site": m.get("site"), "n_val": m.get("n_val"),
                          "mean_loss": m.get("mean_loss")}
                         for m in per_client]})
        logger.info("Combined %d per-client validation reports",
                    len(per_client))
    else:
        clients = [fc.parse_client(c) for c in args.client]
        t_start = fc.interval_start_epoch(args.archive_start,
                                          args.archive_months,
                                          args.interval_months)
        data = [fc.load_client_data(c, t_start,
                                    limit=args.limit_train_sequences)
                for c in clients]
        model = fc.build_model(history.get("seed", 42))
        model.load_state_dict(global_state)
        if torch.cuda.is_available():
            model = model.cuda()
        # Same per-(client, batch) seeding the silo path uses, so the two
        # modes score a checkpoint identically.
        val = fc.generator_val_loss(model, data,
                                    history.get("seed", 42))
    unit = args.round + 1  # 1-indexed round count, mirroring epochs
    history["val_points"].append({"unit": unit, "val_loss": val})
    logger.info("Round %d (unit %d): generator val loss %.6f",
                args.round, unit, val)

    if history["best_val"] is None or val < history["best_val"]:
        history["best_val"] = val
        history["best_unit"] = unit
        best = {"state_dict": global_state, "val": val}
        logger.info("New best checkpoint at unit %d", unit)

    with open(args.history_out, "w") as f:
        json.dump(history, f, indent=2)
    torch.save(best, args.best_out)

    if args.final_best:
        torch.save({"state_dict": best["state_dict"], "history": history},
                   args.final_best)
        logger.info("Final best checkpoint (unit %d, val %s) -> %s",
                    history["best_unit"], history["best_val"],
                    args.final_best)


if __name__ == "__main__":
    main()
