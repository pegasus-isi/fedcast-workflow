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
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.getcwd())  # fedcast_common.py staged into job cwd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Validate global model, chain best checkpoint")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--client", action="append", required=True,
                        help="SITE:sequences_lfn:manifest_lfn (repeatable)")
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

    import torch

    import fedcast_common as fc

    with open(args.history_in) as f:
        history = json.load(f)
    best = torch.load(args.best_in, map_location="cpu")

    clients = [fc.parse_client(c) for c in args.client]
    t_start = fc.interval_start_epoch(args.archive_start,
                                      args.archive_months,
                                      args.interval_months)
    data = [fc.load_client_data(c, t_start,
                                limit=args.limit_train_sequences)
            for c in clients]

    global_state = torch.load(args.global_model, map_location="cpu")
    model = fc.build_model(history.get("seed", 42))
    model.load_state_dict(global_state)

    val = fc.generator_val_loss(model, data)
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
