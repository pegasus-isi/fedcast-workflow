#!/usr/bin/env python3

"""Initialize a federated DGMR run: seeded global model + tracking files.

Emits the round -1 artifacts consumed by the first FL-round SubWorkflow:
  --global-out   seeded initial DGMR weights (torch state dict)
  --history-out  empty validation history JSON
  --best-out     best-so-far file ({"state_dict":..., "val": None})
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
        description="Initialize federated DGMR global model")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--interval-months", type=int, required=True)
    parser.add_argument("--aggregation", required=True,
                        choices=["uniform", "quadratic"])
    parser.add_argument("--global-out", required=True)
    parser.add_argument("--history-out", required=True)
    parser.add_argument("--best-out", required=True)
    args = parser.parse_args()

    import torch

    import fedcast_common as fc

    model = fc.build_model(args.seed)
    state = fc.cpu_state_dict(model)
    torch.save(state, args.global_out)
    torch.save({"state_dict": state, "val": None}, args.best_out)

    history = {"best_val": None, "best_unit": -1,
               "mode": "federated", "aggregation": args.aggregation,
               "seed": args.seed, "interval_months": args.interval_months,
               "val_points": []}
    with open(args.history_out, "w") as f:
        json.dump(history, f, indent=2)

    logger.info("Initialized global model (seed %d) -> %s",
                args.seed, args.global_out)


if __name__ == "__main__":
    main()
