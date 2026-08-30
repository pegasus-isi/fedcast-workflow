#!/usr/bin/env python3

"""Server-side FedAvg aggregation for one FL round (paper Eq. 4 / Eq. 8).

Averages the clients' post-local-training state dicts into the new global
model. Weighting:
  uniform   — every client weight 1 (E1 baseline, Eq. 4)
  quadratic — w_i = max(1, floor(n_i^2 / n_max)) from the client metas
              (E2.1 ablation, Eq. 8)

Runs inside an FL-round SubWorkflow, after all fl_train_client jobs.
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


def average_state_dicts(states, weights):
    """Weighted average of state dicts."""
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


def main():
    parser = argparse.ArgumentParser(
        description="FedAvg aggregation for one FL round")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--aggregation", required=True,
                        choices=["uniform", "quadratic"])
    parser.add_argument("--local-model", action="append", required=True)
    parser.add_argument("--meta", action="append", required=True)
    parser.add_argument("--global-out", required=True)
    args = parser.parse_args()

    if len(args.local_model) != len(args.meta):
        logger.error("Mismatched --local-model (%d) and --meta (%d) counts",
                     len(args.local_model), len(args.meta))
        sys.exit(1)

    import torch

    states, metas = [], []
    for model_path, meta_path in zip(args.local_model, args.meta):
        with open(meta_path) as f:
            metas.append(json.load(f))
        states.append(torch.load(model_path, map_location="cpu"))

    # Clients with no data in the interval carry n_train=0 and must not
    # influence the average (they returned the global model unchanged).
    active = [(s, m) for s, m in zip(states, metas) if m["n_train"] > 0]
    if not active:
        logger.error("Round %d: no client had training data", args.round)
        sys.exit(1)
    states = [s for s, _ in active]
    metas = [m for _, m in active]

    if args.aggregation == "uniform":
        weights = [1.0] * len(states)
    else:  # quadratic (E2.1, Eq. 8)
        n_max = max(m["n_train"] for m in metas)
        weights = [max(1.0, float(m["n_train"] ** 2 // n_max))
                   for m in metas]

    logger.info("Round %d: aggregating %d clients (%s) weights=%s",
                args.round, len(states), args.aggregation,
                {m["site"]: w for m, w in zip(metas, weights)})

    torch.save(average_state_dicts(states, weights), args.global_out)
    logger.info("Global model -> %s", args.global_out)


if __name__ == "__main__":
    main()
