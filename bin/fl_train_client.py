#!/usr/bin/env python3

"""One client's local training for one FL round (paper Sec. IV-C.2).

Loads the current global model, runs ONE local epoch on this client's
interval-filtered training sequences, and emits the post-local-training
state dict plus a metadata JSON (retained-sequence count for server-side
weighting in the quadratic ablation, Eq. 8).

Runs inside an FL-round SubWorkflow, in parallel with the other clients.
"""

import argparse
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

# Everything this job may send back to the (unpinned) aggregator. Enforced
# at run time by fedcast_common.check_export and checked against README's
# "What actually leaves a silo" table by tools/check_export_docs.py.
EXPORT_FIELDS = ("site", "round", "n_train")


def main():
    parser = argparse.ArgumentParser(
        description="One client's local epoch for one FL round")
    parser.add_argument("--client", required=True,
                        help="SITE:sequences_lfn:manifest_lfn")
    parser.add_argument("--client-index", type=int, required=True,
                        help="Stable index of this client (seed derivation)")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--interval-months", type=int, required=True)
    parser.add_argument("--archive-start", required=True, help="YYYY-MM")
    parser.add_argument("--archive-months", type=int, required=True)
    parser.add_argument("--limit-train-sequences", type=int, default=None)
    parser.add_argument("--global-model", required=True)
    parser.add_argument("--local-model-out", required=True)
    parser.add_argument("--meta-out", required=True)
    args = parser.parse_args()

    import fedcast_common as fc

    # Registered before any work: the exit-time guard then covers this
    # output whatever ends up writing it.
    fc.guard_export(args.meta_out, EXPORT_FIELDS, "meta")

    import torch

    import fedcast_common as fc

    client = fc.parse_client(args.client)
    t_start = fc.interval_start_epoch(args.archive_start,
                                      args.archive_months,
                                      args.interval_months)
    data = fc.load_client_data(client, t_start,
                               limit=args.limit_train_sequences)

    meta = {"site": client["name"], "round": args.round,
            "n_train": data["n_train"]}

    if data["n_train"] == 0:
        # No data in interval: return the global model unchanged with
        # n_train=0 so the aggregator gives this client zero/minimal
        # influence. Declared outputs are still written (SPEC c17).
        logger.warning("%s: no training data in interval — passing "
                       "global model through", client["name"])
        state = torch.load(args.global_model, map_location="cpu")
        torch.save(state, args.local_model_out)
        fc.write_export(args.meta_out, meta, EXPORT_FIELDS, "meta")
        return

    # Deterministic per-(round, client) seed.
    seed = args.seed + args.round * 1000 + args.client_index
    model = fc.build_model(seed)
    model.load_state_dict(torch.load(args.global_model,
                                     map_location="cpu"))

    fc.fit_one_epoch(model, fc.make_loader(*data["train"]), epochs=1)

    torch.save(fc.cpu_state_dict(model), args.local_model_out)
    fc.write_export(args.meta_out, meta, EXPORT_FIELDS, "meta")
    logger.info("%s round %d: local epoch done (n_train=%d)",
                client["name"], args.round, data["n_train"])


if __name__ == "__main__":
    main()
