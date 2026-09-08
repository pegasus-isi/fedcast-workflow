#!/usr/bin/env python3

"""One client's share of the post-round validation (silo mode).

In cross-silo placement no federated job moves a client's data, so each
client scores the new global model on its OWN validation split, at the
silo, and ships back a small metrics JSON: the batch-loss sum and mean,
the batch count, and the number of validation sequences. Aggregates
derived from the client's data, not the data itself. fl_validate.py --client-metrics then
combines those into the same mean batch loss it would have computed
centrally (fedcast_common.generator_val_loss), so the checkpoint rule
(SPEC.md constraint 9) is unchanged.

Runs inside the FL-round SubWorkflow on validation rounds only, pinned to
the client's silo worker via its site-catalog tag (see silos.example.yml).
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

# Everything this job may send back to the server. Enforced at run time by
# fedcast_common.check_export and checked against README's "What actually
# leaves a silo" table by tools/check_export_docs.py.
EXPORT_FIELDS = ("site", "round", "n_val", "n_batches", "sum_loss",
                 "mean_loss")


def main():
    parser = argparse.ArgumentParser(
        description="Score the global model on one client's validation split")
    parser.add_argument("--client", required=True,
                        help="SITE:sequences_path:manifest_path")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True,
                        help="Training seed; also seeds validation "
                             "sampling, identically to the central path")
    parser.add_argument("--interval-months", type=int, required=True)
    parser.add_argument("--archive-start", required=True, help="YYYY-MM")
    parser.add_argument("--archive-months", type=int, required=True)
    parser.add_argument("--limit-train-sequences", type=int, default=None)
    parser.add_argument("--global-model", required=True)
    parser.add_argument("--metrics-out", required=True)
    args = parser.parse_args()

    import fedcast_common as fc

    # Registered before any work: the exit-time guard then covers this
    # output whatever ends up writing it.
    fc.guard_export(args.metrics_out, EXPORT_FIELDS, "metrics")

    import torch

    import fedcast_common as fc

    client = fc.parse_client(args.client)
    t_start = fc.interval_start_epoch(args.archive_start,
                                      args.archive_months,
                                      args.interval_months)
    data = fc.load_client_data(client, t_start,
                               limit=args.limit_train_sequences)
    val_x = data["val"][0]
    n_val = 0 if val_x is None else int(val_x.shape[0])

    losses = []
    if n_val:
        global_state = torch.load(args.global_model, map_location="cpu")
        model = fc.build_model(args.seed)
        model.load_state_dict(global_state)
        if torch.cuda.is_available():
            model = model.cuda()
        # Per-(client, batch) seeding, matching fl_validate, so this
        # client's numbers are the ones the central path would compute.
        losses = fc.generator_val_batch_losses(model, [data],
                                               args.seed)
    else:
        logger.warning("%s: no validation sequences in interval",
                       client["name"])

    metrics = {
        "site": client["name"],
        "round": args.round,
        "n_val": n_val,
        "n_batches": len(losses),
        "sum_loss": float(sum(losses)),
        "mean_loss": (float(sum(losses) / len(losses))
                      if losses else None),
    }
    fc.write_export(args.metrics_out, metrics, EXPORT_FIELDS, "metrics",
                    indent=2)
    logger.info("%s round %d: %d val sequences, %d batches, mean loss %s",
                client["name"], args.round, n_val, len(losses),
                metrics["mean_loss"])


if __name__ == "__main__":
    main()
