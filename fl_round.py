#!/usr/bin/env python3

"""Builder for one FL-round Pegasus SubWorkflow.

Each federated round of Fed-Cast is its own sub-DAG (paper Sec. IV-C.2):

    fl_train_client x N   (parallel: one local epoch per client)
        -> fl_aggregate    (FedAvg: uniform or quadratic weights)
        -> fl_validate     (validation rounds only: chained history +
                            best-so-far checkpoint; final round emits the
                            best checkpoint in mct_infer format)

Imported by workflow_generator.py, which writes the returned Workflow to a
YAML file, registers it in the replica catalog, and adds a SubWorkflow job
per round to the top-level DAG.
"""

from Pegasus.api import File, Job, Workflow

COMMON_LFN = "fedcast_common.py"


def round_file_names(method, interval, round_num):
    """Canonical LFNs for one round's chained artifacts."""
    prefix = f"{method}_L{interval}"
    return {
        "global_out": f"{prefix}_global_r{round_num:03d}.pt",
        "history_out": f"{prefix}_history_r{round_num:03d}.json",
        "best_out": f"{prefix}_bestsofar_r{round_num:03d}.pt",
    }


def init_file_names(method, interval):
    """Canonical LFNs for the fl_init artifacts (round -1)."""
    prefix = f"{method}_L{interval}"
    return {
        "global_out": f"{prefix}_global_init.pt",
        "history_out": f"{prefix}_history_init.json",
        "best_out": f"{prefix}_bestsofar_init.pt",
    }


def generate_round_workflow(
    method,
    interval,
    round_num,
    clients,            # list of {"name", "sequences", "manifest"} LFN dicts
    prev_global_lfn,
    prev_history_lfn,
    prev_best_lfn,
    aggregation,        # "uniform" | "quadratic"
    archive_start,
    archive_months,
    seed,
    is_validation_round,
    final_best_lfn=None,        # set on the final round only
    limit_train_sequences=None,  # pilot/CPU smoke tests only
):
    """Build the sub-DAG for one FL round.

    Returns (workflow, names) where names maps the chained artifact LFNs
    produced by this round (global model always; history/best only on
    validation rounds).
    """
    wf = Workflow(f"{method}_L{interval}_r{round_num:03d}")
    names = round_file_names(method, interval, round_num)

    common = File(COMMON_LFN)
    global_in = File(prev_global_lfn)
    global_out = File(names["global_out"])

    interval_args = [
        "--interval-months", str(interval),
        "--archive-start", archive_start,
        "--archive-months", str(archive_months),
    ]
    pilot_args = (
        ["--limit-train-sequences", str(limit_train_sequences)]
        if limit_train_sequences else []
    )

    # -- Parallel client training -------------------------------------------
    local_models, metas, train_jobs = [], [], []
    for idx, client in enumerate(clients):
        site = client["name"]
        local_model = File(
            f"{method}_L{interval}_r{round_num:03d}_local_{site}.pt")
        meta = File(
            f"{method}_L{interval}_r{round_num:03d}_meta_{site}.json")
        seq_f = File(client["sequences"])
        man_f = File(client["manifest"])
        job = (
            Job("fl_train_client",
                _id=f"train_{site}",
                node_label=f"train_{site}_r{round_num:03d}")
            .add_args(
                "--client",
                f"{site}:{client['sequences']}:{client['manifest']}",
                "--client-index", str(idx),
                "--round", str(round_num),
                "--seed", str(seed),
                *interval_args,
                *pilot_args,
                "--global-model", global_in,
                "--local-model-out", local_model,
                "--meta-out", meta,
            )
            .add_inputs(global_in, seq_f, man_f, common)
            .add_outputs(local_model, stage_out=False,
                         register_replica=False)
            .add_outputs(meta, stage_out=False, register_replica=False)
        )
        wf.add_jobs(job)
        train_jobs.append(job)
        local_models.append(local_model)
        metas.append(meta)

    # -- FedAvg aggregation ---------------------------------------------------
    agg_job = (
        Job("fl_aggregate",
            _id="aggregate", node_label=f"aggregate_r{round_num:03d}")
        .add_args(
            "--round", str(round_num),
            "--aggregation", aggregation,
            "--global-out", global_out,
        )
        .add_inputs(common)
        # The new global model leaves the sub-workflow for the next round.
        .add_outputs(global_out, stage_out=True, register_replica=False)
    )
    for local_model, meta in zip(local_models, metas):
        agg_job.add_args("--local-model", local_model,
                         "--meta", meta)
        agg_job.add_inputs(local_model, meta)
    wf.add_jobs(agg_job)
    for tj in train_jobs:
        wf.add_dependency(agg_job, parents=[tj])

    # -- Validation (validation rounds only) ----------------------------------
    if is_validation_round:
        history_in = File(prev_history_lfn)
        best_in = File(prev_best_lfn)
        history_out = File(names["history_out"])
        best_out = File(names["best_out"])
        val_job = (
            Job("fl_validate",
                _id="validate", node_label=f"validate_r{round_num:03d}")
            .add_args(
                "--round", str(round_num),
                *interval_args,
                *pilot_args,
                "--global-model", global_out,
                "--history-in", history_in,
                "--best-in", best_in,
                "--history-out", history_out,
                "--best-out", best_out,
            )
            .add_inputs(global_out, history_in, best_in, common)
            .add_outputs(history_out, stage_out=True,
                         register_replica=False)
            .add_outputs(best_out, stage_out=True, register_replica=False)
        )
        for client in clients:
            val_job.add_args(
                "--client",
                f"{client['name']}:{client['sequences']}"
                f":{client['manifest']}",
            )
            val_job.add_inputs(File(client["sequences"]),
                               File(client["manifest"]))
        if final_best_lfn:
            final_best = File(final_best_lfn)
            val_job.add_args("--final-best", final_best)
            val_job.add_outputs(final_best, stage_out=True,
                                register_replica=False)
        wf.add_jobs(val_job)
        wf.add_dependency(val_job, parents=[agg_job])

    return wf, names
