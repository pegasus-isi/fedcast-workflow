#!/usr/bin/env python3

"""Builder for one FL-round Pegasus SubWorkflow.

Each federated round of Fed-Cast is its own sub-DAG (paper Sec. IV-C.2):

    fl_train_client x N   (parallel: one local epoch per client)
        -> fl_aggregate    (FedAvg: uniform or quadratic weights)
        -> fl_validate     (validation rounds only: chained history +
                            best-so-far checkpoint; final round emits the
                            best checkpoint in mct_infer format)

Two placement models, selected by the caller through the client specs:

  emulated (default)   client shards are Pegasus files staged to whatever
                       worker HTCondor matches, and fl_validate scores all
                       clients' validation splits itself. Faithful to the
                       paper, which emulates federation from a common MRMS
                       archive (paper Sec. III-B, Fig. 1 caption).

  cross-silo (--silos) each client dict carries a "requirements" ClassAd
                       expression and absolute on-worker paths. The shard
                       is never staged: the training job is pinned to the
                       silo holding it and reads it in place, and
                       validation fans out to pinned fl_validate_client
                       jobs that return per-client loss sums and counts.
                       Nothing in this sub-workflow moves a shard, though
                       each client does return small aggregates derived
                       from its data (n_train from training, n_val /
                       n_batches / losses from validation). The pooled
                       arms outside it get one silo_export copy per
                       client. See "Data placement" in README.md for the
                       full list of what leaves a silo.

Imported by workflow_generator.py, which writes the returned Workflow to a
YAML file, registers it in the replica catalog, and adds a SubWorkflow job
per round to the top-level DAG.
"""

from Pegasus.api import File, Job, Namespace, Workflow

COMMON_LFN = "fedcast_common.py"


def _place_client_job(job, client):
    """Pin a client job to its silo, or leave it free-floating.

    In cross-silo mode the client's shard is resident on the silo worker,
    so the job carries an HTCondor requirements expression instead of
    declaring the shard as a staged input. In emulated mode the shard is
    a normal Pegasus file and is declared as an input here.
    """
    if client.get("requirements"):
        job.add_profiles(Namespace.CONDOR, "requirements",
                         client["requirements"])
    else:
        job.add_inputs(File(client["sequences"]),
                       File(client["manifest"]))


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
    clients,            # list of client specs; see _client_specs()
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
            .add_inputs(global_in, common)
            .add_outputs(local_model, stage_out=False,
                         register_replica=False)
            .add_outputs(meta, stage_out=False, register_replica=False)
        )
        _place_client_job(job, client)
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
        val_client_jobs = []
        for client in clients:
            if not client.get("requirements"):
                # Emulated: the server reads this client's split itself.
                val_job.add_args(
                    "--client",
                    f"{client['name']}:{client['sequences']}"
                    f":{client['manifest']}",
                )
                val_job.add_inputs(File(client["sequences"]),
                                   File(client["manifest"]))
                continue
            # Cross-silo: score at the silo, ship metrics only.
            csite = client["name"]
            cmetrics = File(
                f"{method}_L{interval}_r{round_num:03d}_val_{csite}.json")
            cjob = (
                Job("fl_validate_client",
                    _id=f"valclient_{csite}",
                    node_label=f"valclient_{csite}_r{round_num:03d}")
                .add_args(
                    "--client",
                    f"{csite}:{client['sequences']}:{client['manifest']}",
                    "--round", str(round_num),
                    "--seed", str(seed),
                    *interval_args,
                    *pilot_args,
                    "--global-model", global_out,
                    "--metrics-out", cmetrics,
                )
                .add_inputs(global_out, common)
                .add_outputs(cmetrics, stage_out=False,
                             register_replica=False)
            )
            _place_client_job(cjob, client)
            wf.add_jobs(cjob)
            wf.add_dependency(cjob, parents=[agg_job])
            val_client_jobs.append(cjob)
            val_job.add_args("--client-metrics", cmetrics)
            val_job.add_inputs(cmetrics)
        if final_best_lfn:
            final_best = File(final_best_lfn)
            val_job.add_args("--final-best", final_best)
            val_job.add_outputs(final_best, stage_out=True,
                                register_replica=False)
        wf.add_jobs(val_job)
        wf.add_dependency(val_job, parents=[agg_job, *val_client_jobs])

    return wf, names
