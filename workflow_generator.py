#!/usr/bin/env python3

"""Pegasus workflow generator for the Fed-Cast reproduction (fedcast-workflow).

Reproduces the Fed-Cast paper (Xu et al., eScience 2026): federated vs.
centralized DGMR precipitation nowcasting on MRMS PrecipRate, evaluated with
an MCT-style multi-metric + TOPSIS pipeline against a STEPS baseline.
See SPEC.md for the full design, constraints, and validation criteria.

Pipeline phases (SPEC.md Sec. 1.1):
  A. Data       — fetch+crop MRMS per (site, month), build sequences per site
  B. Benchmark  — fetch WPC MPD / LSR / StormEvents, compile frozen event set
  C. Training   — per interval L: centralized DGMR and federated DGMR (Flower),
                  as chains of checkpointed segment jobs (SPEC open question 6b)
  D. Evaluation — MCT-style inference + verification per (method, L), TOPSIS
  E. Ablations  — E2.1 quadratic client weighting, E2.2 SAM centralized

Usage:
    # Pilot (2 sites, 1 month, tiny training budget):
    ./workflow_generator.py --test

    # Full E1 reproduction (7 sites, 48 months, 100 rounds/epochs):
    ./workflow_generator.py --start-month 2021-01 --months 48

    # Include ablations:
    ./workflow_generator.py --start-month 2021-01 --months 48 \
        --experiments e1 e21 e22
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from Pegasus.api import *

from fl_round import (
    COMMON_LFN,
    generate_round_workflow,
    init_file_names,
    round_file_names,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# The seven radar-centered regional clients (paper Sec. III-B).
# Coordinates from the NCEI NEXRAD station list; each client is a 3°x3°
# window centered on the radar, cropped to a 300x300 field at 0.01°.
# NOTE: PAHG (Alaska) is outside the MRMS CONUS domain — see SPEC.md open
# question 3. The fetch wrapper selects the MRMS ALASKA product tree for it.
# ----------------------------------------------------------------------
SITES = {
    "KBYX": {"lat": 24.5975, "lon": -81.7032, "domain": "CONUS",
             "desc": "subtropical maritime convection (Key West, FL)"},
    "KTLX": {"lat": 35.3331, "lon": -97.2778, "domain": "CONUS",
             "desc": "southern Great Plains convection (Oklahoma City, OK)"},
    "KVNX": {"lat": 36.7406, "lon": -98.1279, "domain": "CONUS",
             "desc": "central Great Plains convection (Vance AFB, OK)"},
    "KLGX": {"lat": 47.1158, "lon": -124.1069, "domain": "CONUS",
             "desc": "Pacific coastal / orographic (Langley Hill, WA)"},
    "KENX": {"lat": 42.5865, "lon": -74.0639, "domain": "CONUS",
             "desc": "inland Northeast (Albany, NY)"},
    "KBOX": {"lat": 41.9558, "lon": -71.1369, "domain": "CONUS",
             "desc": "coastal Northeast (Boston, MA)"},
    "PAHG": {"lat": 60.7259, "lon": -151.3512, "domain": "ALASKA",
             "desc": "high-latitude coastal/mountainous (Kenai, AK)"},
}

EVENT_SOURCES = ["mpd", "lsr", "storm_events"]

# Per-tool resource configuration.
TOOL_CONFIGS = {
    "fetch_crop_mrms":      {"memory": "4 GB",  "cores": 1, "container": "data"},
    "preprocess_sequences": {"memory": "16 GB", "cores": 4, "container": "data"},
    "fetch_events":         {"memory": "2 GB",  "cores": 1, "container": "data"},
    "build_benchmark":      {"memory": "4 GB",  "cores": 1, "container": "data"},
    "train_dgmr":           {"memory": "32 GB", "cores": 8, "container": "train",
                             "gpus": 1},
    "fl_init":              {"memory": "8 GB",  "cores": 2, "container": "train"},
    "fl_train_client":      {"memory": "32 GB", "cores": 8, "container": "train",
                             "gpus": 1},
    "fl_aggregate":         {"memory": "16 GB", "cores": 2, "container": "train"},
    "fl_validate":          {"memory": "32 GB", "cores": 8, "container": "train",
                             "gpus": 1},
    "mct_infer":            {"memory": "16 GB", "cores": 4, "container": "eval",
                             "gpus": 1},
    "mct_verify":           {"memory": "8 GB",  "cores": 4, "container": "eval"},
    "mct_topsis":           {"memory": "2 GB",  "cores": 1, "container": "eval"},
    "make_figures":         {"memory": "4 GB",  "cores": 1, "container": "eval"},
    "validate_report":      {"memory": "4 GB",  "cores": 1, "container": "eval"},
}


def month_range(start_month, n_months):
    """Return a list of YYYY-MM strings starting at start_month."""
    year, month = (int(x) for x in start_month.split("-"))
    months = []
    for _ in range(n_months):
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month > 12:
            month = 1
            year += 1
    return months


class FedCastWorkflow:
    """Fed-Cast reproduction workflow (see SPEC.md)."""

    wf = None
    sc = None
    tc = None
    rc = None
    props = None

    wf_name = "fedcast"

    def __init__(self, args):
        self.args = args
        self.dagfile = args.output
        self.wf_dir = str(Path(__file__).parent.resolve())
        self.shared_scratch_dir = os.path.join(self.wf_dir, "scratch")
        self.local_storage_dir = os.path.join(self.wf_dir, "output")

        self.sites = args.sites
        self.months = month_range(args.start_month, args.months)
        self.intervals = sorted(args.intervals)
        self.experiments = args.experiments

        # Per-site sequence/manifest files shared across phases.
        self.site_files = {}
        # Per-method best checkpoints: {(method, L): File}
        self.best_ckpts = {}
        # Benchmark file shared between Phase B and D.
        self.benchmark_file = None
        # Metric CSVs collected for TOPSIS pools: {method: {L: File}}
        self.metric_files = {}
        # FL-round sub-workflow YAMLs live here (generated files).
        self.rounds_dir = os.path.abspath("fl_rounds")
        os.makedirs(self.rounds_dir, exist_ok=True)
        # Path to the sub-workflow planning properties (set by
        # write_subworkflow_conf()).
        self.subwf_conf = None

    def write(self):
        if self.sc is not None:
            self.sc.write()
        self.props.write()
        self.rc.write()
        self.tc.write()
        self.wf.write(file=self.dagfile)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    def create_pegasus_properties(self):
        self.props = Properties()
        self.props["pegasus.transfer.threads"] = "16"
        # Jobs run inside containers whose OS differs from the submit
        # host (Debian 13 / Ubuntu 22 vs Ubuntu 24). Use the staged
        # worker package regardless of platform mismatch instead of
        # attempting a download the containers can't perform.
        self.props["pegasus.transfer.worker.package"] = "true"
        self.props["pegasus.transfer.worker.package.strict"] = "false"
        self.props["pegasus.transfer.worker.package.autodownload"] = \
            "false"
        # Throttle the (site x month) fetch fan-out so we do not hammer the
        # MRMS S3 bucket or the submit host's disk with 336 parallel pulls.
        self.props["dagman.maxjobs"] = str(self.args.max_concurrent_jobs)

    # ------------------------------------------------------------------
    # Site Catalog
    # ------------------------------------------------------------------
    def create_sites_catalog(self, exec_site_name="condorpool"):
        self.sc = SiteCatalog()

        local = Site("local").add_directories(
            Directory(
                Directory.SHARED_SCRATCH, self.shared_scratch_dir
            ).add_file_servers(
                FileServer("file://" + self.shared_scratch_dir, Operation.ALL)
            ),
            Directory(
                Directory.LOCAL_STORAGE, self.local_storage_dir
            ).add_file_servers(
                FileServer("file://" + self.local_storage_dir, Operation.ALL)
            ),
        )

        exec_site = (
            Site(exec_site_name)
            .add_condor_profile(universe="vanilla")
            .add_pegasus_profile(style="condor")
        )

        self.sc.add_sites(local, exec_site)

    # ------------------------------------------------------------------
    # Transformation Catalog
    # ------------------------------------------------------------------
    def create_transformation_catalog(self, exec_site_name="condorpool"):
        self.tc = TransformationCatalog()

        containers = {}
        for name in ("data", "train", "eval"):
            containers[name] = Container(
                f"fedcast_{name}",
                container_type=Container.SINGULARITY,
                image="file://" + os.path.join(
                    self.wf_dir, "Apptainer", f"FedCast_{name}.sif"
                ),
                image_site="local",
            )
            if name in ("train", "eval"):
                # Expose host GPUs inside Apptainer (harmless warning on
                # CPU-only nodes).
                containers[name].add_pegasus_profile(
                    container_arguments="--nv")
        self.tc.add_containers(*containers.values())

        for tool_name, cfg in TOOL_CONFIGS.items():
            memory = cfg["memory"]
            cores = cfg.get("cores", 1)
            gpus = cfg.get("gpus")
            if self.args.test:
                # Pilot mode: keep GPU requests (the pool has GPU workers)
                # but scale memory to fit ~15 GB machines; the FEDCAST_*
                # env shrinks the model to match (fedcast_common.py) and
                # flows into FL-round SubWorkflows via the shared
                # transformation catalog.
                memory = "8 GB" if (gpus or cfg["container"] == "train") \
                    else "2 GB"
                cores = min(cores, 2)
            tx = Transformation(
                tool_name,
                site=exec_site_name,
                pfn=os.path.join(self.wf_dir, f"bin/{tool_name}.py"),
                is_stageable=True,
                container=containers[cfg["container"]],
            ).add_pegasus_profile(memory=memory, cores=cores)
            if gpus:
                tx.add_condor_profile(request_gpus=str(gpus))
            if self.args.test and cfg["container"] in ("train", "eval"):
                tx.add_env(FEDCAST_MODEL_SIZE="128",
                           FEDCAST_BATCH_SIZE="1")
            self.tc.add_transformations(tx)

        # Local bridge for sub-workflow outputs consumed by parent jobs:
        # sub-workflows stage their outputs to the output site, while
        # parent stage-ins expect parent-scratch locations, so a plain
        # local cp re-introduces the file into normal parent staging.
        self.tc.add_transformations(
            Transformation("collect_file", site="local", pfn="/bin/cp",
                           is_stageable=False)
        )
        # Incremental deletion of superseded FL-chain artifacts on the
        # output site (each round's 582 MB global model would otherwise
        # accumulate — ~700 GB at full scale; SPEC open question 8).
        self.tc.add_transformations(
            Transformation("cleanup_file", site="local", pfn="/bin/rm",
                           is_stageable=False)
        )

    # ------------------------------------------------------------------
    # Replica Catalog — no pre-staged data inputs (everything is fetched
    # at runtime from public sources). Registers the shared training
    # helper module and, later, the generated FL-round sub-workflow YAMLs
    # (appended during create_workflow()).
    # ------------------------------------------------------------------
    def create_replica_catalog(self):
        self.rc = ReplicaCatalog()
        self.rc.add_replica(
            "local", COMMON_LFN,
            "file://" + os.path.join(self.wf_dir, "bin", COMMON_LFN),
        )

    # ------------------------------------------------------------------
    # Sub-workflow planning configuration: FL-round SubWorkflows are
    # planned at runtime and need catalog locations + a replica catalog
    # entry for the shared helper module.
    # ------------------------------------------------------------------
    def write_subworkflow_conf(self):
        sub_rc = ReplicaCatalog()
        sub_rc.add_replica(
            "local", COMMON_LFN,
            "file://" + os.path.join(self.wf_dir, "bin", COMMON_LFN),
        )
        sub_rc_path = os.path.abspath("fl_subwf_rc.yml")
        sub_rc.write(sub_rc_path)

        self.subwf_conf = os.path.abspath("fl_subwf.properties")
        with open(self.subwf_conf, "w") as f:
            f.write("pegasus.catalog.transformation=YAML\n")
            f.write("pegasus.catalog.transformation.file="
                    f"{os.path.abspath('transformations.yml')}\n")
            if not self.args.skip_sites_catalog:
                f.write("pegasus.catalog.site=YAML\n")
                f.write("pegasus.catalog.site.file="
                        f"{os.path.abspath('sites.yml')}\n")
            f.write("pegasus.catalog.replica=YAML\n")
            f.write(f"pegasus.catalog.replica.file={sub_rc_path}\n")
            # Sub-workflows are planned with THIS conf, not the parent's
            # pegasus.properties — the worker-package settings must be
            # repeated here or sub-DAG jobs fail inside containers
            # (PegasusLite platform mismatch, no curl in image).
            f.write("pegasus.transfer.threads=16\n")
            f.write("pegasus.transfer.worker.package=true\n")
            f.write("pegasus.transfer.worker.package.strict=false\n")
            f.write("pegasus.transfer.worker.package.autodownload=false\n")

    # ------------------------------------------------------------------
    # Workflow DAG
    # ------------------------------------------------------------------
    def create_workflow(self):
        self.wf = Workflow(self.wf_name, infer_dependencies=True)

        self._add_phase_a_data()
        self._add_phase_b_benchmark()
        self._add_phase_c_training()
        self._add_phase_d_evaluation()

    # -- Phase A: data construction ------------------------------------
    def _add_phase_a_data(self):
        # One fetch job per (domain, month): each MRMS PrecipRate file is
        # downloaded once and cropped for every site in that domain in the
        # same pass (crop-on-ingest — SPEC open question 8). Full-domain
        # GRIB2 files are never persisted.
        domains = {}
        for site in self.sites:
            domains.setdefault(SITES[site]["domain"], []).append(site)

        cropped_files = {site: [] for site in self.sites}
        for domain, domain_sites in domains.items():
            for month in self.months:
                outputs = {
                    site: File(f"{site}_{month}_cropped.nc")
                    for site in domain_sites
                }
                fetch_job = (
                    Job("fetch_crop_mrms",
                        _id=f"fetch_{domain}_{month}",
                        node_label=f"fetch_{domain}_{month}")
                    .add_args(
                        "--domain", domain,
                        "--month", month,
                        "--stride", str(self.args.frame_stride),
                    )
                    .add_pegasus_profiles(label=f"fetch_{domain}")
                    # Required source: wrapper retries transients with
                    # backoff, writes declared outputs even on permanent
                    # failure, then exits non-zero (SPEC constraint 17).
                    .add_dagman_profile(retry="2")
                )
                for site in domain_sites:
                    info = SITES[site]
                    fetch_job.add_args(
                        "--site",
                        f"{site}:{info['lat']}:{info['lon']}"
                        f":{outputs[site].lfn}",
                    )
                    fetch_job.add_outputs(outputs[site], stage_out=False,
                                          register_replica=False)
                    cropped_files[site].append(outputs[site])
                self.wf.add_jobs(fetch_job)

        for site in self.sites:
            month_files = cropped_files[site]
            sequences = File(f"{site}_sequences.npz")
            manifest = File(f"{site}_manifest.json")
            prep_job = (
                Job("preprocess_sequences",
                    _id=f"prep_{site}", node_label=f"prep_{site}")
                .add_args(
                    "--site", site,
                    "--output-sequences", sequences,
                    "--output-manifest", manifest,
                    "--val-seed", str(self.args.split_seed),
                    "--rain-threshold", str(self.args.rain_threshold),
                    "--min-rain-fraction", str(self.args.min_rain_fraction),
                )
                .add_inputs(*month_files)
                .add_outputs(sequences, stage_out=False,
                             register_replica=False)
                # Manifests are validation artifacts (SPEC Tier 0/1).
                .add_outputs(manifest, stage_out=True,
                             register_replica=False)
                .add_pegasus_profiles(label=site)
            )
            for f in month_files:
                prep_job.add_args("--input", f)
            self.wf.add_jobs(prep_job)

            self.site_files[site] = {
                "sequences": sequences, "manifest": manifest
            }

    # -- Phase B: event benchmark ---------------------------------------
    def _add_phase_b_benchmark(self):
        source_files = []
        for source in EVENT_SOURCES:
            events = File(f"events_{source}.json")
            job = (
                Job("fetch_events",
                    _id=f"fetch_events_{source}",
                    node_label=f"fetch_events_{source}")
                .add_args(
                    "--source", source,
                    "--start-month", self.months[0],
                    "--end-month", self.months[-1],
                    "--output", events,
                )
                .add_outputs(events, stage_out=False, register_replica=False)
                # Best-effort source: wrapper degrades gracefully (empty
                # output + exit 0); build_benchmark fails only if ALL
                # sources are empty (SPEC constraint 17).
                .add_dagman_profile(retry="2")
            )
            self.wf.add_jobs(job)
            source_files.append(events)

        self.benchmark_file = File("benchmark_events.csv")
        bench_job = (
            Job("build_benchmark",
                _id="build_benchmark", node_label="build_benchmark")
            .add_args(
                "--output", self.benchmark_file,
                "--seed", str(self.args.split_seed),
                "--max-events-per-site", str(self.args.max_events_per_site),
            )
            .add_inputs(*source_files)
            .add_outputs(self.benchmark_file, stage_out=True,
                         register_replica=False)
        )
        for f in source_files:
            bench_job.add_args("--events", f)
        for site in self.sites:
            info = SITES[site]
            bench_job.add_args(
                "--site", f"{site}:{info['lat']}:{info['lon']}"
            )
        self.wf.add_jobs(bench_job)

    # -- Phase C: training ------------------------------------------------
    def _client_specs(self):
        """LFN dicts for all clients, as consumed by fl_round."""
        return [
            {"name": site,
             "sequences": self.site_files[site]["sequences"].lfn,
             "manifest": self.site_files[site]["manifest"].lfn}
            for site in self.sites
        ]

    def _add_centralized_chain(self, method, interval, extra_args=None):
        """Chain of checkpointed centralized training segment jobs.

        Each segment runs `--segment-size` epochs, carrying a state
        tarball (weights + best-so-far + validation history) to the next
        segment. The final segment emits the best checkpoint (SPEC
        constraint 9: lowest generator validation loss).
        """
        total = self.args.rounds
        seg_size = min(self.args.segment_size, total)
        n_segments = (total + seg_size - 1) // seg_size

        common = File(COMMON_LFN)
        seq_inputs = [common]
        for site in self.sites:
            seq_inputs.append(self.site_files[site]["sequences"])
            seq_inputs.append(self.site_files[site]["manifest"])

        prev_state = None
        best_ckpt = File(f"{method}_L{interval}_best.ckpt")
        for k in range(n_segments):
            state_out = File(f"{method}_L{interval}_seg{k}_state.tar.gz")
            is_last = k == n_segments - 1
            job = (
                Job("train_dgmr",
                    _id=f"train_{method}_L{interval}_seg{k}",
                    node_label=f"train_{method}_L{interval}_seg{k}")
                .add_args(
                    "--interval-months", str(interval),
                    "--archive-start", self.months[0],
                    "--archive-months", str(len(self.months)),
                    "--segment-index", str(k),
                    "--segment-size", str(seg_size),
                    "--total-units", str(total),
                    "--validate-every", str(self.args.validate_every),
                    "--seed", str(self.args.train_seed),
                    "--state-out", state_out,
                )
                .add_inputs(*seq_inputs)
                .add_outputs(state_out, stage_out=False,
                             register_replica=False)
                .add_pegasus_profiles(label=f"{method}_L{interval}")
            )
            for site in self.sites:
                job.add_args(
                    "--client",
                    f"{site}:{self.site_files[site]['sequences'].lfn}"
                    f":{self.site_files[site]['manifest'].lfn}",
                )
            for arg in (extra_args or []):
                job.add_args(*arg)
            if self.args.limit_train_sequences:
                job.add_args("--limit-train-sequences",
                             str(self.args.limit_train_sequences))
            if prev_state is not None:
                job.add_args("--state-in", prev_state)
                job.add_inputs(prev_state)
            if is_last:
                job.add_args("--best-out", best_ckpt)
                job.add_outputs(best_ckpt, stage_out=True,
                                register_replica=False)
            self.wf.add_jobs(job)
            prev_state = state_out

        self.best_ckpts[(method, interval)] = best_ckpt

    def _add_federated_subworkflows(self, method, interval, aggregation):
        """Federated training: one SubWorkflow per FL round.

        fl_init seeds the global model; each round's sub-DAG fans out one
        local epoch per client, aggregates (FedAvg), and — on validation
        rounds — chains the history and best-so-far checkpoint. The final
        round emits {method}_L{interval}_best.ckpt.
        """
        rounds = self.args.rounds
        clients = self._client_specs()
        common = File(COMMON_LFN)

        init_names = init_file_names(method, interval)
        init_global = File(init_names["global_out"])
        init_history = File(init_names["history_out"])
        init_best = File(init_names["best_out"])
        init_job = (
            Job("fl_init",
                _id=f"flinit_{method}_L{interval}",
                node_label=f"flinit_{method}_L{interval}")
            .add_args(
                "--seed", str(self.args.train_seed),
                "--interval-months", str(interval),
                "--aggregation", aggregation,
                "--global-out", init_global,
                "--history-out", init_history,
                "--best-out", init_best,
            )
            .add_inputs(common)
            .add_outputs(init_global, stage_out=False,
                         register_replica=False)
            .add_outputs(init_history, stage_out=False,
                         register_replica=False)
            .add_outputs(init_best, stage_out=False,
                         register_replica=False)
            .add_pegasus_profiles(label=f"{method}_L{interval}")
        )
        self.wf.add_jobs(init_job)

        prev_global = init_names["global_out"]
        prev_history = init_names["history_out"]
        prev_best = init_names["best_out"]
        # The sub-workflow stages its final-best checkpoint to the output
        # site under a "_sub" name; a collect_file bridge job then brings
        # it into parent staging under the canonical LFN (parent stage-ins
        # cannot see sub-workflow output locations directly).
        sub_best_lfn = f"{method}_L{interval}_best_sub.ckpt"
        last_subwf = None

        # Chained artifacts accumulate on the output site (the next
        # round's runtime planning locates them there), so superseded
        # ones are deleted incrementally with a two-round safety window:
        # a rescue replan of round r+1 needs at most round r's global.
        global_chain = []   # global-model LFNs in round order
        val_chain = []      # (history_lfn, bestsofar_lfn) per val round
        cleaned = set()

        def add_cleanup(tag, targets, parent_job):
            targets = [t for t in targets if t not in cleaned]
            if not targets:
                return
            job = Job("cleanup_file",
                      _id=f"clean_{method}_L{interval}_{tag}",
                      node_label=f"clean_{method}_L{interval}_{tag}")
            job.add_args("-f", *[
                os.path.join(self.local_storage_dir, t) for t in targets
            ])
            self.wf.add_jobs(job)
            self.wf.add_dependency(job, parents=[parent_job])
            cleaned.update(targets)

        limit = getattr(self.args, "limit_train_sequences", None)
        for r in range(rounds):
            is_final = r == rounds - 1
            is_validation = ((r + 1) % self.args.validate_every == 0
                             or is_final)

            round_wf, names = generate_round_workflow(
                method=method,
                interval=interval,
                round_num=r,
                clients=clients,
                prev_global_lfn=prev_global,
                prev_history_lfn=prev_history,
                prev_best_lfn=prev_best,
                aggregation=aggregation,
                archive_start=self.months[0],
                archive_months=len(self.months),
                seed=self.args.train_seed,
                is_validation_round=is_validation,
                final_best_lfn=sub_best_lfn if is_final else None,
                limit_train_sequences=limit,
            )
            yml_lfn = f"{method}_L{interval}_r{r:03d}.yml"
            yml_path = os.path.join(self.rounds_dir, yml_lfn)
            round_wf.write(yml_path)
            self.rc.add_replica("local", yml_lfn,
                                "file://" + os.path.abspath(yml_path))

            subwf = SubWorkflow(
                yml_lfn, is_planned=False,
                _id=f"round_{method}_L{interval}_r{r:03d}",
                node_label=f"round_{method}_L{interval}_r{r:03d}",
            )
            subwf.add_args("--conf", self.subwf_conf,
                           "--output-sites", "local")
            subwf.add_inputs(File(prev_global))
            for site in self.sites:
                subwf.add_inputs(self.site_files[site]["sequences"],
                                 self.site_files[site]["manifest"])
            subwf.add_outputs(File(names["global_out"]), stage_out=False,
                              register_replica=False)
            if is_validation:
                subwf.add_inputs(File(prev_history), File(prev_best))
                subwf.add_outputs(File(names["history_out"]),
                                  stage_out=is_final,
                                  register_replica=False)
                subwf.add_outputs(File(names["best_out"]), stage_out=False,
                                  register_replica=False)
                prev_history = names["history_out"]
                prev_best = names["best_out"]
            if is_final:
                subwf.add_outputs(File(sub_best_lfn), stage_out=True,
                                  register_replica=False)
            self.wf.add_jobs(subwf)
            last_subwf = subwf
            prev_global = names["global_out"]

            global_chain.append(names["global_out"])
            if is_validation:
                val_chain.append((names["history_out"],
                                  names["best_out"]))
                # Keep the last two of each chain; delete anything older.
                stale = list(global_chain[:-2])
                for hist, best in val_chain[:-2]:
                    stale.extend([hist, best])
                add_cleanup(f"r{r:03d}", stale, subwf)

        best_ckpt = File(f"{method}_L{interval}_best.ckpt")
        collect_job = (
            Job("collect_file",
                _id=f"collect_{method}_L{interval}",
                node_label=f"collect_{method}_L{interval}")
            .add_args(os.path.join(self.local_storage_dir, sub_best_lfn),
                      best_ckpt)
            .add_outputs(best_ckpt, stage_out=True, register_replica=False)
        )
        self.wf.add_jobs(collect_job)
        # No declared file input (the source is an absolute output-site
        # path), so the ordering edge must be explicit.
        self.wf.add_dependency(collect_job, parents=[last_subwf])

        # Once the canonical best checkpoint is collected, every chained
        # artifact except the final history JSON is superseded.
        final_stale = list(global_chain) + [sub_best_lfn]
        for hist, best in val_chain:
            final_stale.append(best)
        for hist, best in val_chain[:-1]:
            final_stale.append(hist)
        add_cleanup("final", final_stale, collect_job)

        self.best_ckpts[(method, interval)] = best_ckpt

    def _add_phase_c_training(self):
        for interval in self.intervals:
            if "e1" in self.experiments:
                self._add_centralized_chain("cen", interval)
                self._add_federated_subworkflows("fed", interval,
                                                 "uniform")
            if "e21" in self.experiments:
                self._add_federated_subworkflows("fedq", interval,
                                                 "quadratic")
            if "e22" in self.experiments:
                for rho in self.args.sam_rho:
                    method = f"censam{str(rho).replace('0.', '')}"
                    self._add_centralized_chain(
                        method, interval,
                        extra_args=[("--sam-rho", str(rho))],
                    )

    # -- Phase D: evaluation ----------------------------------------------
    def _add_eval_pair(self, method, interval, ckpt=None):
        """Add mct_infer + mct_verify for one (method, interval)."""
        tag = f"{method}_L{interval}" if interval else method
        forecasts = File(f"{tag}_forecasts.npz")

        seq_inputs = []
        for site in self.sites:
            seq_inputs.append(self.site_files[site]["sequences"])
            seq_inputs.append(self.site_files[site]["manifest"])

        infer_job = (
            Job("mct_infer", _id=f"infer_{tag}", node_label=f"infer_{tag}")
            .add_args(
                "--method", method,
                "--benchmark", self.benchmark_file,
                "--ensemble-size",
                str(20 if method == "steps" else self.args.dgmr_ensemble),
                "--output", forecasts,
            )
            .add_inputs(self.benchmark_file, *seq_inputs)
            .add_outputs(forecasts, stage_out=False, register_replica=False)
            .add_pegasus_profiles(label=tag)
        )
        for site in self.sites:
            infer_job.add_args(
                "--client",
                f"{site}:{self.site_files[site]['sequences'].lfn}"
                f":{self.site_files[site]['manifest'].lfn}",
            )
        if ckpt is not None:
            infer_job.add_args("--checkpoint", ckpt)
            infer_job.add_inputs(ckpt)
        if self.args.fallback_test_instances:
            infer_job.add_args("--fallback-test-instances",
                               str(self.args.fallback_test_instances))
        self.wf.add_jobs(infer_job)

        metrics = File(f"{tag}_metrics.csv")
        verify_job = (
            Job("mct_verify", _id=f"verify_{tag}", node_label=f"verify_{tag}")
            .add_args(
                "--method", method,
                "--forecasts", forecasts,
                "--benchmark", self.benchmark_file,
                "--rain-threshold", str(self.args.rain_threshold),
                "--output", metrics,
            )
            .add_inputs(forecasts, self.benchmark_file, *seq_inputs)
            .add_outputs(metrics, stage_out=True, register_replica=False)
            .add_pegasus_profiles(label=tag)
        )
        if interval:
            verify_job.add_args("--interval", str(interval))
        for site in self.sites:
            verify_job.add_args(
                "--client",
                f"{site}:{self.site_files[site]['sequences'].lfn}"
                f":{self.site_files[site]['manifest'].lfn}",
            )
        self.wf.add_jobs(verify_job)

        self.metric_files.setdefault(method, {})[interval] = metrics
        return metrics

    def _add_phase_d_evaluation(self):
        # STEPS is training-free: a single evaluation reused by all pools.
        steps_metrics = self._add_eval_pair("steps", None)

        for (method, interval), ckpt in self.best_ckpts.items():
            self._add_eval_pair(method, interval, ckpt=ckpt)

        # TOPSIS pools — separately normalized per experiment (SPEC
        # constraint 14). E1: cen+fed+steps; E2.1: fedq+cen+steps;
        # E2.2: censam*+fed+steps.
        pools = {}
        if "e1" in self.experiments:
            pools["e1"] = ["cen", "fed"]
        if "e21" in self.experiments:
            pools["e21"] = ["cen", "fedq"]
        if "e22" in self.experiments:
            pools["e22"] = ["fed"] + [
                f"censam{str(rho).replace('0.', '')}"
                for rho in self.args.sam_rho
            ]

        topsis_files = []
        for pool_name, methods in pools.items():
            pool_inputs = [steps_metrics]
            topsis_out = File(f"{pool_name}_topsis.csv")
            topsis_job = (
                Job("mct_topsis",
                    _id=f"topsis_{pool_name}",
                    node_label=f"topsis_{pool_name}")
                .add_args("--pool", pool_name, "--output", topsis_out)
                .add_outputs(topsis_out, stage_out=True,
                             register_replica=False)
            )
            topsis_job.add_args("--metrics", steps_metrics)
            for method in methods:
                for interval, mfile in self.metric_files[method].items():
                    topsis_job.add_args("--metrics", mfile)
                    pool_inputs.append(mfile)
            topsis_job.add_inputs(*pool_inputs)
            self.wf.add_jobs(topsis_job)
            topsis_files.append(topsis_out)

        figures = File("figures.tar.gz")
        fig_job = (
            Job("make_figures", _id="make_figures", node_label="make_figures")
            .add_args("--output", figures)
            .add_inputs(*topsis_files)
            .add_outputs(figures, stage_out=True, register_replica=False)
        )
        for f in topsis_files:
            fig_job.add_args("--topsis", f)
        self.wf.add_jobs(fig_job)

        # Tiered validation report (SPEC Sec. 5).
        report = File("validation_report.md")
        manifests = [self.site_files[s]["manifest"] for s in self.sites]
        val_job = (
            Job("validate_report",
                _id="validate_report", node_label="validate_report")
            .add_args("--output", report)
            .add_inputs(*topsis_files, *manifests, self.benchmark_file)
            .add_outputs(report, stage_out=True, register_replica=False)
        )
        for f in topsis_files:
            val_job.add_args("--topsis", f)
        for m in manifests:
            val_job.add_args("--manifest", m)
        val_job.add_args("--benchmark", self.benchmark_file)
        self.wf.add_jobs(val_job)


# ======================================================================
# main()
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Fed-Cast reproduction workflow generator (see SPEC.md)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --test                          # pilot: 2 sites, 1 month, tiny budget
  %(prog)s --start-month 2021-01 --months 48
  %(prog)s --start-month 2021-01 --months 48 --experiments e1 e21 e22
""",
    )

    # --- Standard Pegasus arguments ---
    parser.add_argument("-s", "--skip-sites-catalog", action="store_true",
                        help="Skip site catalog creation")
    parser.add_argument("-e", "--execution-site-name", metavar="STR",
                        type=str, default="condorpool",
                        help="Execution site name (default: condorpool)")
    parser.add_argument("-o", "--output", metavar="STR", type=str,
                        default="workflow.yml",
                        help="Output file (default: workflow.yml)")

    # --- Data / archive ---
    parser.add_argument("--start-month", type=str, default="2021-01",
                        help="Archive start month YYYY-MM (default: 2021-01; "
                             "SPEC open question 1)")
    parser.add_argument("--months", type=int, default=48,
                        help="Archive length in months (default: 48)")
    parser.add_argument("--sites", type=str, nargs="+",
                        default=list(SITES.keys()), choices=list(SITES.keys()),
                        help="Radar sites / federated clients (default: all 7)")

    # --- Training ---
    parser.add_argument("--intervals", type=int, nargs="+",
                        default=[1, 3, 6, 12, 24, 48],
                        help="Training intervals L in months "
                             "(default: 1 3 6 12 24 48)")
    parser.add_argument("--rounds", type=int, default=100,
                        help="FL rounds / centralized epochs (default: 100)")
    parser.add_argument("--segment-size", type=int, default=10,
                        help="Rounds/epochs per training segment job "
                             "(default: 10)")
    parser.add_argument("--validate-every", type=int, default=5,
                        help="Validation cadence in rounds/epochs "
                             "(default: 5, per paper)")
    parser.add_argument("--experiments", type=str, nargs="+", default=["e1"],
                        choices=["e1", "e21", "e22"],
                        help="Experiment pools to build (default: e1)")
    parser.add_argument("--sam-rho", type=float, nargs="+",
                        default=[0.025, 0.0125],
                        help="SAM perturbation radii for E2.2")
    parser.add_argument("--train-seed", type=int, default=42,
                        help="Training RNG seed, recorded in run metadata")

    # --- Preprocessing / evaluation knobs (SPEC open questions 2, 5) ---
    parser.add_argument("--split-seed", type=int, default=1337,
                        help="Seed for validation-split sampling and "
                             "benchmark event selection")
    parser.add_argument("--rain-threshold", type=float, default=0.1,
                        help="Rain/no-rain threshold in mm/h (default: 0.1)")
    parser.add_argument("--min-rain-fraction", type=float, default=0.05,
                        help="Min wet-pixel fraction for sequence retention "
                             "(calibration knob; SPEC open question 2)")
    parser.add_argument("--max-events-per-site", type=int, default=20,
                        help="Benchmark balancing cap per site "
                             "(SPEC open question 5)")
    parser.add_argument("--dgmr-ensemble", type=int, default=6,
                        help="DGMR stochastic ensemble size K (default: 6)")
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="Keep every Nth 2-min MRMS frame (default: 1 = "
                             "full cadence; >1 subsamples for pilot runs)")
    parser.add_argument("--limit-train-sequences", type=int, default=None,
                        help="PILOT ONLY: cap train/val sequences per "
                             "client in training jobs")
    parser.add_argument("--fallback-test-instances", type=int, default=0,
                        help="PILOT ONLY: mct_infer falls back to N test "
                             "sequences per site when no event matches")
    parser.add_argument("--max-concurrent-jobs", type=int, default=20,
                        help="DAGMan job throttle (default: 20)")

    # --- Pilot mode ---
    parser.add_argument("--test", action="store_true",
                        help="Pilot mode: 2 sites, 1 month, 2 rounds, "
                             "interval [1] — end-to-end smoke test")

    args = parser.parse_args()

    if args.test:
        args.sites = ["KTLX", "KENX"]
        # 2024-01 is a month verified to contain in-window benchmark
        # events for these sites (winter months often have none).
        args.start_month = "2024-01"
        args.months = 1
        args.intervals = [1]
        args.rounds = 2
        args.segment_size = 1
        args.validate_every = 1
        args.max_events_per_site = 2
        args.frame_stride = 15
        args.limit_train_sequences = 4
        args.fallback_test_instances = 2
        logger.info("PILOT MODE: %s, %d month(s), %d round(s)",
                    args.sites, args.months, args.rounds)

    # --- Validation ---
    if max(args.intervals) > args.months:
        print(f"Error: largest interval ({max(args.intervals)}) exceeds "
              f"archive length ({args.months} months)")
        sys.exit(1)
    if "e22" in args.experiments and not args.sam_rho:
        print("Error: --experiments e22 requires at least one --sam-rho")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("FED-CAST WORKFLOW GENERATOR")
    logger.info("=" * 70)
    logger.info(f"Sites: {args.sites}")
    logger.info(f"Archive: {args.start_month} + {args.months} months")
    logger.info(f"Intervals: {args.intervals}")
    logger.info(f"Experiments: {args.experiments}")
    logger.info(f"Training budget: {args.rounds} rounds/epochs in segments "
                f"of {args.segment_size}")
    logger.info(f"Execution site: {args.execution_site_name}")
    logger.info("=" * 70)

    try:
        workflow = FedCastWorkflow(args)
        workflow.create_pegasus_properties()
        if not args.skip_sites_catalog:
            workflow.create_sites_catalog(
                exec_site_name=args.execution_site_name)
        workflow.create_transformation_catalog(
            exec_site_name=args.execution_site_name)
        workflow.create_replica_catalog()
        workflow.write_subworkflow_conf()
        workflow.create_workflow()
        workflow.write()

        logger.info(f"\nWorkflow written to {args.output}")
        logger.info(f"Submit: pegasus-plan --submit "
                    f"-s {args.execution_site_name} -o local {args.output}")
    except Exception as e:
        logger.error(f"Failed to generate workflow: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
