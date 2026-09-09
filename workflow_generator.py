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

Sites. The workflow does not describe any scheduler. Every transformation
states only what it needs — cores, memory, GPUs, runtime — and GPU jobs
carry the Pegasus tag "gpu" (pinned silo jobs carry "silo_<SITE>" or
"silo_<SITE>_gpu"). Where those jobs run, and what a tag means there
(partition, account, constraints, --nodelist, ClassAd requirements), comes
from the site catalog: a hosted one selected in ~/.pegasusrc
(pegasus.catalog.site.repo.file), optionally overlaid by a local sites.yml
written by custom_sites.py. Needs Pegasus >= 5.1.3dev / 6.0.0dev for tags.

Usage:
    # Pilot (2 sites, 1 month, tiny training budget):
    ./workflow_generator.py --test
    pegasus-plan --submit -s compute --output-dir output workflow.yml

    # Full E1 reproduction (7 sites, 48 months, 100 rounds/epochs):
    ./workflow_generator.py --start-month 2021-01 --months 48

    # Include ablations:
    ./workflow_generator.py --start-month 2021-01 --months 48 \
        --experiments e1 e21 e22

    # Cross-silo placement (the site catalog defines the silo tags):
    ./custom_sites.py --style slurm --silos silos.yml
    ./workflow_generator.py --start-month 2021-01 --months 48 \
        --silos silos.yml
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
from silo_map import (
    check_silo_tags, hosted_catalog, load_silo_map, silo_tags,
    site_stages_on_compute, site_submission_style,
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

# Per-tool resource configuration: the only resource statements the
# workflow makes. Everything scheduler-specific (queue/partition, account,
# GPU constraints, node pins) belongs in the site catalog, keyed by tag.
#
# runtime is a wall-clock budget in seconds. Batch sites kill a job that
# exceeds it, so the values are deliberately generous for full-scale inputs
# and doubled on retry (RUNTIME_EXPR); --runtime-scale multiplies them all.
# Sizes stay workflow knobs on purpose: transformation-catalog profiles
# outrank site-catalog ones, so a tag cannot shrink them — what a tag does
# is say how the request is expressed on the pool (partition, constraint).
TOOL_CONFIGS = {
    "fetch_crop_mrms":      {"memory": "4 GB",  "cores": 1, "container": "data",
                             "runtime": 6 * 3600},
    "preprocess_sequences": {"memory": "16 GB", "cores": 4, "container": "data",
                             "runtime": 4 * 3600},
    "fetch_events":         {"memory": "2 GB",  "cores": 1, "container": "data",
                             "runtime": 3600},
    "build_benchmark":      {"memory": "4 GB",  "cores": 1, "container": "data",
                             "runtime": 1800},
    "train_dgmr":           {"memory": "32 GB", "cores": 8, "container": "train",
                             "gpus": 1, "runtime": 12 * 3600},
    "fl_init":              {"memory": "8 GB",  "cores": 2, "container": "train",
                             "runtime": 1800},
    "fl_train_client":      {"memory": "32 GB", "cores": 8, "container": "train",
                             "gpus": 1, "runtime": 2 * 3600},
    "fl_aggregate":         {"memory": "16 GB", "cores": 2, "container": "train",
                             "runtime": 1800},
    "fl_validate":          {"memory": "32 GB", "cores": 8, "container": "train",
                             "gpus": 1, "runtime": 3600},
    "fl_validate_client":   {"memory": "32 GB", "cores": 8, "container": "train",
                             "gpus": 1, "runtime": 3600},
    "mct_infer":            {"memory": "16 GB", "cores": 4, "container": "eval",
                             "gpus": 1, "runtime": 4 * 3600},
    "mct_verify":           {"memory": "8 GB",  "cores": 4, "container": "eval",
                             "runtime": 2 * 3600},
    "mct_topsis":           {"memory": "2 GB",  "cores": 1, "container": "eval",
                             "runtime": 900},
    "make_figures":         {"memory": "4 GB",  "cores": 1, "container": "eval",
                             "runtime": 900},
    "validate_report":      {"memory": "4 GB",  "cores": 1, "container": "eval",
                             "runtime": 900},
}

# Uncontainerized helpers: the shard copy at a silo, and the two submit-host
# bridges. Small and quick, but a batch site still needs a runtime for them.
HELPER_RUNTIME = 1800

# Tag carried by every GPU job. Hosted site catalogs define it (Unity,
# Perlmutter: GPU partition, gpus, --nv); custom_sites.py adds it for pools
# that have no hosted catalog.
GPU_TAG = "gpu"

# Double the wall-clock budget on retry (needs dagman.post.arguments=-U,
# set in the properties, and the pythonsed package on the submit host).
RUNTIME_EXPR = ("int(pegasus_job_runtime * 2) if job_retry > 0 "
                "else pegasus_job_runtime")


def gpu_tag_for(tool_name):
    """The tag a job of this tool carries when not pinned, or None."""
    return GPU_TAG if TOOL_CONFIGS[tool_name].get("gpus") else None


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
    tc = None
    rc = None
    props = None

    wf_name = "fedcast"

    def __init__(self, args):
        self.args = args
        self.dagfile = args.output
        self.wf_dir = str(Path(__file__).parent.resolve())
        # Where staged-out files land on the submit host. Both the parent
        # (pegasus-plan --output-dir) and every FL-round sub-workflow use
        # it, so the collect_file/cleanup_file bridges below can address
        # sub-workflow outputs by path without knowing the site catalog's
        # local-site layout.
        self.local_storage_dir = os.path.abspath(args.output_dir)

        self.sites = args.sites
        # Cross-silo placement map (None = emulated placement, the paper's
        # own model: clients reconstructed from one central MRMS archive).
        self.silos = (load_silo_map(args.silos, args.sites)
                      if args.silos else None)
        self.months = month_range(args.start_month, args.months)
        self.intervals = sorted(args.intervals)
        self.experiments = args.experiments

        # Per-site sequence/manifest files shared across phases.
        self.site_files = {}
        # Per-site preprocess jobs, for explicit ordering edges in silo
        # mode (where shards are not declared files, so Pegasus cannot
        # infer the dependency).
        self.prep_jobs = {}
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
        # A site-catalog overlay in the working directory (custom_sites.py
        # writes one). Pegasus merges it over the hosted catalog for the
        # parent automatically; sub-workflows are told about it explicitly.
        self.local_sites_yml = (os.path.abspath(args.sites_yml)
                                if os.path.isfile(args.sites_yml) else None)

    def log_placement(self):
        """Say how client data will be placed, and what that assumes."""
        if not self.silos:
            return
        logger.info(f"Shard directory: {self.silos['data_dir']}")
        if self.silos["scratch_root"]:
            logger.warning(
                "%s: data_dir %s is under %s, which HTCondor "
                "(MOUNT_UNDER_SCRATCH) and Slurm (job_container/tmpfs) "
                "commonly make private per job and delete when the job "
                "ends. Each job would get its own empty copy and the shard "
                "would be gone before the next round. Use a home-relative "
                "path, or an absolute path outside %s.",
                self.silos["path"], self.silos["data_dir"],
                self.silos["scratch_root"], self.silos["scratch_root"])
        if self.silos["needs_bind"]:
            logger.info(
                "  bind-mounted pool-wide — every worker needs this "
                "directory (tools/silo_worker_setup.sh none)")
        else:
            logger.info(
                "  inside the job user's home, which Apptainer mounts — "
                "no bind, no worker setup needed")
            logger.info(
                "  assumes every job on a worker runs as the same user; "
                "tools/silo_check.py flags pools where it does not")

    def write(self):
        self.props.write()
        self.rc.write()
        self.tc.write()
        self.wf.write(file=self.dagfile)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    def base_properties(self):
        """Properties shared by the parent and the FL-round sub-workflows.

        Starts from ~/.pegasusrc so the hosted-site-catalog selection
        (pegasus.catalog.site.repo.file) and any site credentials the user
        keeps there also reach the sub-workflow planner, which is invoked
        with its own --conf and would otherwise not see them.
        """
        rc = Path.home() / ".pegasusrc"
        props = Properties.load(rc) if rc.is_file() else Properties()
        props["pegasus.transfer.threads"] = "16"
        # Jobs run inside containers whose OS differs from the submit
        # host (Debian 13 / Ubuntu 22 vs Ubuntu 24). Use the staged
        # worker package regardless of platform mismatch instead of
        # attempting a download the containers can't perform.
        props["pegasus.transfer.worker.package"] = "true"
        props["pegasus.transfer.worker.package.strict"] = "false"
        props["pegasus.transfer.worker.package.autodownload"] = "false"
        # Symlink an input instead of copying it when the replica
        # catalog says it already sits on the execution site. A no-op
        # when they differ, so it is always on.
        props["pegasus.transfer.links"] = "true"
        if self.args.shared_filesystem:
            # Let PegasusLite pull inputs straight from the input site
            # rather than through the staging server. This assumes the
            # worker nodes can read that site — true on a cluster with a
            # shared filesystem, false on a condor pool staging over
            # HTCondor file transfer, where it would make jobs chase
            # file:// paths that do not exist on the worker. Worth having
            # where it applies: the three container images are several GB
            # each.
            props["pegasus.transfer.bypass.input.staging"] = "true"
        if self.local_sites_yml:
            # Pegasus merges a local sites.yml over the hosted catalog when
            # it is in the planner's working directory. Naming it here means
            # the overlay — which is where the silo pins live — is found no
            # matter where pegasus-plan is run from, and reaches the
            # sub-workflow planner, which runs elsewhere.
            props["pegasus.catalog.site"] = "YAML"
            props["pegasus.catalog.site.file"] = self.local_sites_yml
        # Retry transients (node failures, preemption, walltime on the
        # first attempt) and let the *.expr profiles rewrite resource
        # requests on retry — RUNTIME_EXPR doubles the wall-clock budget.
        props["dagman.retry"] = str(self.args.retries)
        props["dagman.post.arguments"] = "-U"
        # Throttle the (site x month) fetch fan-out so we do not hammer the
        # MRMS S3 bucket or the submit host's disk with 336 parallel pulls.
        props["dagman.maxjobs"] = str(self.args.max_concurrent_jobs)
        return props

    def create_pegasus_properties(self):
        self.props = self.base_properties()

    # ------------------------------------------------------------------
    # Transformation Catalog
    # ------------------------------------------------------------------
    def create_transformation_catalog(self):
        """Executables, containers and per-tool resource needs.

        Every stageable transformation lives on the local site and is
        shipped to wherever the job runs, so nothing here names an
        execution site. Resource needs are the portable four — cores,
        memory, gpus, runtime — and nothing else: no ClassAds, no
        partitions. Those come from the site catalog by tag.
        """
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
            cargs = []
            if name in ("train", "eval"):
                # Expose host GPUs inside Apptainer (harmless warning on
                # CPU-only nodes).
                cargs.append("--nv")
            if (self.silos and self.silos["needs_bind"]
                    and name in ("data", "train")):
                # Shards outside Apptainer's default mounts have to be
                # bound in, and container.arguments is catalog-scope, so
                # this bind applies to every job using the container —
                # including unpinned ones on workers that hold no data.
                # Each of those workers therefore needs the directory to
                # exist (tools/silo_worker_setup.sh none). A home- or
                # tmp-relative data_dir avoids all of this.
                cargs.append(f"--bind {self.silos['data_dir']}")
            if cargs:
                containers[name].add_pegasus_profile(
                    container_arguments=" ".join(cargs))
        self.tc.add_containers(*containers.values())

        cap_gb = self.args.max_job_memory_gb
        for tool_name, cfg in TOOL_CONFIGS.items():
            memory_gb = int(cfg["memory"].split()[0])
            cores = cfg.get("cores", 1)
            if cap_gb:
                # Cap requests so jobs match small-RAM pools (the FABRIC
                # slice advertises ~14 GB usable per slot).
                memory_gb = min(memory_gb, cap_gb)
                cores = min(cores, self.args.max_job_cores)
            tx = Transformation(
                tool_name,
                site="local",
                pfn=os.path.join(self.wf_dir, f"bin/{tool_name}.py"),
                is_stageable=True,
                container=containers[cfg["container"]],
            ).add_pegasus_profile(
                memory=f"{memory_gb} GB",
                cores=cores,
                runtime=self.runtime(cfg["runtime"]),
                runtime_expr=RUNTIME_EXPR,
            )
            if cfg.get("gpus"):
                # How a GPU is requested (request_gpus, --gpus, a gres
                # string) and which cards qualify are the site's business:
                # the job's "gpu" tag selects that from the site catalog.
                tx.add_pegasus_profile(gpus=cfg["gpus"])
            if cfg["container"] in ("train", "eval"):
                # Model geometry travels as env so it also reaches
                # FL-round SubWorkflow jobs via the shared catalog.
                tx.add_env(FEDCAST_MODEL_SIZE=str(self.args.model_size),
                           FEDCAST_BATCH_SIZE=str(self.args.batch_size))
            self.tc.add_transformations(tx)

        # Local bridge for sub-workflow outputs consumed by parent jobs:
        # sub-workflows stage their outputs to the output site, while
        # parent stage-ins expect parent-scratch locations, so a plain
        # local cp re-introduces the file into normal parent staging.
        self.tc.add_transformations(
            Transformation("collect_file", site="local", pfn="/bin/cp",
                           is_stageable=False)
            .add_pegasus_profile(runtime=HELPER_RUNTIME)
        )
        if self.silos:
            # Explicit egress boundary: copies a resident shard out of its
            # silo into Pegasus staging for the pooled consumers (the
            # centralized baseline and MCT evaluation). The federated arm
            # never uses this path — that asymmetry is the point of the
            # paper's communication-volume comparison (Sec. V, Eq. 7).
            # Staged shell script rather than /bin/cp: it expands a
            # home-relative shard path, which Pegasus does not do for
            # job arguments, and reports a missing shard clearly.
            self.tc.add_transformations(
                Transformation("silo_export", site="local",
                               pfn=os.path.join(self.wf_dir,
                                                "bin/silo_export.sh"),
                               is_stageable=True)
                .add_pegasus_profile(cores=1, memory="1 GB",
                                     runtime=HELPER_RUNTIME)
            )
        # Incremental deletion of superseded FL-chain artifacts on the
        # output site (each round's 582 MB global model would otherwise
        # accumulate — ~700 GB at full scale; SPEC open question 8).
        self.tc.add_transformations(
            Transformation("cleanup_file", site="local", pfn="/bin/rm",
                           is_stageable=False)
            .add_pegasus_profile(runtime=HELPER_RUNTIME)
        )

    def runtime(self, seconds):
        """A tool's wall-clock budget after --runtime-scale."""
        return max(60, int(round(seconds * self.args.runtime_scale)))

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

        # Sub-workflows are planned with THIS conf, not the parent's
        # pegasus.properties, so everything in base_properties() — the
        # hosted site catalog selection, worker-package and transfer
        # settings — is repeated here, plus the catalog locations.
        props = self.base_properties()
        props["pegasus.catalog.transformation"] = "YAML"
        props["pegasus.catalog.transformation.file"] = \
            os.path.abspath("transformations.yml")
        props["pegasus.catalog.replica"] = "YAML"
        props["pegasus.catalog.replica.file"] = sub_rc_path
        self.subwf_conf = os.path.abspath("fl_subwf.properties")
        props.write(self.subwf_conf)

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
                # fedcast_common provides the export-surface check the
                # wrapper runs before writing its manifest.
                .add_inputs(*month_files, File(COMMON_LFN))
                # Manifests are validation artifacts (SPEC Tier 0/1).
                .add_outputs(manifest, stage_out=True,
                             register_replica=False)
                .add_pegasus_profiles(label=site)
            )
            for f in month_files:
                prep_job.add_args("--input", f)

            if self.silos:
                # Cross-silo: build the shard ON the silo that owns this
                # client and leave it there. It is not a Pegasus output,
                # so no federated job ever stages it off the worker.
                silo_dir = self.silo_paths(site)["dir"]
                prep_job.add_args("--silo-dir", silo_dir)
                prep_job.add_pegasus_profile(tag=silo_tags(site)["cpu"])
                self.wf.add_jobs(prep_job)
                self._add_silo_export(site, sequences, prep_job)
            else:
                prep_job.add_outputs(sequences, stage_out=False,
                                     register_replica=False)
                self.wf.add_jobs(prep_job)

            self.prep_jobs[site] = prep_job
            self.site_files[site] = {
                "sequences": sequences, "manifest": manifest
            }

    def silo_paths(self, site):
        """Resident (on-worker) paths for one client's shard."""
        silo_dir = os.path.join(self.silos["data_dir"], site)
        return {
            "dir": silo_dir,
            "sequences": os.path.join(silo_dir, f"{site}_sequences.npz"),
            "manifest": os.path.join(silo_dir, f"{site}_manifest.json"),
        }

    def _add_silo_export(self, site, sequences, prep_job):
        """Copy a resident shard out of its silo for the pooled arms.

        The centralized baseline trains on all seven shards pooled, and
        MCT evaluation is a server-side step, so both need the data
        centrally — that is what the paper compares federation against.
        Making the egress its own pinned DAG node keeps the asymmetry
        visible and countable: the federated arm has no such node.
        """
        job = (
            Job("silo_export",
                _id=f"export_{site}", node_label=f"export_{site}")
            .add_args(self.silo_paths(site)["sequences"], sequences)
            .add_outputs(sequences, stage_out=False, register_replica=False)
            .add_pegasus_profile(tag=silo_tags(site)["cpu"])
            .add_pegasus_profiles(label=site)
        )
        self.wf.add_jobs(job)
        # The source is an absolute on-worker path, not a declared file
        # input, so the ordering edge must be explicit.
        self.wf.add_dependency(job, parents=[prep_job])

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
        """Client specs for fl_round.

        Emulated mode yields LFNs, which fl_round declares as staged job
        inputs, and the plain GPU tag. Cross-silo mode yields absolute
        on-worker paths plus the silo's GPU tag, which the site catalog
        turns into a pin to the worker holding the shard (and that site's
        GPU settings), so the shard is read in place and never staged.
        """
        specs = []
        for site in self.sites:
            if self.silos:
                paths = self.silo_paths(site)
                spec = {"name": site,
                        "sequences": paths["sequences"],
                        "manifest": paths["manifest"],
                        "resident": True,
                        "tag": silo_tags(site)["gpu"]}
            else:
                spec = {"name": site,
                        "sequences": self.site_files[site]["sequences"].lfn,
                        "manifest": self.site_files[site]["manifest"].lfn,
                        "resident": False,
                        "tag": GPU_TAG}
            specs.append(spec)
        return specs

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
                .add_pegasus_profiles(label=f"{method}_L{interval}",
                                      tag=gpu_tag_for("train_dgmr"))
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
            # Operates on the submit host's output directory.
            job.add_profiles(Namespace.SELECTOR, "execution.site", "local")
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
                gpu_tag=GPU_TAG,
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
            # Same output directory as the parent (see __init__), so the
            # collect_file bridge below finds the final checkpoint.
            subwf.add_args("--conf", self.subwf_conf,
                           "--output-dir", self.local_storage_dir)
            subwf.add_inputs(File(prev_global))
            if not self.silos:
                for site in self.sites:
                    subwf.add_inputs(self.site_files[site]["sequences"],
                                     self.site_files[site]["manifest"])
            subwf.add_outputs(File(names["global_out"]), stage_out=False,
                              register_replica=False)
            if is_validation:
                subwf.add_inputs(File(prev_history), File(prev_best))
                # stage_out=False even on the final round: the round
                # itself marks its history JSON for stage-out and is
                # planned with the parent's --output-dir, so it publishes
                # the file. Asking the parent to publish it too makes a
                # second stage-out job sourcing the parent's scratch,
                # where the round leaves no copy — that pair is what
                # failed run0001 at 94%.
                subwf.add_outputs(File(names["history_out"]),
                                  stage_out=False,
                                  register_replica=False)
                subwf.add_outputs(File(names["best_out"]), stage_out=False,
                                  register_replica=False)
                prev_history = names["history_out"]
                prev_best = names["best_out"]
            if is_final:
                # This one stays stage_out=True, and is the reason the FL
                # chain survives planning at all: it is the only output of
                # the last round that anything downstream requires, so
                # with it false the planner prunes every round
                # sub-workflow (run0002 planned with no federated arm).
                # collect_file reads the staged copy from the output
                # directory by path.
                subwf.add_outputs(File(sub_best_lfn), stage_out=True,
                                  register_replica=False)
            self.wf.add_jobs(subwf)
            if self.silos and r == 0:
                # Resident shards are not declared inputs, so the FL chain
                # would otherwise start before the silos are populated.
                # Later rounds inherit the edge through the global-model
                # chain.
                self.wf.add_dependency(
                    subwf,
                    parents=[self.prep_jobs[site] for site in self.sites])
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
            # Reads the sub-workflow's staged output on the submit host.
            .add_profiles(Namespace.SELECTOR, "execution.site", "local")
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
            .add_pegasus_profiles(label=tag, tag=gpu_tag_for("mct_infer"))
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


def cleanup_strategy(args, base_catalog, hosted):
    """Whether pegasus-plan needs --cleanup leaf on this site.

    The FL rounds are deferred sub-workflows, so their per-round
    checkpoints are produced by a planner run that has not happened yet.
    When the compute site stages data through its own scratch
    (data.configuration nonsharedfs or sharedfs — every hosted batch
    catalog, Unity included), the parent planner has no PFN for those
    files and per-file cleanup will not plan at all:

        Unable to determine cleanup url for lfn fed_L1_global_r001.pt
        at site compute

    --cleanup leaf plans, and still removes the site's scratch directory
    at the end of the run. It does mean intermediate files live until
    then; the workflow's own cleanup_file jobs still delete stale
    checkpoints round by round, which is the part that would otherwise
    grow without bound. Under condorio (a plain HTCondor pool) the
    staging site is the submit host and none of this applies.

    Resolution runs from most to least direct, and ends by assuming leaf
    is needed rather than assuming it is not. A catalog that states a
    data configuration settles it; failing that a condor-style site is
    condorio and anything glite-shaped is not; failing that, a hosted
    catalog is configured but not here yet — the first-run case, where
    the site is a batch site by construction (nobody publishes a hosted
    catalog for a personal condor pool) and omitting --cleanup leaf
    means the plan aborts. Leaf plans on either kind of site, so the
    unknown case takes the option that cannot fail to plan.
    """
    site = args.execution_site_name
    stages_on_compute = site_stages_on_compute(
        site, args.sites_yml, base_catalog)
    style = site_submission_style(site, args.sites_yml, base_catalog)

    if stages_on_compute is not None:
        why = "its data.configuration says so"
    elif style in ("condor", "condorc"):
        stages_on_compute, why = False, (
            f"site {site!r} submits to a condor pool, which stages "
            f"through the submit host")
    elif style:
        stages_on_compute, why = True, (
            f"site {site!r} submits through glite, and batch sites stage "
            f"through their own scratch")
    elif hosted:
        stages_on_compute, why = True, (
            f"the hosted catalog {hosted} is named in ~/.pegasusrc but is "
            f"not in this directory yet, so its data configuration cannot "
            f"be read. Hosted catalogs are batch sites; leaf cleanup is "
            f"assumed because it plans on either kind of site, while "
            f"omitting it aborts planning on a batch one. Pass "
            f"--base-catalog {hosted} to decide this from the catalog "
            f"itself")
    else:
        stages_on_compute, why = False, (
            "no site catalog here states a data configuration or a "
            "submission style")

    if stages_on_compute:
        logger.info(
            f"Cleanup: --cleanup leaf — {why}. Per-file cleanup cannot "
            f"plan the FL sub-workflow checkpoints (see README, "
            f"\"Cleanup on a batch site\")")
    else:
        logger.info(
            f"Cleanup: planner default — {why}. If planning fails with "
            f"\"Unable to determine cleanup url\", add --cleanup leaf")
    return stages_on_compute


def check_site_catalog_setup(args, silos):
    """Report where the site catalog comes from; abort if it cannot pin.

    The generator writes no site catalog. Planning needs either a hosted
    one (pegasus.catalog.site.repo.file in ~/.pegasusrc) or a local
    sites.yml, and that is a warning either way — pegasus-plan fails
    plainly when neither exists.

    Cross-silo runs are different and abort here, because none of their
    failure modes fail planning. A pinned job carries a tag and nothing
    else, so a tag that is missing, pins nothing, names the wrong node,
    or pins in a dialect this site's scheduler ignores all produce the
    same outcome: the job runs wherever the scheduler likes, writes its
    shard there, and the run continues as an emulated one that quietly
    loses every later round's shard while still calling itself
    cross-silo. Not being able to *tell* is refused for the same reason.
    """
    hosted, discovered = hosted_catalog()
    hosted_copy = args.base_catalog or discovered
    local = os.path.isfile(args.sites_yml)
    if hosted:
        logger.info(f"Site catalog: hosted {hosted} (~/.pegasusrc)"
                    + (f" + {args.sites_yml} overlay" if local else ""))
    elif local:
        logger.info(f"Site catalog: {args.sites_yml}")
    else:
        logger.warning(
            "No site catalog: set pegasus.catalog.site.repo.file in "
            "~/.pegasusrc (hosted catalog, e.g. unity.yml) or write "
            f"{args.sites_yml} with custom_sites.py --full before planning")

    # Answered before any early return below, because it applies to every
    # run, cross-silo or not.
    leaf_cleanup = cleanup_strategy(args, hosted_copy, hosted)

    if not silos:
        return leaf_cleanup
    if args.skip_silo_tag_check:
        logger.warning(
            "--skip-silo-tag-check: not verifying that the silo tags exist. "
            "Every pinned job must find its tag on site "
            f"{args.execution_site_name!r} at plan time, or it runs "
            "unpinned and the run is not cross-silo.")
        return leaf_cleanup
    if not local:
        raise ValueError(
            f"--silos needs the silo tags in {args.sites_yml}, which does "
            f"not exist. The workflow pins a job only by tag "
            f"(silo_<SITE>, silo_<SITE>_gpu); without them every pinned "
            f"job runs wherever the scheduler likes and no shard stays "
            f"put. Run:\n"
            f"    ./custom_sites.py --style <condor|slurm> --silos "
            f"{args.silos}\n"
            f"(add --base <hosted catalog> or --gpu-queue so pinned GPU "
            f"jobs keep the site's GPU settings), or pass "
            f"--skip-silo-tag-check if the tags come from elsewhere.")

    # No --style here: the site's own catalog says which scheduler it
    # submits to, the local overlay first and then the hosted catalog it
    # overlays, so a tag pointing at the wrong node, one carrying no pin,
    # and a whole catalog written in the wrong dialect are all caught.
    check = check_silo_tags(
        args.sites_yml, args.execution_site_name, args.sites, silos,
        base_catalog=hosted_copy)
    if check.problems:
        raise ValueError(
            f"{len(check.problems)} silo tag problem(s) in "
            f"{args.sites_yml} for site {args.execution_site_name!r} — "
            f"those jobs would run unpinned or pinned to the wrong node:\n"
            + "\n".join(check.problems)
            + f"\nRun ./custom_sites.py --style <condor|slurm> --silos "
              f"{args.silos} --sites {' '.join(args.sites)}, then "
              f"tools/silo_check.py to check the pins themselves.")

    if not check.verified and not args.allow_unverified_style:
        # Fail closed, and note that a first run is exactly when this
        # bites: an overlay states no submission style, so until the
        # hosted catalog it overlays is at hand, a catalog written for
        # the wrong scheduler is indistinguishable from a correct one and
        # would plan cleanly here.
        remedies = [
            "point --base-catalog at the site catalog this overlays"
            + (f", the hosted {hosted} — plan once and the planner leaves "
               f"a copy in this directory, or fetch it from "
               f"github.com/pegasushub/pegasus-site-catalogs"
               if hosted else ""),
            "write the whole compute site with custom_sites.py --full, "
            "which states a submission style",
            "pass --allow-unverified-style to accept the dialect as "
            "given; presence, pins and node names are still checked",
        ]
        raise ValueError(
            f"the silo tags in {args.sites_yml} name every mapped machine "
            f"correctly, but which scheduler site "
            f"{args.execution_site_name!r} submits to could not be "
            f"confirmed: {check.note}. A pin in the other scheduler's "
            f"dialect is ignored, so this cannot be passed as checked. "
            f"Any of:\n" + "\n".join(f"  - {r}" for r in remedies))

    logger.info(f"Silo tags: {2 * len(args.sites)} in {args.sites_yml}, "
                f"each pinning its mapped machine for {check.note}")
    if not check.verified:
        logger.warning(
            "--allow-unverified-style: the pins were checked against a "
            "scheduler no site catalog confirmed. If this site does not "
            "submit that way, every pinned job runs anywhere.")
    logger.info(f"Check the pins against the pool with "
                f"tools/silo_check.py --style <condor|slurm> {args.silos}")
    return leaf_cleanup


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
  %(prog)s --start-month 2021-01 --months 48 --silos silos.yml

Then plan with the command this prints; it names the execution site from
your site catalog (hosted catalogs call it "compute") and adds
--cleanup leaf where the site stages through its own scratch:
  pegasus-plan --submit -s compute --output-dir output workflow.yml
""",
    )

    # --- Standard Pegasus arguments ---
    parser.add_argument("-e", "--execution-site-name", metavar="STR",
                        type=str, default="compute",
                        help="Execution site to name in the printed "
                             "pegasus-plan command (default: compute, the "
                             "hosted site catalogs' convention). The "
                             "workflow itself does not depend on it.")
    parser.add_argument("-o", "--output", metavar="STR", type=str,
                        default="workflow.yml",
                        help="Output file (default: workflow.yml)")
    parser.add_argument("--sites-yml", metavar="FILE", type=str,
                        default="sites.yml",
                        help="local site-catalog overlay written by "
                             "custom_sites.py (default: sites.yml). Named "
                             "in the generated properties, so pegasus-plan "
                             "finds it from any directory.")
    parser.add_argument("--output-dir", metavar="DIR", type=str,
                        default="output",
                        help="Submit-host directory staged outputs land in "
                             "(default: ./output). Plan the parent with the "
                             "same --output-dir; FL-round sub-workflows use "
                             "it automatically.")

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
                        help="PILOT/TIMING ONLY: cap train/val sequences "
                             "per client in training jobs")
    parser.add_argument("--model-size", type=int, default=288,
                        help="DGMR spatial grid; must be divisible by 32 "
                             "(default: 288 = paper-fidelity crop of the "
                             "300x300 window)")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Training batch size (default: 2)")
    parser.add_argument("--max-job-memory-gb", type=int, default=None,
                        help="Cap every job's memory request, for pools "
                             "with small nodes (e.g. 12 on 15.6 GB nodes)")
    parser.add_argument("--max-job-cores", type=int, default=2,
                        help="Core cap applied with --max-job-memory-gb")
    parser.add_argument("--runtime-scale", type=float, default=1.0,
                        help="Multiply every job's wall-clock budget "
                             "(default: 1.0; e.g. 2 on slow GPUs). Per-tag "
                             "overrides belong in the site catalog.")
    parser.add_argument("--retries", type=int, default=1,
                        help="DAGMan retries per job; a retry doubles the "
                             "job's runtime budget (default: 1)")
    parser.add_argument("--fallback-test-instances", type=int, default=0,
                        help="PILOT ONLY: mct_infer falls back to N test "
                             "sequences per site when no event matches")
    parser.add_argument("--max-concurrent-jobs", type=int, default=20,
                        help="DAGMan job throttle (default: 20)")

    # --- Cross-silo placement ---
    parser.add_argument("--silos", metavar="YAML", type=str, default=None,
                        help="Cross-silo placement map (see "
                             "silos.example.yml): pin each client's data "
                             "and training jobs to the worker holding its "
                             "shard, so federated training and validation "
                             "never move it (they return weights plus small "
                             "per-client counts and losses). The centralized "
                             "baseline and evaluation still need shards "
                             "pooled, so each is copied out once by a "
                             "silo_export job. The pins themselves live in "
                             "the site catalog: run custom_sites.py --silos "
                             "with the same map first. Default: emulated "
                             "placement, as in the paper.")

    parser.add_argument("--shared-filesystem", action="store_true",
                        help="the workers can read the submit host's "
                             "filesystem, as on an HPC cluster with a "
                             "shared home or scratch. Lets jobs read "
                             "inputs directly from the input site instead "
                             "of staging every copy through the staging "
                             "server, which matters most for the "
                             "multi-GB container images. Leave off for a "
                             "condor pool that stages over HTCondor file "
                             "transfer.")
    parser.add_argument("--base-catalog", metavar="FILE", default=None,
                        help="the site catalog your sites.yml overlays, "
                             "read to confirm which scheduler the site "
                             "submits to (default: the hosted catalog "
                             "named in ~/.pegasusrc, if the planner has "
                             "left a copy in this directory). Needed with "
                             "--silos, since an overlay states no style "
                             "and a pin in the wrong dialect is ignored.")
    parser.add_argument("--allow-unverified-style", action="store_true",
                        help="with --silos: accept silo pins whose dialect "
                             "no site catalog could confirm. Presence, "
                             "pins and node names are still checked.")
    parser.add_argument("--skip-silo-tag-check", action="store_true",
                        help="with --silos: do not require the silo tags in "
                             "the local site catalog. Only correct if they "
                             "come from somewhere else (a forked hosted "
                             "catalog); otherwise pinned jobs run anywhere.")

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
        args.model_size = 128
        args.batch_size = 1
        args.max_job_memory_gb = 8
        logger.info("PILOT MODE: %s, %d month(s), %d round(s)",
                    args.sites, args.months, args.rounds)

    # --- Validation ---
    if args.model_size % 32 != 0:
        print(f"Error: --model-size ({args.model_size}) must be divisible "
              f"by 32 (DGMR downsamples by 32; see SPEC open question 11)")
        sys.exit(1)
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
    logger.info(f"Runtime budgets x{args.runtime_scale}, "
                f"{args.retries} retr{'y' if args.retries == 1 else 'ies'}")
    logger.info("Input staging: "
                + ("direct from the input site (--shared-filesystem)"
                   if args.shared_filesystem else
                   "through the staging server; pass "
                   "--shared-filesystem on a cluster whose workers can "
                   "read the submit host"))
    if args.silos:
        logger.info(f"Placement: CROSS-SILO ({args.silos})")
    else:
        logger.info("Placement: emulated (shards staged to any worker)")
    logger.info("=" * 70)

    try:
        workflow = FedCastWorkflow(args)
        workflow.log_placement()
        leaf_cleanup = check_site_catalog_setup(args, workflow.silos)
        workflow.create_pegasus_properties()
        workflow.create_transformation_catalog()
        workflow.create_replica_catalog()
        workflow.write_subworkflow_conf()
        workflow.create_workflow()
        workflow.write()

        logger.info(f"\nWorkflow written to {args.output}")
        logger.info(f"Submit: pegasus-plan --submit "
                    f"-s {args.execution_site_name} "
                    + ("--cleanup leaf " if leaf_cleanup else "")
                    + f"--output-dir {workflow.local_storage_dir} "
                    f"{args.output}")
    except ValueError as e:
        # Silo-map problems are user configuration, not bugs.
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to generate workflow: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
