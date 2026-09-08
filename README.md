# fedcast-workflow

Pegasus WMS workflow reproducing **Fed-Cast** (Xu, Mehboob, Zink, Davis —
UMass Amherst, eScience 2026): federated vs. centralized DGMR precipitation
nowcasting on NOAA MRMS PrecipRate across seven climate-diverse radar-centered
regions, evaluated with an MCT-style multi-metric + TOPSIS pipeline against a
PySTEPS STEPS baseline.

- **Design & validation criteria:** [SPEC.md](SPEC.md)
- **Paper summary:** [PAPER_SUMMARY.md](PAPER_SUMMARY.md)

## Pipeline

```
Phase A  fetch_crop_mrms   one job per (domain, month): MRMS PrecipRate from
                           s3://noaa-mrms-pds, cropped to every 3°x3° site
                           window in one pass (crop-on-ingest)
         preprocess_sequences  per site: 16-frame sequences (4 in / 12 out),
                           rain filter, frozen 80/10/10 split + manifest
                           with --silos: pinned to the site's silo, and
                           the shard stays there; silo_export then copies
                           it out for the pooled consumers only
Phase B  fetch_events      WPC MPD, LSR, Storm Events (best-effort each)
         build_benchmark   frozen, balanced event set B
Phase C  train_dgmr        centralized DGMR per interval L, as chains of
                           checkpointed segment jobs
         fl_* + SubWorkflows  federated DGMR per interval L: fl_init, then
                           ONE SubWorkflow PER FL ROUND (fl_round.py) —
                           fl_train_client x N in parallel -> fl_aggregate
                           (FedAvg uniform/quadratic) -> fl_validate
                           (chained best-checkpoint tracking)
                           with --silos: client jobs are pinned to the
                           worker holding their shard, and validation
                           fans out to fl_validate_client x N at the
                           silos, which return metrics only
Phase D  mct_infer         DGMR K=6 ensembles / STEPS 20-member ensemble
         mct_verify        Table-I metric suite per lead time, lead-averaged
         mct_topsis        objective-side-balanced TOPSIS per candidate pool
         make_figures      learning-curve boxplots (paper Figs. 4-6)
         validate_report   tiered reproduction gates (SPEC Sec. 5)
Phase E  ablations         E2.1 quadratic client weighting; E2.2 SAM (TODO)
```

## Quick start

Needs Pegasus 5.1.3dev or 6.0.0dev
([download](https://download.pegasus.isi.edu/pegasus/6.0.0.dev0/)): the
workflow relies on job tags and hosted site catalogs, which older releases
do not have. Install the matching `pegasus-wms` Python package into the
virtualenv you generate from.

```sh
# 1. Build containers (once)
apptainer build Apptainer/FedCast_data.sif  Apptainer/FedCast_data.def
apptainer build Apptainer/FedCast_train.sif Apptainer/FedCast_train.def
apptainer build Apptainer/FedCast_eval.sif  Apptainer/FedCast_eval.def

# 2. Tell Pegasus where jobs run — once per submit host. On a cluster with
#    a hosted site catalog (Unity, Perlmutter, Expanse, ...) this is one
#    line; see "Sites" below for pools without one.
cat >> ~/.pegasusrc <<EOF
pegasus.catalog.site.repo.file = unity.yml
EOF
./custom_sites.py --style slurm --project <your allocation>

# 3. Pilot run (2 sites, 1 month, 2 rounds — end-to-end smoke test)
python3 workflow_generator.py --test
pegasus-plan --submit -s compute --output-dir output workflow.yml

# 4. Full E1 reproduction
python3 workflow_generator.py --start-month 2021-01 --months 48
pegasus-plan --submit -s compute --output-dir output workflow.yml

# 5. With ablations
python3 workflow_generator.py --start-month 2021-01 --months 48 \
    --experiments e1 e21 e22
```

`compute` is the execution site's name in every hosted catalog; the
generator prints the exact `pegasus-plan` line for your `--output-dir`.

Or run the wrappers directly without Pegasus/HTCondor:

```sh
./run_manual.sh          # tiny end-to-end smoke test on the local host
```

## Key options

| Option | Default | Meaning |
|---|---|---|
| `--start-month` | 2021-01 | Archive start (SPEC open question 1) |
| `--months` | 48 | Archive length |
| `--sites` | all 7 | Radar sites / federated clients |
| `--intervals` | 1 3 6 12 24 48 | Training intervals L (months) |
| `--rounds` | 100 | FL rounds / centralized epochs |
| `--segment-size` | 10 | Rounds/epochs per training segment job |
| `--experiments` | e1 | Pools: `e1`, `e21` (quadratic), `e22` (SAM) |
| `--min-rain-fraction` | 0.05 | Sequence retention filter (open question 2) |
| `--frame-stride` | 1 | Subsample MRMS cadence (pilot runs) |
| `--shared-filesystem` | — | Workers can read the submit host; skip staging inputs |
| `--runtime-scale` | 1.0 | Multiply every job's wall-clock budget |
| `--retries` | 1 | DAGMan retries per job; a retry doubles the budget |
| `--output-dir` | `output` | Where staged-out files land; plan with the same |
| `--test` | — | Pilot mode: 2 sites, 1 month, 2 rounds |
| `--silos` | — | Cross-silo placement map (see below) |
| `--base-catalog` | hosted copy | Catalog your `sites.yml` overlays, read to confirm the pin dialect |
| `--allow-unverified-style` | — | Accept silo pins whose dialect no catalog confirmed |

## Sites

The generator writes no site catalog and names no scheduler. Every
transformation states the four portable things — cores, memory, GPUs and a
wall-clock `runtime` (mandatory on batch systems) — and GPU jobs carry the
Pegasus tag `gpu`. Everything else about *where* a job runs and *how* it
asks for resources — partition, account, GPU model constraints, ClassAd
requirements, `--nodelist` — lives in the site catalog, keyed by tag, which
is what makes the same `workflow.yml` plan on an HTCondor pool and on Unity.

**With a hosted catalog.** Pegasus downloads the catalog named by
`pegasus.catalog.site.repo.file` in `~/.pegasusrc` from
[pegasushub/pegasus-site-catalogs](https://github.com/pegasushub/pegasus-site-catalogs/tree/main/conf)
and merges a local `sites.yml`, if present, over it — local entries win key
by key, and `x-tags` merge the same way. `unity.yml`, for instance, already
puts CPU jobs on `queue: cpu` and defines a `gpu` tag with `queue: gpu`,
`gpus: 1` and `--nv`. What you add is what the catalog cannot know:

```sh
./custom_sites.py --style slurm --project my_lab            # account
# and generate with --shared-filesystem on a cluster like Unity
./custom_sites.py --style slurm --project my_lab \
    --gpu pegasus:glite.arguments=--constraint=vram48        # GPU model
```

**Without one** (a personal HTCondor pool, a Slurm cluster nobody has
catalogued), `--full` writes the whole compute site:

```sh
./custom_sites.py --style condor --full \
    --gpu 'condor:requirements=(GPUs_GlobalMemoryMb >= 20000)'
./custom_sites.py --style slurm --full --queue cpu --gpu-queue gpu \
    --project my_lab --scratch /scratch/$USER/fedcast
```

`custom_sites.py` is a thin wrapper over the Pegasus API; anything it does
not have a flag for goes in as `--profile NS:KEY=VALUE` on the site or
`--gpu NS:KEY=VALUE` on the GPU tag, or edit `sites.yml` by hand. It is a
starting point — the
[alphafold3-workflow](https://github.com/baldikacti/alphafold3-workflow)
shows the same pattern at its smallest.

On a cluster whose workers can read the submit host, `--shared-filesystem`
sets `pegasus.transfer.bypass.input.staging`, so jobs read inputs directly
instead of staging every copy through the staging server. That matters
most for the three container images, which are several GB each. It is off
by default and must stay off on a condor pool that stages over HTCondor
file transfer, where the workers cannot reach the submit host's paths and
jobs would chase `file://` URLs that do not exist there.
`pegasus.transfer.links` is always set, and only takes effect when the
replica catalog places an input on the execution site itself, so it costs
nothing where it does not apply.

Runtime budgets are generous defaults for full-scale inputs (six hours for
a month of MRMS, twelve for a ten-epoch centralized segment, two for one
client's local epoch) and are doubled on retry through a `runtime.expr`
profile; `--runtime-scale` multiplies them all. Sizes (memory, cores,
runtime, GPU count) are workflow knobs because transformation-catalog
profiles outrank the site catalog in Pegasus; what a tag controls is how
the request is expressed and where it lands — partition, account,
constraints, pins. Retry rewriting needs the `pythonsed` package on the
submit host.

Nothing here is HTCondor-specific any more, but the pool this has actually
run to completion on is a personal HTCondor pool (`--style condor --full`).
The first Slurm run should be the pilot (`--test`).

## Data placement: emulated vs. cross-silo

The paper *emulates* federation: all seven clients are carved out of one
central MRMS archive, and the figure caption notes that the server icon
"denotes a server-side role, not an actual deployment location". The
default mode here matches that. Client shards are ordinary Pegasus files,
the scheduler places each training job on any free GPU slot, and the shard
is staged to whichever worker won the match, every round.

`--silos silos.yml` switches to a real cross-silo model, where each client
is a data holder and **the federated arm never moves its shard**. The
generator expresses this as a Pegasus tag per pinned job — `silo_KTLX` for
CPU work at that silo, `silo_KTLX_gpu` for GPU work — and
`custom_sites.py --silos` gives those tags their meaning on your scheduler,
so the workflow itself stays scheduler-neutral:

| | emulated (default) | cross-silo (`--silos`) |
|---|---|---|
| Shard location | submit-host staging | resident on the silo worker |
| Client job placement | any matching slot | pinned to its silo |
| Shard staged per FL round | yes, per client | never |
| Validation | server reads every client's split | each client scores at its silo, returns batch-loss sums and counts |
| Leaves the silo, federated arm | shards + weights | model weights plus small per-client metadata (below) |
| Leaves the silo, pooled arms | n/a (already central) | the whole shard, once, via `silo_export` |

**What actually leaves a silo, in full.** The shard itself never moves for
the federated arm, but "only weights" would be wrong. The complete list:

| From | When | Contents |
|---|---|---|
| `fl_train_client` | every round | the post-local-training model weights, and a metadata JSON of `site`, `round`, `n_train` — the client's retained training-sequence count, which the aggregator needs for the quadratic-weighting ablation (Eq. 8) |
| `fl_validate_client` | validation rounds | `site`, `round`, `n_val`, `n_batches`, `sum_loss`, `mean_loss` — counts and losses over the client's validation split |
| `preprocess_sequences` | once | the split manifest, staged out as a reproduction artifact (SPEC Tier 0/1). The largest metadata export, and every field of it by full path: `site`; `retained`; `splits`, `splits.train`, `splits.val`, `splits.test`; `filter`, `filter.rain_threshold_mmh`, `filter.min_rain_fraction`; `val_seed`; `effective_cadence_s`; `retention_stats`, `retention_stats.candidates`, `retention_stats.gap_rejected`, `retention_stats.rain_rejected` — how many candidate sequences the site had and why each was dropped; `sequence_sha256`, a digest of the sequence array alone and **not** of the shard file, whose timestamps and split labels it does not cover; `silo`, `silo.configured_dir`, `silo.resolved_path`, `silo.host`, `silo.user`, `silo.home` — the shard's on-worker path and the identity that resolved it; and the per-sequence `start_epochs` and `split_labels` vectors, a timestamp and split label for **every retained sequence at that site** |
| `preprocess_sequences` | once, only when a site has no usable input | a stand-in manifest of `site` and `error`, written so the declared output exists before the job fails (SPEC constraint 17) |
| `silo_export` | once | the entire shard, for the centralized baseline and MCT evaluation |

Everything above the last row is derived from client data rather than
being the data, but it is not nothing: the manifest in particular
describes when every retained sequence at a site occurred. No formal
privacy claim is made about any of it, and the paper adds no privacy
mechanism either (SPEC non-constraint 8). What cross-silo mode changes is
the *bulk* movement — the shard itself — not every trace of the data.

That list is enforced, not maintained by hand.

**At run time**, in three layers. Each wrapper declares an `EXPORT_FIELDS`
tuple, then:

1. `fedcast_common.write_export()` validates the payload and writes it in
   the same call. Validation walks the real object, recursing through
   nested dictionaries *and* sequences, so it sees a field however it got
   there — inline, assigned later, added by `update()`, produced by a
   helper, or sitting in a dictionary inside a list. An undeclared field
   aborts the job before the file is created.
2. The file is read back and validated again, so a serializer that
   reshapes the payload cannot widen the surface either.
3. `guard_export()` registers the output path up front and re-validates
   the file at interpreter exit. This is the layer that carries the
   guarantee, because it inspects the artifact: it holds whatever wrote
   the file, including an aliased `json.dump`, a hand-rolled `write()`, or
   a guarded call that turned out to be unreachable. An artifact with
   undeclared fields fails the job, so it is never staged out.

Every export must be a JSON object. A payload replaced by a bare array or
string would otherwise expose no field names at all and pass by carrying
nothing the check knows how to name, which is the shape a tampered
artifact would take to smuggle raw data out.

What this controls is field *names*. It does not constrain the type or
size of a declared field, so a declared field holding a large vector — the
manifest's `start_epochs` and `split_labels` genuinely do — cannot be
distinguished from one abused to carry bulk data. That is why those two
are called out in the table above rather than left to the check.

**Before a run**, `tools/check_export_docs.py` requires every declared
field to appear in a body row of the table above, by its exact full path.
Full path, so a nested field is listed by its whole name rather than its
leaf, which stops a declaration being satisfied by an unrelated field that
shares a leaf name. Rows of that one table only, so a field mentioned in
prose — this paragraph included — or listed in some other table further
down the page, but dropped from the export table, still fails. It also
requires
each payload to be registered with `guard_export` and written with
`write_export`, both passing `EXPORT_FIELDS` with the payload argument and
label in agreement. It does not try to prove that no other write path
exists — that is not decidable by reading Python, which is why layer 3
checks the artifact instead. Its `json.dump` detection is a lint for the
obvious cases. Run it after touching any `fl_*` or `preprocess_sequences`
payload:

```sh
tools/check_export_docs.py
```

And after touching either validation path, run the equivalence test. It calls
all three wrapper entry points with real arguments against a shard on disk,
using a stub generator — the per-client validator, then the central one on
both its direct and its recombining branch — and checks that the loss each
records in its history file agrees:

```sh
tools/test_validation_equivalence.py
```

As for the shard: the centralized baseline and MCT evaluation need all
seven in one place, and a run always builds them, so **every silo run
copies each shard out exactly once**. What silo mode buys is that the copy
is one visible, pinned job per client instead of a shard transfer per
client per round — 7 copies rather than 7 x rounds — and that the
federated arm's number is zero.

Both modes select the checkpoint from the same number, and that is
arranged rather than assumed. The validation loss averages a 6-sample
DGMR ensemble, so it depends on the random stream; each batch is therefore
seeded from the training seed, the client name and the batch index, which
makes a batch's loss independent of who computes it and in what order. The
per-client metrics carry batch-loss sums and batch counts, and
`fl_validate` recombines them into exactly the mean the central path
computes. Centralized training uses the same seeding, so all three paths
score a checkpoint identically.

That export is deliberate, not an oversight. Pooling every client's data
is *what the centralized baseline is*, and it is the thing the paper
measures federation against; MCT evaluation is likewise a server-side
step. Giving the egress its own pinned `silo_export` job per client makes
the asymmetry behind the paper's communication-volume argument (Sec. V,
Eq. 7) visible and countable in the DAG rather than assumed. The federated
arm has no such job. If you want a run with no shard egress at all, that
would need a federated-only workflow with the centralized arm and the
pooled evaluation removed, which this generator does not currently build.

What silo mode does not change: **ingest is still shared**. One
`fetch_crop_mrms` job per (domain, month) downloads the MRMS file once and
crops it for every client in that domain, because per-silo ingest would
multiply a multi-terabyte download by the client count. Only the shard —
the thing federated training reads — is resident. Reconstructing seven
regional archives from one public archive is scaffolding either way; the
paper does the same.

### Setting it up

```sh
cp silos.example.yml silos.yml                  # your own machine names
# Fetch the hosted catalog once, so the pin dialect can be verified. The
# planner also leaves a copy here after any plan.
curl -O https://raw.githubusercontent.com/pegasushub/pegasus-site-catalogs/main/conf/unity.yml
./custom_sites.py --style slurm --base unity.yml --silos silos.yml   # tags
tools/silo_check.py --style slurm silos.yml     # confirm the cluster matches
python3 workflow_generator.py --start-month 2021-01 --months 48 \
    --silos silos.yml --base-catalog unity.yml
pegasus-plan --submit -s compute --output-dir output workflow.yml
```

(`--style condor` throughout on an HTCondor pool; drop `--base` if you have
no hosted catalog and pass `--full` plus your GPU settings instead.) That
is the whole procedure. Nothing has to be installed or configured on the
workers, because the defaults avoid needing it:

- **Pinning by machine name** uses a name the scheduler already knows —
  `condor_status -af Machine` on HTCondor, `sinfo -N` on Slurm — so
  nothing has to be advertised. On HTCondor the tag becomes a
  `requirements` expression; on Slurm, `--nodelist=` in the job's glite
  arguments.
- **A home-relative shard directory** (`~/.fedcast/silos`, the default) is
  inside Apptainer's default mounts, so nothing is bind-mounted and the
  preprocess job creates its own directory on first use.

A job carries exactly one tag, which is why each silo gets two: the GPU one
has to say everything the plain `gpu` tag says (partition, `gpus`, `--nv`,
any constraint) *and* the pin. `custom_sites.py --base <hosted catalog>`
copies the hosted `gpu` tag as the starting point — the planner leaves a
copy of the catalog in the working directory — and appends the pin to
whatever key it lands on, so `-C gpu --nodelist=node7` on Perlmutter and
`(GPUs_GlobalMemoryMb >= 20000) && (Machine == "w1")` on HTCondor both
come out right. Without `--base` or `--gpu-queue` on Slurm it warns, because
pinned training would then land in the default partition.

The home-relative default assumes every job on a machine runs as the same
user, which is true of a Slurm cluster and of an ordinary HTCondor pool
but not of one configured with per-slot users. There the preprocess and
training jobs would resolve different directories. On HTCondor
`silo_check.py` queries each matched worker and says so, and an absolute
shard directory is the fix. Do not point the directory at `/tmp` or
`/var/tmp`: HTCondor defaults `MOUNT_UNDER_SCRATCH` to those two, and
Slurm sites commonly enable `job_container/tmpfs`, either of which makes
them private per job and deletes them when the job ends, so the shard
would be gone before the next round. The generator, `custom_sites.py` and
`silo_check.py` all warn if you try.

One thing to be clear-eyed about on a cluster with a shared home: every
node sees `~/.fedcast/silos`, so pinning there decides where a job *runs*,
not what it *can read*. If the point of your run is that a shard is
readable only at its holder, put `data_dir` on node-local storage.

`workflow_generator.py --silos` refuses to write a workflow unless every
client's two tags are present in the site catalog *and* actually pin.
Four ways that fails, all of them refusals:

- the tag is absent;
- it exists but carries no pin at all, which is what a tag copied for its
  queue and GPU settings looks like;
- its pin names something other than the one machine the map gives;
- it pins in the other scheduler's dialect. A `requirements` ClassAd does
  not place a job on a Slurm node, and `glite.arguments` does nothing in a
  vanilla HTCondor pool, so a pin in the wrong dialect is no pin at all.

Which scheduler the site submits to is read from the site catalog, the
local overlay first and then the hosted catalog it overlays. That last
step matters more than it sounds: an overlay states no style of its own,
so a catalog written entirely for the wrong scheduler agrees with itself
and passes every other check.

The hosted copy is the one the planner leaves in the working directory, so
on a first run there may be nothing to read it from — and that is refused
too, rather than warned about, because an unverifiable dialect is the same
silent outcome as a wrong one. Three ways past it: point
`--base-catalog` at the catalog your overlay overlays, write the whole
site with `custom_sites.py --full`, which states a style, or pass
`--allow-unverified-style` to accept the dialect as given, which still
checks presence, pins and node names. `custom_sites.py` also refuses a
`--style` contradicting the catalog it is handed, the earliest point the
mistake can be caught.

It has to be a refusal rather than a warning, because a pinned job carries
a tag and nothing else. An unpinned tag does not fail planning. It makes
the job run wherever the scheduler likes, write its shard there, and take
the run quietly back to emulated placement while still calling itself
cross-silo.

Pins are compared by parsing, not by substring, because
`--nodelist=node7` is a substring of `--nodelist=node70`; a nodelist
naming several nodes or a bracket range is rejected for the same reason a
silo may not span two machines. `--skip-silo-tag-check` overrides all of
this, for tags that genuinely come from somewhere else.

`silo_check.py` repeats those checks and adds the pool. It runs the exact
requirements expression each client's jobs will carry and prints the
machines it matches (HTCondor), or asks `scontrol` whether each named node
exists (Slurm), so a wrong name fails in a second rather than as an idle
job hours later. On HTCondor it then reads each matched worker's
configuration and checks the two ways a shard directory fails to survive
the round, described below; Slurm has no remote config query, so there
that section is reported as unverified.

A worker whose configuration cannot be read is reported as unverified
rather than counted as fine, and that has its own exit status so
`silo_check.py && pegasus-plan ...` is safe by default and automation
cannot mistake an unchecked pool for a clean one:

| Exit | Meaning |
|---|---|
| 0 | every check ran and is fine |
| 1 | a real problem: a silo tag missing or mispinned, a silo matching no worker, a worker that cannot satisfy the bind, or a shard directory that would not survive the round |
| 2 | the tool could not run: bad arguments, `condor_status` missing, or an unusable silo map |
| 3 | placement is fine, but something could not be verified: shard durability on a worker whose config could not be read, or the pin dialect when no site catalog states the site's scheduler |

Pools that deliberately refuse remote config queries always get 3. Pass
`--allow-unverified` there to accept it and exit 0; placement and bind
checks still have to pass, and a real durability problem still exits 1.
An unverified pin dialect is a separate opt-in, `--allow-unverified-style`,
so a script already carrying `--allow-unverified` for a quiet pool does
not thereby start accepting pins whose scheduler nothing confirmed.

If a shard does go missing at run time, the job says so with the worker,
the job user, and the resolved `HOME`, and each client's manifest records
the same three for the preprocess job that wrote the shard. Comparing them
identifies a directory that resolved differently in one line.

### When worker setup is needed

Two departures from the defaults do require `tools/silo_worker_setup.sh`,
and the generator says which mode you are in when it runs.

**An absolute shard directory** outside `~/`, `/tmp/` or `/var/tmp/` — say
`/var/lib/fedcast/silos`, if the shards belong somewhere administered
rather than in a job user's home. Pegasus scopes `container.arguments` to
the catalog, not the job, so that directory is bind-mounted into the data
and train containers **pool-wide**, and Apptainer refuses to start when a
bind source is missing. Every worker then needs it, including ones holding
no data:

```sh
sudo tools/silo_worker_setup.sh KTLX,KVNX /var/lib/fedcast/silos  # holders
sudo tools/silo_worker_setup.sh none      /var/lib/fedcast/silos  # the rest
```

On HTCondor `silo_check.py` verifies this pool-wide and names any worker
that would fail; it skips the check entirely when the map needs no bind.
On Slurm the script does not apply (it writes HTCondor configuration) —
create the directory on every node yourself.

**Pinning by advertised ClassAd** instead of machine name, written as `{}`
entries in the map, or by a raw `requirements:` expression. HTCondor only;
`custom_sites.py` rejects both for Slurm. Useful when you would rather not
hardcode hostnames or want a silo to match any of several machines. Run
the setup script on each data holder with the silos it hosts; it writes
the ClassAd and reconfigures the startd.

Pegasus cannot take over either step for you. Its own symlinking-in-
containers support has the same requirement, documented in the Pegasus
containers guide: the host directory must already exist and be named in
the container's `mounts`.

### Choosing a pool

A silo must resolve to exactly one worker. The preprocess job writes the
shard to one machine and nothing replicates it, so a training job that
later matched a second machine would find nothing there. `silo_check.py`
fails on a silo that matches more than one machine and names it; if you
replicate the shard directory across them yourself, pass
`--allow-multi-worker-silo`. Several clients may share a worker, which is
how seven clients fit a pool with fewer than seven GPU nodes; each still
reads only its own shard.

Pinning trades scheduling freedom for locality, so size the pool by GPU
slots rather than worker count. Seven clients pinned onto two GPU workers
run their rounds in waves, not in parallel, and a GPU constraint on the
`gpu` tag (`custom_sites.py --gpu ...`) narrows the match further by
excluding cards too small for the configured `--model-size`. Check what
each silo actually matched in `silo_check.py` output before committing to
a long run.

## Outputs (staged to `--output-dir`, default `output/`)

- `{site}_manifest.json` — frozen split manifests with SHA-256 (Tier 0/1).
  In cross-silo runs each also records where the shard landed and which
  host, job user and `HOME` resolved that path.
- `benchmark_events.csv` — frozen event benchmark set B
- `{method}_L{L}_best.ckpt` — best-validation-loss checkpoints
- `{method}_L{L}_metrics.csv`, `steps_metrics.csv` — per-instance metrics
- `e1_topsis.csv` (+ `e21`/`e22`) — per-pool TOPSIS scores
- `figures.tar.gz` — learning-curve plots + summary table
- `validation_report.md` — SPEC Sec. 5 gate results

## Running on another cluster

Nothing in the repository hard-codes a host, a path or a scheduler:
catalogs, properties, and the FL-round sub-workflow YAMLs are all generated
from the directory the generator runs in, and they are gitignored. To run
this somewhere else:

1. Clone the repo on the submit host and `pip install -r requirements.txt`
   into a virtualenv (Pegasus 5.1.3dev / 6.0.0dev on the host and in the
   venv).
2. Build the three Apptainer images (Quick start step 1). They are large
   and not in git.
3. Point `~/.pegasusrc` at the cluster's hosted site catalog, or write a
   complete one with `custom_sites.py --full`; add your account and any
   GPU constraint with `custom_sites.py` either way ([Sites](#sites)).
4. Run `workflow_generator.py`, which writes `transformations.yml`,
   `pegasus.properties`, and `fl_subwf.properties` for that host, then
   plan with `--output-dir` as it prints.
5. If the nodes are small, cap requests with `--max-job-memory-gb` and
   `--max-job-cores`; if jobs hit their wall-clock limit, raise
   `--runtime-scale`.
6. For cross-silo runs, copy `silos.example.yml`, put the new cluster's
   machine names in it, run `custom_sites.py --silos` to write the tags,
   and `tools/silo_check.py --style ...` before planning. The defaults
   need nothing installed or configured on the workers; see
   [When worker setup is needed](#when-worker-setup-is-needed) for the two
   cases that do.

The sub-workflow properties repeat everything in `pegasus.properties`
(worker-package settings for containers whose OS differs from the submit
host, the hosted-catalog selection from `~/.pegasusrc`, the path to a
local `sites.yml`) because FL rounds are planned with their own
configuration file, not the parent's. Re-run the generator after changing
`~/.pegasusrc` or writing a new `sites.yml`.

## Known gaps (scaffold state)

- **E2.2 SAM** training is a TODO stub in `bin/train_dgmr.py` (exits with an
  explicit error).
- **Validation loss** uses the grid-cell-regularizer term of the paper's
  Eq. 3; the discriminator hinge term still needs to be added.
- The precipitation-content filter and benchmark balancing rules are
  documented defaults pending author responses (SPEC open questions 2, 5).
- Container package versions are unpinned until the paper's exact releases
  are known (SPEC open question 9).
- Cross-silo mode makes the *shards* resident, not the ingest: MRMS is
  still downloaded once per (domain, month) and cropped for every client
  in that domain (SPEC open question 12).
