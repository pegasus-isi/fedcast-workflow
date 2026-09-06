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

```sh
# 1. Build containers (once)
apptainer build Apptainer/FedCast_data.sif  Apptainer/FedCast_data.def
apptainer build Apptainer/FedCast_train.sif Apptainer/FedCast_train.def
apptainer build Apptainer/FedCast_eval.sif  Apptainer/FedCast_eval.def

# 2. Pilot run (2 sites, 1 month, 2 rounds — end-to-end smoke test)
python3 workflow_generator.py --test
pegasus-plan --submit -s condorpool -o local workflow.yml

# 3. Full E1 reproduction
python3 workflow_generator.py --start-month 2021-01 --months 48
pegasus-plan --submit -s condorpool -o local workflow.yml

# 4. With ablations
python3 workflow_generator.py --start-month 2021-01 --months 48 \
    --experiments e1 e21 e22
```

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
| `--test` | — | Pilot mode: 2 sites, 1 month, 2 rounds |
| `--silos` | — | Cross-silo placement map (see below) |

## Data placement: emulated vs. cross-silo

The paper *emulates* federation: all seven clients are carved out of one
central MRMS archive, and the figure caption notes that the server icon
"denotes a server-side role, not an actual deployment location". The
default mode here matches that. Client shards are ordinary Pegasus files,
HTCondor matches each training job to any free GPU slot, and the shard is
staged to whichever worker won the match, every round.

`--silos silos.yml` switches to a real cross-silo model, where each client
is a data holder and **the federated arm never moves its shard**:

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
both wrappers with real arguments against a shard on disk, with a stub
generator, and checks the per-client losses recombine to the central one:

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
cp silos.example.yml silos.yml    # put your own machine names in
tools/silo_check.py silos.yml     # confirm the pool matches
python3 workflow_generator.py --start-month 2021-01 --months 48 \
    --silos silos.yml
pegasus-plan --submit -s condorpool -o local workflow.yml
```

That is the whole procedure. Nothing has to be installed or configured on
the workers, because the defaults avoid needing it:

- **Pinning by machine name** uses HTCondor's built-in `Machine` attribute,
  so no ClassAd has to be advertised and no `condor_reconfig` is needed.
  Get the names from `condor_status -af Machine`.
- **A home-relative shard directory** (`~/.fedcast/silos`, the default) is
  inside Apptainer's default mounts, so nothing is bind-mounted and the
  preprocess job creates its own directory on first use.

The home-relative default assumes every job on a worker runs as the same
user, which is true of an ordinary pool but not of one configured with
per-slot users. There the preprocess and training jobs would resolve
different directories. `silo_check.py` queries each matched worker and says
so, and an absolute shard directory is the fix. Do not point the directory
at `/tmp` or `/var/tmp`: HTCondor defaults `MOUNT_UNDER_SCRATCH` to those
two, making them private per job and deleting them when the job ends, so
the shard would be gone before the next round. The generator warns if you
try.

`silo_check.py` runs the exact requirements expression each client's jobs
will carry and prints the machines it matches, so a wrong name fails in a
second rather than as an idle job hours later. It then reads each matched worker's
configuration and checks the two ways a shard directory fails to survive
the round, described below.

A worker whose configuration cannot be read is reported as unverified
rather than counted as fine, and that has its own exit status so
`silo_check.py && pegasus-plan ...` is safe by default and automation
cannot mistake an unchecked pool for a clean one:

| Exit | Meaning |
|---|---|
| 0 | every matched worker was checked and is fine |
| 1 | a real problem: a silo matches no worker, a worker cannot satisfy the bind, or a shard directory would not survive the round |
| 2 | the tool could not run: bad arguments, `condor_status` missing, or an unusable silo map |
| 3 | placement is fine, durability unverified on at least one worker |

Pools that deliberately refuse remote config queries always get 3. Pass
`--allow-unverified` there to accept it and exit 0; placement and bind
checks still have to pass, and a real durability problem still exits 1.

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

`silo_check.py` verifies this pool-wide and names any worker that would
fail. It skips the check entirely when the map needs no bind.

**Pinning by advertised ClassAd** instead of machine name, written as `{}`
entries in the map. Useful when you would rather not hardcode hostnames or
want a silo to match any of several machines. Run the setup script on each
data holder with the silos it hosts; it writes the ClassAd and reconfigures
the startd.

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
run their rounds in waves, not in parallel, and `--min-gpu-memory-mb`
narrows the match further by excluding cards too small for the configured
`--model-size`. Check what each silo actually matched in `silo_check.py`
output before committing to a long run.

## Outputs (staged to `output/`)

- `{site}_manifest.json` — frozen split manifests with SHA-256 (Tier 0/1).
  In cross-silo runs each also records where the shard landed and which
  host, job user and `HOME` resolved that path.
- `benchmark_events.csv` — frozen event benchmark set B
- `{method}_L{L}_best.ckpt` — best-validation-loss checkpoints
- `{method}_L{L}_metrics.csv`, `steps_metrics.csv` — per-instance metrics
- `e1_topsis.csv` (+ `e21`/`e22`) — per-pool TOPSIS scores
- `figures.tar.gz` — learning-curve plots + summary table
- `validation_report.md` — SPEC Sec. 5 gate results

## Running on another pool

Nothing in the repository hard-codes a host or a path: catalogs,
properties, and the FL-round sub-workflow YAMLs are all generated from the
directory the generator runs in, and they are gitignored. To run this
somewhere else:

1. Clone the repo on the submit host and `pip install -r requirements.txt`
   into a virtualenv.
2. Build the three Apptainer images (Quick start step 1). They are large
   and not in git.
3. Run `workflow_generator.py`, which writes `sites.yml`,
   `transformations.yml`, `pegasus.properties`, and `fl_subwf.properties`
   for that host.
4. If the pool's nodes are small, cap requests with `--max-job-memory-gb`
   and `--max-job-cores`; if it mixes GPU models, set
   `--min-gpu-memory-mb` so training does not land on a card too small
   for the configured `--model-size`.
5. For cross-silo runs, copy `silos.example.yml`, put the new pool's
   machine names in it, and run `tools/silo_check.py` before planning. The
   defaults need nothing installed or configured on the workers; see
   [When worker setup is needed](#when-worker-setup-is-needed) for the two
   cases that do.

Worker packages are configured for containers whose OS differs from the
submit host (`pegasus.transfer.worker.package.strict=false`), which the
sub-workflow properties repeat because FL rounds are planned with their
own configuration file, not the parent's.

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
