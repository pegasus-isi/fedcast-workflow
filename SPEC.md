# SPEC — fedcast-workflow

Pegasus WMS workflow to reproduce the results of **Fed-Cast: Federated Training and
Evaluation Workflow for Generative Precipitation Nowcasting Across Climate-Diverse
Regions** (Xu, Mehboob, Zink, Davis — UMass Amherst, eScience 2026; see
`_CR_escience_2026.pdf` and `PAPER_SUMMARY.md`).

**Goal:** reproduce the paper's central result — federated DGMR outperforms
centralized DGMR on the event-driven TOPSIS benchmark for 1–24 month training
intervals (0.8637 vs. 0.7019 at 1 month) and remains within ~0.56% at 48 months —
using an independent, fully automated Pegasus pipeline.

---

## 1. Workflow overview

The workflow mirrors the paper's three-stage architecture (paper Fig. 3): **Data →
Training → Evaluation (MCT)**, plus the two ablations (E2.1, E2.2). It is
parameterized by:

| Parameter | Values | Source |
|---|---|---|
| Radar sites | KBYX, KTLX, KVNX, KLGX, KENX, KBOX, PAHG | paper Sec. III-B |
| Training intervals `L` | {1, 3, 6, 12, 24, 48} months | paper Sec. IV-A |
| Methods | DGMR-centralized, DGMR-federated (Fed-Cast), STEPS | paper Sec. IV-A |
| Experiments | E1 (primary), E2.1 (quadratic weighting), E2.2 (SAM ρ ∈ {0.025, 0.0125}) | paper Sec. V |

### 1.1 DAG structure

```
Phase A — Data construction (per site × month; 7 × 48 = 336 chunks)
  A1. list_mrms(site, month)        # index MRMS PrecipRate files on s3://noaa-mrms-pds
  A2. fetch_mrms(site, month)       # download GRIB2 chunks (anonymous S3; retry w/ backoff)
  A3. grib2_to_netcdf(site, month)  # standardize precipitation fields
  A4. crop_subdomain(site, month)   # 3°×3° radar-centered window → 300×300 @ 0.01°
  A5. freeze_manifest(site)         # site-month file index; content-hashed for determinism
  A6. preprocess_sequences(site)    # 16-frame sequences @ 2-min cadence (4 in / 12 out),
                                    # precipitation-content filter, 80/10/10 split:
                                    # test = first 3 calendar days of each month,
                                    # validation = fixed-seed sample of remainder

Phase B — Event benchmark construction (once)
  B1. fetch_wpc_mpd                 # WPC Mesoscale Precipitation Discussions (Iowa Mesonet)
  B2. fetch_lsr                     # Local Storm Reports (Iowa Mesonet GeoJSON)
  B3. fetch_storm_events            # NOAA/NCEI Storm Events Database CSVs
  B4. build_event_table             # unify → {event ID, source, UTC window, footprint}
  B5. compile_benchmark             # balanced event selection → fixed benchmark set B

Phase C — Training (per L; GPU jobs)
  C1. train_centralized(L)          # DGMR on pooled 7-site data, 100 epochs,
                                    # validate every 5, keep lowest generator val-loss ckpt;
                                    # chain of checkpointed segment jobs (10 epochs/job)
  C2. train_federated(L)            # synchronous FedAvg: 7 clients, uniform weights,
                                    # 100 rounds × 1 local epoch, all clients per round,
                                    # validate every 5 rounds, same checkpoint rule;
                                    # ONE SubWorkflow PER ROUND (fl_round.py): per-client
                                    # fan-out → fl_aggregate → fl_validate, chained via
                                    # the global-model file.
                                    # With --silos (Q12): client jobs are pinned to the
                                    # worker holding their shard and read it in place,
                                    # and validation fans out to fl_validate_client at
                                    # the silos, which return batch-loss sums only
  (STEPS requires no training — it runs at inference time in Phase D)

Phase D — Evaluation via MCT (per method × L × event)
  D1. mct_infer(method, L, event)   # DGMR: K=6 stochastic ensemble; STEPS: 20-member
                                    # PySTEPS ensemble (6 cascade levels, BPS perturbations)
  D2. mct_verify(method, L, event)  # per-lead-time metrics via PySTEPS + TorchMetrics
                                    # (+ METplus hooks); threshold θ = 0.1 mm/h
  D3. topsis_aggregate(experiment)  # lead-time-averaged metrics → objective-side-balanced
                                    # TOPSIS per candidate pool (Table I metric suite)
  D4. make_figures                  # learning-curve boxplots (paper Figs. 4–6),
                                    # communication-volume estimate (Eq. 7)

Phase E — Ablations (reuse Phases A/B/D)
  E2.1: train_federated_quadratic(L)   # w_i = max(1, ⌊n_i²/n_max⌋) aggregation
  E2.2: train_centralized_sam(L, ρ)    # generator-side SAM, ρ ∈ {0.025, 0.0125}
```

The step names above are the design's, not the implementation's: A1–A5 are fused
into one `fetch_crop_mrms` job per (domain, month) so full-CONUS files are never
persisted (open question 8), and the fetch fans out per domain rather than per
site because one download serves every site in that domain. See §1.2 for the
jobs as built.

Phase A fans out per (site, month) and is independent across sites; Phase C depends
on all Phase A outputs for the sites/months inside its interval `L`; Phase D depends
on Phase B and the relevant Phase C checkpoint. Ablations are separate sub-DAGs
gated on E1 completion (they reuse E1 data and benchmark artifacts).

### 1.2 Repository layout (as built)

```
fedcast-workflow/
├── workflow_generator.py      # Pegasus DAG generator (Pegasus.api)
├── fl_round.py                # builder for one FL-round SubWorkflow
├── silos.example.yml          # cross-silo placement map template (§6 Q12)
├── bin/                       # job wrappers, staged to the workers
│   ├── fedcast_common.py      # shared model/data helpers for the fl_* jobs
│   ├── fetch_crop_mrms.py     # S3 fetch + crop-on-ingest, one job per
│   │                          # (domain, month); fuses planned A1-A4
│   ├── preprocess_sequences.py  # sequences, rain filter, frozen split,
│   │                          # manifest; --silo-dir keeps the shard resident
│   ├── fetch_events.py        # MPD / LSR / StormEvents (best-effort per source)
│   ├── build_benchmark.py     # balanced event selection → benchmark set B
│   ├── train_dgmr.py          # centralized DGMR segment job (Lightning)
│   ├── fl_init.py             # seeds the global model for an FL chain
│   ├── fl_train_client.py     # one client's local epoch for one round
│   ├── fl_aggregate.py        # FedAvg, uniform or quadratic weights
│   ├── fl_validate.py         # chained history + best-so-far checkpoint
│   ├── fl_validate_client.py  # cross-silo: score at the silo, return metrics
│   ├── silo_export.sh         # cross-silo: shard egress for the pooled arms
│   ├── mct_infer.py           # forecast adapters: DGMR / PySTEPS
│   ├── mct_verify.py          # metric computation
│   ├── mct_topsis.py          # TOPSIS per Eq. 5-6 (no clipping / no epsilon)
│   ├── make_figures.py
│   └── validate_report.py     # tiered reproduction gates (§5)
├── tools/                     # submit-host helpers, not workflow jobs
│   ├── check_export_docs.py   # fails if a silo job exports an undocumented field
│   ├── silo_check.py          # cross-silo preflight (placement + durability)
│   ├── silo_worker_setup.sh   # worker prep, only for non-default silo maps
│   └── timing_extrapolate.py  # project full-study wall-clock from a run dir
├── Apptainer/                 # FedCast_{data,train,eval}.def
├── run_manual.sh              # tiny end-to-end smoke test without Pegasus
├── requirements.txt           # submit-host only; job deps live in the images
├── SPEC.md                    # this file
├── PAPER_SUMMARY.md
└── README.md
```

Generated and gitignored: `workflow.yml`, the catalogs (`sites.yml`,
`transformations.yml`, `replicas.yml`), `pegasus.properties`, the FL-round
sub-workflow files (`fl_rounds/`, `fl_subwf.properties`, `fl_subwf_rc.yml`),
and the `scratch/` and `output/` directories. All are written relative to the
directory the generator runs in, so nothing in the repository hard-codes a
host or a path.

Containers are built locally from `Apptainer/*.def` into `Apptainer/*.sif`
(gitignored — several GB each) rather than pulled from a registry, so the
image the jobs run is the one on the submit host.

---

## 2. Constraints (must match the paper for a valid reproduction)

**Data**
1. MRMS `PrecipRate` product only (0.01° grid, nominal 2-min cadence) — not Level-II
   volumetric data.
2. Exactly the seven radar-centered 3°×3° subdomains at the WSR-88D coordinates of
   KBYX, KTLX, KVNX, KLGX, KENX, KBOX, PAHG → 300×300 fields.
3. 48-month contiguous archive; training intervals selected as suffixes/subsets of it
   with matched periods across paradigms.
4. Sequence definition: 16 frames at 2-min cadence — 4 input (8-min context), 12
   target (24-min horizon).
5. Split rule: test = first three available days of each month (contiguous held-out
   calendar blocks, ≈10%); validation = comparable-size fixed-random-seed sample of
   the remainder; split manifests frozen and **shared identically** between
   centralized and federated runs.

**Training**
6. Same DGMR architecture and hyperparameters for both paradigms (openclimatefix
   `skillful_nowcasting` implementation; `forecast_steps=12`; λ_grid = 20; PyTorch
   Lightning defaults elsewhere). Only the training workflow may differ.
7. Centralized: 100 epochs on pooled data, validation every 5 epochs.
8. Federated: synchronous FedAvg via Flower; 7 clients; **uniform client weighting**
   (E1 baseline, Eq. 4); 100 rounds; 1 local epoch/client/round; full participation
   every round; validation every 5 rounds.
9. Checkpoint selection: lowest **generator validation loss** (Eq. 3) in both
   paradigms.
10. Fixed nominal outer-loop count (100 epochs ≡ 100 rounds) is the matched budget —
    not wall-clock, GPU-hours, or optimizer steps.

**Evaluation**
11. All three methods evaluated on the **same frozen event-driven benchmark**
    (WPC MPD + LSR + NOAA Storm Events), same MCT pipeline, same TOPSIS criteria.
12. DGMR inference: K=6 stochastic ensemble (`num_samples=6`); ensemble mean for
    deterministic metrics; full ensemble for CRPS. STEPS: 20-member PySTEPS ensemble,
    6 cascade levels, nonparametric noise, Bowler–Pierce–Seed velocity perturbations,
    incremental precipitation mask.
13. Metrics computed **per lead time** then averaged over the 12 leads; rain/no-rain
    threshold θ = 0.1 mm/h; metric suite exactly per paper Table I.
14. TOPSIS: objective-side-balanced weighting (benefit side and cost side each get
    total weight 0.5, split equally within side); vector normalization; ideals fitted
    per candidate pool; HK and BIAS converted to |x−1| deviations; no clipping or
    epsilon stabilization. E1, E2.1, E2.2 are **separately normalized pools** — never
    compare scores across pools.
15. Ablation definitions: E2.1 quadratic weighting `w_i = max(1, ⌊n_i²/n_max⌋)`
    (Eq. 8); E2.2 generator-side SAM at ρ ∈ {0.025, 0.0125} with everything else
    unchanged.

**Engineering (repo-wide rules)**
16. Credentials/API keys reach jobs only via `add_env(...)` at generation time —
    never via submit-shell exports. (MRMS S3 is anonymous; Iowa Mesonet/NCEI need no
    keys, so this mainly applies if a mirror requiring auth is added.)
17. Fetch jobs retry transients with backoff and carry
    `add_dagman_profile(retry="2")`. MRMS fetches are **required** sources:
    write the declared (possibly empty) output, then exit non-zero on permanent
    failure — never exit without the declared output. Event-source fetches (B1–B3)
    are **best-effort**: empty + ERROR log + exit 0; `build_event_table` fails only
    if all three sources are empty.
18. All jobs containerized; deterministic seeds recorded in run metadata; every
    derived artifact (manifests, splits, benchmark set, checkpoints) content-hashed.

---

## 3. Non-constraints (explicitly free to differ)

1. **Wall-clock time, GPU model, node count, site placement.** The budget is matched
   in epochs/rounds, not hardware. Any CUDA-capable site (Chameleon, FABRIC, local
   HTCondor pool) is acceptable.
2. **Geographic distribution of clients.** The paper itself *emulates* federation
   from a common MRMS archive (Sec. III-B; Fig. 1's caption states the server icon
   "denotes a server-side role, not an actual deployment location"), so running all
   7 clients on shared GPU nodes is faithful and remains the default.
   **IMPLEMENTED (2026-09-05) as an option beyond the paper:** `--silos <map>`
   pins each client's data and jobs to the worker holding its shard, so no client
   data is staged for the federated arm — see open question 12.
3. **MCT as a software artifact.** MCT is not publicly released (as of Aug 2026); we
   reimplement its *behavior* (adapters → fixed benchmark → per-lead metrics →
   TOPSIS) from the paper's specification rather than reuse its code.
4. **METplus/MET integration.** The paper's TOPSIS scores use PySTEPS/TorchMetrics-
   computable quantities; MODE object diagnostics are explicitly excluded from the
   reported rankings. METplus is optional plumbing, not needed for reproduction.
5. **Exact TOPSIS score values.** GAN training is stochastic and the paper's seeds
   are unpublished; we target the *ordering and effect sizes* (see §5), not
   digit-level score equality.
6. **Storage/file layout, intermediate formats** (NetCDF chunking, tensor
   serialization), and the Flower version — any synchronous FedAvg-faithful
   implementation qualifies.
7. **The paper's exact event count / benchmark composition**, since the balanced-
   selection procedure is not fully specified (see §6). We freeze *our own*
   benchmark set once and use it identically across all methods, which preserves
   the paper's internal-validity design.
8. **Privacy mechanisms.** The paper adds none (no DP, no secure aggregation); we
   don't either.

---

## 4. Expected outcomes

**Artifacts**
- A frozen, hash-stamped 7-site sequence dataset with split manifests, and retained-
  sequence counts per site to compare with the paper's (KBYX 542, KTLX 478, KVNX
  489, KLGX 885, KENX 839, KBOX 715, PAHG 831 at 48 months).
- 6 centralized + 6 federated DGMR checkpoints (one per `L`), plus E2.1 federated
  and E2.2 SAM-centralized checkpoint sets.
- A frozen event benchmark table and per-(method, L, event) metric records.
- TOPSIS learning-curve figures reproducing the *shape* of paper Figs. 4–6, and a
  communication-volume table reproducing Eq. 7's ~0.853 TB federated payload
  estimate against the 147–245 TB raw-transfer bound.

**Scientific claims to reproduce**
- **R1 (primary):** federated DGMR ≥ centralized DGMR on TOPSIS for every
  L ∈ {1, 3, 6, 12, 24}, with the largest gap at L = 1 month.
- **R2:** at L = 48, centralized ≥ federated with a gap ≲ 1–2% (paper: 0.56%).
- **R3:** federated scores occupy a narrower range across L than centralized
  (stability of the federated curve).
- **R4:** STEPS scores highest on the reported composite (an artifact of the
  gridpoint metric inventory at θ = 0.1 mm/h, per the paper's own interpretation).
- **R5 (E2.1):** the federated-over-centralized short-window ordering persists under
  quadratic client weighting.
- **R6 (E2.2):** SAM alters the centralized short-window trajectory but the analysis
  remains inconclusive as a substitute for federation — we expect qualitative
  agreement, not specific SAM curves.

---

## 5. Validation criteria

Tiered — each tier is a pass/fail gate evaluated by a final `validate_report` job.

**Tier 0 — Pipeline determinism (hard gate)**
- Re-running Phase A from the same MRMS object list yields byte-identical manifests
  and split files (hash check).
- Centralized and federated runs for a given `L` consume identical train/val/test
  manifests (hash equality asserted at training-job start).
- The benchmark set B hash is identical across all Phase D jobs.

**Tier 1 — Data fidelity**
- Sequence tensor shape (16, 300, 300), 2-min cadence, and per-site subdomain
  bounds verified programmatically.
- Retained-sequence counts per site within ±15% of the paper's values (exact
  equality is unlikely since the filtering rule is under-specified; a large
  deviation signals a wrong filter and fails this tier).
- Paired-instance counts per L reported alongside the paper's n = (2, 16, 8, 4, 2, 1)
  for L = (1, 3, 6, 12, 24, 48).

**Tier 2 — Primary result (the reproduction claim)**
- R1 holds: federated TOPSIS > centralized TOPSIS at every L ∈ {1,…,24} within the
  E1 pool. **This is the headline pass/fail criterion.**
- The L = 1 gap is large and positive: federated − centralized ≥ 0.05 TOPSIS
  (paper: 0.16).
- R2 holds: |centralized − federated| ≤ 0.02 at L = 48.
- R4 holds: STEPS ranks first in the E1 pool.

**Tier 3 — Secondary/ablation results**
- R5: E2.1 preserves the short-window ordering.
- R6: E2.2 centralized-SAM trajectories differ measurably from vanilla centralized
  at short windows (any direction; the paper treats this as exploratory).
- Communication-volume estimate reproduces Eq. 7 to within serialization-size
  differences (payload = R·K·(S↓+S↑) with our measured checkpoint size S).

**Tier 4 — Sanity checks on metrics**
- CSI/POD/FAR monotonic degradation with lead time for all methods.
- CRPS(ensemble) ≤ MAE(ensemble mean) per event (proper-score sanity).
- STEPS fields smoother than DGMR fields (RAPSD comparison), matching the paper's
  qualitative Fig. 2 analysis.

Given the small n at large L (one paired instance at 48 months in the paper), Tier 2
is evaluated on medians of our date-tagged distributions, and the report must state
our n per interval — no statistical-significance claims, mirroring the paper's own
framing.

---

## 6. Open design questions

1. **Which 48-month window?** The MRMS AWS archive begins 2020-10-14 and the paper's
   example event is dated 2024-01-10, but the exact study interval is unstated.
   *Proposal:* 2021-01 through 2024-12 (fully contained in the archive, contains the
   Fig. 2 event). Needs confirmation — or contact with the authors.
2. **Precipitation-content filter.** "Preprocessing filters out data with
   insufficient precipitation information" is not quantified (no threshold, coverage
   fraction, or per-sequence rule given). This directly drives the retained-sequence
   counts in Tier 1. *Proposal:* calibrate a (rain-fraction ≥ p at θ = 0.1 mm/h)
   rule to approximate the published counts, and document it as a deviation.
3. **PAHG (Alaska) coverage.** ~~Must verify PrecipRate exists for the Alaska
   domain over the chosen window.~~ **RESOLVED (2026-08-31, verified against the
   live bucket):** `ALASKA/PrecipRate_00.00/` exists with the same archive start
   as CONUS (2020-10-14), full 2-min cadence (720 files/day) at the start,
   middle, and end of the proposed 2021-01→2024-12 window, and an identical
   0.01° grid (2200×5000, lat 50–72, lon −176 to −126). A live decode+crop test
   of a 2024-01-10 frame yielded an exact 300×300 PAHG window with 100% valid
   pixels. No code changes required.
4. **MCT reimplementation risk.** Since MCT is unreleased, subtle choices
   (ensemble-mean thresholding order, per-event vs. pooled contingency tables,
   TOPSIS candidate-pool membership per "experiment bundle") must be inferred. Worth
   emailing the authors for the tool or the exact TOPSIS input tables.
5. **Benchmark "balanced events" selection.** The compile step "balanced events →
   benchmark set B" (paper Fig. 3) doesn't specify balancing dimensions (per site?
   per season? per source?) or the event count. *Proposal:* balance per site ×
   source, cap events per site, freeze with a fixed seed.
6. **Federated training inside Pegasus.** ~~Options: (a) one monolithic GPU job per
   (paradigm, L); (b) checkpointed segments; (c) per-round SubWorkflows with
   per-client jobs.~~ **DECIDED (2026-08-30): per-round SubWorkflows (c).** Each FL
   round is a sub-DAG (`fl_round.py`): fl_train_client × 7 in parallel →
   fl_aggregate (FedAvg) → fl_validate (validation rounds chain history +
   best-so-far; final round emits the best checkpoint). Rounds are chained through
   the global-model file. Centralized training keeps checkpointed segment chains
   (b) — it has no client structure to express. Trade-offs accepted: ~100 sub-DAG
   plannings per (method, L) and per-round staging of the global model and client
   sequence files; gained: the per-client job structure that cross-silo placement
   later builds on (open question 12), per-round retry granularity, and per-round
   visibility.
   Follow-up RESOLVED (2026-08-30, pilot run0004): sub-to-sub chaining works
   natively (round r+1 consumes round r's global model via runtime planning),
   but parent jobs cannot consume sub-workflow outputs directly — subs stage
   outputs to the output site while parent stage-ins expect parent scratch.
   Bridged with a local `collect_file` (/bin/cp) job that re-introduces the
   final best checkpoint into parent staging under the canonical LFN.
7. **GPU budget.** 6 intervals × (centralized + federated) + E2.1 (6) + E2.2 (2 ρ ×
   6) ≈ 30 DGMR training runs of 100 epochs/rounds each on ~4.5K–33K sequences.
   Need an estimate of hours-per-run on available GPUs (Chameleon A100? FABRIC?)
   before committing; may need to stage E1 first and gate ablations on results.
8. **Storage footprint.** 48 months × 2-min CONUS PrecipRate GRIB2 is tens of TB
   before cropping. *Proposal:* crop-on-ingest (A2–A4 fused in one job per chunk,
   discarding full-CONUS files immediately) so only 7 × 300×300 subdomain archives
   persist. Decide whether the cropped archive (~hundreds of GB) lives on the
   submit host, a shared FS, or S3-compatible staging.
   **Partially resolved (2026-08-31):** the FL-chain side is fixed — each round's
   ~582 MB global model must be staged to the output site (the next round's
   runtime planning locates it there), so `cleanup_file` (/bin/rm) jobs now
   delete superseded chain artifacts incrementally with a two-round rescue-safe
   window; per-chain high-water mark is ~2.3 GB instead of ~58 GB (~14 GB total
   for E1 instead of ~700 GB). The cropped-archive placement question remains.
9. **DGMR hyperparameters beyond the cited defaults.** The paper defers to the
   openclimatefix implementation + Lightning defaults; those defaults have changed
   across releases. Pin exact package versions (and record them in the containers)
   — which release corresponds to the paper is unknown.
10. **Event-window → input-sequence mapping.** How MCT picks the forecast
    initialization time(s) within each event's UTC window (one init per event?
    all inits in the window?) is unspecified; this changes per-event sample sizes.
    *Proposal:* all valid 16-frame sequences whose target window intersects the
    event window, documented as our rule.
11. **300×300 fields vs. DGMR's 32-divisibility requirement.** The openclimatefix
    DGMR requires spatial dims divisible by 32 (its latent/conditioning stacks
    downsample by 32); 300×300 is not, and feeding it fails with a ConvGRU shape
    mismatch (confirmed empirically 2026-08-30). How did the paper feed 300×300
    MRMS windows into DGMR — padding to 320, resizing/cropping to 256 or 288, or
    a modified architecture? *Our documented rule:* center-crop to 288×288
    (= 9×32) at the model boundary, applied identically to every method
    (including STEPS) so the evaluation grid stays uniform.

12. **Client data placement.** ~~The per-round SubWorkflow structure gives each
    client its own job, but nothing made a client's shard *stay* anywhere: shards
    were built centrally and re-staged to whatever worker HTCondor matched, every
    round, and `fl_validate` read every client's validation split on one node.
    That is emulated FL, not cross-silo FL.~~ **RESOLVED (2026-09-05): both models
    are supported, emulated by default.**

    `--silos <map>` (see `silos.example.yml`) makes each client a real data holder:

    - `preprocess_sequences` runs pinned to the client's silo and writes the shard
      into an on-worker directory (`--silo-dir`). The shard is not a Pegasus file,
      so nothing can stage it implicitly.
    - `fl_train_client` and the new `fl_validate_client` carry the silo's HTCondor
      requirements expression and read the resident shard in place. No federated
      job moves the shard; what those jobs do return, besides the model weights,
      is small per-client metadata — `n_train` from training (the aggregator
      needs it for the Eq. 8 quadratic-weighting ablation) and
      `n_val`/`n_batches`/loss sum and mean from validation.
    - Validation is split: each client scores the new global model on its own split
      at its silo and returns batch-loss sums; `fl_validate --client-metrics`
      recombines them. `fedcast_common.combine_client_val_metrics` reconstructs
      exactly the mean batch loss the central path computes, so constraint 9's
      checkpoint rule is unchanged and the two modes are comparable.
    - **The federated arm's zero-egress property does not extend to the run as a
      whole.** The centralized baseline and MCT evaluation need shards pooled —
      that is the thing the paper measures federation against — and every run
      builds them, so silo mode adds an UNCONDITIONAL pinned `silo_export` job
      per client: each shard is copied out exactly once. What the mode buys is
      7 copies instead of 7 x rounds, and egress that is an explicit DAG node on
      the centralized side and absent on the federated side, making the asymmetry
      behind Eq. 7 structural rather than assumed. A run with no shard egress at
      all would need a federated-only workflow (no centralized arm, no pooled
      evaluation), which the generator does not currently build. The per-client
      per-client metadata above also leaves the silo, as does the split manifest
      — which is staged out as a Tier 0/1 reproduction artifact and is the
      largest metadata export, carrying per-sequence `start_epochs` and
      `split_labels` vectors (a timestamp and split label for every retained
      sequence at that site), the shard's on-worker path and resolving identity
      under `silo`, and `sequence_sha256`, which digests the sequence array
      alone and so does not verify the shard file. All of it is derived from client data
      rather than being the data, but no privacy claim is made about any of it
      (non-constraint 8 — the paper adds no privacy mechanism either). What
      cross-silo mode changes is the bulk movement, not every trace of the data;
      README's "Data placement" section lists the exports in full, field by
      field, and that list is enforced in two halves rather than maintained by
      hand. Each wrapper declares an `EXPORT_FIELDS` tuple, and enforcement is
      at run time in three layers: `fedcast_common.write_export()` validates
      and writes in one call (validation recurses through nested mappings *and*
      sequences, so a dict inside a list is covered, and an undeclared field
      aborts before the file exists); the file is read back and validated
      again; and `guard_export()` registers the path up front and re-validates
      the artifact at interpreter exit, which is the layer that carries the
      guarantee because it holds whatever wrote the file — an aliased
      `json.dump`, a hand-rolled write, or a guarded call that proved
      unreachable. Every export must be a JSON object: a payload replaced by a
      bare array or scalar exposes no field names and would otherwise pass by
      carrying nothing the check can name. The check controls field *names*
      only — not the type or size of a declared field, which is why the
      manifest's per-sequence `start_epochs`/`split_labels` vectors are
      disclosed in README's table rather than left to it. Statically, `tools/check_export_docs.py` requires each field
      in README's export section and each payload to be registered with
      `guard_export` and written with `write_export`, both passing
      `EXPORT_FIELDS` with payload and label in agreement. It deliberately does
      NOT claim to prove the absence of other write paths — not decidable by
      reading Python — and its `json.dump` detection is a lint, not a proof.
      An earlier version of that tool tried to derive the surface by analysing
      how each payload was built; that cannot be completed in a dynamic language
      (successive reviews found an unresolvable name, `dict(...)` in place of a
      literal, then post-construction mutation), which is why the authoritative
      check now runs against the real object. `preprocess_sequences` therefore
      takes `fedcast_common.py` as an input, which is the only DAG change: two
      extra staged-input entries per site in the pilot.
    - **No worker-side setup is needed for the default map.** Silos pin by
      HTCondor's built-in `Machine` attribute (nothing to advertise, no
      `condor_reconfig`), and the default shard directory `~/.fedcast/silos` is
      inside the job user's home, which Apptainer mounts, so nothing is
      bind-mounted and the preprocess job creates its own directory. Shard
      paths therefore reach jobs unexpanded; `fedcast_common.parse_client`,
      `preprocess_sequences.py` and `bin/silo_export.sh` expand them in the job's
      own environment, because Pegasus does not expand job arguments. The shell
      wrapper uses parameter substitution, never `eval` — the path comes from a
      config file, and `eval` would execute anything in it.
    - **The home-relative default's one assumption** is that every job on a
      worker runs as the same user. A pool with per-slot users would resolve the
      directory differently for the preprocess and training jobs; `silo_check.py`
      queries `SLOT_USER` per matched worker and says so, and an absolute
      `data_dir` is the fix. `/tmp` and `/var/tmp` are explicitly NOT usable
      despite Apptainer mounting them: HTCondor defaults `MOUNT_UNDER_SCRATCH`
      to `/tmp,/var/tmp`, making both private per job and deleting them at job
      end, so a shard there would not survive to the next round. The generator
      warns on such a `data_dir` (matching the directory itself as well as
      anything under it) and `silo_check.py` checks it per worker, reporting a
      worker whose config cannot be read as unverified rather than as fine —
      with its own exit status (3, vs 1 for a real problem) so
      `silo_check.py && pegasus-plan` is safe by default and an unchecked pool
      is never mistaken for a clean one; `--allow-unverified` accepts it on
      pools that refuse remote config queries. `load_silo_map` raises ValueError
      for every way the map can be wrong (unparseable YAML, wrong top-level
      shape, non-mapping `silos`, a non-mapping or non-string silo entry), so
      both the generator and the preflight report a configuration mistake
      instead of a traceback.
      Failures are made self-diagnosing rather than mysterious: a missing shard
      reports host, job user and `HOME`, and each manifest records the same three
      for the preprocess job that wrote it.
    - Two departures still need `tools/silo_worker_setup.sh`: an absolute
      `data_dir` elsewhere (Pegasus scopes `container.arguments` to the catalog,
      not the job, so it is bind-mounted pool-wide and EVERY worker needs it —
      `silo_worker_setup.sh none`), and ClassAd pinning (`{}` entries). Pegasus
      cannot own either: its own symlinking-in-containers support has the same
      requirement that the host directory pre-exist and be named in the
      container's `mounts`. `tools/silo_check.py` resolves every silo against
      `condor_status` and, only when the map needs a bind, checks it pool-wide.

    Deliberately out of scope: **ingest is still shared** (one fetch job per
    (domain, month) crops for all clients in that domain, since per-silo ingest
    multiplies a multi-TB download by the client count), and a silo is one worker
    unless its directory is replicated by hand. Both are documented in README.md.
    Note this exceeds the paper, which does no silo pinning of its own; the
    reproduction claim rests on the emulated default.

---

## 7. References / provenance

- Paper: `_CR_escience_2026.pdf` (this directory); summary in `PAPER_SUMMARY.md`.
- MRMS on AWS: [Registry of Open Data — noaa-mrms-pds](https://registry.opendata.aws/noaa-mrms-pds/)
  (archive begins 2020-10-14).
- DGMR implementation: [openclimatefix/skillful_nowcasting](https://github.com/openclimatefix/skillful_nowcasting).
- Federated framework: Flower (Beutel et al. 2020); FedAvg (McMahan et al.).
- STEPS baseline: [PySTEPS](https://github.com/pySTEPS/pysteps).
- Event sources: Iowa Mesonet WPC MPD + LSR services; NOAA/NCEI Storm Events CSVs.
