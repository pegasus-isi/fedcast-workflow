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
                                    # the global-model file
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

Phase A fans out per (site, month) and is independent across sites; Phase C depends
on all Phase A outputs for the sites/months inside its interval `L`; Phase D depends
on Phase B and the relevant Phase C checkpoint. Ablations are separate sub-DAGs
gated on E1 completion (they reuse E1 data and benchmark artifacts).

### 1.2 Repository layout (per repo conventions)

```
fedcast-workflow/
├── workflow_generator.py      # Pegasus DAG generator (Pegasus.api)
├── bin/
│   ├── fetch_mrms.py          # S3 fetch wrapper (retry + fail-loud, see §2)
│   ├── grib2_to_netcdf.py
│   ├── crop_subdomain.py
│   ├── preprocess_sequences.py
│   ├── fetch_events.py        # MPD / LSR / StormEvents (best-effort per source)
│   ├── build_benchmark.py
│   ├── train_centralized.py   # PyTorch Lightning DGMR (openclimatefix impl)
│   ├── train_federated.py     # Flower simulation driver
│   ├── mct_infer.py           # forecast adapters: DGMR / PySTEPS / persistence
│   ├── mct_verify.py          # metric computation
│   └── topsis.py              # TOPSIS per Eq. 5–6 (no clipping / no ε-stabilization)
├── Docker/
│   ├── Dockerfile.data        # wgrib2/eccodes, xarray, boto3
│   ├── Dockerfile.train       # CUDA + PyTorch Lightning + Flower + DGMR
│   └── Dockerfile.eval        # PySTEPS, TorchMetrics, METplus (optional), pandas
├── requirements.txt
├── SPEC.md                    # this file
├── PAPER_SUMMARY.md
└── README.md
```

Container images published to Docker Hub under `kthare10/` (e.g.
`kthare10/fedcast-data`, `kthare10/fedcast-train`, `kthare10/fedcast-eval`).

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
   from a common MRMS archive; running all 7 Flower clients as a simulation on one
   GPU node is faithful. True multi-site deployment is a stretch goal, not a
   requirement.
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
3. **PAHG (Alaska) coverage.** MRMS CONUS products do not cover Alaska; MRMS has a
   separate Alaska domain with its own product set and possibly different cadence/
   availability on `noaa-mrms-pds`. Must verify PrecipRate exists for the Alaska
   domain over the chosen window; if not, decide between substituting a CONUS site
   (deviation) or a different Alaska product (deviation).
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
   sequence files; gained: per-client job placement (true multi-site federation
   becomes possible), per-round retry granularity, and per-round visibility.
   Follow-up: verify parent↔sub-workflow file-flow semantics (stage_out /
   --output-sites) on the cluster at pilot scale before full runs.
7. **GPU budget.** 6 intervals × (centralized + federated) + E2.1 (6) + E2.2 (2 ρ ×
   6) ≈ 30 DGMR training runs of 100 epochs/rounds each on ~4.5K–33K sequences.
   Need an estimate of hours-per-run on available GPUs (Chameleon A100? FABRIC?)
   before committing; may need to stage E1 first and gate ablations on results.
8. **Storage footprint.** 48 months × 2-min CONUS PrecipRate GRIB2 is tens of TB
   before cropping. *Proposal:* crop-on-ingest (A2–A4 fused in one job per chunk,
   discarding full-CONUS files immediately) so only 7 × 300×300 subdomain archives
   persist. Decide whether the cropped archive (~hundreds of GB) lives on the
   submit host, a shared FS, or S3-compatible staging.
9. **DGMR hyperparameters beyond the cited defaults.** The paper defers to the
   openclimatefix implementation + Lightning defaults; those defaults have changed
   across releases. Pin exact package versions (and record them in the containers)
   — which release corresponds to the paper is unknown.
10. **Event-window → input-sequence mapping.** How MCT picks the forecast
    initialization time(s) within each event's UTC window (one init per event?
    all inits in the window?) is unspecified; this changes per-event sample sizes.
    *Proposal:* all valid 16-frame sequences whose target window intersects the
    event window, documented as our rule.

---

## 7. References / provenance

- Paper: `_CR_escience_2026.pdf` (this directory); summary in `PAPER_SUMMARY.md`.
- MRMS on AWS: [Registry of Open Data — noaa-mrms-pds](https://registry.opendata.aws/noaa-mrms-pds/)
  (archive begins 2020-10-14).
- DGMR implementation: [openclimatefix/skillful_nowcasting](https://github.com/openclimatefix/skillful_nowcasting).
- Federated framework: Flower (Beutel et al. 2020); FedAvg (McMahan et al.).
- STEPS baseline: [PySTEPS](https://github.com/pySTEPS/pysteps).
- Event sources: Iowa Mesonet WPC MPD + LSR services; NOAA/NCEI Storm Events CSVs.
