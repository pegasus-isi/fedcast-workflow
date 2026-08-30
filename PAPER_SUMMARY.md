# Paper Summary — Fed-Cast

**Fed-Cast: Federated Training and Evaluation Workflow for Generative Precipitation Nowcasting Across Climate-Diverse Regions**

- **Authors:** Zhe Xu, Talha Mehboob, Michael Zink, and Christopher Davis (Department of Electrical and Computer Engineering, University of Massachusetts Amherst)
- **Venue:** eScience 2026 (camera-ready)
- **Source:** `_CR_escience_2026.pdf`

## The problem

Short-term precipitation nowcasting (0–2 hour rainfall forecasts) needs massive, climate-diverse radar archives for training. Centralizing that data is increasingly impractical: a single active radar produces ~0.6–1.0 GB/h of raw observations, and aggregating dozens of radars creates severe bandwidth, storage, and data-governance burdens, plus a single point of failure.

## What Fed-Cast is

A federated training-and-evaluation workflow that keeps radar data at its regional source and exchanges only model parameters. Key components:

- **Data:** NOAA MRMS `PrecipRate` product (0.01° grid, 2-min cadence), partitioned into **seven radar-centered regional clients** spanning diverse climates:
  - Subtropical maritime convection — KBYX (542 retained sequences)
  - Southern/central Great Plains convection — KTLX (478), KVNX (489)
  - Pacific coastal and orographic — KLGX (885)
  - Inland and coastal Northeast — KENX (839), KBOX (715)
  - High-latitude coastal/mountainous Alaska — PAHG (831)

  Each client covers a 3°×3° window (300×300 field at MRMS resolution); models take 4 input frames (8 min of context) and predict 12 frames (24 min ahead) at 2-min cadence. Train/validation/test split is ~80/10/10, with the first three days of each month reserved as a deterministic held-out test block.
- **Model:** Each client locally trains a **DGMR** (Deep Generative Model of Radar, a conditional GAN nowcaster); a central server synchronizes via **FedAvg with uniform client weighting**, orchestrated with the Flower framework — 100 rounds, one local epoch per client per round, all seven clients participating each round. Checkpoint selection uses lowest generator validation loss.
- **Evaluation:** A custom **Model-Compare-Tool (MCT)** standardizes inference, event-driven benchmarking (weather events from WPC Mesoscale Precipitation Discussions, Local Storm Reports, and the NOAA/NCEI Storm Events Database), and multi-metric verification (CSI, POD, FAR, HSS, GSS, MCC, F1, SEDI, CRPS, PSNR, RAPSD, bias, runtime, etc., built on PySTEPS, TorchMetrics, and MET/METplus), summarized into a single ranking via **TOPSIS** with objective-side-balanced weighting. DGMR is evaluated as a 6-member stochastic ensemble; the STEPS baseline uses a 20-member PySTEPS ensemble.

## Key results

Comparing federated DGMR vs. centralized DGMR vs. a classical STEPS baseline over training intervals L ∈ {1, 3, 6, 12, 24, 48} months:

- **Fed-Cast beats centralized DGMR for 1–24 month training intervals.** The gap is largest with only 1 month of data: TOPSIS 0.8637 (federated) vs. 0.7019 (centralized) — a 57.53% peak instance-level advantage.
- **At 48 months the centralized model edges ahead only slightly** (0.8524 vs. 0.8476, ~0.56%), so federation stays highly competitive at scale.
- **Communication savings are order-of-magnitude:** federated training moves ~0.853 TB of serialized model states total (609.1 MB per client per round, bidirectional, 7 clients × 100 rounds), versus an estimated 147–245 TB of raw Level-II radar transfer over the 48-month period — roughly 173–288× more. (The authors flag this as an order-of-magnitude comparison, since the experiments use MRMS PrecipRate rather than Level-II data.)
- **STEPS still scores highest on the reported TOPSIS composite (~0.92)**, but the authors attribute that to the metric inventory's gridpoint bias and low rain-rate threshold (0.1 mm/h), which double-penalize sharp-but-slightly-displaced generative forecasts, while smoother STEPS fields retain greater pixelwise overlap.
- **Ablations:**
  - *E2.1 (aggregation rule):* switching to a quadratic sample-count-derived client weighting does not remove the federated advantage — uniform client weighting is not responsible for the short-window ordering.
  - *E2.2 (optimization):* adding generator-side Sharpness-Aware Minimization (ρ = 0.025 and 0.0125) to centralized training alters but does not cleanly reproduce the federated short-window behavior — the mechanism behind the advantage remains open.

## Takeaway and caveats

The paper's contribution is framed as a **reproducible workflow and controlled testbed**, not a new FL algorithm or nowcasting architecture. It demonstrates that federated generative nowcasting is feasible under regional distributional heterogeneity without sacrificing skill, while drastically cutting data movement.

Caveats the authors note explicitly:

- Paired-sample sizes shrink at longer intervals (n = 2, 16, 8, 4, 2, 1 for L = 1, 3, 6, 12, 24, 48 months; only one paired instance at 48 months), so curves describe evaluated runs rather than statistical significance.
- The seven regions do not isolate climatology as a causal factor; results establish workflow feasibility, not that federation is intrinsically more data-efficient.
- TOPSIS normalization is fitted per candidate pool, so scores are not comparable across experiment figures.
- Future work: matched aggregation and SAM controls, higher rain-rate thresholds, metric-family sensitivity analyses, and MODE-based object diagnostics.

## Relevance to this repository

The paper's pipeline (Fig. 3) maps naturally onto a Pegasus DAG for a `fedcast-workflow`:

1. MRMS PrecipRate download by site and split
2. GRIB2 → NetCDF conversion
3. Radar-centered subdomain cropping
4. Freeze manifests (site-month file index)
5. Sequence preprocessing (train/validation/test)
6. Event table construction (MPD + LSR + StormEvents)
7. Federated DGMR training loop (Flower server + 7 clients, checkpointing)
8. MCT evaluation → per-metric verification → TOPSIS ranking

## Funding

U.S. National Science Foundation Award No. CNS-2335335, and Jerome M. and Linda L. Paros through the Paros Center for Atmospheric Research.
