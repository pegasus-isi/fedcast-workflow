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
Phase B  fetch_events      WPC MPD, LSR, Storm Events (best-effort each)
         build_benchmark   frozen, balanced event set B
Phase C  train_dgmr        centralized DGMR per interval L, as chains of
                           checkpointed segment jobs
         fl_* + SubWorkflows  federated DGMR per interval L: fl_init, then
                           ONE SubWorkflow PER FL ROUND (fl_round.py) —
                           fl_train_client x N in parallel -> fl_aggregate
                           (FedAvg uniform/quadratic) -> fl_validate
                           (chained best-checkpoint tracking)
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

## Outputs (staged to `output/`)

- `{site}_manifest.json` — frozen split manifests with SHA-256 (Tier 0/1)
- `benchmark_events.csv` — frozen event benchmark set B
- `{method}_L{L}_best.ckpt` — best-validation-loss checkpoints
- `{method}_L{L}_metrics.csv`, `steps_metrics.csv` — per-instance metrics
- `e1_topsis.csv` (+ `e21`/`e22`) — per-pool TOPSIS scores
- `figures.tar.gz` — learning-curve plots + summary table
- `validation_report.md` — SPEC Sec. 5 gate results

## Known gaps (scaffold state)

- **E2.2 SAM** training is a TODO stub in `bin/train_dgmr.py` (exits with an
  explicit error).
- **Validation loss** uses the grid-cell-regularizer term of the paper's
  Eq. 3; the discriminator hinge term still needs to be added.
- The precipitation-content filter and benchmark balancing rules are
  documented defaults pending author responses (SPEC open questions 2, 5).
- Container package versions are unpinned until the paper's exact releases
  are known (SPEC open question 9).
