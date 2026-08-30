#!/usr/bin/env bash
# Manual end-to-end smoke test of the fedcast wrappers WITHOUT Pegasus.
# Runs a tiny slice: 2 sites, one month of MRMS at stride 15, 2 training
# rounds capped at a few sequences per client (CPU-feasible).
#
# By default runs each phase inside the Apptainer containers if the .sif
# images exist in Apptainer/; otherwise falls back to bare python3 (which
# then needs the dev deps installed — see requirements.txt).
set -euo pipefail

cd "$(dirname "$0")"
ROOT=$(pwd)
WORK=manual_run
mkdir -p "$WORK"

# Phase runners: apptainer exec if the images are present.
if [ -f "$ROOT/Apptainer/FedCast_data.sif" ]; then
    PY_DATA="apptainer exec --bind $ROOT $ROOT/Apptainer/FedCast_data.sif python3"
    PY_TRAIN="apptainer exec --bind $ROOT $ROOT/Apptainer/FedCast_train.sif python3"
    PY_EVAL="apptainer exec --bind $ROOT $ROOT/Apptainer/FedCast_eval.sif python3"
    echo "Using Apptainer containers"
else
    PY_DATA="python3"; PY_TRAIN="python3"; PY_EVAL="python3"
    echo "Containers not found — using bare python3"
fi

cd "$WORK"
BIN=$ROOT/bin
MONTH="2024-01"
SITES_CONUS="KTLX:35.3331:-97.2778:KTLX_${MONTH}_cropped.nc \
KENX:42.5865:-74.0639:KENX_${MONTH}_cropped.nc"

echo "== Phase A: fetch + crop (stride 15 => ~30-min cadence) =="
if [ -f "KTLX_${MONTH}_cropped.nc" ] && [ -f "KENX_${MONTH}_cropped.nc" ]; then
    echo "cropped NetCDFs already present — skipping fetch"
else
    # shellcheck disable=SC2086
    $PY_DATA "$BIN/fetch_crop_mrms.py" --domain CONUS --month "$MONTH" \
        --stride 15 $(for s in $SITES_CONUS; do echo --site "$s"; done)
fi
ls -lh ./*_cropped.nc

echo "== Phase A: sequences =="
for SITE in KTLX KENX; do
    $PY_DATA "$BIN/preprocess_sequences.py" --site "$SITE" \
        --input "${SITE}_${MONTH}_cropped.nc" \
        --output-sequences "${SITE}_sequences.npz" \
        --output-manifest "${SITE}_manifest.json" \
        --val-seed 1337 --rain-threshold 0.1 --min-rain-fraction 0.05
done

echo "== Phase B: events + benchmark =="
for SRC in mpd lsr storm_events; do
    $PY_DATA "$BIN/fetch_events.py" --source "$SRC" \
        --start-month "$MONTH" --end-month "$MONTH" \
        --output "events_${SRC}.json"
done
$PY_DATA "$BIN/build_benchmark.py" \
    --events events_mpd.json --events events_lsr.json \
    --events events_storm_events.json \
    --site KTLX:35.3331:-97.2778 --site KENX:42.5865:-74.0639 \
    --seed 1337 --max-events-per-site 2 --output benchmark_events.csv
cat benchmark_events.csv

CLIENTS="--client KTLX:KTLX_sequences.npz:KTLX_manifest.json \
--client KENX:KENX_sequences.npz:KENX_manifest.json"

echo "== Phase C: training (capped sequences — CPU-feasible) =="
# The wrappers import fedcast_common.py from their cwd (Pegasus stages it
# into job dirs); mirror that here.
cp "$BIN/fedcast_common.py" .

echo "-- centralized: 2 epochs in one segment"
# shellcheck disable=SC2086
$PY_TRAIN "$BIN/train_dgmr.py" $CLIENTS \
    --interval-months 1 --archive-start "$MONTH" --archive-months 1 \
    --segment-index 0 --segment-size 2 --total-units 2 \
    --validate-every 1 --seed 42 \
    --limit-train-sequences 4 \
    --state-out cen_L1_seg0_state.tar.gz \
    --best-out cen_L1_best.ckpt

echo "-- federated: init + 1 FL round (the fl_* wrappers a round"
echo "   SubWorkflow runs: train per client -> aggregate -> validate)"
$PY_TRAIN "$BIN/fl_init.py" --seed 42 --interval-months 1 \
    --aggregation uniform \
    --global-out fed_L1_global_init.pt \
    --history-out fed_L1_history_init.json \
    --best-out fed_L1_bestsofar_init.pt

IDX=0
for SITE in KTLX KENX; do
    $PY_TRAIN "$BIN/fl_train_client.py" \
        --client "${SITE}:${SITE}_sequences.npz:${SITE}_manifest.json" \
        --client-index $IDX --round 0 --seed 42 \
        --interval-months 1 --archive-start "$MONTH" --archive-months 1 \
        --limit-train-sequences 4 \
        --global-model fed_L1_global_init.pt \
        --local-model-out "fed_L1_r000_local_${SITE}.pt" \
        --meta-out "fed_L1_r000_meta_${SITE}.json"
    IDX=$((IDX + 1))
done

$PY_TRAIN "$BIN/fl_aggregate.py" --round 0 --aggregation uniform \
    --local-model fed_L1_r000_local_KTLX.pt \
    --meta fed_L1_r000_meta_KTLX.json \
    --local-model fed_L1_r000_local_KENX.pt \
    --meta fed_L1_r000_meta_KENX.json \
    --global-out fed_L1_global_r000.pt

# shellcheck disable=SC2086
$PY_TRAIN "$BIN/fl_validate.py" --round 0 $CLIENTS \
    --interval-months 1 --archive-start "$MONTH" --archive-months 1 \
    --limit-train-sequences 4 \
    --global-model fed_L1_global_r000.pt \
    --history-in fed_L1_history_init.json \
    --best-in fed_L1_bestsofar_init.pt \
    --history-out fed_L1_history_r000.json \
    --best-out fed_L1_bestsofar_r000.pt \
    --final-best fed_L1_best.ckpt

echo "== Phase D: inference + verification =="
# shellcheck disable=SC2086
$PY_EVAL "$BIN/mct_infer.py" --method steps $CLIENTS \
    --benchmark benchmark_events.csv --ensemble-size 20 \
    --output steps_forecasts.npz
# shellcheck disable=SC2086
$PY_EVAL "$BIN/mct_verify.py" --method steps --interval "" \
    --forecasts steps_forecasts.npz --benchmark benchmark_events.csv \
    --output steps_metrics.csv

for M in cen fed; do
    # shellcheck disable=SC2086
    $PY_EVAL "$BIN/mct_infer.py" --method "$M" $CLIENTS \
        --checkpoint "${M}_L1_best.ckpt" \
        --benchmark benchmark_events.csv --ensemble-size 6 \
        --output "${M}_L1_forecasts.npz"
    # shellcheck disable=SC2086
    $PY_EVAL "$BIN/mct_verify.py" --method "$M" --interval 1 \
        --forecasts "${M}_L1_forecasts.npz" \
        --benchmark benchmark_events.csv \
        --output "${M}_L1_metrics.csv"
done

echo "== Phase D: TOPSIS + figures + report =="
$PY_EVAL "$BIN/mct_topsis.py" --pool e1 \
    --metrics steps_metrics.csv --metrics cen_L1_metrics.csv \
    --metrics fed_L1_metrics.csv --output e1_topsis.csv
$PY_EVAL "$BIN/make_figures.py" --topsis e1_topsis.csv \
    --output figures.tar.gz
$PY_EVAL "$BIN/validate_report.py" --topsis e1_topsis.csv \
    --manifest KTLX_manifest.json --manifest KENX_manifest.json \
    --benchmark benchmark_events.csv --output validation_report.md

echo "== Done. Key outputs in $(pwd): =="
ls -lh e1_topsis.csv figures.tar.gz validation_report.md
