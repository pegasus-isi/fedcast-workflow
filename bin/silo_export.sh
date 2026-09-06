#!/bin/sh
# Copy a resident client shard out of its silo into Pegasus staging.
#
#   silo_export.sh <resident-shard-path> <output-lfn>
#
# This is the explicit data-egress boundary of a cross-silo run: the
# centralized baseline and MCT evaluation need every client's shard
# pooled, which is exactly what the paper compares federation against.
# The federated arm never runs this.
#
# A shell script rather than /bin/cp so a home-relative shard path
# (~/.fedcast/silos/...) is expanded — Pegasus does not expand job
# arguments — and so a missing shard reports what actually went wrong.
set -eu

if [ $# -ne 2 ]; then
    echo "usage: $0 <resident-shard-path> <output-lfn>" >&2
    exit 2
fi

# Expand a leading ~ against the job user's real home, which is the host
# home Apptainer mounts. Done with parameter substitution, never eval: the
# path comes from a config file, and eval would run anything in it.
case $1 in
    "~/"*) SRC="$HOME/${1#"~/"}" ;;
    "~")   SRC="$HOME" ;;
    *)     SRC="$1" ;;
esac
DST=$2

if [ ! -f "$SRC" ]; then
    echo "silo_export: shard not found at $SRC" >&2
    echo "  host=$(hostname) user=$(id -un) HOME=$HOME" >&2
    echo "  This client's preprocess job should have written it here." >&2
    echo "  Likely causes: this job and the preprocess job did not run on" >&2
    echo "  the same worker (check tools/silo_check.py <silo map>), or the" >&2
    echo "  pool gives each slot its own user, so a home-relative shard" >&2
    echo "  directory resolved differently for the two jobs. In the second" >&2
    echo "  case switch data_dir to an absolute path and run" >&2
    echo "  tools/silo_worker_setup.sh on the workers." >&2
    exit 1
fi

cp "$SRC" "$DST"
echo "silo_export: $SRC -> $DST ($(wc -c < "$DST") bytes) on $(hostname)"
