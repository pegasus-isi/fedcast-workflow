#!/usr/bin/env python3

"""Preflight for --silos, run on the submit host before pegasus-plan.

Checks, which otherwise surface hours later as an idle job or a container
that will not start:

1. Every silo resolves to exactly one machine. For each client, runs the
   exact HTCondor requirements expression workflow_generator.py will
   attach to that client's jobs and lists the machines it matches. More
   than one is a misconfiguration, not a bonus: the shard is written by
   one preprocess job to one machine and is never replicated, so a
   training job that later matches a different machine finds no shard.
   Pass --allow-multi-worker-silo if you replicate the shard directory
   yourself.

2. Every worker can satisfy the bind — only when the map needs one. A
   home-relative shard directory is inside Apptainer's default mounts, so
   there is no bind and nothing to prepare. An absolute path is
   bind-mounted pool-wide, and Apptainer refuses to start when the source
   is missing, so every worker needs the directory even if it holds no
   data. Workers advertise it as FEDCAST_SILO_DIR.

3. The shard directory survives the job. A home-relative directory assumes
   every job on a worker runs as the same user; a pool with per-slot users
   would resolve it differently for the preprocess and training jobs. And
   HTCondor's MOUNT_UNDER_SCRATCH (default /tmp,/var/tmp) makes those
   directories private per job and deletes them when the job ends, so a
   shard written under one would be gone by the next round. Both are
   checked per pinned machine, advisory only, since a pool may refuse
   remote config queries.

    tools/silo_check.py silos.yml [--sites KTLX KENX ...]

Exit status, so that `silo_check.py && pegasus-plan ...` is safe by
default and automation can tell the cases apart:

    0  every matched worker was checked and is fine
    1  a real problem — a silo matches no worker, a worker cannot satisfy
       the bind, or a shard directory would not survive the round
    2  the tool could not run: bad arguments, condor_status missing, or a
       silo map that is unreadable, unparseable, the wrong shape, or
       missing one of the requested clients
    3  placement is fine but durability could not be verified on at least
       one worker, because its configuration could not be read. Not
       evidence of a problem and not evidence of safety. Pass
       --allow-unverified to accept this and exit 0, on a pool that
       deliberately refuses remote config queries.
"""

EXIT_OK = 0
EXIT_PROBLEM = 1
EXIT_CANNOT_RUN = 2
EXIT_UNVERIFIED = 3

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))

from workflow_generator import (  # noqa: E402
    SITES, load_silo_map, under_scratch_mount,
)

SILO_DIR_ATTR = "FEDCAST_SILO_DIR"


def condor_status(args):
    try:
        out = subprocess.run(["condor_status", *args],
                             capture_output=True, text=True, check=True)
    except FileNotFoundError:
        print("condor_status not found — run this on the submit host",
              file=sys.stderr)
        sys.exit(EXIT_CANNOT_RUN)
    except subprocess.CalledProcessError as exc:
        print(f"condor_status failed: {exc.stderr.strip() or exc}",
              file=sys.stderr)
        sys.exit(EXIT_CANNOT_RUN)
    return [line.split(None, len(args)) for line in out.stdout.splitlines()
            if line.strip()]


def check_silos(silos, sites):
    """Report which machines each silo pins to.

    Counted by machine name, not by ad: condor_status returns one ad per
    slot, and a single worker's slots can advertise different TotalGpus
    (a partitionable slot and the dynamic slots carved from it, or an ad
    where the attribute is undefined). Counting distinct (Machine, GPUs)
    pairs would call one worker several machines and reject a perfectly
    good map.

    Returns (unresolved sites, ambiguous sites, matched machine names).
    """
    missing, ambiguous = [], []
    matched = set()
    for site in sites:
        expr = silos["requirements"][site]
        rows = condor_status(["-constraint", expr,
                              "-af", "Machine", "TotalGpus"])

        machines = {}
        for row in rows:
            gpus = row[1] if len(row) > 1 else "?"
            machines.setdefault(row[0], set()).add(gpus)

        def describe(machine):
            seen = sorted(v for v in machines[machine]
                          if v not in ("?", "", "undefined"))
            return f"{machine} (GPUs={','.join(seen) if seen else '?'})"

        if len(machines) > 1:
            print(f"  {site:5s} -> {len(machines)} MACHINES: " + ", ".join(
                describe(m) for m in sorted(machines)))
            ambiguous.append(site)
        elif machines:
            machine = next(iter(machines))
            slots = len(rows)
            suffix = f", {slots} slot ads" if slots > 1 else ""
            print(f"  {site:5s} -> {describe(machine)}{suffix}")
        else:
            print(f"  {site:5s} -> NO MATCH   [{expr}]")
            missing.append(site)
        matched.update(machines)
    return missing, ambiguous, matched


def check_bind_dir(expected):
    """Report machines that cannot satisfy the container bind."""
    rows = condor_status(["-af", "Machine", SILO_DIR_ATTR])
    seen = {}
    for r in rows:
        seen.setdefault(r[0], r[1] if len(r) > 1 else "undefined")
    bad = {m: v for m, v in seen.items() if v != expected}
    for machine, value in sorted(bad.items()):
        why = ("does not advertise it" if value in ("undefined", "")
               else f"advertises {value}")
        print(f"  {machine}: {why}")
    return bad, len(seen)


def config_val(machine, macro):
    """One config macro from a worker, or None if it cannot be queried."""
    try:
        out = subprocess.run(
            ["condor_config_val", "-name", machine, "-startd", macro],
            capture_output=True, text=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    value = out.stdout.strip()
    return value or None


def check_durability(silos, machines):
    """Check that a shard directory survives from one round to the next.

    Returns (problems, unverified). A worker lands in `unverified` when its
    configuration cannot be read at all — that is not evidence of safety,
    so the caller must never report such a run as checked.
    """
    data_dir = silos["data_dir"]
    problems, unverified = [], []

    for machine in sorted(machines):
        # UID_DOMAIN is always defined, so it distinguishes "no config
        # access" from "macro genuinely not set" for the two below.
        if config_val(machine, "UID_DOMAIN") is None:
            unverified.append(machine)
            continue

        scratch = config_val(machine, "MOUNT_UNDER_SCRATCH")
        if scratch:
            roots = [d.strip() for d in scratch.split(",") if d.strip()]
            hit = under_scratch_mount(data_dir, roots)
            if hit:
                problems.append(
                    f"  {machine}: MOUNT_UNDER_SCRATCH={scratch} makes "
                    f"{hit} private per job and deletes it when the job "
                    f"ends — the shard would not survive the round")

        slot_user = config_val(machine, "SLOT_USER")
        if slot_user and silos["home_relative"]:
            problems.append(
                f"  {machine}: SLOT_USER={slot_user} means slots run as "
                f"different users, so a home-relative shard directory can "
                f"resolve differently for two jobs on this worker — use an "
                f"absolute data_dir with tools/silo_worker_setup.sh")

    if problems:
        print("\n".join(problems))
    checked = len(machines) - len(unverified)
    if unverified:
        print(f"  NOT VERIFIED on {len(unverified)} of {len(machines)} "
              f"worker(s) — config could not be read: "
              f"{', '.join(unverified)}")
    if not problems and checked:
        print(f"  no problems on the {checked} worker(s) that could be "
              f"checked")
    return problems, unverified


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("silos", help="silo map YAML (see silos.example.yml)")
    parser.add_argument("--sites", nargs="+", default=list(SITES.keys()),
                        choices=list(SITES.keys()),
                        help="clients to check (default: all 7)")
    parser.add_argument("--allow-multi-worker-silo", action="store_true",
                        help="accept a silo that matches several machines. "
                             "Only correct if you replicate the shard "
                             "directory across them yourself — nothing in "
                             "the workflow does.")
    parser.add_argument("--allow-unverified", action="store_true",
                        help="exit 0 instead of 3 when a worker's config "
                             "cannot be read, for pools that refuse remote "
                             "config queries. Placement and bind checks "
                             "still have to pass.")
    args = parser.parse_args()

    try:
        silos = load_silo_map(args.silos, args.sites)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(EXIT_CANNOT_RUN)

    data_dir = silos["data_dir"]
    print(f"silo map    : {silos['path']}")
    print(f"shard dir   : {data_dir}")
    if silos["needs_bind"]:
        print("              bind-mounted into the containers pool-wide")
    else:
        print("              inside the job user's home, which Apptainer "
              "mounts — no bind, no worker setup")
    print()

    print("client placement:")
    unresolved, ambiguous, matched = check_silos(silos, args.sites)
    if ambiguous and args.allow_multi_worker_silo:
        print(f"  ({len(ambiguous)} multi-machine silo(s) accepted via "
              f"--allow-multi-worker-silo)")
        ambiguous = []

    print("\nshard durability on the matched workers:")
    durability, unverified = check_durability(silos, matched)

    bad = {}
    if silos["needs_bind"]:
        print(f"\nworkers missing {data_dir} (container bind would fail):")
        bad, total = check_bind_dir(data_dir)
        if not bad:
            print(f"  none — all {total} machine(s) advertise it")

    if unresolved or ambiguous or bad or durability:
        print()
        if unresolved:
            print(f"{len(unresolved)} silo(s) match no worker: "
                  f"{' '.join(unresolved)}. Run tools/silo_worker_setup.sh "
                  f"on the intended worker(s), or fix {args.silos}.",
                  file=sys.stderr)
        if ambiguous:
            print(f"{len(ambiguous)} silo(s) match more than one machine: "
                  f"{' '.join(ambiguous)}. The shard is written to one "
                  f"machine and never replicated, so a later job matching "
                  f"another would find nothing. Narrow the map (pin by "
                  f"machine name), or pass --allow-multi-worker-silo if you "
                  f"replicate the shard directory yourself.",
                  file=sys.stderr)
        if bad:
            print(f"{len(bad)} worker(s) cannot satisfy the bind. Run "
                  f"'sudo tools/silo_worker_setup.sh none {data_dir}' on "
                  f"each — every worker needs the directory, not just the "
                  f"data holders.", file=sys.stderr)
        if durability:
            print(f"{len(durability)} durability problem(s): the shard "
                  f"would not survive from one round to the next on those "
                  f"workers. Fix data_dir before submitting.",
                  file=sys.stderr)
        sys.exit(EXIT_PROBLEM)

    if unverified:
        print("\nplacement resolves and every worker that could be checked "
              "is fine, but durability is UNVERIFIED on "
              f"{len(unverified)} worker(s) — read their MOUNT_UNDER_SCRATCH "
              "and SLOT_USER by hand before a long run")
        if args.allow_unverified:
            print("accepted via --allow-unverified")
            sys.exit(EXIT_OK)
        print("exiting 3 so this is not mistaken for a clean check; pass "
              "--allow-unverified to accept it")
        sys.exit(EXIT_UNVERIFIED)

    print("\nready: shards will stay on the listed workers")
    sys.exit(EXIT_OK)


if __name__ == "__main__":
    main()
