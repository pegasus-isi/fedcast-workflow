#!/usr/bin/env python3

"""Write the site-specific half of a Fed-Cast run: sites.yml.

workflow_generator.py never names a scheduler. Its jobs state cores,
memory, GPUs and runtime, and carry a Pegasus tag — "gpu" on GPU jobs,
"silo_<SITE>" / "silo_<SITE>_gpu" on jobs pinned to a client's data holder
in cross-silo runs. What a tag means on a given pool (partition, account,
GPU constraint, node pin) is the site catalog's business, and this script
writes that catalog.

Pegasus merges a local sites.yml over the hosted catalog named in
~/.pegasusrc (pegasus.catalog.site.repo.file), local entries winning key
by key, and x-tags merge the same way. So on a cluster with a hosted
catalog this script writes only what differs; on a pool without one
(--full) it writes the whole compute site.

    # Unity (hosted unity.yml already has queue=cpu and a gpu tag):
    ./custom_sites.py --style slurm --project my_lab \\
        --gpu pegasus:glite.arguments=--constraint=vram48

    # Same, cross-silo — the silo tags need the full GPU settings, so
    # inherit the hosted gpu tag from the downloaded copy:
    ./custom_sites.py --style slurm --project my_lab \\
        --base unity.yml --silos silos.yml

    # A local HTCondor pool with no hosted catalog:
    ./custom_sites.py --style condor --full \\
        --gpu 'condor:requirements=(GPUs_GlobalMemoryMb >= 20000)'

    # A local Slurm cluster with no hosted catalog:
    ./custom_sites.py --style slurm --full --queue cpu --gpu-queue gpu \\
        --project my_lab --scratch /scratch/$USER/fedcast

Run it before workflow_generator.py (the generator tells the FL-round
sub-workflows where the file is) and before pegasus-plan. Re-run whenever
the map or the pool changes; the file is overwritten.
"""

import argparse
import os
import sys

from Pegasus.api import (
    Directory, FileServer, Namespace, Operation, Site, SiteCatalog,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from silo_map import (  # noqa: E402
    STYLES, hosted_catalog, load_silo_map, pin_profile, read_site_entry,
    read_site_style, read_site_tags, silo_tags,
)

# Radar sites, for --sites validation; imported lazily so this script
# stays usable without the generator's dependencies being importable.
ALL_SITES = ["KBYX", "KTLX", "KVNX", "KLGX", "KENX", "KBOX", "PAHG"]

GPU_TAG = "gpu"

NAMESPACES = {ns.value: ns for ns in Namespace}


def parse_profile(text):
    """'ns:key=value' or 'key=value' (pegasus namespace) -> (ns, key, value)."""
    key, sep, value = text.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(
            f"{text!r}: expected NS:KEY=VALUE or KEY=VALUE")
    ns, colon, bare = key.partition(":")
    if not colon:
        ns, bare = "pegasus", key
    if ns not in NAMESPACES:
        raise argparse.ArgumentTypeError(
            f"{text!r}: unknown namespace {ns!r}; one of "
            f"{', '.join(sorted(NAMESPACES))}")
    return ns, bare, value


def check_style_against(catalog, site_name, style, parser):
    """Refuse a --style the site catalog contradicts.

    The earliest place the mistake can be caught. An overlay states no
    submission style of its own, so pins written for the wrong scheduler
    would agree with each other and read as correct everywhere
    downstream; the catalog being overlaid is what actually knows.
    """
    try:
        found = read_site_style(read_site_entry(catalog, site_name)[0])
    except ValueError as exc:
        parser.error(str(exc))
        return
    if found is None:
        return
    if found.startswith("batch:"):
        parser.error(
            f"{catalog}: site {site_name!r} submits to "
            f"{found.split(':')[1]}, which this script cannot write silo "
            f"pins for — it writes HTCondor requirements or Slurm "
            f"--nodelist only")
    if found != style:
        parser.error(
            f"--style {style} contradicts {catalog}, where site "
            f"{site_name!r} submits to {found}. Pins in the wrong dialect "
            f"are ignored by the scheduler, so every pinned job would run "
            f"anywhere. Use --style {found}.")


def hosted_tag_profiles(path, site_name, tag):
    """The x-tag profiles for `tag` on `site_name` in a catalog file.

    Returns {namespace: {key: value}}; empty if the file has no such tag.
    """
    return read_site_tags(path, site_name).get(tag, {})


def add_tag(site, tag, profiles):
    """Attach {namespace: {key: value}} to a tag on the site."""
    for ns, keys in profiles.items():
        for key, value in keys.items():
            site.add_tag_profiles(tag, NAMESPACES[ns], key=key, value=value)


def with_pin(profiles, pin):
    """GPU-tag profiles plus a node pin, combining rather than replacing.

    A tag can carry one value per key, and the pin lands on a key the GPU
    settings may already use (glite.arguments on Slurm: "-C gpu" plus
    "--nodelist=..."; condor requirements: a VRAM floor plus the machine).
    """
    ns, key, value = pin
    merged = {n: dict(k) for n, k in profiles.items()}
    existing = merged.get(ns, {}).get(key)
    if existing:
        if key == "requirements":
            value = f"({existing}) && {value}"
        else:
            value = f"{existing} {value}"
    merged.setdefault(ns, {})[key] = value
    return merged


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n\n", 1)[1])
    parser.add_argument("--style", choices=STYLES, required=True,
                        help="how the compute site is submitted to: a local "
                             "HTCondor pool, or Slurm through glite")
    parser.add_argument("--site", default="compute",
                        help="compute site name (default: compute, the "
                             "hosted catalogs' convention)")
    parser.add_argument("--full", action="store_true",
                        help="write a complete compute site rather than an "
                             "overlay, for a pool with no hosted catalog")
    parser.add_argument("--scratch", metavar="DIR",
                        help="with --full on slurm: shared scratch the "
                             "workers and submit host both see "
                             "(default: $PWD/work)")
    parser.add_argument("--storage", metavar="DIR",
                        help="with --full on slurm: the site's output "
                             "storage (default: $PWD/storage; unused when "
                             "planning with --output-dir)")
    parser.add_argument("--queue", metavar="Q",
                        help="partition/queue for CPU jobs (pegasus.queue)")
    parser.add_argument("--gpu-queue", metavar="Q",
                        help="partition/queue for the gpu tag and the silo "
                             "GPU tags")
    parser.add_argument("--project", metavar="ACCOUNT",
                        help="allocation charged (pegasus.project; "
                             "--account on Slurm)")
    parser.add_argument("--profile", action="append", default=[],
                        type=parse_profile, metavar="NS:KEY=VALUE",
                        help="extra profile on the site itself, e.g. "
                             "pegasus:glite.arguments=--constraint=avx512 or "
                             "env:PYTHONUNBUFFERED=1; repeatable")
    parser.add_argument("--gpu", action="append", default=[],
                        type=parse_profile, metavar="NS:KEY=VALUE",
                        help="extra profile on the gpu tag (and every silo "
                             "GPU tag), e.g. "
                             "pegasus:glite.arguments=--constraint=vram48 or "
                             "'condor:requirements=(GPUs_GlobalMemoryMb >= "
                             "20000)'; repeatable")
    parser.add_argument("--base", metavar="CATALOG.yml",
                        help="start the gpu settings from this catalog's gpu "
                             "x-tag (the hosted file pegasus-plan downloads "
                             "into the working directory). Needed so silo "
                             "GPU tags carry the site's full GPU settings.")
    parser.add_argument("--silos", metavar="YAML",
                        help="cross-silo map (see silos.example.yml): write "
                             "a CPU and a GPU tag per client that pin its "
                             "jobs to the machine holding its shard")
    parser.add_argument("--sites", nargs="+", default=ALL_SITES,
                        choices=ALL_SITES,
                        help="clients in the map to write tags for "
                             "(default: all 7)")
    parser.add_argument("-o", "--output", default="sites.yml",
                        help="where to write (default: sites.yml — the "
                             "name Pegasus picks up from the working "
                             "directory)")
    args = parser.parse_args()

    # A --style that contradicts the catalog being overlaid is the one
    # mistake nothing downstream can catch on its own, so check it here
    # against whatever catalog is at hand.
    reference = args.base or hosted_catalog()[1]
    if reference:
        check_style_against(reference, args.site, args.style, parser)

    site = Site(args.site)

    # -- The compute site itself --------------------------------------
    if args.full:
        if args.style == "slurm":
            scratch = os.path.abspath(args.scratch or "work")
            storage = os.path.abspath(args.storage or "storage")
            site.add_directories(
                Directory(Directory.SHARED_SCRATCH, scratch,
                          shared_file_system=False)
                .add_file_servers(FileServer("file://" + scratch,
                                             Operation.ALL)),
                Directory(Directory.LOCAL_STORAGE, storage,
                          shared_file_system=False)
                .add_file_servers(FileServer("file://" + storage,
                                             Operation.ALL)),
            )
            site.add_condor_profile(grid_resource="batch slurm")
            site.add_pegasus_profile(style="glite",
                                     data_configuration="nonsharedfs",
                                     auxillary_local="true")
            if not args.queue:
                parser.error("--full on slurm needs --queue (the "
                             "partition CPU jobs submit to)")
        else:
            site.add_condor_profile(universe="vanilla")
            site.add_pegasus_profile(style="condor")
    if args.queue:
        site.add_pegasus_profile(queue=args.queue)
    if args.project:
        site.add_pegasus_profile(project=args.project)
    for ns, key, value in args.profile:
        site.add_profiles(NAMESPACES[ns], key=key, value=value)

    # -- GPU settings, as one tag ---------------------------------------
    # Hosted catalogs define "gpu" already, and a local tag merges into it
    # key by key, so only what the user asked for is written. The silo GPU
    # tags are new names and get no such merge: they need everything.
    gpu = {}
    if args.base:
        try:
            gpu = hosted_tag_profiles(args.base, args.site, GPU_TAG)
        except ValueError as exc:
            parser.error(str(exc))
        if not gpu:
            parser.error(f"{args.base}: no gpu x-tag on site {args.site!r}")
    requested = {}
    if args.gpu_queue:
        requested.setdefault("pegasus", {})["queue"] = args.gpu_queue
    for ns, key, value in args.gpu:
        requested.setdefault(ns, {})[key] = value
    if args.full and not args.base:
        # Nothing to merge into: state the GPU request outright.
        requested.setdefault("pegasus", {}).setdefault("gpus", 1)
        requested["pegasus"].setdefault("container.arguments", "--nv")
    for ns, keys in requested.items():
        gpu.setdefault(ns, {}).update(keys)
    if requested or args.full:
        add_tag(site, GPU_TAG, requested if not args.full else gpu)

    # -- Silo pins ----------------------------------------------------
    silos = None
    if args.silos:
        try:
            silos = load_silo_map(args.silos, args.sites)
            pins = {s: pin_profile(s, silos["entries"][s], args.style,
                                   silos["attribute"])
                    for s in args.sites}
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(2)
        if not args.full and not reference:
            print("warning: this overlay states no submission style and no "
                  "hosted catalog was found here, so nothing can confirm "
                  f"the silo pins are in {args.style} dialect. Pass --base "
                  "<hosted catalog> (plan once to download it), or write "
                  "the whole site with --full.", file=sys.stderr)
        if silos["scratch_root"]:
            print(f"warning: data_dir {silos['data_dir']} is under "
                  f"{silos['scratch_root']}, which schedulers commonly make "
                  f"private per job and delete at job end — the shard would "
                  f"not survive to the next round", file=sys.stderr)
        if args.style == "slurm" and not (args.base or args.gpu_queue):
            print("warning: silo GPU tags carry only gpus/--nv — pass "
                  "--base <hosted catalog> or --gpu-queue so pinned "
                  "training jobs reach the GPU partition", file=sys.stderr)
        if not gpu.get("pegasus", {}).get("gpus"):
            gpu.setdefault("pegasus", {})["gpus"] = 1
            gpu["pegasus"].setdefault("container.arguments", "--nv")
        for s in args.sites:
            tags = silo_tags(s)
            ns, key, value = pins[s]
            site.add_tag_profiles(tags["cpu"], NAMESPACES[ns],
                                  key=key, value=value)
            add_tag(site, tags["gpu"], with_pin(gpu, pins[s]))

    sc = SiteCatalog()
    sc.add_sites(site)
    sc.write(args.output)

    print(f"wrote {args.output}: site {args.site} "
          f"({'complete' if args.full else 'overlay on the hosted catalog'}, "
          f"{args.style})")
    for tag in site.tags:
        print(f"  tag {tag}: " + "; ".join(
            f"{ns}.{k}={v}" for ns, keys in site.tags[tag].items()
            for k, v in keys.items()))
    if silos:
        print(f"  silo pins from {silos['path']}; check them with "
              f"tools/silo_check.py --style {args.style} {args.silos}")


if __name__ == "__main__":
    main()
