#!/usr/bin/env python3

"""Cross-silo placement map: parsing, tag names, and per-scheduler pins.

Shared by workflow_generator.py (which only needs the silo list, the shard
directory and the tag names), custom_sites.py (which turns each silo into
site-catalog tag profiles for the scheduler in use) and tools/silo_check.py
(the preflight).

The workflow itself never names a scheduler. A pinned job carries a Pegasus
``tag`` — ``silo_KTLX`` for CPU work at that silo, ``silo_KTLX_gpu`` for GPU
work — and the site catalog's ``x-tags`` say what that means on the pool at
hand: an HTCondor ``requirements`` expression, or a Slurm ``--nodelist``
through glite. custom_sites.py writes those entries from the same map.
"""

import os
import shlex
from collections import namedtuple
from pathlib import Path

import yaml

DEFAULT_SILO_ATTRIBUTE = "FEDCAST_SILOS"
DEFAULT_SILO_DATA_DIR = "~/.fedcast/silos"

# Apptainer mounts the job user's home from the host by default, so a
# shard kept under it needs no --bind and therefore no pre-created
# directory on the worker: the preprocess job makes its own. Anywhere else
# the bind is required, and Apptainer refuses to start when its source is
# missing.
#
# /tmp and /var/tmp are deliberately NOT here even though Apptainer mounts
# them too: HTCondor defaults MOUNT_UNDER_SCRATCH to "/tmp,/var/tmp", which
# makes both private to each job and deletes them when the job ends, and
# Slurm sites commonly do the same through job_container/tmpfs. A shard
# written there would be gone before the next round read it.
AUTO_MOUNTED_PREFIXES = ("~/",)

# Same reason — warn if someone points data_dir at one of these anyway.
SCRATCH_MOUNTED_DIRS = ("/tmp", "/var/tmp")

# Schedulers custom_sites.py knows how to pin on.
STYLES = ("condor", "slurm")


def under_scratch_mount(data_dir, roots=SCRATCH_MOUNTED_DIRS):
    """The scratch-mounted root containing data_dir, or None.

    Matches the directory itself as well as anything under it, so a bare
    "/tmp" is caught alongside "/tmp/fedcast". Trailing slashes and "." or
    ".." segments are normalized away first, since a path that only looks
    different still resolves into the same private per-job mount.
    """
    if data_dir.startswith("~"):
        return None
    target = os.path.normpath(data_dir)
    for root in roots:
        root = os.path.normpath(root)
        if target == root or target.startswith(root + os.sep):
            return root
    return None


def needs_container_bind(data_dir):
    """True if data_dir must be bind-mounted into the containers."""
    return not data_dir.startswith(AUTO_MOUNTED_PREFIXES)


def silo_tags(site):
    """The two tag names a silo's pinned jobs carry.

    A job carries exactly one Pegasus tag, and pinned GPU jobs need the
    site's GPU settings (queue, --nv, ...) as well as the pin, so each silo
    gets a CPU tag and a GPU tag. custom_sites.py defines both.
    """
    return {"cpu": f"silo_{site}", "gpu": f"silo_{site}_gpu"}


def load_silo_map(path, sites):
    """Parse a silo map (see silos.example.yml) into placement rules.

    Returns a dict with ``data_dir`` (the on-worker root holding resident
    shards), ``attribute`` (the ClassAd name used by ``{}`` entries),
    ``entries`` mapping each requested site to its validated map entry,
    and the bind/home flags the generator and preflight report on.

    Every way the file can be wrong — unreadable, unparseable, or the
    right YAML but the wrong shape — is raised as ValueError with a
    message naming the file, so callers report configuration mistakes
    rather than tracebacks.
    """
    try:
        with open(path) as f:
            doc = yaml.safe_load(f)
    except OSError as exc:
        # Missing, unreadable, or a directory. Raised as ValueError like
        # every other failure here so every entry point reports it the
        # same way; see this function's contract above.
        raise ValueError(
            f"{path}: cannot read the silo map: {exc.strerror or exc} "
            f"(start from silos.example.yml)"
        ) from exc
    except UnicodeDecodeError as exc:
        # A ValueError already, so it would surface without a file name.
        raise ValueError(
            f"{path}: not a text file ({exc.reason}) — expected YAML"
        ) from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: not valid YAML: {exc}") from exc

    if doc is None:
        doc = {}
    if not isinstance(doc, dict):
        raise ValueError(
            f"{path}: expected a mapping at the top level, found "
            f"{type(doc).__name__} — see silos.example.yml"
        )

    data_dir = doc.get("data_dir", DEFAULT_SILO_DATA_DIR)
    attribute = doc.get("attribute", DEFAULT_SILO_ATTRIBUTE)
    for key, value in (("data_dir", data_dir), ("attribute", attribute)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{path}: '{key}' must be a non-empty string, found "
                f"{value!r}"
            )

    silos = doc.get("silos")
    if silos is None:
        silos = {}
    if not isinstance(silos, dict):
        raise ValueError(
            f"{path}: 'silos' must be a mapping of site name to placement, "
            f"found {type(silos).__name__} — see silos.example.yml"
        )

    missing = [s for s in sites if s not in silos]
    if missing:
        raise ValueError(
            f"{path}: no silo defined for {', '.join(missing)} — every "
            f"client in --sites needs an entry under 'silos:'"
        )

    entries = {}
    for site in sites:
        entry = silos[site]
        if entry is None:
            entry = {}
        if not isinstance(entry, dict):
            raise ValueError(
                f"{path}: silo '{site}' must be a mapping such as "
                f"{{machine: \"host\"}} or {{}}, found "
                f"{type(entry).__name__}"
            )
        for key in ("machine", "requirements"):
            if key in entry and not isinstance(entry[key], str):
                raise ValueError(
                    f"{path}: silo '{site}' has a non-string '{key}': "
                    f"{entry[key]!r}"
                )
        entries[site] = entry

    return {"data_dir": data_dir, "attribute": attribute,
            "entries": entries, "path": os.path.abspath(path),
            "needs_bind": needs_container_bind(data_dir),
            "home_relative": data_dir.startswith("~"),
            "scratch_root": under_scratch_mount(data_dir)}


def read_site_entry(path, site_name):
    """One site's own profiles and x-tags from a site catalog.

    Returns (profiles, tags), each {namespace: {key: value}} and
    {tag: {namespace: {key: value}}}. Raises ValueError if the file is
    missing, unparseable, or has no such site — every caller treats those
    as configuration mistakes.
    """
    try:
        with open(path) as f:
            doc = yaml.safe_load(f) or {}
    except OSError as exc:
        raise ValueError(
            f"{path}: cannot read the site catalog: {exc.strerror or exc}"
        ) from exc
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"{path}: not valid YAML: {exc}") from exc
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")

    def namespaces(block):
        return {ns: dict(keys) for ns, keys in (block or {}).items()
                if isinstance(keys, dict)}

    for site in doc.get("sites") or []:
        if isinstance(site, dict) and site.get("name") == site_name:
            tags = {}
            for entry in site.get("x-tags") or []:
                if isinstance(entry, dict) and entry.get("name"):
                    tags[entry["name"]] = namespaces(entry.get("profiles"))
            return namespaces(site.get("profiles")), tags
    raise ValueError(f"{path}: no site named {site_name!r}")


def read_site_tags(path, site_name):
    """Just the x-tags of one site; see read_site_entry."""
    return read_site_entry(path, site_name)[1]


# Data configurations under which a job's files are staged through the
# compute site's own scratch rather than the submit host. Every hosted
# batch catalog uses one of them; a plain HTCondor pool uses condorio,
# where the staging site is "local".
STAGE_ON_COMPUTE = ("nonsharedfs", "sharedfs")


def stages_on_compute_site(profiles):
    """Whether a site's own profiles put its scratch in the data path.

    Matters because the planner can only build a cleanup URL for a file
    it placed on the staging site itself. The FL chain files are produced
    by deferred sub-workflows, so the parent has no PFN for them, and
    per-file cleanup refuses to plan at all ("Unable to determine cleanup
    url for lfn ... at site <compute>"). Under condorio the staging site
    is the submit host and the question never arises. See README,
    "Cleanup on a batch site".
    """
    dc = (profiles.get("pegasus") or {}).get("data.configuration")
    return str(dc) in STAGE_ON_COMPUTE


def site_stages_on_compute(site_name, sites_yml=None, base_catalog=None):
    """Resolve stages_on_compute_site over an overlay and what it overlays.

    Returns True/False, or None when neither file names a data
    configuration — the caller should say so rather than assume.
    """
    for path in (sites_yml, base_catalog):
        if not path or not os.path.isfile(path):
            continue
        try:
            profiles, _ = read_site_entry(path, site_name)
        except ValueError:
            continue
        if (profiles.get("pegasus") or {}).get("data.configuration"):
            return stages_on_compute_site(profiles)
    return None


def site_submission_style(site_name, sites_yml=None, base_catalog=None):
    """The scheduler a site submits to, over an overlay and its base.

    Same resolution order as site_stages_on_compute: the local overlay
    first, then the catalog it overlays. None when neither states a
    style, which is the normal first-run case for an overlay whose
    hosted catalog has not been downloaded yet.
    """
    for path in (sites_yml, base_catalog):
        if not path or not os.path.isfile(path):
            continue
        try:
            profiles, _ = read_site_entry(path, site_name)
        except ValueError:
            continue
        style = read_site_style(profiles)
        if style:
            return style
    return None


def split_nodelist(spec):
    """Split a Slurm node spec on commas outside [] brackets.

    "a,b" is two nodes; "node[1,3]" is one spec standing for two, and the
    comma inside the brackets must not split it.
    """
    parts, depth, cur = [], 0, ""
    for ch in spec:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p for p in (x.strip() for x in parts) if p]


def nodelist_nodes(arguments):
    """The nodes named by --nodelist / -w in a glite arguments string.

    Returns a list of node specs in the order given, so a caller can tell
    "exactly the one node I asked for" from "some other node", "that node
    and another", or "no pin at all". A spec containing "[" is returned as
    it stands: a bracket range can name several nodes, so it is never a
    pin to the one machine holding a shard.
    """
    try:
        tokens = shlex.split(arguments)
    except ValueError:
        # Unbalanced quotes in a hand-edited catalog: fall back to a plain
        # split rather than reporting no pin at all.
        tokens = arguments.split()

    nodes, expect_value = [], False
    for token in tokens:
        if expect_value:
            nodes.extend(split_nodelist(token))
            expect_value = False
            continue
        if token in ("--nodelist", "-w"):
            expect_value = True
        elif token.startswith("--nodelist="):
            nodes.extend(split_nodelist(token[len("--nodelist="):]))
        elif token.startswith("-w") and len(token) > 2:
            nodes.extend(split_nodelist(token[2:]))
    return nodes


def pin_problem(site, entry, style, attribute, value):
    """Why `value` fails to pin this silo's jobs, or None if it does.

    `value` is what the site catalog currently has for the key
    pin_profile() names. Checked by parsing, not by substring: on Slurm
    "--nodelist=node7" is a substring of "--nodelist=node70", so a tag
    left pointing at a different node would otherwise pass, and a list or
    a bracket range would pass while letting the job land on a node that
    holds no shard. A silo must resolve to exactly one machine.

    HTCondor pins are compared as substrings, which is sound there
    because the generated clause is delimited — `(Machine == "w1")` is
    not a substring of `(Machine == "w10")` — and a raw `requirements:`
    expression from the map is the user's own text.
    """
    ns, key, pin = pin_profile(site, entry, style, attribute)
    if style == "slurm":
        machine = entry["machine"]
        nodes = nodelist_nodes(value)
        if not nodes:
            return (f"{ns}.{key}={value!r} names no node "
                    f"(expected {pin!r})")
        if nodes != [machine]:
            found = ", ".join(nodes)
            if len(nodes) > 1:
                return (f"{ns}.{key} pins to {len(nodes)} nodes "
                        f"({found}), not to {machine} alone — the shard "
                        f"is written to one node and never replicated")
            if "[" in nodes[0]:
                return (f"{ns}.{key} pins to the range {found}, which can "
                        f"match several nodes, not to {machine} alone")
            return (f"{ns}.{key} pins to {found}, not to {machine}")
        return None
    if pin not in value:
        return (f"{ns}.{key}={value!r} does not contain {pin!r}")
    return None


# What check_silo_tags found: the problems, a note naming the scheduler it
# checked against and how that was determined, whether that determination
# came from a site catalog (unverified means nothing authoritative said,
# so a catalog written for the wrong scheduler could not be told apart),
# and the scheduler itself.
SiloTagCheck = namedtuple("SiloTagCheck",
                          "problems note verified style")

# Where a pin lives, per scheduler: (namespace, key, style). Used both to
# write a pin (pin_profile) and to recognise one in a catalog.
PIN_KEYS = (
    ("condor", "requirements", "condor"),
    ("pegasus", "glite.arguments", "slurm"),
)

# Pegasus submission styles that mean "a local HTCondor pool". Anything
# glite-shaped is a batch system, and which one comes from grid_resource.
CONDOR_STYLES = ("condor", "condorc")


def read_site_style(profiles):
    """The scheduler a site entry submits to, from its own profiles.

    Returns "condor", "slurm", "batch:<lrms>" for a batch system --silos
    cannot pin on, or None when the entry does not say — which is the
    normal case for an overlay, since it inherits ``style`` from the
    hosted catalog.
    """
    grid = str((profiles.get("condor") or {}).get("grid_resource", "")).split()
    if len(grid) >= 2 and grid[0] == "batch":
        return "slurm" if grid[1] == "slurm" else f"batch:{grid[1]}"
    style = (profiles.get("pegasus") or {}).get("style")
    if style in CONDOR_STYLES:
        return "condor"
    if style == "glite":
        # glite without a grid_resource: Slurm is the only batch system
        # this workflow's pins are written for, so assume it and let the
        # pin check confirm the tags actually use --nodelist.
        return "slurm"
    return None


def hosted_catalog(pegasusrc=None, cwd=None):
    """The hosted site catalog in use: (name, local copy path or None).

    The planner downloads the file named by pegasus.catalog.site.repo.file
    into the directory it runs in, so once a workflow has been planned
    there is a local copy to read the site's real submission style from.
    Returns (None, None) when no hosted catalog is configured.
    """
    rc = Path(pegasusrc) if pegasusrc else Path.home() / ".pegasusrc"
    name = None
    try:
        for line in rc.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == "pegasus.catalog.site.repo.file":
                name = value.strip()
    except OSError:
        return None, None
    if not name:
        return None, None
    path = Path(cwd or ".") / name
    return name, str(path) if path.is_file() else None


def tag_pin_styles(profiles):
    """The styles a tag's profiles carry a pin key for; [] if none do.

    Read off the keys rather than the site's ``style`` profile, because an
    overlay inherits its style from the hosted catalog and so often does
    not state one. A tag carrying no pin key at all is the case that
    matters most: it looks like a silo tag, and pins nothing.
    """
    return [style for ns, key, style in PIN_KEYS
            if key in (profiles.get(ns) or {})]


def check_silo_tags(sites_yml, site_name, sites, silos, style=None,
                    base_catalog=None):
    """Problems with the silo tags in a site catalog; empty list if none.

    Every pinned job carries a tag and nothing else, so each silo tag has
    to be present AND actually pin, in the way this site's scheduler
    honours. Reported failures:

    * the tag is absent;
    * it exists but carries no pin key, so the job runs wherever the
      scheduler likes;
    * its pin names something other than the single mapped machine;
    * it pins in the other scheduler's dialect, which this site ignores.
      A ``requirements`` ClassAd does not place a job on a Slurm node and
      ``glite.arguments`` does nothing in a vanilla HTCondor pool, so a
      catalog mixing the two has at least one silently unpinned tag.

    Which scheduler to check against is the site's own, read from the
    local catalog if it states a style and otherwise from `base_catalog`,
    the local copy of the hosted catalog it overlays. That beats the
    caller's `style`, and a contradiction is reported: an overlay usually
    states no style, so pins written for the wrong scheduler agree with
    each other and pass every other check here. With neither available
    the tags decide and must agree, which the returned note calls
    unverified.

    Returns a SiloTagCheck. `verified` is False when no site catalog
    stated the scheduler, which callers must not treat as a pass: it is
    the one case where a catalog written wholly in the wrong dialect
    cannot be distinguished from a correct one.
    """
    try:
        profiles, tags = read_site_entry(sites_yml, site_name)
    except ValueError as exc:
        return SiloTagCheck([f"  {exc}"],
                            f"{sites_yml} could not be read", False, None)

    problems = []

    # The site's real submission style, in order of authority: the local
    # overlay if it states one, else the hosted catalog it overlays. An
    # overlay usually states none — which is exactly why the hosted copy
    # has to be consulted: without it a catalog written for the wrong
    # scheduler agrees with itself and looks correct.
    authoritative, source = read_site_style(profiles), sites_yml
    if authoritative is None and base_catalog:
        try:
            authoritative = read_site_style(
                read_site_entry(base_catalog, site_name)[0])
            source = base_catalog
        except ValueError as exc:
            problems.append(f"  {exc}")
    if authoritative and authoritative.startswith("batch:"):
        problems.append(
            f"  site {site_name!r} submits to {authoritative.split(':')[1]} "
            f"(per {source}), which --silos cannot pin on: it writes "
            f"HTCondor requirements or Slurm --nodelist only")
        authoritative = None

    wanted = {tag: kind for site in sites
              for kind, tag in silo_tags(site).items()}
    tag_styles = {tag: tag_pin_styles(p) for tag, p in tags.items()
                  if tag in wanted}
    seen = {st for sts in tag_styles.values() for st in sts}

    if style and authoritative and style != authoritative:
        problems.append(
            f"  --style {style} contradicts {source}, where site "
            f"{site_name!r} submits to {authoritative} — the catalog "
            f"decides, and {style} pins would be ignored")
    effective = authoritative or style
    if effective is None:
        if len(seen) > 1:
            named = "; ".join(
                f"{st}: " + ", ".join(sorted(
                    t for t, sts in tag_styles.items() if st in sts))
                for st in sorted(seen))
            problems.append(
                f"  silo tags pin in {len(seen)} different schedulers' "
                f"dialects ({named}) — a site submits to one, so the "
                f"others' pins are ignored and those jobs run anywhere. "
                f"Re-run custom_sites.py with a single --style")
        elif seen:
            effective = next(iter(seen))

    verified = authoritative is not None
    if verified:
        note = f"{effective} (per {source})"
    elif style:
        note = (f"{style} (from --style; UNVERIFIED — {sites_yml} states "
                f"no submission style and no hosted catalog was read, so "
                f"nothing here can tell a correct dialect from a wrong "
                f"one)")
    elif effective:
        note = (f"{effective} (assumed from the tags; UNVERIFIED — "
                f"neither {sites_yml} nor a hosted catalog states a "
                f"submission style, and tags written wholly in the wrong "
                f"dialect agree with each other)")
    else:
        note = "UNVERIFIED — undetermined"

    for site in sites:
        entry = silos["entries"][site]
        for kind, tag in silo_tags(site).items():
            if tag not in tags:
                problems.append(f"  {site:5s} {kind}: tag {tag} missing")
                continue
            carried = tag_styles.get(tag, [])
            if not carried:
                keys = ", ".join(f"{ns}.{key}" for ns, key, _ in PIN_KEYS)
                problems.append(
                    f"  {site:5s} {kind}: tag {tag} carries no pin "
                    f"({keys} all absent), so the job would run wherever "
                    f"the scheduler likes — re-run custom_sites.py")
                continue
            for stray in (st for st in carried if effective
                          and st != effective):
                ns, key, _ = next((n, k, s) for n, k, s in PIN_KEYS
                                  if s == stray)
                problems.append(
                    f"  {site:5s} {kind}: tag {tag} pins with {ns}.{key}, "
                    f"which a {effective} site ignores — that pin has no "
                    f"effect here")
            for st in ([effective] if effective else carried):
                try:
                    ns, key, _ = pin_profile(site, entry, st,
                                             silos["attribute"])
                except ValueError as exc:
                    # A map entry this scheduler cannot express. Reported
                    # per tag rather than raised: the caller asked what is
                    # wrong with the catalog, and this is one answer.
                    problems.append(f"  {site:5s} {kind}: {exc}")
                    continue
                problem = pin_problem(
                    site, entry, st, silos["attribute"],
                    str((tags[tag].get(ns) or {}).get(key, "")))
                if problem:
                    problems.append(f"  {site:5s} {kind}: tag {tag} "
                                    f"{problem} — re-run custom_sites.py")
    return SiloTagCheck(problems, note, verified, effective)


def condor_requirements(site, entry, attribute):
    """HTCondor requirements expression pinning a silo's jobs."""
    if entry.get("requirements"):
        return entry["requirements"]
    if entry.get("machine"):
        return f'(Machine == "{entry["machine"]}")'
    # Worker advertises its hosted silos as a comma-separated string in
    # `attribute` (tools/silo_worker_setup.sh). stringListMember matches
    # whole entries, so silo "KTLX" never matches a worker hosting only
    # "KTLXX", and the =?= keeps the expression False (not undefined) on
    # workers that do not advertise the attribute at all.
    return f'(stringListMember("{site}", {attribute}) =?= True)'


def pin_profile(site, entry, style, attribute=DEFAULT_SILO_ATTRIBUTE):
    """The (namespace, key, value) profile that pins a silo's jobs.

    ``style`` is the Pegasus submission style of the compute site:
    ``condor`` pins through a ClassAd requirements expression; ``slurm``
    pins through ``--nodelist`` in glite.arguments. Slurm has no
    equivalent of a raw requirements expression or an advertised
    attribute, so a map that uses either is rejected for it.
    """
    if style == "condor":
        return ("condor", "requirements",
                condor_requirements(site, entry, attribute))
    if style == "slurm":
        if entry.get("requirements"):
            raise ValueError(
                f"silo '{site}' pins with a ClassAd 'requirements' "
                f"expression, which has no Slurm equivalent — use "
                f"{{machine: \"<nodename>\"}}")
        if not entry.get("machine"):
            raise ValueError(
                f"silo '{site}' pins by advertised ClassAd ({{}}), which "
                f"has no Slurm equivalent — use {{machine: \"<nodename>\"}}")
        return ("pegasus", "glite.arguments",
                f"--nodelist={entry['machine']}")
    raise ValueError(f"unknown site style {style!r}; expected one of "
                     f"{', '.join(STYLES)}")
