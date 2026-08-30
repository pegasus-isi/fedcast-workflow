#!/usr/bin/env python3

"""Tiered validation report against SPEC.md Sec. 5 criteria.

Reads the TOPSIS pool CSVs, site manifests, and the benchmark table, and
evaluates the reproduction gates:

  Tier 1 — data fidelity: retained-sequence counts vs. the paper's values
           (checked only for the full 7-site / 48-month configuration).
  Tier 2 — primary result: R1 (federated > centralized for L in 1..24),
           the L=1 gap >= 0.05, R2 (|gap| <= 0.02 at L=48), R4 (STEPS
           ranks first in the E1 pool).
  Tier 3/4 gates that need raw fields or ablation pools are reported as
           SKIPPED when their inputs are absent.

Tier 0 (manifest hash determinism) is checked ACROSS runs, so this job
only records the hashes for later comparison.

Output: validation_report.md. Exit code is 0 even on gate failures — the
report is the deliverable; failing gates are findings, not job errors.
"""

import argparse
import csv
import json
import logging
from collections import defaultdict

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Paper's retained-sequence counts at 48 months (Sec. III-B).
PAPER_COUNTS = {"KBYX": 542, "KTLX": 478, "KVNX": 489, "KLGX": 885,
                "KENX": 839, "KBOX": 715, "PAHG": 831}
COUNT_TOLERANCE = 0.15  # +/-15% (SPEC Tier 1)

SHORT_WINDOW = {1, 3, 6, 12, 24}  # R1 intervals
L1_MIN_GAP = 0.05                 # SPEC Tier 2
L48_MAX_GAP = 0.02                # SPEC Tier 2


def load_pool(path):
    groups = defaultdict(list)
    pool = None
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            pool = row["pool"]
            interval = int(row["interval"]) if row["interval"] else None
            groups[(row["method"], interval)].append(float(row["topsis"]))
    return pool, groups


def median(groups, method, interval):
    scores = groups.get((method, interval))
    return float(np.median(scores)) if scores else None


def main():
    parser = argparse.ArgumentParser(
        description="Build the tiered validation report (SPEC Sec. 5)")
    parser.add_argument("--topsis", action="append", required=True)
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    lines = ["# Fed-Cast Reproduction — Validation Report", "",
             "Gates from SPEC.md Sec. 5. PASS/FAIL/SKIP per criterion.", ""]

    # ---------------- Tier 0/1: manifests -----------------------------------
    lines.append("## Tier 0/1 — Data fidelity")
    lines.append("")
    manifests = {}
    for path in args.manifest:
        try:
            with open(path) as f:
                m = json.load(f)
            manifests[m.get("site", path)] = m
        except Exception as exc:  # noqa: BLE001
            lines.append(f"- ERROR reading manifest `{path}`: {exc}")

    lines.append("| Site | Retained | Paper (48 mo) | Within +/-15% | "
                 "SHA-256 |")
    lines.append("|---|---|---|---|---|")
    full_config = set(manifests) == set(PAPER_COUNTS)
    for site, m in sorted(manifests.items()):
        retained = m.get("retained", 0)
        paper = PAPER_COUNTS.get(site)
        if paper and full_config:
            ok = abs(retained - paper) <= COUNT_TOLERANCE * paper
            verdict = "PASS" if ok else "FAIL"
        else:
            verdict = "SKIP (not full 7x48 config)"
        digest = str(m.get("sequence_sha256", ""))[:12]
        lines.append(f"| {site} | {retained} | {paper or '-'} | {verdict} "
                     f"| `{digest}` |")
    lines.append("")
    lines.append("Tier 0 note: sequence SHA-256 values above must be "
                 "byte-identical across re-runs of Phase A (compare "
                 "reports between runs).")
    lines.append("")

    # ---------------- Tier 2: primary result (E1) ---------------------------
    lines.append("## Tier 2 — Primary result (E1 pool)")
    lines.append("")
    e1 = None
    pools = {}
    for path in args.topsis:
        pool, groups = load_pool(path)
        if pool:
            pools[pool] = groups
    e1 = pools.get("e1")

    if e1 is None:
        lines.append("- SKIP: no E1 pool found.")
    else:
        intervals = sorted({L for (m, L) in e1 if L is not None})
        # R1: fed > cen for all short-window intervals present.
        r1_checks = []
        for L in intervals:
            if L not in SHORT_WINDOW:
                continue
            fed, cen = median(e1, "fed", L), median(e1, "cen", L)
            if fed is None or cen is None:
                r1_checks.append((L, None, None, "SKIP"))
            else:
                r1_checks.append((L, fed, cen,
                                  "PASS" if fed > cen else "FAIL"))
        lines.append("| Gate | Detail | Verdict |")
        lines.append("|---|---|---|")
        for L, fed, cen, verdict in r1_checks:
            detail = (f"L={L}: fed {fed:.4f} vs cen {cen:.4f}"
                      if fed is not None else f"L={L}: missing data")
            lines.append(f"| R1 fed > cen | {detail} | {verdict} |")

        fed1, cen1 = median(e1, "fed", 1), median(e1, "cen", 1)
        if fed1 is not None and cen1 is not None:
            gap = fed1 - cen1
            verdict = "PASS" if gap >= L1_MIN_GAP else "FAIL"
            lines.append(f"| L=1 gap >= {L1_MIN_GAP} | gap {gap:.4f} "
                         f"(paper: 0.16) | {verdict} |")
        else:
            lines.append(f"| L=1 gap >= {L1_MIN_GAP} | missing | SKIP |")

        fed48, cen48 = median(e1, "fed", 48), median(e1, "cen", 48)
        if fed48 is not None and cen48 is not None:
            gap = abs(cen48 - fed48)
            verdict = "PASS" if gap <= L48_MAX_GAP else "FAIL"
            lines.append(f"| R2 L=48 gap <= {L48_MAX_GAP} | gap {gap:.4f} "
                         f"(paper: 0.0048) | {verdict} |")
        else:
            lines.append(f"| R2 L=48 gap <= {L48_MAX_GAP} | L=48 not run "
                         "| SKIP |")

        # R4: STEPS ranks first.
        steps_med = median(e1, "steps", None)
        if steps_med is not None:
            dgmr_meds = [median(e1, m, L) for (m, L) in e1
                         if L is not None]
            dgmr_meds = [x for x in dgmr_meds if x is not None]
            verdict = ("PASS" if dgmr_meds and
                       steps_med >= max(dgmr_meds) else "FAIL")
            lines.append(f"| R4 STEPS first | steps {steps_med:.4f} vs "
                         f"max DGMR {max(dgmr_meds):.4f} | {verdict} |")
        else:
            lines.append("| R4 STEPS first | no STEPS scores | SKIP |")
    lines.append("")

    # ---------------- Tier 3: ablations --------------------------------------
    lines.append("## Tier 3 — Ablations")
    lines.append("")
    e21 = pools.get("e21")
    if e21 is None:
        lines.append("- R5 (E2.1 quadratic weighting): SKIP — pool not run.")
    else:
        intervals = sorted({L for (m, L) in e21 if L is not None})
        fails = []
        for L in intervals:
            if L not in SHORT_WINDOW:
                continue
            fedq, cen = median(e21, "fedq", L), median(e21, "cen", L)
            if fedq is not None and cen is not None and fedq <= cen:
                fails.append(L)
        verdict = "PASS" if not fails else f"FAIL at L={fails}"
        lines.append(f"- R5 (E2.1 ordering persists): {verdict}")
    if pools.get("e22") is None:
        lines.append("- R6 (E2.2 SAM): SKIP — pool not run "
                     "(SAM training TODO in train_dgmr.py).")
    else:
        lines.append("- R6 (E2.2 SAM): pool present — inspect figures "
                     "(exploratory; no hard gate).")
    lines.append("")

    # ---------------- Benchmark provenance -----------------------------------
    with open(args.benchmark, newline="") as f:
        n_events = sum(1 for _ in csv.DictReader(f))
    lines.append("## Benchmark")
    lines.append("")
    lines.append(f"- Frozen benchmark: {n_events} events "
                 f"(`{args.benchmark}`).")
    lines.append("")
    lines.append("---")
    lines.append("Generated by validate_report.py — gates per SPEC.md "
                 "Sec. 5; small-n caveats apply (no significance claims).")

    with open(args.output, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Validation report -> %s", args.output)


if __name__ == "__main__":
    main()
