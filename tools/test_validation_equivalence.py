#!/usr/bin/env python3

"""Exercise the two validation paths through the wrappers themselves.

    tools/test_validation_equivalence.py     # 0 = pass

Checks two things that have to hold for cross-silo mode to be usable:

1. **All three wrapper entry points run**, each called with real argv
   against a real shard on disk: `fl_validate_client.main()` per client,
   then `fl_validate.main()` twice — once on its `--client` branch, which
   scores every split centrally, and once on its `--client-metrics`
   branch, which recombines what the silos returned. A previous bug passed
   a single dataset dict where a list of them was expected, crashing for
   every client that had validation data, and was invisible to a test that
   called `generator_val_batch_losses` directly with a correctly shaped
   argument. Testing the helper is not testing the call site, and the
   recombination branch is a call site of its own.
2. **The two paths agree**, compared where it actually matters: the
   validation loss each one records in the history file it writes. That
   is the number the checkpoint rule reads, so agreement there is what
   lets either mode select the same checkpoint.

DGMR itself is replaced by a stub with the same interface: this is about
the plumbing and the arithmetic, not the model. Runs on CPU in seconds and
needs only torch and numpy.
"""

import json
import os
import pathlib
import sys
import tempfile

os.environ.setdefault("FEDCAST_MODEL_SIZE", "32")
os.environ.setdefault("FEDCAST_BATCH_SIZE", "2")

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bin"))

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402

import fedcast_common as fc                                  # noqa: E402

SITES = ["KTLX", "KENX", "KBYX"]
SEED = 42
START_MONTH, ARCHIVE_MONTHS, INTERVAL = "2024-01", 1, 1


class StubDGMR(torch.nn.Module):
    """Same interface as the real generator, and RNG-dependent like it."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        base = x.mean(dim=1, keepdim=True).repeat(1, fc.FORECAST_STEPS,
                                                  1, 1, 1)
        return base * self.scale + torch.randn(base.shape) * 0.1


def write_shard(directory, site, n_sequences):
    """A shard in the exact shape preprocess_sequences writes."""
    size = fc.MODEL_SIZE
    rng = np.random.default_rng(abs(hash(site)) % 1000)
    seqs = rng.random(
        (n_sequences, fc.INPUT_FRAMES + fc.FORECAST_STEPS, size, size)
    ).astype(np.float16)
    # Everything inside the interval, alternating train/val so both splits
    # are populated.
    start = np.full(n_sequences, 1.9e9)
    split = np.array([i % 2 for i in range(n_sequences)], dtype=np.int8)

    seq_path = directory / f"{site}_sequences.npz"
    man_path = directory / f"{site}_manifest.json"
    np.savez_compressed(seq_path, sequences=seqs, start_epoch=start,
                        split=split)
    man_path.write_text(json.dumps({"site": site, "retained": n_sequences}))
    return seq_path, man_path


def main():
    failures = []
    fc.build_model = lambda seed: StubDGMR()          # no dgmr dependency

    with tempfile.TemporaryDirectory() as tmp:
        work = pathlib.Path(tmp)
        clients = []
        for site, n in zip(SITES, (6, 4, 5)):
            seq, man = write_shard(work, site, n)
            clients.append((site, seq, man))

        global_model = work / "global.pt"
        torch.save(StubDGMR().state_dict(), global_model)

        interval_args = [
            "--interval-months", str(INTERVAL),
            "--archive-start", START_MONTH,
            "--archive-months", str(ARCHIVE_MONTHS),
        ]

        # ---- path 1: each client scores at its silo -------------------
        import fl_validate_client

        per_client = []
        for site, seq, man in clients:
            out = work / f"metrics_{site}.json"
            argv = [
                "fl_validate_client.py",
                "--client", f"{site}:{seq}:{man}",
                "--round", "4",
                "--seed", str(SEED),
                *interval_args,
                "--global-model", str(global_model),
                "--metrics-out", str(out),
            ]
            saved, sys.argv = sys.argv, argv
            try:
                fl_validate_client.main()
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    failures.append(f"fl_validate_client({site}) exited "
                                    f"{exc.code}")
            except Exception as exc:                        # noqa: BLE001
                failures.append(f"fl_validate_client({site}) raised "
                                f"{type(exc).__name__}: {exc}")
            finally:
                sys.argv = saved

            if not out.exists():
                failures.append(f"fl_validate_client({site}) wrote no "
                                f"metrics")
                continue
            metrics = json.loads(out.read_text())
            if not metrics.get("n_batches"):
                failures.append(f"fl_validate_client({site}) reported "
                                f"n_batches={metrics.get('n_batches')} for a "
                                f"client that has validation data")
            per_client.append(metrics)

        # ---- the central wrapper, both of its branches ---------------
        import fl_validate

        history_in = work / "history_in.json"
        history_in.write_text(json.dumps({
            "best_val": None, "best_unit": -1, "mode": "federated",
            "aggregation": "uniform", "seed": SEED,
            "interval_months": INTERVAL, "val_points": [],
        }))
        best_in = work / "best_in.pt"
        torch.save({"state_dict": StubDGMR().state_dict(), "val": None},
                   best_in)

        def run_central(tag, extra_args):
            """Drive fl_validate.main() and return the loss it recorded."""
            history_out = work / f"history_{tag}.json"
            argv = [
                "fl_validate.py",
                "--round", "4",
                *interval_args,
                "--global-model", str(global_model),
                "--history-in", str(history_in),
                "--best-in", str(best_in),
                "--history-out", str(history_out),
                "--best-out", str(work / f"best_{tag}.pt"),
                *extra_args,
            ]
            saved, sys.argv = sys.argv, argv
            try:
                fl_validate.main()
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    failures.append(f"fl_validate({tag}) exited {exc.code}")
                    return None
            except Exception as exc:                        # noqa: BLE001
                failures.append(f"fl_validate({tag}) raised "
                                f"{type(exc).__name__}: {exc}")
                return None
            finally:
                sys.argv = saved

            if not history_out.exists():
                failures.append(f"fl_validate({tag}) wrote no history")
                return None
            points = json.loads(history_out.read_text())["val_points"]
            if not points:
                failures.append(f"fl_validate({tag}) recorded no val point")
                return None
            return points[-1]["val_loss"]

        # branch 1: the server reads every client's split itself
        central = run_central("emulated", [
            arg for site, seq, man in clients
            for arg in ("--client", f"{site}:{seq}:{man}")
        ])

        # branch 2: the server recombines what the silos returned
        recombined = run_central("silo", [
            arg for site, _, _ in clients
            for arg in ("--client-metrics", str(work / f"metrics_{site}.json"))
        ])

        # ---- they must agree, at the number the checkpoint rule reads --
        if central is not None and recombined is not None:
            if abs(central - recombined) > 1e-9:
                failures.append(
                    f"validation paths disagree in the history they write: "
                    f"emulated {central!r} vs cross-silo {recombined!r}")
            else:
                print(f"  both branches of fl_validate recorded "
                      f"{central:.9f}")
            batches = sum(m["n_batches"] for m in per_client)
            print(f"  {len(per_client)} clients, {batches} validation "
                  f"batches scored through the wrappers")

    if failures:
        print("\nFAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nboth validation paths run through their wrappers and agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
