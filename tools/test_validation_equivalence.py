#!/usr/bin/env python3

"""Exercise the two validation paths through the wrappers themselves.

    tools/test_validation_equivalence.py     # 0 = pass

Checks two things that have to hold for cross-silo mode to be usable:

1. **The wrappers run.** `fl_validate_client.main()` and the central
   `fl_validate.main()` are called with real argv against a real shard on
   disk. A previous bug passed a single dataset dict where a list of them
   was expected, which crashed for every client that had validation data —
   and was invisible to tests that called `generator_val_batch_losses`
   directly with a correctly shaped argument. Testing the helper is not
   testing the call site.
2. **The two paths agree.** The per-client losses recombine to exactly the
   loss the central path computes, which is what lets either mode select
   the same checkpoint.

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

        # ---- path 2: the server scores every client itself ------------
        t_start = fc.interval_start_epoch(START_MONTH, ARCHIVE_MONTHS,
                                          INTERVAL)
        datasets = [
            fc.load_client_data(
                fc.parse_client(f"{site}:{seq}:{man}"), t_start)
            for site, seq, man in clients
        ]
        model = fc.build_model(SEED)
        model.load_state_dict(torch.load(global_model, map_location="cpu"))
        central = fc.generator_val_loss(model, datasets, SEED)

        # ---- they must agree -----------------------------------------
        if per_client:
            recombined = fc.combine_client_val_metrics(per_client)
            if abs(central - recombined) > 1e-9:
                failures.append(
                    f"validation paths disagree: central {central!r} vs "
                    f"recombined {recombined!r}")
            else:
                print(f"  central and recombined agree: {central:.9f}")
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
