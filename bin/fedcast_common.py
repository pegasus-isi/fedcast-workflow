"""Shared helpers for the Fed-Cast training wrappers.

This module is staged into each training job's working directory via the
Pegasus replica catalog (LFN ``fedcast_common.py``); wrappers import it from
the job's cwd. Keep it dependency-light: numpy + torch + lightning + dgmr.
"""

import logging
import os
import sys
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)

INPUT_FRAMES = 4
FORECAST_STEPS = 12
GRID_LAMBDA = 20.0  # grid-cell regularizer weight (paper Sec. IV-C)

# DGMR's latent/conditioning stacks require spatial dims divisible by 32;
# 300 is not, so we center-crop the 300x300 windows to 288x288 (= 9 * 32),
# the largest fitting size. Documented deviation — how the paper fed
# 300x300 fields into DGMR is an open author question (SPEC Sec. 6).
#
# FEDCAST_MODEL_SIZE / FEDCAST_BATCH_SIZE are PILOT-ONLY overrides for
# CPU/low-memory smoke tests (e.g. 128 / 1). Reproduction runs must use
# the 288 / 2 defaults.
MODEL_SIZE = int(os.environ.get("FEDCAST_MODEL_SIZE", "288"))
BATCH_SIZE = int(os.environ.get("FEDCAST_BATCH_SIZE", "2"))
if MODEL_SIZE != 288 or BATCH_SIZE != 2:
    logger.warning("PILOT overrides active: MODEL_SIZE=%d BATCH_SIZE=%d — "
                   "not valid for reproduction runs",
                   MODEL_SIZE, BATCH_SIZE)


def center_crop(arr, size=MODEL_SIZE):
    """Center-crop the last two (H, W) dims to size x size."""
    h, w = arr.shape[-2], arr.shape[-1]
    top = max(0, (h - size) // 2)
    left = max(0, (w - size) // 2)
    return arr[..., top:top + size, left:left + size]


def parse_client(spec):
    """Parse SITE:sequences:manifest into a client dict.

    In emulated mode the two paths are plain LFNs in the job sandbox. In
    cross-silo mode they are resident paths on the silo worker and may be
    home-relative, which Pegasus does not expand for job arguments — so
    expand here, where the job's own environment is in scope.
    """
    import os

    name, seq, manifest = spec.split(":")
    return {"name": name,
            "sequences": os.path.expanduser(seq),
            "manifest": os.path.expanduser(manifest)}


def interval_start_epoch(archive_start, archive_months, interval_months):
    """Epoch seconds of the first month inside the LAST L months.

    The training interval L uses the last L months of the archive
    (SPEC.md open question 11 — our documented rule).
    """
    year, mon = (int(x) for x in archive_start.split("-"))
    total = year * 12 + (mon - 1) + archive_months - interval_months
    y, m = divmod(total, 12)
    return datetime(y, m + 1, 1, tzinfo=timezone.utc).timestamp()


def escape_export_segment(key):
    """Escape a key so a dotted path cannot be forged.

    Without this a top-level key literally named ``splits.train`` would
    render as the path ``splits.train`` and satisfy a declaration meant
    for the nested field. Backslash, dot and ``[`` are escaped, so a
    real dot in a key name reads as ``\.`` and never collides with the
    separator.
    """
    return (str(key).replace("\\", "\\\\")
            .replace(".", "\\.")
            .replace("[", "\\["))


def export_field_paths(value, prefix=""):
    """Dotted paths of every field in a payload, nested ones included.

    Inspects the real object at run time, so it sees fields however they
    got there — literal, assignment, update(), a helper's return value —
    which static analysis of the construction site cannot promise.

    Descends through sequences as well as mappings: a dict inside a list
    exports its keys just as much as one nested directly, and is reported
    with a ``[]`` segment (``events[].site``). Key names are escaped by
    escape_export_segment so a dot inside a key cannot impersonate the
    separator.
    """
    paths = []
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            segment = escape_export_segment(key)
            dotted = f"{prefix}.{segment}" if prefix else segment
            paths.append(dotted)
            paths.extend(export_field_paths(value[key], dotted))
    elif isinstance(value, (list, tuple, set, frozenset)):
        # One entry for the element shape, not one per element.
        seen = set()
        for item in value:
            for path in export_field_paths(item, f"{prefix}[]"):
                if path not in seen:
                    seen.add(path)
                    paths.append(path)
    return paths


_EXPORT_GUARDS = {}


def write_export(path, payload, declared, label, indent=None):
    """Validate a payload against its declaration, then write it as JSON.

    Checking and writing are one operation on purpose: a wrapper cannot
    validate one object and write another, and cannot mutate a payload
    between the check and the write.

    The file is then read back and validated again, so a serializer that
    reshapes the payload cannot widen the export surface either. Finally
    the path is registered for a re-check at interpreter exit: whatever
    writes it afterwards — this wrapper, an aliased json.dump, anything —
    the job fails rather than shipping undeclared fields. That last check
    inspects the artifact, which is the only way to cover write paths
    static analysis cannot enumerate.
    """
    import json

    check_export(payload, declared, label)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=indent)

    verify_export_file(path, declared, label)
    guard_export(path, declared, label)
    return payload


def verify_export_file(path, declared, label):
    """Fail unless the JSON at `path` carries only declared fields."""
    import json

    try:
        with open(path) as handle:
            written = json.load(handle)
    except (OSError, ValueError) as exc:
        raise SystemExit(
            f"{label}: cannot verify the payload written to {path}: {exc}"
        ) from exc
    check_export(written, declared, f"{label} (as written to {path})")
    return written


def guard_export(path, declared, label):
    """Re-verify an export at interpreter exit, however it got written.

    Call this once per output path, early — before the work that produces
    the payload. Registration is what makes the guarantee independent of
    the write: if the guarded write never runs and something else
    produces the file, the exit check still inspects it. A path that
    does not exist at exit is skipped, since nothing left the silo;
    Pegasus already fails a job that omits a declared output.
    """
    import atexit
    import os

    first = not _EXPORT_GUARDS
    _EXPORT_GUARDS[os.path.abspath(path)] = (tuple(declared), label)

    if not first:
        return

    def _final_check():
        failed = []
        for guarded, (fields, name) in sorted(_EXPORT_GUARDS.items()):
            if not os.path.exists(guarded):
                continue
            try:
                verify_export_file(guarded, fields, name)
            except SystemExit as exc:
                failed.append(str(exc))
        if failed:
            for message in failed:
                sys.stderr.write(f"export guard: {message}\n")
            sys.stderr.write(
                "export guard: an output carries undeclared fields; "
                "failing the job rather than shipping it.\n")
            sys.stderr.flush()
            # os._exit so the status is not lost in interpreter shutdown.
            os._exit(70)

    atexit.register(_final_check)


def check_export(payload, declared, label):
    """Fail unless a payload is an object whose fields are all declared.

    The guarantee is one-directional: no field may leave that was not
    declared. A declared field that is absent is fine.

    Cross-silo mode's claim is about what leaves a data holder, so each
    wrapper declares the fields it may export. Prefer write_export(),
    which validates and writes in one step; `tools/check_export_docs.py`
    checks the declarations against the documentation.

    The top level must be a mapping. Without that, a payload replaced by
    a bare array or string would expose no field paths at all and pass
    review by carrying nothing this function knows how to name — the
    exact shape a tampered artifact would take to smuggle raw data out.

    Scope, so it is not mistaken for more: this controls field *names*.
    It does not constrain the type or size of a declared field, so a
    declared field legitimately holding a large vector (the manifest's
    `start_epochs` and `split_labels` do) is indistinguishable from one
    abused to carry bulk data. That is why README documents those two
    explicitly rather than relying on this check alone.
    """
    if not isinstance(payload, dict):
        raise SystemExit(
            f"{label}: refusing a payload whose top level is "
            f"{type(payload).__name__}, not an object. Exports must be "
            f"JSON objects so every field they carry can be named and "
            f"checked against EXPORT_FIELDS."
        )
    actual = set(export_field_paths(payload))
    declared = set(declared)
    undeclared = sorted(actual - declared)
    if undeclared:
        raise SystemExit(
            f"{label}: refusing to write a payload with undeclared "
            f"field(s): {', '.join(undeclared)}. Add them to "
            f"EXPORT_FIELDS and to the \"What actually leaves a silo\" "
            f"table in README.md, or stop exporting them."
        )
    # The declaration is an upper bound, not an exact set: a payload need
    # not carry every field it is allowed to. The manifest and its
    # no-usable-input stand-in legitimately share one declaration and each
    # omits most of the other's fields, so absence is not worth a warning.
    missing = sorted(declared - actual)
    if missing:
        logger.debug("%s: declared field(s) not present: %s", label,
                     ", ".join(missing))
    return payload


def job_identity():
    """Who and where this job is running as.

    Recorded by the preprocess job and printed when a resident shard is
    missing, so a home-relative path that resolved differently for two
    jobs is diagnosable from the logs alone.
    """
    import getpass
    import socket

    try:
        user = getpass.getuser()
    except Exception:                                   # noqa: BLE001
        user = os.environ.get("USER", "unknown")
    return {"host": socket.gethostname(), "user": user,
            "home": os.path.expanduser("~")}


def require_shard(client):
    """Fail with a diagnosis when a client's shard is not readable."""
    path = client["sequences"]
    if os.path.exists(path):
        return
    ident = job_identity()
    logger.error("%s: shard not found at %s", client["name"], path)
    logger.error("  host=%s user=%s HOME=%s", ident["host"], ident["user"],
                 ident["home"])
    if os.path.isabs(path):
        logger.error(
            "  This client's preprocess job should have written it here. "
            "Likely causes: this job did not run on the same worker as "
            "that preprocess job (check tools/silo_check.py <silo map>), "
            "or the pool gives each slot its own user, so a home-relative "
            "shard directory resolved differently for the two jobs. In "
            "the second case switch data_dir to an absolute path and run "
            "tools/silo_worker_setup.sh on the workers.")
    raise FileNotFoundError(path)


def load_client_data(client, t_start, limit=None):
    """Return dict with train/val tensors for one client.

    Filters to sequences starting at/after ``t_start``. ``limit`` caps
    train/val sequences per client (pilot/CPU smoke tests only).
    """
    import torch

    require_shard(client)
    with np.load(client["sequences"]) as data:
        seqs = data["sequences"]
        starts = data["start_epoch"]
        split = data["split"]
    keep = starts >= t_start
    seqs, split = seqs[keep], split[keep]

    def to_tensor(mask):
        arr = seqs[mask].astype(np.float32)
        if limit:
            arr = arr[:limit]
        if arr.shape[0] == 0:
            return None, None
        arr = center_crop(arr)  # DGMR needs dims divisible by 32
        # (N, T, H, W) -> inputs (N, 4, 1, H, W), targets (N, 12, 1, H, W)
        x = torch.from_numpy(arr[:, :INPUT_FRAMES])[:, :, None]
        y = torch.from_numpy(arr[:, INPUT_FRAMES:])[:, :, None]
        return x, y

    train_x, train_y = to_tensor(split == 0)
    val_x, val_y = to_tensor(split == 1)
    n_train = 0 if train_x is None else train_x.shape[0]
    logger.info("%s: %d train / %d val sequences in interval",
                client["name"], n_train,
                0 if val_x is None else val_x.shape[0])
    return {"name": client["name"], "train": (train_x, train_y),
            "val": (val_x, val_y), "n_train": n_train}


def build_model(seed):
    """Instantiate DGMR (openclimatefix skillful_nowcasting), seeded."""
    import torch
    from dgmr import DGMR

    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    return DGMR(forecast_steps=FORECAST_STEPS, output_shape=MODEL_SIZE)


def make_loader(x, y):
    import torch

    ds = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE,
                                       shuffle=True)


def fit_one_epoch(model, loader, epochs=1):
    """Run Lightning fit for a fixed number of epochs on one loader."""
    import pytorch_lightning as pl
    import torch

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(model, loader)


def _val_batch_seed(val_seed, client_name, batch_index):
    """Deterministic RNG seed for one client's validation batch.

    The loss draws a 6-sample DGMR ensemble, so it depends on the RNG
    state. Deriving that state from (seed, client, batch) instead of
    letting a continuous stream run makes the number independent of who
    computes it and in what order — which is what lets the emulated and
    cross-silo paths select the same checkpoint. crc32, not hash(), so it
    is stable across processes.
    """
    import zlib

    digest = zlib.crc32(str(client_name).encode("utf-8"))
    return (int(val_seed) + digest + batch_index * 7919) % (2 ** 31)


def generator_val_batch_losses(model, datasets, val_seed):
    """Per-batch grid-cell-regularizer validation losses.

    Shared by the central validator (all clients' datasets at once) and
    the silo-mode per-client validator (one dataset at the silo). Each
    batch's ensemble is drawn from a seed derived from the client name
    and the batch index, so both paths produce identical numbers: the
    checkpoint loss is the mean over these batch losses, and a mean of
    means weighted by batch count reconstructs it exactly.

    TODO: add the discriminator hinge term to fully match the paper's
    Eq. 3; the grid-cell term (lambda=20, intensity-weighted MAE on the
    ensemble mean of 6 samples) is the dominant, checkpoint-driving
    component.
    """
    import torch

    if isinstance(datasets, dict):
        # A single load_client_data() result rather than a list of them.
        # Iterating it would walk the key strings and fail obscurely deep
        # in the loop, so say what is wrong here instead.
        raise TypeError(
            "generator_val_batch_losses expects a list of client datasets; "
            "got one dataset dict. Wrap it: [data]."
        )

    model.eval()
    device = next(model.parameters()).device
    losses = []
    with torch.no_grad():
        for d in datasets:
            val_x, val_y = d["val"]
            if val_x is None:
                continue
            for batch_index, i in enumerate(
                    range(0, val_x.shape[0], BATCH_SIZE)):
                torch.manual_seed(
                    _val_batch_seed(val_seed, d["name"], batch_index))
                x = val_x[i:i + BATCH_SIZE].to(device)
                y = val_y[i:i + BATCH_SIZE].to(device)
                preds = torch.stack(
                    [model(x) for _ in range(6)]
                ).mean(dim=0)
                weight = torch.clamp(y + 1.0, max=24.0)
                grid_loss = (torch.abs(preds - y) * weight).mean()
                losses.append(float(GRID_LAMBDA * grid_loss))
    return losses


def generator_val_loss(model, datasets, val_seed):
    """Grid-cell-regularizer validation loss over all clients' val sets."""
    losses = generator_val_batch_losses(model, datasets, val_seed)
    return float(np.mean(losses)) if losses else float("inf")


def combine_client_val_metrics(metrics):
    """Combine per-client validation metrics into one global loss.

    ``metrics`` is a list of dicts written by fl_validate_client.py
    ({"sum_loss", "n_batches", ...}). The result equals the mean batch
    loss generator_val_loss() would return if every client's validation
    split were scored in one process.
    """
    total_batches = sum(int(m.get("n_batches") or 0) for m in metrics)
    if not total_batches:
        return float("inf")
    total = sum(float(m.get("sum_loss") or 0.0) for m in metrics)
    return float(total / total_batches)


def cpu_state_dict(model):
    """Detached CPU copy of a model's state dict."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
