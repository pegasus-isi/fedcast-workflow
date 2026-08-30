"""Shared helpers for the Fed-Cast training wrappers.

This module is staged into each training job's working directory via the
Pegasus replica catalog (LFN ``fedcast_common.py``); wrappers import it from
the job's cwd. Keep it dependency-light: numpy + torch + lightning + dgmr.
"""

import logging
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)

INPUT_FRAMES = 4
FORECAST_STEPS = 12
GRID_LAMBDA = 20.0  # grid-cell regularizer weight (paper Sec. IV-C)
BATCH_SIZE = 2      # conservative default for 300x300 fields


def parse_client(spec):
    """Parse SITE:sequences_lfn:manifest_lfn."""
    name, seq_lfn, manifest_lfn = spec.split(":")
    return {"name": name, "sequences": seq_lfn, "manifest": manifest_lfn}


def interval_start_epoch(archive_start, archive_months, interval_months):
    """Epoch seconds of the first month inside the LAST L months.

    The training interval L uses the last L months of the archive
    (SPEC.md open question 11 — our documented rule).
    """
    year, mon = (int(x) for x in archive_start.split("-"))
    total = year * 12 + (mon - 1) + archive_months - interval_months
    y, m = divmod(total, 12)
    return datetime(y, m + 1, 1, tzinfo=timezone.utc).timestamp()


def load_client_data(client, t_start, limit=None):
    """Return dict with train/val tensors for one client.

    Filters to sequences starting at/after ``t_start``. ``limit`` caps
    train/val sequences per client (pilot/CPU smoke tests only).
    """
    import torch

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
    return DGMR(forecast_steps=FORECAST_STEPS)


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


def generator_val_loss(model, datasets):
    """Grid-cell-regularizer validation loss over all clients' val sets.

    TODO: add the discriminator hinge term to fully match the paper's
    Eq. 3; the grid-cell term (lambda=20, intensity-weighted MAE on the
    ensemble mean of 6 samples) is the dominant, checkpoint-driving
    component.
    """
    import torch

    model.eval()
    device = next(model.parameters()).device
    losses = []
    with torch.no_grad():
        for d in datasets:
            val_x, val_y = d["val"]
            if val_x is None:
                continue
            for i in range(0, val_x.shape[0], BATCH_SIZE):
                x = val_x[i:i + BATCH_SIZE].to(device)
                y = val_y[i:i + BATCH_SIZE].to(device)
                preds = torch.stack(
                    [model(x) for _ in range(6)]
                ).mean(dim=0)
                weight = torch.clamp(y + 1.0, max=24.0)
                grid_loss = (torch.abs(preds - y) * weight).mean()
                losses.append(float(GRID_LAMBDA * grid_loss))
    return float(np.mean(losses)) if losses else float("inf")


def cpu_state_dict(model):
    """Detached CPU copy of a model's state dict."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
