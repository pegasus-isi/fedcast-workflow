#!/usr/bin/env python3

"""MCT verification: compute the Table-I metric suite for one (method, L).

Metrics are computed PER LEAD TIME and then averaged across the 12 lead
times (SPEC.md constraint 13), per forecast instance. Deterministic scores
use the ensemble mean; CRPS uses the full ensemble (constraint 12).
Rain/no-rain threshold: --rain-threshold (0.1 mm/h).

Output CSV columns:
  method, interval, event_id, site, start_epoch, metric, value

The interval L is parsed from the forecasts filename tag by the caller and
passed via --interval ("" for STEPS).
"""

import argparse
import csv
import logging
import sys

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

EPS = 1e-12


def contingency(fcst, obs, thr):
    """Hits, misses, false alarms, correct negatives on one 2D field."""
    f = fcst > thr
    o = obs > thr
    h = float(np.sum(f & o))
    m = float(np.sum(~f & o))
    fa = float(np.sum(f & ~o))
    cn = float(np.sum(~f & ~o))
    return h, m, fa, cn


def categorical_scores(h, m, fa, cn):
    """All contingency-table scores of Table I."""
    n = h + m + fa + cn
    pod = h / (h + m + EPS)
    far = fa / (h + fa + EPS)          # false alarm ratio
    fa_rate = fa / (fa + cn + EPS)     # false alarm rate (F)
    csi = h / (h + m + fa + EPS)
    acc = (h + cn) / (n + EPS)
    bias = (h + fa) / (h + m + EPS)    # event-frequency bias (target 1)
    hk = pod - fa_rate                 # Hanssen-Kuipers (target 1)
    # HSS
    exp = ((h + m) * (h + fa) + (cn + m) * (cn + fa)) / (n + EPS)
    hss = (h + cn - exp) / (n - exp + EPS)
    # GSS (Gilbert)
    h_rand = (h + m) * (h + fa) / (n + EPS)
    gss = (h - h_rand) / (h + m + fa - h_rand + EPS)
    # MCC
    denom = np.sqrt((h + fa) * (h + m) * (cn + fa) * (cn + m)) + EPS
    mcc = (h * cn - fa * m) / denom
    # F1
    f1 = 2 * h / (2 * h + fa + m + EPS)
    # SEDI
    hr = np.clip(pod, EPS, 1 - EPS)
    f_r = np.clip(fa_rate, EPS, 1 - EPS)
    sedi = ((np.log(f_r) - np.log(hr) - np.log(1 - f_r) + np.log(1 - hr))
            / (np.log(f_r) + np.log(hr) + np.log(1 - f_r)
               + np.log(1 - hr) + EPS))
    return {"POD": pod, "FAR": far, "FA": fa_rate, "CSI": csi, "ACC": acc,
            "BIAS": bias, "HK": hk, "HSS": hss, "GSS": gss, "MCC": mcc,
            "F1": f1, "SEDI": sedi}


def crps_ensemble(ens, obs):
    """Ensemble CRPS averaged over pixels (fair, standard estimator)."""
    k = ens.shape[0]
    term1 = np.mean(np.abs(ens - obs[None]), axis=0)
    term2 = 0.0
    for i in range(k):
        term2 = term2 + np.mean(np.abs(ens[i][None] - ens), axis=0)
    return float(np.mean(term1 - 0.5 * term2 / 1.0))


def psnr(fcst, obs, data_range):
    mse = float(np.mean((fcst - obs) ** 2))
    if mse <= EPS:
        return 60.0
    return float(10.0 * np.log10((data_range ** 2) / mse))


def rapsd_distance(fcst, obs):
    """Mean |log10 PSD ratio| between forecast and observed fields."""
    from pysteps.utils.spectral import rapsd

    psd_f = rapsd(np.nan_to_num(fcst), fft_method=np.fft)
    psd_o = rapsd(np.nan_to_num(obs), fft_method=np.fft)
    psd_f = np.clip(psd_f, EPS, None)
    psd_o = np.clip(psd_o, EPS, None)
    return float(np.mean(np.abs(np.log10(psd_f) - np.log10(psd_o))))


def main():
    parser = argparse.ArgumentParser(
        description="Compute the MCT metric suite for one (method, L)")
    parser.add_argument("--method", required=True)
    # nargs="?" tolerates a bare `--interval` flag: Pegasus drops
    # empty-string arguments when serializing job args.
    parser.add_argument("--interval", nargs="?", const="", default="")
    parser.add_argument("--forecasts", required=True)
    parser.add_argument("--benchmark", required=True)  # provenance input
    parser.add_argument("--client", action="append", default=[])
    parser.add_argument("--rain-threshold", type=float, default=0.1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with np.load(args.forecasts, allow_pickle=False) as data:
        if data["forecasts"].shape[0] == 0:
            logger.error("Empty forecasts file %s", args.forecasts)
            with open(args.output, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["method", "interval", "event_id", "site",
                     "start_epoch", "metric", "value"])
            sys.exit(1)
        forecasts = data["forecasts"].astype(np.float32)   # (N,K,12,H,W)
        observations = data["observations"].astype(np.float32)
        exec_times = data["exec_time_s"]
        event_ids = data["event_id"]
        sites = data["site"]
        start_epochs = data["start_epoch"]

    thr = args.rain_threshold
    rows = []
    n_inst, _, n_leads = forecasts.shape[:3]
    for i in range(n_inst):
        ens = forecasts[i]                      # (K, 12, H, W)
        obs = observations[i]                   # (12, H, W)
        mean = ens.mean(axis=0)                 # (12, H, W)
        data_range = max(float(obs.max()), float(mean.max()), 1.0)

        per_lead = {}
        for lead in range(n_leads):
            h, m, fa, cn = contingency(mean[lead], obs[lead], thr)
            scores = categorical_scores(h, m, fa, cn)
            scores["CRPS"] = crps_ensemble(ens[:, lead], obs[lead])
            scores["PSNR"] = psnr(mean[lead], obs[lead], data_range)
            try:
                scores["RAPSD"] = rapsd_distance(mean[lead], obs[lead])
            except Exception:  # noqa: BLE001
                scores["RAPSD"] = np.nan
            for key, val in scores.items():
                per_lead.setdefault(key, []).append(val)

        # Lead-time averaging (SPEC constraint 13).
        summary = {k: float(np.nanmean(v)) for k, v in per_lead.items()}
        summary["Executing_Time"] = float(exec_times[i])

        for metric, value in summary.items():
            rows.append([args.method, args.interval,
                         str(event_ids[i]), str(sites[i]),
                         float(start_epochs[i]), metric, value])

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "interval", "event_id", "site",
                         "start_epoch", "metric", "value"])
        writer.writerows(rows)

    logger.info("%s L=%s: %d instances x %d metrics -> %s",
                args.method, args.interval or "-", n_inst,
                len(rows) // max(n_inst, 1), args.output)


if __name__ == "__main__":
    main()
