"""TW3Cast — exact replication of any submitted GIFT-Eval line.

Single entrypoint: downloads the served forecast quantiles of a configuration from the model
release, evaluates them with the official GIFT-Eval harness, and checks every submitted metric
for equality.

Usage:
    pip install -r requirements.txt
    export GIFT_EVAL=<path to the GIFT-Eval datasets>
    python replicate.py --config "m4_weekly/W/short"        # one configuration
    python replicate.py --all                                # all 97 configurations

The served quantiles (`served/<config>.npy`, shape (n_windows, 9, horizon)) are the exact
arrays behind the submission: evaluating them reproduces the submitted metrics to full
precision. They are downloaded from https://huggingface.co/TW3PartnersLLM/TW3Cast (or read
from --artifacts if given), together with the submitted score file
(`submission/all_results.csv`). `predict.py` remains the reference for recomputing forecasts
from the released expert checkpoints; serving-time context variants make its outputs match the
submission only approximately, which is why exact replication goes through the served arrays.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

RELEASE = "TW3PartnersLLM/TW3Cast"
QL = [round(0.1 * i, 1) for i in range(1, 10)]
TOL = 1e-6


def fetch(path, artifacts=None):
    if artifacts:
        fp = os.path.join(artifacts, path)
        if os.path.exists(fp):
            return fp
    from huggingface_hub import hf_hub_download
    return hf_hub_download(RELEASE, path)


def evaluate(config, q):
    from gift_eval.data import Dataset
    from gluonts.ev.metrics import (MAE, MAPE, MASE, MSE, MSIS, ND, NRMSE, RMSE, SMAPE,
                                    MeanWeightedSumQuantileLoss)
    from gluonts.model import evaluate_forecasts
    from gluonts.model.forecast import QuantileForecast
    from gluonts.time_feature import get_seasonality
    from predict import NAME_MAP
    name, term = config.rsplit("/", 1)
    name = NAME_MAP.get(name, name)
    d0 = Dataset(name=name, term=term, to_univariate=False)
    d = Dataset(name=name, term=term, to_univariate=d0.target_dim > 1)
    fcs = [QuantileForecast(forecast_arrays=np.vstack([q9, q9[4][None]]),
                            forecast_keys=[str(x) for x in QL] + ["mean"],
                            start_date=e["start"], item_id=e.get("item_id"))
           for e, q9 in zip(d.test_data.label, q)]
    mets = [MSE(forecast_type="mean"), MSE(forecast_type=0.5), MAE(), MASE(), MAPE(), SMAPE(),
            MSIS(), RMSE(), NRMSE(), ND(), MeanWeightedSumQuantileLoss(quantile_levels=QL)]
    try:
        r = evaluate_forecasts(fcs, test_data=d.test_data, metrics=mets, batch_size=1024,
                               axis=None, mask_invalid_label=True, allow_nan_forecast=False,
                               seasonality=get_seasonality(d.freq))
    except Exception:
        mets = mets[:-5] + [MSIS(alpha=0.2)] + mets[-4:]
        r = evaluate_forecasts(fcs, test_data=d.test_data, metrics=mets, batch_size=1024,
                               axis=None, mask_invalid_label=True, allow_nan_forecast=False,
                               seasonality=get_seasonality(d.freq))
    return {k: float(v) for k, v in r.reset_index(drop=True).to_dict("records")[0].items()}


def replicate(config, submitted, artifacts=None):
    q = np.load(fetch(f"served/{config.replace('/', '_')}.npy", artifacts))
    got = evaluate(config, q)
    row = submitted[submitted.dataset == config].iloc[0]
    ok = True
    print(f"\n{config}")
    print(f"{'metric':44s} {'replicated':>14s} {'submitted':>14s} {'rel. diff':>10s}")
    for col in submitted.columns:
        key = col.replace("eval_metrics/", "")
        if not col.startswith("eval_metrics/") or key not in got:
            continue
        a, b = got[key], float(row[col])
        diff = abs(a - b) / max(abs(b), 1e-12)
        ok &= diff < TOL
        print(f"{key:44s} {a:14.6f} {b:14.6f} {diff:10.1e}")
    print("EXACT MATCH" if ok else "MISMATCH")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--artifacts", help="local copy of the release (optional)")
    args = ap.parse_args()
    submitted = pd.read_csv(fetch("submission/all_results.csv", args.artifacts))
    configs = list(submitted.dataset) if args.all else [args.config]
    if not configs[0]:
        raise SystemExit("give --config <dataset/freq/term> or --all")
    bad = [c for c in configs if not replicate(c, submitted, args.artifacts)]
    print(f"\n{len(configs) - len(bad)}/{len(configs)} configurations replicated exactly")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
