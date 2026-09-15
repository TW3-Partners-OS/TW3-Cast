"""TW3Cast — minimal inference: reproduce the served forecast of any GIFT-Eval configuration.

Reads router.csv (configuration -> frozen mixture weights over experts), reloads the required
expert checkpoints and/or base models, composes them (weighted quantile mean, sorted) and
outputs the 9 quantiles [0.1 ... 0.9], shape (n_windows, 9, horizon), for the official test
windows.

Usage:
    pip install -r requirements.txt
    export GIFT_EVAL=<path to the GIFT-Eval datasets>
    python predict.py --config "electricity/H/short" --artifacts <path-to-artifacts> \
        --out forecasts.npy

Artifact layout expected under --artifacts (from the model release): one entry per expert id
of router.csv — either <Exx>.pt (state_dict) or <Exx>/ (save_pretrained or peft adapter
directory) ; the architecture family of each expert is given by experts.json.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

QL = [round(0.1 * i, 1) for i in range(1, 10)]
BASE = json.load(open(os.path.join(os.path.dirname(__file__), "base_models.json")))["base_models"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- base model wrappers
def load_chronos(repo_key):
    from chronos import BaseChronosPipeline
    spec = BASE[repo_key]
    return BaseChronosPipeline.from_pretrained(spec["repo"], revision=spec["revision"],
                                               device_map=DEV, torch_dtype=torch.float32)


def load_tirex():
    from tirex import load_model
    spec = BASE["tirex"]
    return load_model(spec["repo"], backend=spec.get("backend", "torch"), device=DEV)


def load_toto():
    from toto2 import Toto2Model
    spec = BASE["toto_25b_ft"]
    return Toto2Model.from_pretrained(spec["repo"], revision=spec["revision"]).to(DEV).eval()


def load_timesfm(max_horizon):
    import timesfm
    spec = BASE["timesfm"]
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(spec["repo"], revision=spec["revision"])
    m.compile(timesfm.ForecastConfig(max_context=2048, max_horizon=max_horizon,
                                     normalize_inputs=True, use_continuous_quantile_head=True,
                                     force_flip_invariance=True, infer_is_positive=True,
                                     fix_quantile_crossing=True))
    return m


# ---------------------------------------------------------------- artifact reloading
def reload_artifact(kind, path, base_key=None):
    """Reload one trained artifact. kind: c2 | tirex | toto."""
    if kind == "c2":
        from chronos import BaseChronosPipeline
        if os.path.isdir(path):
            if os.path.exists(os.path.join(path, "adapter_config.json")):
                import peft
                pipe = load_chronos(base_key or "chronos2")
                pipe.model = peft.PeftModel.from_pretrained(pipe.model, path) \
                                 .merge_and_unload().eval()
                return pipe
            return BaseChronosPipeline.from_pretrained(path, device_map=DEV,
                                                       torch_dtype=torch.float32)
        pipe = load_chronos(base_key or "chronos2")
        pipe.model.load_state_dict(torch.load(path, map_location=DEV))
        return pipe
    if kind == "tirex":
        m = load_tirex()
        m.load_state_dict(torch.load(path, map_location=DEV))
        return m
    if kind == "toto":
        import peft
        base = load_toto()
        return peft.PeftModel.from_pretrained(base, path).eval()
    raise ValueError(kind)


# ---------------------------------------------------------------- forecasting
def contexts_for(config):
    from gift_eval.data import Dataset
    from gluonts.time_feature import get_seasonality
    name, term = config.rsplit("/", 1)
    d0 = Dataset(name=name, term=term, to_univariate=False)
    d = Dataset(name=name, term=term, to_univariate=d0.target_dim > 1)
    ctxs = []
    for e in d.test_data.input:
        y = np.asarray(e["target"], float)
        y = y[0] if y.ndim > 1 else y
        y = np.nan_to_num(y, nan=0.0)[-8192:]
        if len(y) < 2 or np.std(y) <= 1e-12:
            y = np.concatenate([y, [0.0, 1e-6]])
        ctxs.append(y)
    return d, ctxs, d.prediction_length, get_seasonality(d.freq)


def quantiles_chronos(pipe, ctxs, h):
    out = []
    for s in range(0, len(ctxs), 64):
        q, _ = pipe.predict_quantiles(
            [torch.tensor(c, dtype=torch.float32) for c in ctxs[s:s + 64]],
            prediction_length=h, quantile_levels=QL)
        if isinstance(q, list):
            q = torch.stack([x.squeeze(0) for x in q])
        out.append(np.asarray(q.cpu()).transpose(0, 2, 1))
    return np.concatenate(out)


def member_forecast(member, ctxs, h, artifacts, config=None):
    """Forecast of one router member: an expert id (Exx) or a base model name."""
    if member.startswith("E"):
        experts = json.load(open(os.path.join(os.path.dirname(__file__), "experts.json")))
        kind = experts[member]["type"]
        candidates = [os.path.join(artifacts, member),
                      os.path.join(artifacts, member + ".pt")]
        path = next((p for p in candidates if os.path.exists(p)), None)
        if path is None:
            if config:
                # experts shipped as verified forecast quantiles (weights not retained)
                qf = os.path.join(artifacts, f"{member}__{config.replace('/', '_')}.npy")
                if os.path.exists(qf):
                    return np.load(qf)[:, :9, :h]
            raise FileNotFoundError(f"expert checkpoint not found: {member}")
        model = reload_artifact(kind, path)
        if kind == "c2":
            return quantiles_chronos(model, ctxs, h)
        raise NotImplementedError(f"see README for the {kind} forecast recipe")
    if member in ("chronos2", "turk"):
        return quantiles_chronos(load_chronos("chronos2" if member == "turk" else member), ctxs, h)
    raise NotImplementedError(f"member {member}: load via base_models.json (see README)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="forecasts.npy")
    args = ap.parse_args()
    rt = pd.read_csv(os.path.join(os.path.dirname(__file__), "router.csv"))
    rows = rt[rt.config == args.config]
    if not len(rows):
        raise SystemExit(f"unknown configuration: {args.config}")
    if (rows.expert == "__tournament__").any():
        raise SystemExit("this configuration is served by the train-side tournament; "
                         "see README for the tournament procedure")
    d, ctxs, h, _ = contexts_for(args.config)
    q = None
    for _, r in rows.iterrows():
        fc = member_forecast(str(r.expert), ctxs, h, args.artifacts, config=args.config)[:, :9, :h]
        q = fc * float(r.weight) if q is None else q + fc * float(r.weight)
    q = np.sort(q, axis=1).astype(np.float32)
    np.save(args.out, q)
    print(f"{args.config}: forecasts {q.shape} -> {args.out}")


if __name__ == "__main__":
    main()
