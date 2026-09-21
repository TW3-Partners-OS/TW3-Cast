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
of router.csv -- either <Exx>.pt (state_dict), <Exx>/ (save_pretrained or peft adapter
directory), or <Exx>__<config-slug>.npy (verified forecast quantiles, for the one expert
whose weights were not retained); the architecture family of each expert is in experts.json.

These are reference recipes: they reload each artifact and forecast with the corresponding
base model's native path. The submitted lines were additionally composed with long-context
and multivariate serving variants on some configurations, so small decimal differences with
the submission are expected there.
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

# benchmark configuration name -> GIFT-Eval dataset directory
NAME_MAP = {
    "kdd_cup_2018/D": "kdd_cup_2018_with_missing/D", "kdd_cup_2018/H": "kdd_cup_2018_with_missing/H",
    "loop_seattle/D": "LOOP_SEATTLE/D", "loop_seattle/H": "LOOP_SEATTLE/H",
    "loop_seattle/5T": "LOOP_SEATTLE/5T", "m_dense/D": "M_DENSE/D", "m_dense/H": "M_DENSE/H",
    "sz_taxi/H": "SZ_TAXI/H", "sz_taxi/15T": "SZ_TAXI/15T",
    "temperature_rain/D": "temperature_rain_with_missing", "restaurant/D": "restaurant",
    "saugeen/D": "saugeenday/D", "saugeen/W": "saugeenday/W", "saugeen/M": "saugeenday/M",
    "car_parts/M": "car_parts_with_missing", "m4_daily/D": "m4_daily", "m4_weekly/W": "m4_weekly",
    "m4_hourly/H": "m4_hourly", "m4_quarterly/Q": "m4_quarterly", "m4_monthly/M": "m4_monthly",
    "m4_yearly/A": "m4_yearly", "hospital/M": "hospital", "covid_deaths/D": "covid_deaths",
    "bizitobs_application/10S": "bizitobs_application", "bizitobs_service/10S": "bizitobs_service",
    "us_births/D": "us_births/D",
}


# ---------------------------------------------------------------- base model loaders
def load_chronos(repo_key):
    from chronos import BaseChronosPipeline
    spec = BASE[repo_key]
    return BaseChronosPipeline.from_pretrained(spec["repo"], revision=spec["revision"],
                                               device_map=DEV, torch_dtype=torch.float32)


def load_tirex():
    from tirex import load_model
    spec = BASE["tirex"]
    return load_model(spec["repo"], backend=spec.get("backend", "torch"), device=DEV)


def load_tirex2():
    try:  # pre-Ampere GPUs: the fused flashrnn kernel is unavailable, force the pure-torch path
        import tirex2.model.component.flashrnn_slstm as F
        F._flashrnn_backend = lambda device: "vanilla"
    except Exception:
        pass
    from tirex2 import load_model
    return load_model(BASE["tirex2"]["repo"], device=DEV)


def load_turk():
    """Public LoRA adapter of chronos-2, merged into the base model."""
    import peft
    pipe = load_chronos("chronos2")
    spec = BASE["turk"]
    pipe.model = peft.PeftModel.from_pretrained(pipe.model, spec["repo"],
                                                revision=spec["revision"]) \
                     .merge_and_unload().eval()
    return pipe


def load_toto():
    from toto2 import Toto2Model
    spec = BASE["toto_25b_ft"]
    return Toto2Model.from_pretrained(spec["repo"], revision=spec["revision"]).to(DEV).eval()


def load_timesfm(max_horizon):
    import timesfm
    spec = BASE["timesfm"]
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(spec["repo"], revision=spec["revision"])
    m.compile(timesfm.ForecastConfig(max_context=2048, max_horizon=max(64, max_horizon),
                                     normalize_inputs=True, use_continuous_quantile_head=True,
                                     force_flip_invariance=True, infer_is_positive=True,
                                     fix_quantile_crossing=True))
    return m


# ---------------------------------------------------------------- forecasting recipes
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


def quantiles_tirex(model, ctxs, h):
    out = []
    with torch.no_grad():
        for s in range(0, len(ctxs), 256):
            q, _ = model.forecast(
                context=[np.asarray(c, dtype=np.float32) for c in ctxs[s:s + 256]],
                prediction_length=h, output_type="numpy", batch_size=256)
            out.append(np.sort(np.asarray(q).transpose(0, 2, 1), axis=1))   # (b, h, 9) -> (b, 9, h)
    return np.concatenate(out)


def quantiles_tirex2(model, ctxs, h, max_h=320):
    """Native horizon cap 320: autoregressive chunks, median fed back."""
    from tirex2 import TimeseriesType
    def bloc(cs, hh):
        qs = []
        for i in range(0, len(cs), 256):
            ts = [TimeseriesType(target=torch.as_tensor(np.asarray(c, dtype=np.float32)).unsqueeze(0),
                                 past_covariates=None, future_covariates=None) for c in cs[i:i + 256]]
            out = model.forecast(ts, prediction_length=hh, output_type="numpy")
            qs.extend(np.asarray(x)[0] for x in out)
        return np.stack(qs)                                                 # (n, 9, hh)
    cur, parts, left = [np.asarray(c, dtype=np.float32) for c in ctxs], [], h
    while left > 0:
        hh = min(left, max_h)
        q = bloc(cur, hh)
        parts.append(q)
        left -= hh
        if left > 0:
            cur = [np.concatenate([c, q[i, 4]]) for i, c in enumerate(cur)]
    return np.sort(np.concatenate(parts, axis=2), axis=1)


def quantiles_toto(model, ctxs, h, max_h=128):
    """Univariate Toto serving (native quantiles); horizon cap 128: AR chunks, median fed back."""
    patch = int(getattr(model.config, "patch_size", 64) or 64)

    def fn(cs, hh):
        x = torch.stack([torch.tensor(np.asarray(c, dtype=np.float32), device=DEV)[None]
                         for c in cs])
        T = x.shape[-1]
        x = (torch.cat([x[..., :1].expand(*x.shape[:-1], patch - T), x], dim=-1) if T < patch
             else x[..., -((T // patch) * patch):])
        inp = {"target": x, "series_ids": torch.zeros(len(cs), 1, dtype=torch.long, device=DEV),
               "target_mask": torch.ones_like(x, dtype=torch.bool)}
        with torch.no_grad():
            o = model.forecast(inp, horizon=hh)
        return np.asarray(o.detach().float().cpu())[:, :, 0, :].transpose(1, 0, 2)   # (b, 9, hh)

    def ar(lot):
        parts, left = [], h
        while left > 0:
            hh = min(left, max_h)
            q = fn(lot, hh)
            parts.append(q)
            left -= hh
            if left > 0:
                lot = [np.concatenate([c, q[i, 4]]) for i, c in enumerate(lot)]
        return np.concatenate(parts, axis=2)

    # Toto batches require equal lengths: group windows by context length
    out = [None] * len(ctxs)
    groupes = {}
    for i, c in enumerate(ctxs):
        groupes.setdefault(len(c), []).append(i)
    for idx in groupes.values():
        for s in range(0, len(idx), 16):
            part = idx[s:s + 16]
            q = ar([np.asarray(ctxs[i], dtype=np.float32) for i in part])
            for j, i in enumerate(part):
                out[i] = q[j]
    return np.sort(np.stack(out), axis=1)


def quantiles_timesfm(model, ctxs, h):
    out = []
    for s in range(0, len(ctxs), 64):
        _, q = model.forecast(horizon=h,
                              inputs=[np.asarray(c, dtype=np.float32) for c in ctxs[s:s + 64]])
        out.append(np.asarray(q)[:, :h, 1:].transpose(0, 2, 1))   # (b, h, 10) -> (b, 9, h)
    return np.concatenate(out)


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
        return peft.PeftModel.from_pretrained(base, path).merge_and_unload().eval()
    raise ValueError(kind)


# ---------------------------------------------------------------- forecasting
def contexts_for(config):
    from gift_eval.data import Dataset
    from gluonts.time_feature import get_seasonality
    name, term = config.rsplit("/", 1)
    name = NAME_MAP.get(name, name)
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
        if kind == "tirex":
            return quantiles_tirex(model, ctxs, h)
        if kind == "toto":
            return quantiles_toto(model, ctxs, h)
        raise ValueError(kind)
    if member == "chronos2":
        return quantiles_chronos(load_chronos("chronos2"), ctxs, h)
    if member == "turk":
        return quantiles_chronos(load_turk(), ctxs, h)
    if member == "tirex":
        return quantiles_tirex(load_tirex(), ctxs, h)
    if member == "tirex2":
        return quantiles_tirex2(load_tirex2(), ctxs, h)
    if member in ("toto_25b_ft", "toto_uni"):
        return quantiles_toto(load_toto(), ctxs, h)
    if member == "timesfm":
        return quantiles_timesfm(load_timesfm(h), ctxs, h)
    raise ValueError(f"unknown member: {member}")


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
