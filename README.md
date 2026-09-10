# TW3Cast

Time-series forecasting system evaluated on [GIFT-Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval)
(97 configurations) — **MASE_Rank: 2nd**, best non-agentic system on the leaderboard
(no agents, no LLM at inference: a lightweight per-configuration router over specialized models).

## How it works

For each benchmark configuration, the system serves one of:

1. **A specialized model** — a fine-tune or LoRA of a public foundation model, trained only on
   the configuration's train split. Training is never naive: train series are cleaned
   (degenerate series and scale outliers excluded), data-poor configurations borrow the train
   splits of sibling configurations with target over-weighting, and training runs on sampled
   windows with robustly normalized residuals and a pinball loss over the 9 quantiles used by
   the benchmark. Four base architectures are specialized (Chronos-2, TiRex, Chronos-Bolt,
   Toto 2.5B via LoRA adapters), and cross-architecture quantile blends are used where they win.
2. **A selection tournament** — where no specialist wins, a multi-window backtest played
   exclusively on the train split selects the served model among the base pool (asymmetric
   margin against locally trained candidates, probabilistic tie-breaking), with a per-window
   fallback gate at prediction time.

The system is shipped as a **frozen router**: `router.csv` gives, for every configuration,
the mixture weights over experts (one-hot for a single designated expert, fractional for
blends, `__tournament__` for configurations resolved by the train-side tournament).
`experts.json` lists each expert's architecture family; expert checkpoints are shipped in
the model release under their expert id.

## Using the model

- **Base models** are public; download them at the pinned revisions listed in
  `base_models.json`.
- **Trained artifacts** (fine-tuned checkpoints and LoRA adapters) are distributed in the
  project's model release (link to come in this README).
- `predict.py` shows how to reload each expert type and how to reproduce the served
  forecast of any configuration: it composes the experts with the weights of `router.csv`
  (weighted quantile mean, sorted) and outputs the 9 quantiles [0.1 … 0.9], shape
  `(n_windows, 9, horizon)`.

```bash
pip install -r requirements.txt
python predict.py --config "electricity/H/short" --artifacts <path-to-artifacts>
```

Evaluation follows the official GIFT-Eval harness (`gluonts.model.evaluate_forecasts`,
`axis=None`, `mask_invalid_label=True`, `allow_nan_forecast=False`).

## Results

MASE_Rank 2nd on the leaderboard; first non-agentic system; ahead of all published LLM-agent
systems except one, and roughly ten mean-rank points ahead of the best public foundation
model served alone.

## License & contact

Code and artifacts by TW3 Partners. Base models remain under their original licenses.
