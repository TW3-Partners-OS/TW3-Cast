# TW3Cast

Time-series forecasting system evaluated on [GIFT-Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval)
(97 configurations) — **MASE_Rank: 3rd as of 2026-09-14**, best non-agentic system on the
leaderboard, with **zero test-data leakage**: no component was trained on data overlapping the
benchmark's test sets (no agents, no LLM at inference: a lightweight per-configuration router
over specialized models).

## How it works

**Full technical report: [REPORT.md](REPORT.md)** (method, diagrams, protocol).


For each benchmark configuration, the system serves one of:

1. **A specialized model** — a fine-tune or LoRA of a public foundation model, trained only on
   the configuration's train split. Training is never naive: train series are cleaned
   (degenerate series and scale outliers excluded), data-poor configurations borrow the train
   splits of sibling configurations with target over-weighting, and training runs on sampled
   windows with robustly normalized residuals and a pinball loss over the 9 quantiles used by
   the benchmark. Three base architectures are specialized (Chronos-2, TiRex, and Toto 2.5B via
   LoRA adapters), and cross-architecture quantile blends are used where they win.
2. **A selection tournament** — where no specialist wins, a multi-window backtest played
   exclusively on the train split selects the served model among the base pool (asymmetric
   margin against locally trained candidates, probabilistic tie-breaking), with a per-window
   fallback gate at prediction time.

All decisions are taken on a backtest **extracted from the train split**; a specialist is
designated for a configuration as soon as it beats the tournament on that backtest.
Specialists are never naive fine-tunes — the data preparation is the core of the method
(see [REPORT.md](REPORT.md)).

The system is shipped as a **frozen router**: `router.csv` gives, for every configuration,
the mixture weights over experts (one-hot for a single designated expert, fractional for
blends, `__tournament__` for configurations resolved by the train-side tournament).
`experts.json` lists each expert's architecture family; expert checkpoints are shipped in
the model release under their expert id.

## Using the model

- **Base models** are public; download them at the pinned revisions listed in
  `base_models.json`.
- **Trained artifacts** (fine-tuned checkpoints and LoRA adapters) are distributed in the
  project's model release:
  [huggingface.co/TW3-Partners/TW3Cast](https://huggingface.co/TW3-Partners/TW3Cast).
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

**MASE_Rank: 3rd, as of 2026-09-14.** First non-agentic system — only two LLM-agent pipelines
rank above it — with zero test-data leakage. Per-configuration figures are in the GIFT-Eval
submission (`results/TW3Cast/all_results.csv` on the benchmark repository).

## License & contact

Code and artifacts by TW3 Partners. Base models remain under their original licenses.
