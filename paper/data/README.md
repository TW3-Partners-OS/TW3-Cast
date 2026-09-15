# data/ — inputs of make_figures.py

- `public_scores.csv` — per-configuration MASE and WQL of the 129 public GIFT-Eval entries with a complete
  `all_results.csv`, downloaded from the `results/` folder of huggingface.co/spaces/Salesforce/GIFT-Eval
  (date in SNAPSHOT_DATE.txt). VISIT-2.0 has no result file and is excluded.
- `public_configs/*.json` — the `config.json` of each entry (declared type and test-data leakage).
- `router.csv`, `experts.json`, `base_models.json` — copied from the TW3-Cast release repository.
- `reported_standing.csv` — the top of the leaderboard with TW3Cast inserted, as computed by the author on
  2026-09-14 from the public per-configuration scores. Used by make_figures.py ONLY while
  `tw3cast_all_results.csv` is absent.
- `tw3cast_all_results.csv` — (to add) the submission file of TW3Cast in the GIFT-Eval format. When present,
  make_figures.py recomputes the whole standing from data and ignores reported_standing.csv.
