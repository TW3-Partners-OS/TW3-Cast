# arXiv, champs a copier-coller

Un bloc = un champ du formulaire. Copier tel quel. Televerser `TW3Cast_arxiv.tar.gz`
(le source LaTeX, pas le PDF) : arXiv compile avec pdflatex, `main.bbl` est inclus.

## Title

TW3Cast: A Frozen Router of Lightly Fine-Tuned Foundation Models for Time-Series Forecasting on GIFT-Eval, Selected Entirely on the Training Split

## Author(s)

Prenom d'abord. Sans accent sur ce champ.

Nathan Thierry

## Abstract

1833 caracteres (arXiv refuse au-dela de 1920). Identique a l'abstract du PDF.

TW3Cast is a time-series forecasting system that reaches position 3 of 130 entries on the GIFT-Eval benchmark by mean MASE rank, as of 2026-09-14. The two entries above it belong to the leaderboard's agentic category, multi-step systems that use agents or language models to reason about, generate or select forecasts, and both run a large language model. TW3Cast runs none. Its selection is a table computed once on the training split and then frozen, its experts are public foundation models lightly fine-tuned on those training splits. For each of the 97 dataset, frequency and horizon configurations, the table serves one of four modes: a specialist, which is a LoRA or full fine-tune of Chronos-2, TiRex or Toto whose training data was cleaned and enriched by explicit rules; a quantile blend that contains a specialist; a blend of base models; or a selection tournament played on a backtest carved from the training split. Every decision in the table was taken on that backtest. A specialist is admitted the moment it beats the tournament there, so a candidate costs a few megabytes and minutes of GPU time, and a failed candidate changes nothing. Three guarded mechanisms protect the selection from its own biases: a dual accuracy and calibration criterion, an asymmetric margin against candidates that saw the series during training, and conservative per-window gates. The selection rules themselves were chosen inside a temporal meta-backtest. The best base model served alone reaches a mean MASE rank of 33.8, the tournament served on every configuration reaches 38.0, and the full router reaches 19.4. The routing table, the expert index, the pinned base-model revisions, the submitted score file and the dated snapshot of the public scores are released, and every number in this paper regenerates from them by one script.

## Comments

16 pages, 6 figures, 6 tables. Code, routing table and score files: https://github.com/TW3-Partners-OS/TW3-Cast

## Categories

Primaire : cs.LG
Croisees : stat.ML

## Report number / Journal ref / DOI / ACM / MSC

Laisser vide.

## License

arXiv.org perpetual, non-exclusive license (choix par defaut), ou CC BY 4.0 si TW3 Partners
veut autoriser la reutilisation.
