# TW3Cast paper, delivery notes (2026-09-15)

Author: Nathan Thierry (TW3 Partners). Format: house arXiv format (11pt article, natbib
author-year, run-in results, Problem statement, Threats to validity, Reproducibility), same
template as the GEO Score and OCR-routing papers.

## What is in this folder

| File | Role |
|---|---|
| `main.tex`, `refs.bib`, `numbers.tex`, `tables/*.tex`, `figures/*.pdf` | the paper source (all of it goes to arXiv) |
| `main.pdf` (16 pages), `main.bbl` | build outputs; `main.bbl` is shipped so arXiv needs no bibtex |
| `make_figures.py` | regenerates every figure, table and number from `data/` in a few seconds |
| `data/` | the artifacts: released router and expert index, the submitted TW3Cast score file, the tournament-only grid, the dated snapshot of the 129 public per-configuration score files and their description files |
| `arxiv/` and `TW3Cast_arxiv.tar.gz` | exactly what to upload to arXiv (source, not the PDF) |
| `ARXIV_FORM.md` | the form fields to paste (title, author, abstract under 1,920 characters, comments, categories) |

Build: `tectonic --keep-intermediates main.tex` (arXiv compiles with pdflatex; line 2 of
`main.tex` declares it). Regenerate: `python3 make_figures.py` then rebuild.

## What is measured and where it comes from

Every number in the paper is produced by `make_figures.py` from `data/`:

- Leaderboard standing (Table 1, Figure 5, abstract, Results 1): the 129 public
  `all_results.csv` files of the GIFT-Eval space downloaded on 2026-09-14 (VISIT-2.0 has no
  score file and is excluded), plus `data/tw3cast_all_results.csv` (the submitted file).
  Ranking rule: per configuration, rank by MASE (ties share the lowest rank), then mean over
  the 97 configurations. Independent recomputation gives position 3 of 130 and mean rank
  19.4, identical to the author's own top-10 file (`data/leaderboard_top10_MASE_rank_2026-09-14.csv`).
- Pool members (Table 2) and per-configuration oracle (Figure 6, Result 2): same field.
- Ablation (Table 3, Result 3): `data/tournament_only_per_config.csv` inserted into the same
  field. Recomputed 38.0 on all 97 and 19.8 on the 27 tournament configurations, versus 37.0
  and 19.7 in the author's `ablation_*.csv` files. The difference is the field (the author's
  public field is slightly different); the paper uses one field for every table.
- Router composition (Figure 2, Section 4, Appendix B): `data/router.csv`, `data/experts.json`.

Prose lint (`prose_lint.py main.tex`): clean. Build: 0 undefined references, 0 overfull boxes.

## Before uploading (author checklist)

1. Make the GitHub repository public (it is private today; the paper links it) and add this
   folder to it as `paper/` so that the Reproducibility section is true. A pull request with
   the folder is the simplest way.
2. `router.csv` uses the member `toto_uni` (ett1/H/medium, ett2/H/medium, and once in the
   tournament grid) which is not in `base_models.json` and not handled by `predict.py`. Add it
   to `base_models.json` (repo and revision). The appendix currently describes it as "the
   univariate serving of Toto 2.0 2.5B FT"; correct that sentence in `main.tex` if it is
   something else.
3. `predict.py` implements the Chronos-2 forecast path in full and raises
   `NotImplementedError` for TiRex, Toto and TimesFM members. The paper says the forecast
   call of each family is the standard one of its library; finishing the three calls in
   `predict.py` is a small job and makes the sentence unquestionable.
4. The model release URL in the README (huggingface.co/TW3-Partners/TW3Cast) returns 401 and
   the organisation page 404. The paper only says "the model release linked from the
   repository"; make that link live before upload.
5. Submit the score file to the GIFT-Eval leaderboard, or keep the paper's wording (which
   says the rank is computed by inserting the submitted file into a dated snapshot).

## Claims kept qualitative on purpose

Four claims have no released table and the paper says so in Threats to validity: the temporal
versus random meta-backtest ordering, the preparation ablation on a fixed base model,
cross-architecture blends versus single-family seed ensembles, and the per-window gate switch
counts. If the tables exist, drop them in `data/` and the corresponding paragraphs can carry
numbers; nothing else in the paper depends on them.

## Title and abstract

The earlier title ("Logits, not weights") described the table as logits; it stores mixture
weights, so the title was changed. The new title names the system, the task, the benchmark,
the method and its key property, with no rank in the title (ranks age). The rank is the first
sentence of the abstract.
