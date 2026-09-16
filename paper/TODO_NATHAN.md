# TODO Nathan, papier TW3Cast (16/09/2026)

Le papier est fini et compilé (`main.pdf`, 16 pages). Le paquet arXiv est `TW3Cast_arxiv.tar.gz`
(source LaTeX avec `main.bbl` et figures PDF ; c'est le source qu'on téléverse, pas le PDF).
Les champs du formulaire sont dans `ARXIV_FORM.md`. Tout chiffre du papier se régénère avec
`python3 make_figures.py` depuis `data/`, qui contient tes cinq fichiers de `results_support/`
à l'identique.

Dans l'ordre :

1. **Rendre le dépôt public** : https://github.com/TW3-Partners-OS/TW3-Cast (le papier le cite).
2. **Fusionner la PR #1** (dossier `paper/` avec le source, les données et le script) :
   https://github.com/TW3-Partners-OS/TW3-Cast/pull/1
3. **Ajouter `toto_uni` à `base_models.json`** (dépôt et révision) : `router.csv` le sert sur
   ett1/H/medium et ett2/H/medium.
4. **Compléter `predict.py`** pour les membres TiRex, Toto et TimesFM (aujourd'hui seul
   Chronos-2 est implémenté, les autres lèvent `NotImplementedError`).
5. **Release Hugging Face** : rendre vivant le lien du README (huggingface.co/TW3-Partners/TW3Cast
   renvoie 401) ou retirer le lien et déposer les checkpoints dans le dépôt.
6. **Soumettre sur arXiv** : téléverser `TW3Cast_arxiv.tar.gz`, coller les champs de
   `ARXIV_FORM.md` (cs.LG, croisé stat.ML).
7. Optionnel, après publication : soumettre le fichier `data/tw3cast_all_results.csv` au
   leaderboard GIFT-Eval sous le nom TW3Cast, pour que la position publique rejoigne celle du
   papier.
