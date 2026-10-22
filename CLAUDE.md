# Projet ML : Adult + CART from scratch

Projet de groupe (Lucas, Manuel, Grégoire, Amine) pour le cours « Mathematics & Statistics for
Machine Learning » d'Alassane Diallo (Albert School, MSc 1). Réponds en français.

## Dates et barème
- Rendu (rapport PDF + code) : 6 novembre 2026 au plus tard. Oral : 13 novembre 2026.
- Barème : rapport 20 %, implémentation from scratch 25 %, expériences et analyse 25 %, oral 15 %, questions 15 %.

## Règles de la consigne
- Le CART (`src/cart.py`) est écrit avec NumPy uniquement. scikit-learn ne sert qu'à la comparaison,
  aux métriques et au découpage (`train_test_split`).
- Mêmes splits et mêmes hyperparamètres pour les deux implémentations.
- Hyperparamètres choisis sur la validation, jamais sur le test. Pas de fuite de données.
- Ne pas committer les données (`data/` est ignoré) : on donne l'URL Kaggle
  https://www.kaggle.com/datasets/uciml/adult-census-income
- Graine unique `SEED = 42` (dans `src/data.py`).

## Façon de travailler
- On code ensemble : Claude peut écrire le code, mais on réfléchit d'abord à l'approche, et chaque
  choix de `src/cart.py` est expliqué (maths du cours, pourquoi ce choix) pour que le groupe puisse
  tout défendre à l'oral.
- Suivre le cours (Lecture 3B) : Gini G = 1 − Σ p_k², score d'un split
  G_split = (n_L/n)·G_L + (n_R/n)·G_R, construction gloutonne, feuille = classe majoritaire.
- Style du Lab 1 : code vectorisé, vérifications par `assert` / tests.
- Avant de dire que quelque chose marche, lancer `pytest` et le script concerné.

## Commandes
Dépendances gérées avec uv (`pyproject.toml` + `uv.lock`). Jamais `pip install` : `uv add <paquet>`.
```bash
uv sync                                       # installe l'environnement
uv run python -m src.data                     # télécharge Adult (OpenML) et affiche les splits
uv run pytest                                 # tests du CART
uv run python -m experiments.compare_sklearn  # comparaison avec scikit-learn
```

## Git
- Commits, PR et push au nom de Lucas uniquement : aucune mention de Claude (pas de
  `Co-Authored-By`, pas de lien de session, pas de « Generated with Claude Code », pas de
  branche `claude/...`).
- Une branche par tâche, fusion via pull request.

## Sources du cours
Les PDF du cours et la consigne sont dans `docs/` (non versionné, à ajouter localement).
