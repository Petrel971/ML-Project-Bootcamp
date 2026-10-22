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
- `src/cart.py` est écrit par le groupe : il doit pouvoir l'expliquer à l'oral. Sur ce fichier,
  explique, guide, relis et propose des tests, mais n'écris pas l'algorithme à leur place sauf si on te le demande explicitement.
- Suivre le cours (Lecture 3B) : Gini G = 1 − Σ p_k², score d'un split
  G_split = (n_L/n)·G_L + (n_R/n)·G_R, construction gloutonne, feuille = classe majoritaire.
- Style du Lab 1 : code vectorisé, vérifications par `assert` / tests.
- Avant de dire que quelque chose marche, lancer `pytest` et le script concerné.

## Commandes
```bash
pip install -r requirements.txt
python -m src.data                     # télécharge Adult (OpenML) et affiche les splits
pytest                                 # tests du CART
python -m experiments.compare_sklearn  # comparaison avec scikit-learn
```

## Sources du cours
Les PDF du cours et la consigne sont dans `docs/` (non versionné, à ajouter localement).
