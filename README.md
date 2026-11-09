# Prédiction du revenu (Adult) avec un arbre CART from scratch

Projet de machine learning (cours d'Alassane Diallo).
Équipe : Lucas Petrelluzzi, Manuel Andrade, Amine Ennadzim.

## Objectif

Classification binaire : prédire si le revenu annuel d'une personne dépasse 50 000 $
à partir de ses caractéristiques socio-démographiques (recensement américain de 1994).

Question de recherche : l'éducation est-elle le principal déterminant d'un revenu supérieur à
50K $, ou d'autres facteurs (situation familiale, âge, temps de travail) pèsent-ils davantage ?

Résultat : notre CART donne les mêmes métriques que scikit-learn sur le test (accuracy 85,7 %,
F1 0,669, AUC 0,906 ; 99,98 % de prédictions identiques). La situation familiale arrive en
premier, l'éducation en second. Rapport : `report/report.pdf`.

## Données

- Kaggle : https://www.kaggle.com/datasets/uciml/adult-census-income
- Source d'origine : UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/2
- Chargement dans le code : OpenML (`fetch_openml(name="adult", version=2)`), mis en cache dans `data/`.
  Ce sont **les mêmes données** que le CSV Kaggle (même extraction UCI de 48 842 lignes, train + test
  d'origine réunis) ; OpenML permet un téléchargement automatique sans compte Kaggle
  (dossier ignoré par git, les données ne sont pas redistribuées).

## Structure

```
notebooks/01_cart_from_scratch.ipynb   construction du CART pas à pas
notebooks/02_experiments.ipynb         expériences et réponse à la question
src/data.py                            chargement et préparation des données
src/cart.py                            arbre CART from scratch
tests/test_cart.py                     tests du CART
report/report.pdf                      rapport (généré par report/build_report.py)
report/figures/                        figures écrites par le notebook 02
CONTEXT.md                             contexte complet du projet et des décisions
PLAN.md                                plan du projet
pyproject.toml                         dépendances
uv.lock                                versions exactes des dépendances
```

## Reproduire les résultats

Les dépendances sont gérées avec [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock`
pour des versions identiques chez tout le monde).

```bash
uv sync                                       # crée .venv et installe les versions du uv.lock
uv run python -m src.data                     # télécharge Adult et affiche la taille des splits
uv run pytest                                 # vérifications de l'implémentation
# puis exécuter notebooks/01 puis notebooks/02 du début à la fin (kernel .venv)
uv run --with reportlab python report/build_report.py   # régénère le rapport PDF
```

Ajouter une dépendance : `uv add <paquet>` (jamais `pip install`).

Graine aléatoire : `SEED = 42` dans `src/data.py`.
