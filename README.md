# Prédiction du revenu (Adult) avec un arbre CART from scratch

Projet de machine learning (cours d'Alassane Diallo).
Équipe : Lucas, Manuel, Grégoire, Amine.

## Objectif

Classification binaire : prédire si le revenu annuel d'une personne dépasse 50 000 $
à partir de ses caractéristiques socio-démographiques (recensement américain de 1994).

Question de recherche (provisoire) : quels facteurs déterminent un revenu supérieur à 50K $,
et un arbre de classification CART implémenté from scratch atteint-il les mêmes
performances que `DecisionTreeClassifier` de scikit-learn sur les mêmes splits ?

## Données

- Kaggle : https://www.kaggle.com/datasets/uciml/adult-census-income
- Source d'origine : UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/2
- Chargement dans le code : OpenML (`fetch_openml(name="adult", version=2)`), mis en cache dans `data/`
  (dossier ignoré par git, les données ne sont pas redistribuées).

## Structure

```
src/data.py                     chargement, encodage, split train/val/test (graine 42)
src/cart.py                     arbre CART from scratch (NumPy)
experiments/compare_sklearn.py  comparaison avec scikit-learn sur les mêmes splits
tests/test_cart.py              vérifications numériques (Gini, premier split vs sklearn)
report/                         rapport PDF
```

## Reproduire les résultats

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.data                     # télécharge Adult et affiche la taille des splits
pytest                                 # vérifications de l'implémentation
python -m experiments.compare_sklearn  # comparaison from scratch vs scikit-learn
```

Graine aléatoire : `SEED = 42` dans `src/data.py`.
