"""Arbre de classification CART implémenté from scratch (NumPy uniquement).

À compléter par l'équipe. Les signatures suivent celles de
sklearn.tree.DecisionTreeClassifier pour faciliter la comparaison.

Composants exigés par la consigne (section 3.1) :
- formulation du modèle : règle de décision x[feature] <= threshold à chaque nœud ;
- fonction objectif : impureté de Gini (et entropie en option) ;
- entraînement : recherche exhaustive du meilleur split + construction récursive ;
- prédiction : parcours de l'arbre jusqu'à une feuille.
"""

from dataclasses import dataclass

import numpy as np


def gini(y):
    """Impureté de Gini d'un vecteur d'étiquettes : 1 - somme_k p_k^2."""
    raise NotImplementedError


def entropy(y):
    """Entropie d'un vecteur d'étiquettes : - somme_k p_k log2 p_k."""
    raise NotImplementedError


@dataclass
class Node:
    """Nœud de l'arbre. Une feuille a feature = None."""
    feature: int | None = None
    threshold: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None
    proba: np.ndarray | None = None  # proportion de chaque classe dans le nœud
    n_samples: int = 0
    impurity: float = 0.0


class CARTClassifier:
    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2,
                 min_samples_leaf=1):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root_ = None
        self.n_classes_ = None
        self.feature_importances_ = None

    def fit(self, X, y):
        """Construit l'arbre sur (X, y). Renvoie self."""
        raise NotImplementedError

    def _best_split(self, X, y):
        """Renvoie (feature, threshold, gain) du meilleur split, ou None.

        Conseil : pour chaque variable, trier les valeurs puis utiliser des
        sommes cumulées pour évaluer tous les seuils en une passe vectorisée.
        Seuil = milieu entre deux valeurs consécutives distinctes (comme sklearn).
        """
        raise NotImplementedError

    def _build(self, X, y, depth):
        """Construction récursive ; s'arrête sur max_depth, min_samples ou nœud pur."""
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
