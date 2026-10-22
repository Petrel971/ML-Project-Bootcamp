"""Vérifications numériques du CART from scratch. Lancer : pytest"""

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from src.cart import CARTClassifier, gini


def test_gini_valeurs_connues():
    assert gini(np.array([0, 0, 0, 0])) == pytest.approx(0.0)
    assert gini(np.array([0, 0, 1, 1])) == pytest.approx(0.5)
    assert gini(np.array([0, 1, 1, 1])) == pytest.approx(0.375)


def test_premier_split_identique_a_sklearn():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = (X[:, 1] > 0.3).astype(int)
    ours = CARTClassifier(max_depth=1).fit(X, y)
    ref = DecisionTreeClassifier(max_depth=1).fit(X, y)
    assert ours.root_.feature == ref.tree_.feature[0]
    assert ours.root_.threshold == pytest.approx(ref.tree_.threshold[0])


def test_arbre_profond_apprend_le_train():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 2))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    model = CARTClassifier().fit(X, y)
    assert (model.predict(X) == y).all()
