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


def test_naive_equals_vectorised():
    from src.cart import best_split, best_split_naive
    rng = np.random.default_rng(2)
    for _ in range(20):
        X = rng.integers(0, 8, size=(80, 4)).astype(float)
        y = rng.integers(0, 2, size=80)
        naive, fast = best_split_naive(X, y, 3), best_split(X, y, 2, 3)
        assert naive[2] == pytest.approx(fast[2])


def test_same_predictions_and_importances_as_sklearn():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(500, 5))
    y = (X[:, 0] + X[:, 1] ** 2 + rng.normal(scale=0.5, size=500) > 1).astype(int)
    for depth in (1, 2, 3, 4):
        ours = CARTClassifier(max_depth=depth).fit(X, y)
        ref = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X, y)
        assert (ours.predict(X) == ref.predict(X)).all()
        assert ours.feature_importances_ == pytest.approx(ref.feature_importances_)


def test_pruning_matches_sklearn_leaf_count():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(400, 3))
    y = (X[:, 0] * X[:, 1] + rng.normal(scale=0.3, size=400) > 0).astype(int)
    ours = CARTClassifier(ccp_alpha=0.01).fit(X, y)
    ref = DecisionTreeClassifier(ccp_alpha=0.01, random_state=0).fit(X, y)
    assert ours.get_n_leaves() == ref.get_n_leaves()
