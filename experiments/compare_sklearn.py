"""Comparaison CART from scratch vs sklearn.tree.DecisionTreeClassifier.

Même splits, mêmes hyperparamètres, mêmes métriques. À lancer depuis la racine :
    python -m experiments.compare_sklearn
"""

import time

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier

from src.cart import CARTClassifier
from src.data import SEED, get_splits

PARAMS = dict(criterion="gini", max_depth=8, min_samples_split=2, min_samples_leaf=1)


def evaluate(model, X_train, y_train, X_eval, y_eval):
    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start
    y_pred = model.predict(X_eval)
    return {
        "fit_time_s": fit_time,
        "accuracy": accuracy_score(y_eval, y_pred),
        "precision": precision_score(y_eval, y_pred),
        "recall": recall_score(y_eval, y_pred),
        "f1": f1_score(y_eval, y_pred),
        "confusion": confusion_matrix(y_eval, y_pred),
        "y_pred": y_pred,
    }


def main():
    splits, _ = get_splits()
    args = (splits["X_train"], splits["y_train"], splits["X_val"], splits["y_val"])
    ours = evaluate(CARTClassifier(**PARAMS), *args)
    ref = evaluate(DecisionTreeClassifier(random_state=SEED, **PARAMS), *args)

    for name, res in (("from scratch", ours), ("scikit-learn", ref)):
        print(f"--- {name}")
        for k in ("fit_time_s", "accuracy", "precision", "recall", "f1"):
            print(f"{k:11s} {res[k]:.4f}")
        print(res["confusion"])
    agreement = (ours["y_pred"] == ref["y_pred"]).mean()
    print(f"Accord des prédictions : {agreement:.2%}")

    # TODO : courbe train/val selon max_depth, importance des variables, évaluation finale sur le test.


if __name__ == "__main__":
    main()
