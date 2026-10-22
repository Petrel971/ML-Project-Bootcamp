"""Chargement et préparation du dataset Adult (Census Income).

Source : UCI Adult, via OpenML (name="adult", version=2).
Kaggle : https://www.kaggle.com/datasets/uciml/adult-census-income

Les deux implémentations (CART from scratch et scikit-learn) reçoivent
exactement les mêmes matrices produites par ce module.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

SEED = 42
DATA_HOME = "data"

# fnlwgt : poids d'échantillonnage du recensement, pas une caractéristique de la personne.
# education : redondante avec education-num (même information, encodée en entier).
DROPPED_COLUMNS = ["fnlwgt", "education"]


def load_adult(data_home=DATA_HOME):
    """Télécharge (ou relit depuis le cache) Adult et renvoie X (DataFrame) et y (0/1)."""
    adult = fetch_openml(name="adult", version=2, as_frame=True, data_home=data_home)
    X = adult.data.drop(columns=DROPPED_COLUMNS)
    y = (adult.target.astype(str).str.strip() == ">50K").astype(int).to_numpy()
    return X, y


def preprocess(X_train, X_other):
    """One-hot encode les variables catégorielles.

    Les valeurs manquantes deviennent une modalité "Unknown". Les colonnes sont
    déterminées sur le train uniquement, puis appliquées aux autres jeux
    (pas de fuite d'information du test vers le train). Pas de normalisation :
    un arbre de décision est insensible à l'échelle des variables.
    """
    def encode(df):
        df = df.copy()
        cat_cols = df.select_dtypes(exclude="number").columns
        for col in cat_cols:
            df[col] = df[col].astype(object).fillna("Unknown").astype(str).str.strip()
            df[col] = df[col].replace("?", "Unknown")
        return pd.get_dummies(df, columns=list(cat_cols), dtype=float)

    train_enc = encode(X_train)
    others_enc = [encode(X).reindex(columns=train_enc.columns, fill_value=0.0) for X in X_other]
    return train_enc, others_enc


def get_splits(test_size=0.2, val_size=0.2, seed=SEED):
    """Split stratifié train / validation / test (60/20/20 par défaut).

    Renvoie un dict de tableaux NumPy et la liste des noms de variables.
    """
    X, y = load_adult()
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_size / (1 - test_size), stratify=y_tmp, random_state=seed
    )
    X_train, (X_val, X_test) = preprocess(X_train, [X_val, X_test])
    feature_names = list(X_train.columns)
    splits = {
        "X_train": X_train.to_numpy(), "y_train": y_train,
        "X_val": X_val.to_numpy(), "y_val": y_val,
        "X_test": X_test.to_numpy(), "y_test": y_test,
    }
    return splits, feature_names


if __name__ == "__main__":
    splits, names = get_splits()
    for key in ("train", "val", "test"):
        X, y = splits[f"X_{key}"], splits[f"y_{key}"]
        print(f"{key:5s}: {X.shape[0]:6d} lignes, {X.shape[1]} variables, {y.mean():.1%} de >50K")
