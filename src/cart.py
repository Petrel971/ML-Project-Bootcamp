"""CART classification tree implemented from scratch (NumPy only).

The signatures follow sklearn.tree.DecisionTreeClassifier so that both models
can be compared on the same splits.

Components required by the brief (section 3.1):
- model: a decision rule x[feature] <= threshold at each internal node,
  a class distribution at each leaf;
- objective: Gini impurity (entropy as an option);
- training: exhaustive search of the best split + greedy recursive growth;
- prediction: each observation goes down the tree to a leaf.
"""

from dataclasses import dataclass

import numpy as np

# Two values closer than this are treated as equal (same rule as scikit-learn),
# so that no threshold is placed between two floats that only differ by rounding.
FEATURE_THRESHOLD = 1e-7
EPSILON = np.finfo(np.float64).eps


# ---------------------------------------------------------------------------
# Impurity measures
# ---------------------------------------------------------------------------

def gini(y):
    """Gini impurity of a label vector: 1 - sum_k p_k^2."""
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    p = np.bincount(y) / y.size
    return 1.0 - np.sum(p ** 2)


def entropy(y):
    """Entropy of a label vector: - sum_k p_k log2 p_k."""
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    p = np.bincount(y) / y.size
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _impurity_from_counts(counts, criterion):
    """Impurity computed from class counts. `counts` has shape (..., n_classes),
    so the same function scores one node or every candidate threshold at once."""
    n = counts.sum(axis=-1, keepdims=True)
    p = np.divide(counts, n, out=np.zeros_like(counts, dtype=float), where=n > 0)
    if criterion == "gini":
        return 1.0 - np.sum(p ** 2, axis=-1)
    if criterion == "entropy":
        logp = np.log2(p, out=np.zeros_like(p), where=p > 0)
        return -np.sum(p * logp, axis=-1)
    raise ValueError(f"unknown criterion: {criterion!r}")


def split_score(y_left, y_right, criterion="gini"):
    """Weighted impurity of a split: n_L/n * G_L + n_R/n * G_R."""
    impurity = gini if criterion == "gini" else entropy
    n_left, n_right = len(y_left), len(y_right)
    n = n_left + n_right
    return n_left / n * impurity(y_left) + n_right / n * impurity(y_right)


# ---------------------------------------------------------------------------
# Best split search
# ---------------------------------------------------------------------------

def best_split_naive(X, y, min_samples_leaf=1, criterion="gini"):
    """Reference version with plain loops: for every feature and every midpoint,
    split the data and score it. O(d * n^2). Returns (feature, threshold, score)
    or None. Used to check the vectorised version, not for training."""
    n, d = X.shape
    best = None
    for j in range(d):
        values = np.unique(X[:, j])
        for a, b in zip(values[:-1], values[1:]):
            if b <= a + FEATURE_THRESHOLD:
                continue
            threshold = (a + b) / 2
            mask = X[:, j] <= threshold
            n_left = mask.sum()
            if n_left < min_samples_leaf or n - n_left < min_samples_leaf:
                continue
            score = split_score(y[mask], y[~mask], criterion)
            if best is None or score < best[2]:
                best = (j, threshold, score)
    return best


def best_split(X, y, n_classes, min_samples_leaf=1, criterion="gini"):
    """Vectorised search of the best split. Returns (feature, threshold, score)
    or None if no valid split exists.

    For each feature: sort the values once, then cumulative sums of the one-hot
    labels give the class counts on the left of every threshold at once (right =
    total - left). Cost O(d * n log n) instead of O(d * n^2).
    """
    n, d = X.shape
    onehot = np.eye(n_classes)[y]                     # (n, K)
    total = onehot.sum(axis=0)                        # (K,)
    best = None
    for j in range(d):
        order = np.argsort(X[:, j], kind="stable")
        x = X[order, j]
        left = np.cumsum(onehot[order], axis=0)[:-1]  # counts left of position i+1
        right = total - left
        n_left = np.arange(1, n)
        # Candidate positions: between two distinct values, both leaves big enough.
        valid = (x[1:] > x[:-1] + FEATURE_THRESHOLD) \
            & (n_left >= min_samples_leaf) & (n - n_left >= min_samples_leaf)
        if not valid.any():
            continue
        scores = (n_left * _impurity_from_counts(left, criterion)
                  + (n - n_left) * _impurity_from_counts(right, criterion)) / n
        scores = np.where(valid, scores, np.inf)
        i = int(np.argmin(scores))                    # first minimum, as sklearn
        if best is None or scores[i] < best[2]:
            threshold = (x[i] + x[i + 1]) / 2
            if threshold == x[i + 1]:                 # rounding: keep x <= threshold on the left
                threshold = x[i]
            best = (j, float(threshold), float(scores[i]))
    return best


# ---------------------------------------------------------------------------
# The tree
# ---------------------------------------------------------------------------

@dataclass
class Node:
    """Tree node. A leaf has feature = None."""
    feature: int | None = None
    threshold: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None
    proba: np.ndarray | None = None  # share of each class in the node
    n_samples: int = 0
    impurity: float = 0.0

    @property
    def is_leaf(self):
        return self.feature is None


class CARTClassifier:
    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_impurity_decrease=0.0, ccp_alpha=0.0):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        # Pre-pruning: a node is split only if the weighted impurity decrease
        # (N_t/N)·(G_t - G_split) reaches this value (same definition as sklearn).
        self.min_impurity_decrease = min_impurity_decrease
        # Cost-complexity post-pruning: after fit, subtrees whose gain per removed
        # leaf is below alpha are collapsed (Lecture 3B).
        self.ccp_alpha = ccp_alpha
        self.root_ = None
        self.n_classes_ = None
        self.feature_importances_ = None

    # -- training -----------------------------------------------------------

    def fit(self, X, y):
        """Grows the tree on (X, y). Returns self."""
        # sklearn stores X in float32; we do the same so that thresholds match.
        X = np.asarray(X, dtype=np.float32).astype(np.float64)
        y = np.asarray(y).astype(int)
        self.n_classes_ = int(y.max()) + 1
        self.n_features_in_ = X.shape[1]
        self._n_total = len(y)
        self.root_ = self._build(X, y, depth=0)
        if self.ccp_alpha > 0:
            self._prune(self.root_)
        self.feature_importances_ = self._compute_importances()
        return self

    def _best_split(self, X, y):
        """Returns (feature, threshold, score) of the best split, or None."""
        return best_split(X, y, self.n_classes_, self.min_samples_leaf, self.criterion)

    def _build(self, X, y, depth):
        """Greedy recursive growth. Stops on max_depth, min_samples_split,
        min_samples_leaf, pure node or impurity decrease < min_impurity_decrease."""
        n = len(y)
        counts = np.bincount(y, minlength=self.n_classes_).astype(float)
        node = Node(proba=counts / n, n_samples=n,
                    impurity=float(_impurity_from_counts(counts, self.criterion)))

        if ((self.max_depth is not None and depth >= self.max_depth)
                or n < self.min_samples_split
                or n < 2 * self.min_samples_leaf
                or node.impurity <= EPSILON):
            return node

        split = self._best_split(X, y)
        if split is None:
            return node
        feature, threshold, score = split
        decrease = n / self._n_total * (node.impurity - score)
        if decrease + EPSILON < self.min_impurity_decrease:
            return node

        mask = X[:, feature] <= threshold
        node.feature, node.threshold = feature, threshold
        node.left = self._build(X[mask], y[mask], depth + 1)
        node.right = self._build(X[~mask], y[~mask], depth + 1)
        return node

    # -- cost-complexity pruning ---------------------------------------------

    def _leaves_risk(self, node):
        """(number of leaves, sum of weighted leaf impurities R(T_t)) of a subtree."""
        if node.is_leaf:
            return 1, node.n_samples / self._n_total * node.impurity
        nl, rl = self._leaves_risk(node.left)
        nr, rr = self._leaves_risk(node.right)
        return nl + nr, rl + rr

    def _weakest_link(self, node):
        """Internal node with the smallest g(t) = (R(t) - R(T_t)) / (|T_t| - 1)."""
        if node.is_leaf:
            return None, np.inf
        n_leaves, risk_subtree = self._leaves_risk(node)
        risk_node = node.n_samples / self._n_total * node.impurity
        best = (node, (risk_node - risk_subtree) / (n_leaves - 1))
        for child in (node.left, node.right):
            cand = self._weakest_link(child)
            if cand[1] < best[1]:
                best = cand
        return best

    def _prune(self, node):
        """Weakest-link pruning: collapse the weakest subtree while its g(t) <= ccp_alpha."""
        while not node.is_leaf:
            weakest, g = self._weakest_link(node)
            if g > self.ccp_alpha:
                break
            weakest.feature = weakest.threshold = weakest.left = weakest.right = None

    # -- importances ---------------------------------------------------------

    def _compute_importances(self):
        """Sum over the nodes using feature j of (N_t/N)·(G_t - G_split), normalised to 1."""
        imp = np.zeros(self.n_features_in_)

        def visit(node):
            if node.is_leaf:
                return
            w = lambda c: c.n_samples / self._n_total * c.impurity
            imp[node.feature] += w(node) - w(node.left) - w(node.right)
            visit(node.left)
            visit(node.right)

        visit(self.root_)
        return imp / imp.sum() if imp.sum() > 0 else imp

    # -- prediction ----------------------------------------------------------

    def predict_proba(self, X):
        """Sends every row down the tree (vectorised by node) and returns the leaf proportions."""
        X = np.asarray(X, dtype=np.float32).astype(np.float64)
        out = np.empty((len(X), self.n_classes_))

        def route(node, idx):
            if node.is_leaf:
                out[idx] = node.proba
                return
            mask = X[idx, node.feature] <= node.threshold
            route(node.left, idx[mask])
            route(node.right, idx[~mask])

        route(self.root_, np.arange(len(X)))
        return out

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    # -- inspection ----------------------------------------------------------

    def get_depth(self):
        def depth(node):
            return 0 if node.is_leaf else 1 + max(depth(node.left), depth(node.right))
        return depth(self.root_)

    def get_n_leaves(self):
        return self._leaves_risk(self.root_)[0]

    def print_tree(self, feature_names=None, max_depth=None):
        """Prints the rules of the tree, one line per node."""
        names = feature_names or [f"x[{j}]" for j in range(self.n_features_in_)]

        def show(node, depth):
            pad = "|   " * depth
            if node.is_leaf or (max_depth is not None and depth >= max_depth):
                print(f"{pad}leaf: n={node.n_samples}, P(>50K)={node.proba[-1]:.2f}")
                return
            print(f"{pad}{names[node.feature]} <= {node.threshold:.2f}  (n={node.n_samples}, gini={node.impurity:.3f})")
            show(node.left, depth + 1)
            print(f"{pad}{names[node.feature]} >  {node.threshold:.2f}")
            show(node.right, depth + 1)

        show(self.root_, 0)
