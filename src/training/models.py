"""Fixed-hyperparameter tabular models for Task 1 (model set v2).

Four real models — logistic regression, a smoothed decision tree, a random forest and a small neural
network — plus a prior-only baseline that is kept as the reference line, not as a model.

Hyperparameters were chosen a priori for a very small dataset (strong regularization, shallow
trees, a single small hidden layer) and are never tuned on evaluation folds. Every learned
preprocessing step (median imputation, zero-variance filtering, scaling) sits inside the Pipeline,
so it is fitted on the training part of each fold only. Missing values are imputed inside the
pipeline; the dataset's explicit ``<variable>_missing`` flags keep missingness visible to the models.

K-Means is deliberately absent: it is not a supervised mortality predictor.
"""

from typing import Dict

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

MODEL_ORDER = ("dummy_prior", "logistic_regression", "decision_tree", "random_forest", "neural_network")
REFERENCE_MODEL = "dummy_prior"  # predicts the training death rate; shown as a reference, not a model


class SmoothedDecisionTree(ClassifierMixin, BaseEstimator):
    """A shallow decision tree whose leaf death rates are shrunk toward the training death rate.

    With few deaths, many leaves contain none and a plain tree outputs exactly 0% (or 100%). Each leaf
    instead reports (deaths + m * prior) / (stays + m) — the m-estimate — so small leaves move toward
    the overall rate while the tree's rules stay unchanged and readable.
    """

    def __init__(self, max_depth: int = 3, min_samples_leaf: int = 8, smoothing: float = 5.0, random_state=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.smoothing = smoothing
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        self.tree_ = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                            random_state=self.random_state).fit(X, y)
        self.classes_ = self.tree_.classes_
        self.prior_ = float(y.mean())
        leaves = self.tree_.apply(X)
        self.leaf_rate_ = {
            int(leaf): (float(y[leaves == leaf].sum()) + self.smoothing * self.prior_) / (float((leaves == leaf).sum()) + self.smoothing)
            for leaf in np.unique(leaves)
        }
        return self

    def predict_proba(self, X):
        rates = np.array([self.leaf_rate_.get(int(leaf), self.prior_) for leaf in self.tree_.apply(X)])
        return np.column_stack([1.0 - rates, rates])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _impute():
    return ("impute", SimpleImputer(strategy="median", keep_empty_features=True))


def _drop_constant():
    return ("drop_constant", VarianceThreshold(threshold=0.0))


def candidate_models(seed: int) -> Dict[str, Pipeline]:
    """Returns unfitted pipelines keyed by model name, in MODEL_ORDER."""
    return {
        "dummy_prior": Pipeline([("model", DummyClassifier(strategy="prior"))]),
        "logistic_regression": Pipeline([
            _impute(), _drop_constant(), ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.1, max_iter=5000)),
        ]),
        "decision_tree": Pipeline([
            _impute(), _drop_constant(),
            ("model", SmoothedDecisionTree(max_depth=3, min_samples_leaf=8, smoothing=5.0, random_state=seed)),
        ]),
        "random_forest": Pipeline([
            _impute(), _drop_constant(),
            ("model", RandomForestClassifier(
                n_estimators=300, max_depth=4, min_samples_leaf=3, max_features="sqrt",
                random_state=seed, n_jobs=1,
            )),
        ]),
        "neural_network": Pipeline([
            _impute(), _drop_constant(), ("scale", StandardScaler()),
            ("model", MLPClassifier(hidden_layer_sizes=(16,), activation="relu", solver="lbfgs", alpha=1.0,
                                    max_iter=2000, random_state=seed)),
        ]),
    }


def describe_models(seed: int) -> Dict[str, Dict[str, object]]:
    """JSON-friendly description of each pipeline's steps and non-default parameters."""
    description = {}
    for name, pipeline in candidate_models(seed).items():
        steps = []
        for step_name, step in pipeline.steps:
            defaults = type(step)().get_params(deep=False)
            # repr comparison: NaN defaults (e.g. SimpleImputer.missing_values) never compare equal
            params = {k: v for k, v in step.get_params(deep=False).items() if repr(v) != repr(defaults.get(k))}
            steps.append({"step": step_name, "estimator": type(step).__name__, "non_default_params": params})
        description[name] = {"steps": steps, "hyperparameters_tuned": False,
                             "role": "reference line (training death rate)" if name == REFERENCE_MODEL else "model"}
    return description
