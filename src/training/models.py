"""Fixed-hyperparameter tabular baselines for Task 1.

Hyperparameters were chosen a priori for a very small dataset (strong regularization, shallow
trees) and are never tuned on evaluation folds. Every learned preprocessing step (median
imputation, zero-variance filtering, scaling) sits inside the Pipeline, so it is fitted on the
training part of each fold only. Missing values are imputed inside the pipeline; the dataset's
explicit ``<variable>_missing`` flags keep missingness visible to the models.

K-Means is deliberately absent: it is not a supervised mortality predictor.
"""

from typing import Dict

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

MODEL_ORDER = ("dummy_prior", "logistic_regression", "decision_tree", "random_forest")


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
            ("model", DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=seed)),
        ]),
        "random_forest": Pipeline([
            _impute(), _drop_constant(),
            ("model", RandomForestClassifier(
                n_estimators=300, max_depth=4, min_samples_leaf=3, max_features="sqrt",
                random_state=seed, n_jobs=1,
            )),
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
        description[name] = {"steps": steps, "hyperparameters_tuned": False}
    return description
