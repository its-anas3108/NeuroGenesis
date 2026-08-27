"""
Classical machine-learning baselines (Section 18, baselines 1 and 2).
====================================================================

Non-deep reference models over the flattened per-ROI morphometric features:

=========================  ====================================================
``logistic_regression``    Multinomial logistic regression, balanced class
                           weights. The interpretable linear reference.
``gradient_boosting``      Gradient-boosted trees. Stands in for XGBoost, which
                           is not installed; the substitution is recorded in the
                           result so no table can claim XGBoost was run.
``random_forest``          Bagged trees, balanced class weights.
``mlp``                    Two-layer perceptron. The graph-free neural
                           reference, implemented independently of the PyTorch
                           stack so that agreement between it and ablation A0 is
                           informative rather than tautological.
=========================  ====================================================

Fairness of the comparison
--------------------------

Every baseline receives:

* the **same** subject-wise splits as the deep variants,
* the **same** scaler, fitted on the same training split,
* the **same** class-imbalance handling in spirit (``class_weight="balanced"``
  where the estimator supports it, which is inverse-frequency weighting on the
  training split — identical to what the deep models use).

Hyper-parameters are left at sensible defaults rather than tuned. This is stated
plainly because it cuts against the proposed model: a tuned baseline might do
better, and the comparison should be read as "the proposed model versus
reasonable off-the-shelf baselines", not "versus the best possible baseline".
:func:`baseline_caveats` returns this wording for the report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from modules.common.logging_utils import get_logger
from modules.common.roi_constants import N_STAGE, STAGE_ORDER
from modules.training.metrics import (
    ClassificationMetrics,
    aggregate_metrics,
    compute_metrics,
)
from modules.common.serialization import json_safe

logger = get_logger(__name__)

#: Baseline name -> human-readable description.
BASELINE_DESCRIPTIONS: Dict[str, str] = {
    "logistic_regression": "Morphometry + Logistic Regression",
    "gradient_boosting": "Morphometry + Gradient Boosting (XGBoost substitute)",
    "random_forest": "Morphometry + Random Forest",
    "mlp": "Morphometry + MLP (scikit-learn)",
}


@dataclass
class BaselineRun:
    """One baseline evaluated on one split."""

    name: str
    repeat: int
    metrics: ClassificationMetrics
    n_train: int
    n_test: int
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "name": self.name, "repeat": self.repeat,
            "n_train": self.n_train, "n_test": self.n_test,
            "metrics": self.metrics.to_dict(), "notes": list(self.notes),
        }


def available_baselines() -> Dict[str, bool]:
    """Report which baselines can run in this environment."""
    try:
        import sklearn  # noqa: F401

        has_sklearn = True
    except ImportError:
        has_sklearn = False
    try:
        import xgboost  # noqa: F401

        has_xgboost = True
    except ImportError:
        has_xgboost = False

    return {
        "logistic_regression": has_sklearn,
        "gradient_boosting": has_sklearn,
        "random_forest": has_sklearn,
        "mlp": has_sklearn,
        "xgboost": has_xgboost,
    }


def _build_estimator(name: str, seed: int) -> Tuple[Any, List[str]]:
    """Instantiate one baseline estimator.

    Returns:
        ``(estimator, notes)``.

    Raises:
        ImportError: If scikit-learn is unavailable.
        KeyError: If the baseline name is unknown.
    """
    try:
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "scikit-learn is required for the classical baselines. "
            "Install it, or omit the baselines from the run."
        ) from exc

    notes: List[str] = []
    if name == "logistic_regression":
        # `multi_class` is deliberately not passed: it was removed in
        # scikit-learn 1.8, and multinomial is the default for the lbfgs solver
        # in every version this project supports.
        return LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=seed,
        ), notes
    if name == "gradient_boosting":
        notes.append(
            "XGBoost is not installed; scikit-learn's GradientBoostingClassifier "
            "was used instead. This is a different implementation with different "
            "defaults and must not be reported as XGBoost."
        )
        notes.append(
            "GradientBoostingClassifier has no class_weight parameter, so this "
            "baseline is the one model in the comparison without explicit "
            "imbalance correction. Its recall on the AD class should be read "
            "with that in mind."
        )
        return GradientBoostingClassifier(random_state=seed), notes
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=seed,
            n_jobs=1,
        ), notes
    if name == "mlp":
        notes.append(
            "MLPClassifier has no class_weight parameter; the class imbalance "
            "is uncorrected for this baseline."
        )
        return MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=2000, random_state=seed,
            early_stopping=False,
        ), notes
    raise KeyError(
        f"Unknown baseline {name!r}. Available: {sorted(BASELINE_DESCRIPTIONS)}"
    )


def run_baseline(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    repeat: int = 0,
    seed: int = 42,
) -> BaselineRun:
    """Fit and evaluate one baseline on one split.

    Args:
        name: Baseline key.
        x_train: ``(n_train, N_ROI * N_FEATURES)`` standardised features.
        y_train: Training labels.
        x_test: Test features.
        y_test: Test labels.
        repeat: Repeat index.
        seed: RNG seed.

    Returns:
        A :class:`BaselineRun`.
    """
    estimator, notes = _build_estimator(name, seed)
    x_train = np.asarray(x_train, dtype=np.float64)
    x_test = np.asarray(x_test, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.int64).ravel()
    y_test = np.asarray(y_test, dtype=np.int64).ravel()

    # Non-finite values would make several estimators raise; they are replaced
    # with 0, which after standardisation is the fitted training median.
    x_train = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0)
    x_test = np.nan_to_num(x_test, nan=0.0, posinf=0.0, neginf=0.0)

    estimator.fit(x_train, y_train)
    predictions = estimator.predict(x_test)

    probabilities: Optional[np.ndarray] = None
    if hasattr(estimator, "predict_proba"):
        raw = np.asarray(estimator.predict_proba(x_test), dtype=np.float64)
        # An estimator that never saw a class omits its column; expand back to
        # the full N_STAGE width so the AUC computation stays aligned to
        # STAGE_ORDER rather than to whichever classes happened to be present.
        probabilities = np.zeros((raw.shape[0], N_STAGE), dtype=np.float64)
        for column, class_index in enumerate(estimator.classes_):
            if 0 <= int(class_index) < N_STAGE:
                probabilities[:, int(class_index)] = raw[:, column]
        missing = set(range(N_STAGE)) - set(int(c) for c in estimator.classes_)
        if missing:
            notes.append(
                f"Class(es) {sorted(STAGE_ORDER[i] for i in missing)} were "
                "absent from this training split, so the estimator cannot "
                "predict them and their probability column is zero."
            )

    return BaselineRun(
        name=name,
        repeat=repeat,
        metrics=compute_metrics(y_test, predictions, probabilities),
        n_train=int(x_train.shape[0]),
        n_test=int(x_test.shape[0]),
        notes=notes,
    )


class BaselineStudy:
    """Run every classical baseline over repeated splits."""

    def __init__(self) -> None:
        self.runs: List[BaselineRun] = []
        self.failures: List[Dict[str, Any]] = []

    def run(
        self,
        splits_data: Sequence[Tuple[int, np.ndarray, np.ndarray,
                                    np.ndarray, np.ndarray]],
        baselines: Optional[Sequence[str]] = None,
        seed: int = 42,
    ) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
        """Evaluate baselines across repeats.

        Args:
            splits_data: One tuple per repeat:
                ``(repeat, x_train, y_train, x_test, y_test)``. Features must
                already be standardised with a scaler fitted on that repeat's
                training split.
            baselines: Baseline keys. Defaults to all four.
            seed: Base seed.

        Returns:
            ``{baseline: aggregated metrics}``.
        """
        names = list(baselines or BASELINE_DESCRIPTIONS)
        capability = available_baselines()

        self.runs = []
        self.failures = []
        for name in names:
            if not capability.get(name, False):
                self.failures.append({
                    "name": name,
                    "error": "required package not installed",
                })
                logger.warning("Baseline %s skipped: dependency missing.", name)
                continue
            for repeat, x_tr, y_tr, x_te, y_te in splits_data:
                try:
                    run = run_baseline(
                        name, x_tr, y_tr, x_te, y_te,
                        repeat=repeat, seed=seed + repeat,
                    )
                    self.runs.append(run)
                    logger.info(
                        "  %-20s repeat %d: acc %.4f macro-F1 %.4f",
                        name, repeat, run.metrics.accuracy,
                        run.metrics.macro_f1 or float("nan"),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error("  %s repeat %d FAILED: %s", name, repeat, exc)
                    self.failures.append({
                        "name": name, "repeat": repeat,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
        return self.aggregate()

    def aggregate(self) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
        """Aggregate runs per baseline."""
        by_name: Dict[str, List[BaselineRun]] = {}
        for run in self.runs:
            by_name.setdefault(run.name, []).append(run)
        return {
            name: aggregate_metrics(
                [r.metrics for r in runs],
                ("accuracy", "macro_f1", "balanced_accuracy", "roc_auc_macro"),
            )
            for name, runs in by_name.items()
        }

    def table(self) -> pd.DataFrame:
        """Return a baseline comparison table."""
        aggregated = self.aggregate()
        rows = []
        for name, metrics in sorted(aggregated.items()):
            def fmt(metric: str) -> str:
                cell = metrics.get(metric, {})
                mean, sd = cell.get("mean"), cell.get("sd")
                if mean is None:
                    return "n/a"
                return f"{mean:.4f} +/- {sd:.4f}" if sd is not None \
                    else f"{mean:.4f}"

            runs = [r for r in self.runs if r.name == name]
            rows.append({
                "Method": BASELINE_DESCRIPTIONS.get(name, name),
                "Baseline": name,
                "Accuracy": fmt("accuracy"),
                "Macro-F1": fmt("macro_f1"),
                "Balanced Accuracy": fmt("balanced_accuracy"),
                "ROC-AUC": fmt("roc_auc_macro"),
                "Repeats": len(runs),
            })
        return pd.DataFrame(rows)

    def per_repeat_values(self, metric: str = "macro_f1"
                          ) -> Dict[str, List[Optional[float]]]:
        """Return ``{baseline: [value per repeat]}`` for paired comparison."""
        out: Dict[str, List[Optional[float]]] = {}
        by_name: Dict[str, List[BaselineRun]] = {}
        for run in self.runs:
            by_name.setdefault(run.name, []).append(run)
        for name, runs in by_name.items():
            out[name] = [
                r.metrics.get(metric) for r in sorted(runs, key=lambda r: r.repeat)
            ]
        return out

    def save(self, out_dir: Path) -> Dict[str, Path]:
        """Persist baseline runs and the comparison table."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written: Dict[str, Path] = {}

        payload = {
            "descriptions": dict(BASELINE_DESCRIPTIONS),
            "environment": available_baselines(),
            "caveats": baseline_caveats(),
            "aggregate": self.aggregate(),
            "runs": [r.to_dict() for r in self.runs],
            "failures": list(self.failures),
        }
        path = out_dir / "baseline_results.json"
        path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
        written["results_json"] = path

        table = self.table()
        if not table.empty:
            path = out_dir / "baseline_comparison.csv"
            table.to_csv(path, index=False)
            written["comparison_csv"] = path

        logger.info("Baseline results saved to %s", out_dir)
        return written


def baseline_caveats() -> List[str]:
    """Return the mandatory caveats for the baseline comparison."""
    return [
        "Baseline hyper-parameters are library defaults and were not tuned. A "
        "tuned baseline could perform better, so this comparison should be read "
        "as 'the proposed model versus reasonable off-the-shelf baselines', not "
        "'versus the best achievable baseline'.",
        "XGBoost is not installed in this environment; scikit-learn's "
        "GradientBoostingClassifier was substituted and is labelled as such.",
        "GradientBoostingClassifier and MLPClassifier have no class-weight "
        "parameter, so those two baselines run without the imbalance correction "
        "that every other model in the comparison receives.",
        "All baselines use the same subject-wise splits and the same "
        "training-split-fitted scaler as the deep variants.",
    ]


__all__ = [
    "BASELINE_DESCRIPTIONS",
    "BaselineRun",
    "BaselineStudy",
    "available_baselines",
    "run_baseline",
    "baseline_caveats",
]
