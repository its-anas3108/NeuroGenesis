"""
Classification metrics (Sections 11, 26).
=========================================

Implements the full metric set the design requires — accuracy, macro-F1,
balanced accuracy, per-class precision / recall / specificity / F1, ROC-AUC and
the confusion matrix — with the small-sample edge cases handled explicitly
rather than silently.

Small-sample honesty
--------------------

The test split contains roughly 6 AD sessions. At that size several metrics are
genuinely undefined in ways a library default would paper over:

* **A class with no predictions** makes precision ``0/0``. scikit-learn's default
  substitutes ``0.0``, which reads as "the model was wrong about this class"
  rather than "the model never attempted it". Here it is ``None``, and the
  reason appears in :attr:`ClassificationMetrics.notes`.
* **A class absent from the labels** makes recall and one-vs-rest AUC undefined.
  Reported as ``None``.
* **Macro-F1 and balanced accuracy** are averaged only over classes that are
  actually present, and the count of averaged classes is reported, so a
  three-class macro-F1 is never quietly computed over two.

``None`` propagates into the tables as an explicit blank. That is the point: a
fabricated 0.0 in a results table is worse than an admitted gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from modules.common.logging_utils import get_logger
from modules.common.roi_constants import N_STAGE, STAGE_ORDER

logger = get_logger(__name__)


@dataclass
class ClassMetrics:
    """Per-class metrics. ``None`` marks a genuinely undefined quantity."""

    stage: str
    support: int
    n_predicted: int
    precision: Optional[float] = None
    recall: Optional[float] = None
    specificity: Optional[float] = None
    f1: Optional[float] = None
    auc: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "stage": self.stage, "support": self.support,
            "n_predicted": self.n_predicted, "precision": self.precision,
            "recall": self.recall, "specificity": self.specificity,
            "f1": self.f1, "auc": self.auc,
        }


@dataclass
class ClassificationMetrics:
    """Complete metric set for one evaluation."""

    n_samples: int
    accuracy: float
    balanced_accuracy: Optional[float] = None
    macro_f1: Optional[float] = None
    macro_precision: Optional[float] = None
    macro_recall: Optional[float] = None
    roc_auc_macro: Optional[float] = None
    #: ``(N_STAGE, N_STAGE)`` matrix; rows are true, columns predicted.
    confusion_matrix: Optional[np.ndarray] = None
    per_class: List[ClassMetrics] = field(default_factory=list)
    #: Number of classes each macro average was actually taken over.
    n_classes_averaged: Dict[str, int] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "n_samples": self.n_samples,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_f1": self.macro_f1,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "roc_auc_macro": self.roc_auc_macro,
            "confusion_matrix": (
                self.confusion_matrix.tolist()
                if self.confusion_matrix is not None else None
            ),
            "per_class": [c.to_dict() for c in self.per_class],
            "n_classes_averaged": dict(self.n_classes_averaged),
            "notes": list(self.notes),
        }

    def get(self, name: str) -> Optional[float]:
        """Return a scalar metric by name, for model selection.

        Raises:
            KeyError: If the name is not a known scalar metric.
        """
        table = {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_f1": self.macro_f1,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "roc_auc_macro": self.roc_auc_macro,
        }
        if name not in table:
            raise KeyError(
                f"Unknown metric {name!r}. Available: {sorted(table)}"
            )
        return table[name]

    def summary(self) -> str:
        """Human-readable summary."""
        def fmt(v: Optional[float]) -> str:
            return "  n/a" if v is None else f"{v:.4f}"

        lines = [
            f"n = {self.n_samples}",
            f"  accuracy          {fmt(self.accuracy)}",
            f"  balanced accuracy {fmt(self.balanced_accuracy)}",
            f"  macro F1          {fmt(self.macro_f1)}",
            f"  macro ROC-AUC     {fmt(self.roc_auc_macro)}",
            "  per class:",
        ]
        for c in self.per_class:
            lines.append(
                f"    {c.stage:<4s} n={c.support:<4d} pred={c.n_predicted:<4d} "
                f"P={fmt(c.precision)} R={fmt(c.recall)} "
                f"Spec={fmt(c.specificity)} F1={fmt(c.f1)} AUC={fmt(c.auc)}"
            )
        for n in self.notes:
            lines.append(f"  NOTE: {n}")
        return "\n".join(lines)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                     n_classes: int = N_STAGE) -> np.ndarray:
    """Compute a confusion matrix. Rows are true classes, columns predicted."""
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()):
        if 0 <= int(t) < n_classes and 0 <= int(p) < n_classes:
            matrix[int(t), int(p)] += 1
    return matrix


def _binary_auc(scores: np.ndarray, positive: np.ndarray) -> Optional[float]:
    """One-vs-rest ROC-AUC via the rank (Mann-Whitney U) formulation.

    Implemented directly rather than via scikit-learn so that ties are handled
    with mid-ranks and the undefined case returns ``None`` instead of raising or
    silently returning 0.5.

    Args:
        scores: Predicted score for the positive class.
        positive: Boolean mask of true positives.

    Returns:
        AUC, or ``None`` when one of the two groups is empty.
    """
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)

    # Mid-ranks for ties; without this, a model that outputs the same score for
    # many subjects gets an AUC that depends on array order.
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        mid = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = mid
        i = j + 1

    sum_pos_ranks = ranks[positive].sum()
    u = sum_pos_ranks - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Optional[np.ndarray] = None,
    n_classes: int = N_STAGE,
) -> ClassificationMetrics:
    """Compute the full metric set.

    Args:
        y_true: True class indices.
        y_pred: Predicted class indices.
        y_prob: ``(n_samples, n_classes)`` predicted probabilities, needed for
            ROC-AUC. ``None`` leaves every AUC as ``None``.
        n_classes: Number of classes.

    Returns:
        A :class:`ClassificationMetrics`.

    Raises:
        ValueError: If the input lengths disagree.
    """
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true has {y_true.shape[0]} entries but y_pred has "
            f"{y_pred.shape[0]}"
        )
    n = int(y_true.size)
    if n == 0:
        return ClassificationMetrics(
            n_samples=0, accuracy=float("nan"),
            notes=["No samples were evaluated; every metric is undefined."],
        )
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=np.float64)
        if y_prob.shape[0] != n:
            raise ValueError(
                f"y_prob has {y_prob.shape[0]} rows but there are {n} samples"
            )

    cm = confusion_matrix(y_true, y_pred, n_classes)
    accuracy = float(np.trace(cm) / n)

    per_class: List[ClassMetrics] = []
    notes: List[str] = []
    recalls: List[float] = []
    precisions: List[float] = []
    f1s: List[float] = []
    aucs: List[float] = []

    for idx in range(n_classes):
        stage = STAGE_ORDER[idx] if idx < len(STAGE_ORDER) else f"class_{idx}"
        tp = int(cm[idx, idx])
        support = int(cm[idx, :].sum())
        n_predicted = int(cm[:, idx].sum())
        fp = n_predicted - tp
        fn = support - tp
        tn = n - tp - fp - fn

        metrics = ClassMetrics(stage=stage, support=support,
                               n_predicted=n_predicted)

        if n_predicted > 0:
            metrics.precision = float(tp / n_predicted)
            precisions.append(metrics.precision)
        else:
            notes.append(
                f"{stage}: precision is undefined — the model made no {stage} "
                f"prediction (support {support})."
            )
        if support > 0:
            metrics.recall = float(tp / support)
            recalls.append(metrics.recall)
        else:
            notes.append(
                f"{stage}: recall is undefined — no {stage} sample is present "
                "in this split."
            )
        if (tn + fp) > 0:
            metrics.specificity = float(tn / (tn + fp))
        if metrics.precision is not None and metrics.recall is not None:
            denom = metrics.precision + metrics.recall
            metrics.f1 = float(
                2 * metrics.precision * metrics.recall / denom
            ) if denom > 0 else 0.0
            f1s.append(metrics.f1)
        elif support > 0:
            # The class exists but was never predicted: F1 is legitimately 0,
            # not undefined, and must count toward the macro average.
            metrics.f1 = 0.0
            f1s.append(0.0)

        if y_prob is not None and y_prob.shape[1] > idx:
            auc = _binary_auc(y_prob[:, idx], y_true == idx)
            metrics.auc = auc
            if auc is not None:
                aucs.append(auc)

        per_class.append(metrics)

    result = ClassificationMetrics(
        n_samples=n,
        accuracy=accuracy,
        balanced_accuracy=float(np.mean(recalls)) if recalls else None,
        macro_f1=float(np.mean(f1s)) if f1s else None,
        macro_precision=float(np.mean(precisions)) if precisions else None,
        macro_recall=float(np.mean(recalls)) if recalls else None,
        roc_auc_macro=float(np.mean(aucs)) if aucs else None,
        confusion_matrix=cm,
        per_class=per_class,
        n_classes_averaged={
            "balanced_accuracy": len(recalls),
            "macro_f1": len(f1s),
            "macro_precision": len(precisions),
            "roc_auc_macro": len(aucs),
        },
        notes=notes,
    )

    for name, count in result.n_classes_averaged.items():
        if 0 < count < n_classes:
            result.notes.append(
                f"{name} was averaged over {count} of {n_classes} classes; the "
                "remaining class(es) were undefined on this split."
            )
    return result


#: Metrics that are mathematically confined to ``[0, 1]``. A confidence
#: interval on one of these is clamped to that range, because reporting an
#: F1 upper bound of 1.047 is indefensible even when the arithmetic produces it.
BOUNDED_METRICS: Dict[str, Tuple[float, float]] = {
    "accuracy": (0.0, 1.0),
    "balanced_accuracy": (0.0, 1.0),
    "macro_f1": (0.0, 1.0),
    "macro_precision": (0.0, 1.0),
    "macro_recall": (0.0, 1.0),
    "roc_auc_macro": (0.0, 1.0),
}


def aggregate_metrics(
    runs: Sequence[ClassificationMetrics],
    metric_names: Sequence[str] = ("accuracy", "macro_f1",
                                   "balanced_accuracy", "roc_auc_macro"),
    confidence: float = 0.95,
) -> Dict[str, Dict[str, Optional[float]]]:
    """Aggregate repeated runs into mean, SD and a confidence interval.

    The interval is a normal-approximation CI on the mean across repeats
    (``mean +/- z * SD / sqrt(k)``). With ``k`` around 10 this is an
    approximation, and the returned ``n`` lets a reader judge it. It describes
    variability **across splits of the same 235-session cohort**, not sampling
    variability of the wider population — the same subjects recur in every
    repeat.

    Bounds are enforced. Every metric here lives in ``[0, 1]``, but the normal
    approximation does not know that: at high means and small ``k`` it happily
    produces an upper bound above 1. Such an interval is clamped, and
    ``ci_clamped`` records that it was, so a reader can see the approximation
    hitting its limit rather than seeing an impossible number.

    Args:
        runs: Per-repeat metric objects.
        metric_names: Scalar metrics to aggregate.
        confidence: Interval level; 0.90, 0.95 and 0.99 are supported.

    Returns:
        ``{metric: {"mean", "sd", "ci_low", "ci_high", "ci_clamped", "n",
        "values"}}``.
    """
    z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)
    out: Dict[str, Dict[str, Optional[float]]] = {}

    for name in metric_names:
        values = [r.get(name) for r in runs]
        clean = [v for v in values if v is not None and np.isfinite(v)]
        if not clean:
            out[name] = {"mean": None, "sd": None, "ci_low": None,
                         "ci_high": None, "ci_clamped": False,
                         "n": 0, "values": []}
            continue
        arr = np.asarray(clean, dtype=np.float64)
        mean = float(arr.mean())
        sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        half = z * sd / np.sqrt(arr.size) if arr.size > 1 else 0.0
        low, high = mean - half, mean + half

        clamped = False
        if name in BOUNDED_METRICS:
            lower, upper = BOUNDED_METRICS[name]
            clamped = low < lower or high > upper
            low, high = max(low, lower), min(high, upper)

        out[name] = {
            "mean": mean, "sd": sd,
            "ci_low": float(low), "ci_high": float(high),
            "ci_clamped": bool(clamped),
            "n": int(arr.size), "values": [float(v) for v in arr],
        }
    return out


__all__ = [
    "BOUNDED_METRICS",
    "ClassMetrics",
    "ClassificationMetrics",
    "confusion_matrix",
    "compute_metrics",
    "aggregate_metrics",
]
