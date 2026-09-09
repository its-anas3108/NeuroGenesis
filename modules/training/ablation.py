"""
M18 — Ablation and baseline study (Sections 17, 18).
===================================================

Runs each model variant over the **same** repeated stratified subject-wise
splits and aggregates the results, so every variant sees identical data,
identical training procedure and identical selection criteria. The only thing
that differs is the component under test.

Why repeated splits are mandatory here
--------------------------------------

The labelled OASIS-1 cohort has 30 AD sessions, giving roughly 6 AD subjects in
a 20% test split. A single split's accuracy on 47 test sessions has a standard
error of several percentage points, and the difference between two ablation rungs
will routinely be smaller than that. A single-split ranking of A0-A7 would
therefore be mostly noise. Every comparison in this module is consequently
reported as mean +/- SD with a confidence interval over repeats, and paired
across repeats.

Paired comparison
-----------------

Because every variant is evaluated on the *same* splits, the comparison against
the reference variant is **paired**, which removes between-split variance and is
far more sensitive than an unpaired test at this sample size. The Wilcoxon
signed-rank test is used rather than a paired t-test: with ~10 repeats there is
no way to establish normality of the difference distribution, and the ranked test
does not require it. The matched-pairs rank-biserial correlation is reported as
the effect size.

.. note::

   The confidence intervals and p-values describe variability **across splits of
   one fixed 235-session cohort**. They are not confidence intervals over the
   population of Alzheimer's patients: the same 235 subjects appear in every
   repeat. :meth:`AblationStudy.caveats` returns this wording for the report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from modules.common.config import NeuroGenesisConfig
from modules.common.logging_utils import get_logger
from modules.common.roi_constants import STAGE_ORDER
from modules.common.seeds import set_all_seeds
from modules.m01_dataset.splits import (
    SplitManifest,
    repeated_subject_splits,
    stratified_subject_folds,
)
from modules.model import (
    ABLATION_LADDER,
    ABLATION_SPECS,
    BASELINE_SPECS,
    build_model,
)
from modules.training.metrics import (
    ClassificationMetrics,
    aggregate_metrics,
    compute_metrics,
)
from modules.training.trainer import Trainer
from modules.common.serialization import json_safe

logger = get_logger(__name__)

#: Metrics aggregated for every variant.
METRIC_NAMES: Tuple[str, ...] = (
    "accuracy", "macro_f1", "balanced_accuracy", "roc_auc_macro",
    "macro_precision", "macro_recall",
)


def _at_best(result: Any, attribute: str) -> Optional[float]:
    """Read one validation metric from the selected epoch's record.

    Returns ``None`` rather than the last epoch's value when the best
    epoch is out of range, so a failed or zero-epoch run cannot contribute
    a misleading number to a selection decision.
    """
    history = getattr(result, "history", None) or []
    best = getattr(result, "best_epoch", -1)
    if not 0 < best <= len(history):
        return None
    value = getattr(history[best - 1], attribute, None)
    return None if value is None else float(value)


@dataclass
class VariantRun:
    """One variant evaluated on one split."""

    variant: str
    repeat: int
    seed: int
    metrics: ClassificationMetrics
    best_epoch: int
    n_parameters: int
    train_seconds: float
    stage_geometry_ordered: Optional[bool] = None
    #: The monitored metric on the **validation** split at the selected
    #: epoch. This is the only signal any configuration choice may be
    #: made on; test metrics above must never be used for selection.
    val_monitor: Optional[str] = None
    val_best_value: Optional[float] = None
    val_accuracy: Optional[float] = None
    val_balanced_accuracy: Optional[float] = None
    val_macro_f1: Optional[float] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "variant": self.variant,
            "repeat": self.repeat,
            "seed": self.seed,
            "best_epoch": self.best_epoch,
            "n_parameters": self.n_parameters,
            "train_seconds": self.train_seconds,
            "stage_geometry_ordered": self.stage_geometry_ordered,
            "val_monitor": self.val_monitor,
            "val_best_value": self.val_best_value,
            "val_accuracy": self.val_accuracy,
            "val_balanced_accuracy": self.val_balanced_accuracy,
            "val_macro_f1": self.val_macro_f1,
            "metrics": self.metrics.to_dict(),
            "warnings": list(self.warnings),
        }


@dataclass
class VariantSummary:
    """Aggregated results for one variant across repeats."""

    variant: str
    description: str
    n_repeats: int
    n_parameters: int
    #: metric -> {mean, sd, ci_low, ci_high, n, values}
    aggregate: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict)
    #: Comparison against the reference variant, when one was set.
    comparison: Optional[Dict[str, Any]] = None
    notes: List[str] = field(default_factory=list)

    def value(self, metric: str, key: str = "mean") -> Optional[float]:
        """Return one aggregated statistic, or ``None`` when unavailable."""
        return self.aggregate.get(metric, {}).get(key)

    def formatted(self, metric: str, digits: int = 4) -> str:
        """Return ``"mean +/- sd"`` or ``"n/a"`` for a table cell."""
        mean = self.value(metric, "mean")
        sd = self.value(metric, "sd")
        if mean is None:
            return "n/a"
        if sd is None:
            return f"{mean:.{digits}f}"
        return f"{mean:.{digits}f} +/- {sd:.{digits}f}"

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "variant": self.variant,
            "description": self.description,
            "n_repeats": self.n_repeats,
            "n_parameters": self.n_parameters,
            "aggregate": self.aggregate,
            "comparison": self.comparison,
            "notes": list(self.notes),
        }


def paired_comparison(
    reference: Sequence[Optional[float]],
    candidate: Sequence[Optional[float]],
    metric: str,
) -> Dict[str, Any]:
    """Compare two variants paired across repeats.

    Args:
        reference: Reference variant's per-repeat values.
        candidate: Candidate variant's per-repeat values, same repeat order.
        metric: Metric name, for the record.

    Returns:
        Dict with the mean difference, Wilcoxon signed-rank p-value, the
        matched-pairs rank-biserial effect size and the number of usable pairs.
        ``p_value`` is ``None`` when the test cannot be run — fewer than three
        pairs, or all differences exactly zero.
    """
    pairs = [
        (r, c) for r, c in zip(reference, candidate)
        if r is not None and c is not None
        and np.isfinite(r) and np.isfinite(c)
    ]
    out: Dict[str, Any] = {
        "metric": metric,
        "n_pairs": len(pairs),
        "mean_difference": None,
        "p_value": None,
        "test": "not_run",
        "effect_size": None,
        "effect_size_name": None,
        "note": None,
    }
    if len(pairs) < 3:
        out["note"] = (
            f"Only {len(pairs)} usable pair(s); a paired test needs at least 3."
        )
        return out

    ref = np.array([p[0] for p in pairs], dtype=np.float64)
    cand = np.array([p[1] for p in pairs], dtype=np.float64)
    diff = cand - ref
    out["mean_difference"] = float(diff.mean())
    out["sd_difference"] = float(diff.std(ddof=1)) if diff.size > 1 else 0.0

    if np.allclose(diff, 0.0):
        out["test"] = "identical"
        out["note"] = (
            "The two variants produced identical values on every repeat, so no "
            "test is meaningful."
        )
        out["effect_size"] = 0.0
        out["effect_size_name"] = "rank_biserial_matched_pairs"
        return out

    try:
        result = stats.wilcoxon(cand, ref, alternative="two-sided",
                                zero_method="wilcox")
        out["test"] = "wilcoxon_signed_rank"
        out["statistic"] = float(result.statistic)
        out["p_value"] = float(result.pvalue)
    except ValueError as exc:
        out["test"] = "failed"
        out["note"] = f"Wilcoxon test failed: {exc}"
        return out

    # Matched-pairs rank-biserial: (positive rank sum - negative rank sum) over
    # the total rank sum of the non-zero differences.
    nonzero = diff[diff != 0]
    ranks = stats.rankdata(np.abs(nonzero))
    positive = ranks[nonzero > 0].sum()
    negative = ranks[nonzero < 0].sum()
    total = positive + negative
    out["effect_size"] = float((positive - negative) / total) if total else 0.0
    out["effect_size_name"] = "rank_biserial_matched_pairs"
    return out


class AblationStudy:
    """Run and aggregate the ablation and baseline comparison.

    Args:
        cohort: Cohort table.
        features: Standardised long-format feature table. Note that the scaler
            must be re-fitted per repeat by ``dataset_factory`` — a scaler fitted
            once over a fixed training split would leak into the other repeats'
            test splits.
        outputs_root: Root outputs directory.
        cfg: Framework configuration.
        dataset_factory: Callable ``(split, spec_uses_cnn) -> (train_ds, val_ds,
            test_ds)``. Injected so this class does not need to know how datasets
            are assembled and so the caller controls the per-repeat scaler fit.
    """

    def __init__(
        self,
        cohort: pd.DataFrame,
        outputs_root: Path,
        cfg: Optional[NeuroGenesisConfig] = None,
        dataset_factory: Optional[Callable] = None,
    ) -> None:
        self.cohort = cohort
        self.outputs_root = Path(outputs_root)
        self.cfg = cfg or NeuroGenesisConfig()
        self.dataset_factory = dataset_factory
        self.runs: List[VariantRun] = []
        self.summaries: Dict[str, VariantSummary] = {}
        self.splits: List[SplitManifest] = []
        self.failures: List[Dict[str, Any]] = []

    # ── Execution ─────────────────────────────────────────────────────────

    def run(
        self,
        variants: Optional[Sequence[str]] = None,
        n_repeats: Optional[int] = None,
        epochs: Optional[int] = None,
        reference: str = "A7",
        verbose: bool = False,
    ) -> Dict[str, VariantSummary]:
        """Train and evaluate every variant on every repeat.

        Args:
            variants: Variant keys. Defaults to the full A0-A7 ladder.
            n_repeats: Number of repeated splits. Defaults to
                ``cfg.data.n_repeats``.
            epochs: Override the epoch budget for every variant. All variants
                share it, so no variant gets a training advantage.
            reference: Variant that others are compared against.
            verbose: Per-epoch logging.

        Returns:
            ``{variant: VariantSummary}``.

        Raises:
            RuntimeError: If no ``dataset_factory`` was supplied.
        """
        if self.dataset_factory is None:
            raise RuntimeError(
                "AblationStudy needs a dataset_factory to build per-repeat "
                "datasets. See run.py for the standard factory."
            )
        from modules.m06_spatial_encoder.patch_dataset import make_loader
        from modules.m04_feature_extraction.feature_spec import FEATURE_ORDER

        variants = list(variants or ABLATION_LADDER)
        n_repeats = int(n_repeats if n_repeats is not None
                        else self.cfg.data.n_repeats)

        # Section 9: k-fold is the default because every subject is tested
        # exactly once per pass, so the spread across partitions reflects model
        # variance rather than the accident of who was held out. Repeated random
        # draws remain available for backward comparability.
        if self.cfg.data.split_scheme == "folds":
            self.splits = list(stratified_subject_folds(
                self.cohort,
                n_folds=self.cfg.data.n_folds,
                val_fraction=self.cfg.data.val_fraction,
                seed=self.cfg.repro.seed,
            ))
        else:
            self.splits = list(repeated_subject_splits(
                self.cohort,
                n_repeats=n_repeats,
                val_fraction=self.cfg.data.val_fraction,
                test_fraction=self.cfg.data.test_fraction,
                base_seed=self.cfg.repro.seed,
            ))
        logger.info(
            "Ablation: %d variant(s) x %d repeat(s) = %d training runs",
            len(variants), len(self.splits), len(variants) * len(self.splits),
        )

        self.runs = []
        self.failures = []

        for split in self.splits:
            for variant in variants:
                spec = ABLATION_SPECS.get(variant) or BASELINE_SPECS.get(variant)
                if spec is None:
                    self.failures.append(
                        {"variant": variant, "repeat": split.repeat,
                         "error": "unknown variant"}
                    )
                    continue
                # Re-seed per (variant, repeat) so initialisation is
                # reproducible and identical across variants for a given repeat.
                set_all_seeds(
                    self.cfg.repro.seed + 1000 * (split.repeat or 0),
                    deterministic=self.cfg.repro.deterministic,
                    cudnn_benchmark=self.cfg.repro.cudnn_benchmark,
                )
                try:
                    train_ds, val_ds, test_ds = self.dataset_factory(
                        split, spec.use_cnn
                    )
                    model = build_model(
                        len(FEATURE_ORDER), self.cfg, variant, list(FEATURE_ORDER)
                    )
                    trainer = Trainer(model, self.cfg)
                    result = trainer.fit(
                        make_loader(
                            train_ds, self.cfg.train.batch_size, shuffle=True,
                            seed=self.cfg.repro.seed + (split.repeat or 0),
                            drop_last=len(train_ds) > 2 * self.cfg.train.batch_size,
                        ),
                        make_loader(val_ds, self.cfg.train.batch_size),
                        epochs=epochs,
                        verbose=verbose,
                    )
                    evaluation = trainer.evaluate(
                        make_loader(test_ds, self.cfg.train.batch_size),
                        collect_predictions=False,
                    )
                    self.runs.append(VariantRun(
                        variant=variant,
                        repeat=split.repeat or 0,
                        seed=split.seed,
                        metrics=evaluation["metrics"],
                        best_epoch=result.best_epoch,
                        n_parameters=model.n_parameters(),
                        train_seconds=result.total_seconds,
                        stage_geometry_ordered=(
                            result.stage_geometry.get("ordering_respected")
                            if result.stage_geometry else None
                        ),
                        val_monitor=result.monitor,
                        val_best_value=result.best_value,
                        val_accuracy=_at_best(result, "val_accuracy"),
                        val_balanced_accuracy=_at_best(
                            result, "val_balanced_accuracy"
                        ),
                        val_macro_f1=_at_best(result, "val_macro_f1"),
                        warnings=result.warnings,
                    ))
                    logger.info(
                        "  %-10s repeat %d: acc %.4f  macro-F1 %.4f  bal-acc %.4f",
                        variant, split.repeat or 0,
                        evaluation["metrics"].accuracy,
                        evaluation["metrics"].macro_f1 or float("nan"),
                        evaluation["metrics"].balanced_accuracy or float("nan"),
                    )
                except Exception as exc:  # noqa: BLE001 - one variant must not
                    # abort the whole study; the failure is recorded and
                    # surfaced instead of silently reducing the repeat count.
                    logger.error("  %s repeat %s FAILED: %s", variant,
                                 split.repeat, exc)
                    self.failures.append({
                        "variant": variant, "repeat": split.repeat,
                        "error": f"{type(exc).__name__}: {exc}",
                    })

        return self.aggregate(reference=reference)

    # ── Aggregation ───────────────────────────────────────────────────────

    def aggregate(self, reference: str = "A7") -> Dict[str, VariantSummary]:
        """Aggregate completed runs into per-variant summaries."""
        by_variant: Dict[str, List[VariantRun]] = {}
        for run in self.runs:
            by_variant.setdefault(run.variant, []).append(run)

        ref_values: Dict[str, List[Optional[float]]] = {}
        if reference in by_variant:
            ordered = sorted(by_variant[reference], key=lambda r: r.repeat)
            for metric in METRIC_NAMES:
                ref_values[metric] = [r.metrics.get(metric) for r in ordered]

        self.summaries = {}
        for variant, runs in by_variant.items():
            runs = sorted(runs, key=lambda r: r.repeat)
            spec = ABLATION_SPECS.get(variant) or BASELINE_SPECS.get(variant)
            summary = VariantSummary(
                variant=variant,
                description=spec.description if spec else variant,
                n_repeats=len(runs),
                n_parameters=runs[0].n_parameters if runs else 0,
                aggregate=aggregate_metrics(
                    [r.metrics for r in runs], METRIC_NAMES
                ),
            )
            if variant != reference and ref_values:
                summary.comparison = {
                    metric: paired_comparison(
                        ref_values.get(metric, []),
                        [r.metrics.get(metric) for r in runs],
                        metric,
                    )
                    for metric in METRIC_NAMES
                }
                summary.notes.append(
                    f"Compared against {reference}, paired across the same "
                    "repeated splits."
                )
            geometry = [r.stage_geometry_ordered for r in runs
                        if r.stage_geometry_ordered is not None]
            if geometry and not all(geometry):
                summary.notes.append(
                    f"Stage prototype ordering was violated in "
                    f"{geometry.count(False)}/{len(geometry)} repeat(s); "
                    "propensity outputs from those repeats reflect "
                    "under-trained geometry."
                )
            self.summaries[variant] = summary

        expected = len(self.splits)
        for variant, summary in self.summaries.items():
            if expected and summary.n_repeats < expected:
                summary.notes.append(
                    f"Only {summary.n_repeats} of {expected} repeat(s) "
                    "completed; the remainder failed and are listed in the "
                    "failures record."
                )
        return self.summaries

    # ── Tables ────────────────────────────────────────────────────────────

    def table2(self) -> pd.DataFrame:
        """Return Table 2: main classification performance per method."""
        rows = []
        for variant in sorted(self.summaries):
            s = self.summaries[variant]
            rows.append({
                "Method": s.description,
                "Variant": variant,
                "Accuracy": s.formatted("accuracy"),
                "Macro-F1": s.formatted("macro_f1"),
                "Balanced Accuracy": s.formatted("balanced_accuracy"),
                "ROC-AUC": s.formatted("roc_auc_macro"),
                "Precision": s.formatted("macro_precision"),
                "Recall": s.formatted("macro_recall"),
                "Repeats": s.n_repeats,
                "Parameters": s.n_parameters,
            })
        return pd.DataFrame(rows)

    def table3(self, variant: str = "A7") -> pd.DataFrame:
        """Return Table 3: class-wise performance for one variant.

        Per-class values are averaged across repeats. A class that was undefined
        in a repeat is excluded from its own average, and the count of
        contributing repeats is shown so a reader can see when an average rests
        on fewer repeats than the others.
        """
        runs = [r for r in self.runs if r.variant == variant]
        if not runs:
            return pd.DataFrame()

        rows = []
        for index, stage in enumerate(STAGE_ORDER):
            cells: Dict[str, List[float]] = {
                k: [] for k in ("precision", "recall", "specificity", "f1", "auc")
            }
            supports = []
            for run in runs:
                if index >= len(run.metrics.per_class):
                    continue
                per_class = run.metrics.per_class[index]
                supports.append(per_class.support)
                for key in cells:
                    value = getattr(per_class, key)
                    if value is not None and np.isfinite(value):
                        cells[key].append(float(value))

            def fmt(key: str) -> str:
                values = cells[key]
                if not values:
                    return "n/a"
                sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                return f"{np.mean(values):.4f} +/- {sd:.4f}"

            rows.append({
                "Stage": stage,
                "Mean support": float(np.mean(supports)) if supports else 0.0,
                "Precision": fmt("precision"),
                "Recall": fmt("recall"),
                "Specificity": fmt("specificity"),
                "F1": fmt("f1"),
                "AUC": fmt("auc"),
                "Repeats contributing (F1)": len(cells["f1"]),
            })
        return pd.DataFrame(rows)

    def table4(self) -> pd.DataFrame:
        """Return Table 4: the NeuroProp-X ablation ladder in order."""
        ladder = [
            ("A0", "Baseline (morphometry only)"),
            ("A1", "+ Anatomical Prior"),
            ("A2", "+ Standard GAT"),
            ("A3", "+ Learned Attention (AP-LAF)"),
            ("A4", "+ SRVE"),
            ("A5", "+ ANP"),
            ("A7", "Full NeuroProp-X + SAEG-GATv2 + 3D CNN"),
        ]
        rows = []
        for variant, label in ladder:
            s = self.summaries.get(variant)
            if s is None:
                rows.append({"Configuration": label, "Variant": variant,
                             "Accuracy": "not run", "Macro-F1": "not run",
                             "Balanced Accuracy": "not run", "ROC-AUC": "not run"})
                continue
            rows.append({
                "Configuration": label,
                "Variant": variant,
                "Accuracy": s.formatted("accuracy"),
                "Macro-F1": s.formatted("macro_f1"),
                "Balanced Accuracy": s.formatted("balanced_accuracy"),
                "ROC-AUC": s.formatted("roc_auc_macro"),
            })
        return pd.DataFrame(rows)

    def table9(self, metric: str = "macro_f1",
               reference: str = "A7") -> pd.DataFrame:
        """Return Table 9: final model comparison with CI, p-value and effect size."""
        rows = []
        for variant in sorted(self.summaries):
            s = self.summaries[variant]
            ci_low = s.value(metric, "ci_low")
            ci_high = s.value(metric, "ci_high")
            comparison = (s.comparison or {}).get(metric, {}) if s.comparison \
                else {}
            rows.append({
                "Method": s.description,
                "Variant": variant,
                "Accuracy Mean+/-SD": s.formatted("accuracy"),
                "Macro-F1 Mean+/-SD": s.formatted("macro_f1"),
                "AUC Mean+/-SD": s.formatted("roc_auc_macro"),
                f"95% CI ({metric})": (
                    f"[{ci_low:.4f}, {ci_high:.4f}]"
                    if ci_low is not None and ci_high is not None else "n/a"
                ),
                "p-value (paired vs "
                f"{reference})": comparison.get("p_value"),
                "Effect Size": comparison.get("effect_size"),
                "Test": comparison.get("test", "reference"),
            })
        return pd.DataFrame(rows)

    # ── Persistence ───────────────────────────────────────────────────────

    @staticmethod
    def caveats() -> List[str]:
        """Return the mandatory caveats for every ablation table."""
        return [
            "Confidence intervals and p-values describe variability across "
            "repeated splits of one fixed cohort. The same subjects appear in "
            "every repeat, so these are not confidence intervals over the "
            "population of Alzheimer's patients.",
            "Comparisons against the reference variant are paired across "
            "identical splits and use the Wilcoxon signed-rank test, which does "
            "not assume normality of the paired differences.",
            "With a small AD class, a non-significant difference is not "
            "evidence that two variants perform equally.",
        ]

    def save(self, out_dir: Path, reference: str = "A7") -> Dict[str, Path]:
        """Persist runs, summaries and every table.

        Args:
            out_dir: Destination directory.
            reference: Reference variant for the comparison columns.

        Returns:
            Mapping of logical name -> written path.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written: Dict[str, Path] = {}

        payload = {
            "n_variants": len(self.summaries),
            "n_repeats": len(self.splits),
            "reference_variant": reference,
            "caveats": self.caveats(),
            "summaries": {k: v.to_dict() for k, v in self.summaries.items()},
            "runs": [r.to_dict() for r in self.runs],
            "failures": list(self.failures),
        }
        path = out_dir / "ablation_results.json"
        path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
        written["results_json"] = path

        for name, frame in (
            ("table2_main_performance", self.table2()),
            ("table3_class_wise", self.table3(reference)),
            ("table4_ablation", self.table4()),
            ("table9_model_comparison", self.table9(reference=reference)),
        ):
            if frame.empty:
                continue
            path = out_dir / f"{name}.csv"
            frame.to_csv(path, index=False)
            written[name] = path

        logger.info("Ablation results saved to %s (%d files)",
                    out_dir, len(written))
        return written


__all__ = [
    "METRIC_NAMES",
    "VariantRun",
    "VariantSummary",
    "paired_comparison",
    "AblationStudy",
]
