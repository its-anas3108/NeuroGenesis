"""
M17 — Stage-wise statistical analysis (Section 16).
==================================================

For every ROI x feature combination, tests the three stage contrasts
CN vs MCI, MCI vs AD and CN vs AD, and reports p-values, FDR-adjusted p-values,
effect sizes and the direction of the trend.

Test selection is data-driven, not fixed
----------------------------------------

Section 16 explicitly forbids blindly using t-tests. For each contrast the
procedure is:

1. **Screen normality** in each group with the Shapiro-Wilk test (or D'Agostino
   for larger groups, where Shapiro-Wilk becomes over-sensitive).
2. **Screen equality of variance** with Levene's test, which is robust to
   non-normality.
3. **Choose accordingly:**

   * both groups plausibly normal, variances comparable -> **Student's t-test**
   * both plausibly normal, variances differ -> **Welch's t-test**
   * either group non-normal -> **Mann-Whitney U** (rank-based)

   The chosen test is recorded per cell in the ``test`` column, so a reader can
   see exactly what was run rather than having to trust a blanket claim.

4. **Guard the small-sample case.** With AD n=30 in the full cohort — and fewer
   in any subgroup — a group below ``min_group_n`` is not tested at all. The cell
   reports ``insufficient_data`` instead of a p-value. A normality screen on
   n=4 has almost no power, so "we could not reject normality" would be
   meaningless there.

Effect size matches the test
----------------------------

Hedges' *g* (bias-corrected Cohen's *d*) accompanies a parametric test; rank
biserial correlation accompanies Mann-Whitney. Reporting Cohen's *d* next to a
rank-based p-value would pair an assumption-laden effect size with a test chosen
precisely because those assumptions failed.

Multiple comparisons
--------------------

Benjamini-Hochberg FDR is applied across **all** ROI x feature x contrast tests
in one family, which is the honest correction: with 5 ROIs x 14 features x 3
contrasts there are up to 210 tests, and per-contrast correction would leave the
family-wise error rate uncontrolled. Both raw and adjusted p-values are
reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from modules.common.config import StatsConfig
from modules.common.logging_utils import get_logger
from modules.common.roi_constants import (
    ROI_ORDER,
    STAGE_ORDER,
    roi_short,
    stage_pairs,
)
from modules.m04_feature_extraction.feature_spec import (
    FEATURE_DEFINITIONS,
    FEATURE_ORDER,
)

logger = get_logger(__name__)


@dataclass
class TestResult:
    """One statistical comparison between two stages for one ROI x feature."""

    roi: str
    feature: str
    group_a: str
    group_b: str
    n_a: int
    n_b: int
    mean_a: Optional[float] = None
    sd_a: Optional[float] = None
    mean_b: Optional[float] = None
    sd_b: Optional[float] = None
    median_a: Optional[float] = None
    median_b: Optional[float] = None
    test: str = "not_run"
    statistic: Optional[float] = None
    p_value: Optional[float] = None
    p_adjusted: Optional[float] = None
    effect_size: Optional[float] = None
    effect_size_name: Optional[str] = None
    #: ``"increase"``, ``"decrease"`` or ``None`` — direction from A to B.
    trend: Optional[str] = None
    normality_a_p: Optional[float] = None
    normality_b_p: Optional[float] = None
    variance_equal_p: Optional[float] = None
    note: Optional[str] = None

    @property
    def significant(self) -> bool:
        """True if the FDR-adjusted p-value is available and below 0.05."""
        return self.p_adjusted is not None and self.p_adjusted < 0.05

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "roi": self.roi, "roi_short": roi_short(self.roi),
            "feature": self.feature,
            "contrast": f"{self.group_a} vs {self.group_b}",
            "n_a": self.n_a, "n_b": self.n_b,
            "mean_a": self.mean_a, "sd_a": self.sd_a,
            "mean_b": self.mean_b, "sd_b": self.sd_b,
            "median_a": self.median_a, "median_b": self.median_b,
            "test": self.test, "statistic": self.statistic,
            "p_value": self.p_value, "p_adjusted": self.p_adjusted,
            "effect_size": self.effect_size,
            "effect_size_name": self.effect_size_name,
            "trend": self.trend,
            "normality_a_p": self.normality_a_p,
            "normality_b_p": self.normality_b_p,
            "variance_equal_p": self.variance_equal_p,
            "significant_fdr": self.significant,
            "note": self.note,
        }


def hedges_g(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Bias-corrected standardised mean difference (Hedges' g).

    Cohen's *d* is upward-biased at small *n*, and this study's AD group is
    small enough for the correction to matter.

    Returns:
        Hedges' *g*, or ``None`` when the pooled standard deviation is zero.
    """
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return None
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)
    pooled = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    if pooled <= 0:
        return None
    d = (b.mean() - a.mean()) / np.sqrt(pooled)
    correction = 1.0 - (3.0 / (4.0 * (n_a + n_b) - 9.0))
    return float(d * correction)


def rank_biserial(a: np.ndarray, b: np.ndarray,
                  u_statistic: float) -> Optional[float]:
    """Rank biserial correlation, the effect size matching Mann-Whitney U.

    Ranges over ``[-1, 1]``: 0 means complete overlap, +/-1 complete separation.

    Args:
        a: First group.
        b: Second group.
        u_statistic: The U statistic for group ``a`` as returned by
            ``scipy.stats.mannwhitneyu(a, b)``.

    Returns:
        The correlation, or ``None`` if either group is empty.
    """
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return None
    # Sign flipped so that a positive value means group b tends to be larger,
    # matching the sign convention of Hedges' g above.
    return float(-(2.0 * u_statistic / (n_a * n_b) - 1.0))


def _normality_p(values: np.ndarray) -> Optional[float]:
    """Return a normality-test p-value, choosing the test by sample size."""
    n = len(values)
    if n < 3:
        return None
    if float(np.std(values)) == 0.0:
        # A constant vector is degenerate; Shapiro-Wilk raises on it.
        return 0.0
    try:
        if n <= 50:
            return float(stats.shapiro(values).pvalue)
        # Shapiro-Wilk rejects almost any large real sample; D'Agostino's
        # omnibus test is the appropriate screen above ~50 observations.
        return float(stats.normaltest(values).pvalue)
    except ValueError:
        return None


def compare_groups(
    values_a: np.ndarray,
    values_b: np.ndarray,
    roi: str,
    feature: str,
    group_a: str,
    group_b: str,
    cfg: Optional[StatsConfig] = None,
) -> TestResult:
    """Compare one ROI x feature between two stages with an appropriate test.

    Args:
        values_a: Feature values for the first stage.
        values_b: Feature values for the second stage.
        roi: ROI name.
        feature: Feature name.
        group_a: First stage name.
        group_b: Second stage name.
        cfg: Statistics configuration.

    Returns:
        A :class:`TestResult`. When a group is too small the result carries
        ``test="insufficient_data"`` and no p-value — never a fabricated one.
    """
    cfg = cfg or StatsConfig()
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    result = TestResult(
        roi=roi, feature=feature, group_a=group_a, group_b=group_b,
        n_a=int(a.size), n_b=int(b.size),
    )
    if a.size:
        result.mean_a = float(a.mean())
        result.sd_a = float(a.std(ddof=1)) if a.size > 1 else 0.0
        result.median_a = float(np.median(a))
    if b.size:
        result.mean_b = float(b.mean())
        result.sd_b = float(b.std(ddof=1)) if b.size > 1 else 0.0
        result.median_b = float(np.median(b))

    if a.size < cfg.min_group_n or b.size < cfg.min_group_n:
        result.test = "insufficient_data"
        result.note = (
            f"At least one group is below the minimum of {cfg.min_group_n} "
            f"(n={a.size}, {b.size}). No test was run: a normality screen at "
            "this size has negligible power, so any test choice would be "
            "arbitrary."
        )
        return result

    if float(np.std(np.concatenate([a, b]))) == 0.0:
        result.test = "constant"
        result.note = (
            "The feature is constant across both groups; there is nothing to "
            "test."
        )
        return result

    result.normality_a_p = _normality_p(a)
    result.normality_b_p = _normality_p(b)
    try:
        result.variance_equal_p = float(stats.levene(a, b).pvalue)
    except ValueError:
        result.variance_equal_p = None

    alpha = cfg.normality_alpha
    normal_a = result.normality_a_p is not None and result.normality_a_p >= alpha
    normal_b = result.normality_b_p is not None and result.normality_b_p >= alpha
    equal_var = (
        result.variance_equal_p is not None
        and result.variance_equal_p >= alpha
    )

    try:
        if normal_a and normal_b:
            outcome = stats.ttest_ind(a, b, equal_var=equal_var)
            result.test = "students_t" if equal_var else "welch_t"
            result.statistic = float(outcome.statistic)
            result.p_value = float(outcome.pvalue)
            result.effect_size = hedges_g(a, b)
            result.effect_size_name = "hedges_g"
        else:
            outcome = stats.mannwhitneyu(a, b, alternative="two-sided")
            result.test = "mann_whitney_u"
            result.statistic = float(outcome.statistic)
            result.p_value = float(outcome.pvalue)
            result.effect_size = rank_biserial(a, b, float(outcome.statistic))
            result.effect_size_name = "rank_biserial"
            result.note = (
                "Rank-based test chosen because normality was rejected in "
                + ("both groups" if not normal_a and not normal_b
                   else f"group {group_a if not normal_a else group_b}")
                + "."
            )
    except ValueError as exc:
        result.test = "failed"
        result.note = f"Test failed: {exc}"
        return result

    if result.mean_a is not None and result.mean_b is not None:
        if result.mean_b > result.mean_a:
            result.trend = "increase"
        elif result.mean_b < result.mean_a:
            result.trend = "decrease"
        else:
            result.trend = "no_change"

    return result


def benjamini_hochberg(p_values: Sequence[Optional[float]]
                       ) -> List[Optional[float]]:
    """Benjamini-Hochberg FDR adjustment, preserving ``None`` entries.

    Implemented directly rather than via ``statsmodels`` so that the statistics
    module has no optional dependency and so that untested cells stay ``None``
    instead of being dropped or imputed.

    Args:
        p_values: Raw p-values; ``None`` for cells that were not tested.

    Returns:
        Adjusted p-values in the input order, with ``None`` where the input was
        ``None``.
    """
    indexed = [(i, p) for i, p in enumerate(p_values)
               if p is not None and np.isfinite(p)]
    out: List[Optional[float]] = [None] * len(p_values)
    if not indexed:
        return out

    indexed.sort(key=lambda kv: kv[1])
    m = len(indexed)
    # Step-up procedure with the monotonicity enforcement that keeps adjusted
    # p-values non-decreasing in rank.
    previous = 1.0
    for rank in range(m, 0, -1):
        original_index, p = indexed[rank - 1]
        adjusted = min(previous, p * m / rank)
        out[original_index] = float(min(adjusted, 1.0))
        previous = out[original_index]
    return out


@dataclass
class StatisticsReport:
    """Full stage-wise statistical analysis."""

    results: List[TestResult] = field(default_factory=list)
    n_tests_run: int = 0
    n_tests_skipped: int = 0
    n_significant: int = 0
    group_sizes: Dict[str, int] = field(default_factory=dict)
    fdr_method: str = "benjamini_hochberg"
    test_counts: Dict[str, int] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        """Return the long-format results table."""
        return pd.DataFrame([r.to_dict() for r in self.results])

    def table5(self) -> pd.DataFrame:
        """Return Table 5: stage-wise ROI statistical analysis.

        One row per ROI x feature, with a column per contrast.
        """
        by_cell: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for r in self.results:
            key = (r.roi, r.feature)
            row = by_cell.setdefault(key, {
                "ROI": roi_short(r.roi),
                "Feature": r.feature,
            })
            label = f"{r.group_a} vs {r.group_b}"
            row[f"{label} p"] = r.p_value
            row[f"{label} FDR p"] = r.p_adjusted
            row[f"{label} effect"] = r.effect_size
            row[f"{label} test"] = r.test
            if label == "CN vs AD":
                row["Trend"] = r.trend
                row["Effect Size"] = r.effect_size
                row["FDR p"] = r.p_adjusted
        return pd.DataFrame(list(by_cell.values()))

    def table7(self) -> pd.DataFrame:
        """Return Table 7: morphometric differences with mean +/- SD per stage."""
        cells: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for r in self.results:
            key = (r.roi, r.feature)
            row = cells.setdefault(key, {
                "ROI": roi_short(r.roi), "Feature": r.feature,
            })

            def fmt(mean: Optional[float], sd: Optional[float]) -> Optional[str]:
                if mean is None:
                    return None
                return f"{mean:.4g} +/- {sd:.4g}" if sd is not None \
                    else f"{mean:.4g}"

            row[f"{r.group_a} Mean+/-SD"] = fmt(r.mean_a, r.sd_a)
            row[f"{r.group_b} Mean+/-SD"] = fmt(r.mean_b, r.sd_b)
            if r.group_a == "CN" and r.group_b == "AD":
                row["p-value"] = r.p_value
                row["FDR p"] = r.p_adjusted
                row["Effect Size"] = r.effect_size
                row["Test"] = r.test
        return pd.DataFrame(list(cells.values()))

    def significant_findings(self) -> List[Dict[str, Any]]:
        """Return FDR-significant results, largest effect first."""
        found = [r.to_dict() for r in self.results if r.significant]
        found.sort(
            key=lambda r: abs(r["effect_size"] or 0.0), reverse=True
        )
        return found

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "n_tests_run": self.n_tests_run,
            "n_tests_skipped": self.n_tests_skipped,
            "n_significant_fdr": self.n_significant,
            "group_sizes": dict(self.group_sizes),
            "fdr_method": self.fdr_method,
            "test_counts": dict(self.test_counts),
            "results": [r.to_dict() for r in self.results],
            "notes": list(self.notes),
        }

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Group sizes: {self.group_sizes}",
            f"Tests run: {self.n_tests_run} | skipped: "
            f"{self.n_tests_skipped} | FDR-significant: {self.n_significant}",
            f"Tests used: {self.test_counts}",
        ]
        for n in self.notes:
            lines.append(f"NOTE: {n}")
        return "\n".join(lines)


def run_stagewise_analysis(
    features: pd.DataFrame,
    cohort: pd.DataFrame,
    cfg: Optional[StatsConfig] = None,
    feature_list: Optional[List[str]] = None,
    session_ids: Optional[List[str]] = None,
) -> StatisticsReport:
    """Run the full stage-wise analysis over every ROI x feature x contrast.

    Args:
        features: Long-format feature table with ``session_id``, ``roi_name``
            and feature columns. **Unstandardised** values should be used so
            that means and SDs are in interpretable physical units.
        cohort: Cohort table with ``session_id`` and ``stage``.
        cfg: Statistics configuration.
        feature_list: Features to test. Defaults to
            :data:`~modules.m04_feature_extraction.feature_spec.FEATURE_ORDER`.
        session_ids: Restrict to these sessions. Use the training split when the
            analysis must stay independent of the test set; use the full cohort
            for a descriptive characterisation, which is the usual choice for a
            table describing the *data* rather than evaluating the model.

    Returns:
        A :class:`StatisticsReport` with FDR applied across the whole family.

    Raises:
        KeyError: If a required column is missing.
    """
    cfg = cfg or StatsConfig()
    for col in ("session_id", "roi_name"):
        if col not in features.columns:
            raise KeyError(f"features is missing required column {col!r}")
    for col in ("session_id", "stage"):
        if col not in cohort.columns:
            raise KeyError(f"cohort is missing required column {col!r}")

    feature_list = [
        f for f in (feature_list or FEATURE_ORDER) if f in features.columns
    ]
    report = StatisticsReport(fdr_method=cfg.fdr_method)

    stage_map = dict(zip(cohort["session_id"].astype(str),
                         cohort["stage"].astype(str)))
    table = features.copy()
    table["session_id"] = table["session_id"].astype(str)
    if session_ids is not None:
        table = table[table["session_id"].isin(set(map(str, session_ids)))]
    table["stage"] = table["session_id"].map(stage_map)
    table = table[table["stage"].notna()]

    report.group_sizes = {
        stage: int(table.loc[table["stage"] == stage, "session_id"].nunique())
        for stage in STAGE_ORDER
    }

    if table.empty:
        report.notes.append(
            "No feature rows could be matched to a labelled session; no test "
            "was run."
        )
        return report

    for roi in ROI_ORDER:
        roi_rows = table[table["roi_name"] == roi]
        if roi_rows.empty:
            report.notes.append(f"No feature rows for ROI {roi}.")
            continue
        for feature in feature_list:
            for stage_a, stage_b in stage_pairs():
                values_a = pd.to_numeric(
                    roi_rows.loc[roi_rows["stage"] == stage_a, feature],
                    errors="coerce",
                ).values
                values_b = pd.to_numeric(
                    roi_rows.loc[roi_rows["stage"] == stage_b, feature],
                    errors="coerce",
                ).values
                report.results.append(compare_groups(
                    values_a, values_b, roi, feature, stage_a, stage_b, cfg
                ))

    raw = [r.p_value for r in report.results]
    adjusted = benjamini_hochberg(raw)
    for r, p in zip(report.results, adjusted):
        r.p_adjusted = p

    report.n_tests_run = sum(1 for r in report.results if r.p_value is not None)
    report.n_tests_skipped = len(report.results) - report.n_tests_run
    report.n_significant = sum(1 for r in report.results if r.significant)
    counts: Dict[str, int] = {}
    for r in report.results:
        counts[r.test] = counts.get(r.test, 0) + 1
    report.test_counts = counts

    report.notes.append(
        f"Benjamini-Hochberg FDR was applied across all {report.n_tests_run} "
        "tests in one family (every ROI x feature x contrast), not per contrast."
    )
    if report.n_tests_skipped:
        report.notes.append(
            f"{report.n_tests_skipped} cell(s) were not tested because a group "
            f"fell below the minimum of {cfg.min_group_n} or the feature was "
            "constant. Those cells carry no p-value rather than a fabricated one."
        )
    smallest = min(
        (v for v in report.group_sizes.values() if v > 0), default=0
    )
    if 0 < smallest < 30:
        report.notes.append(
            f"The smallest stage group has {smallest} subject(s). Effect-size "
            "estimates at this sample size have wide confidence intervals, and "
            "a non-significant result should not be read as evidence of no "
            "difference."
        )

    logger.info("Stage-wise statistics:\n%s", report.summary())
    return report


__all__ = [
    "TestResult",
    "StatisticsReport",
    "hedges_g",
    "rank_biserial",
    "compare_groups",
    "benjamini_hochberg",
    "run_stagewise_analysis",
]
