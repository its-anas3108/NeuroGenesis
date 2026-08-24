"""M17 — Stage-wise statistical analysis with FDR correction and effect sizes."""

from modules.m09_statistics.stagewise import (
    StatisticsReport,
    TestResult,
    benjamini_hochberg,
    compare_groups,
    hedges_g,
    rank_biserial,
    run_stagewise_analysis,
)

__all__ = [
    "TestResult",
    "StatisticsReport",
    "hedges_g",
    "rank_biserial",
    "compare_groups",
    "benjamini_hochberg",
    "run_stagewise_analysis",
]
