"""M19 — Subject report generation with enforced language discipline."""

from modules.m11_report.report import (
    DISCLAIMER,
    PROHIBITED_PHRASES,
    ReportGenerator,
    ReportInputs,
    validate_language,
)

__all__ = [
    "ReportGenerator",
    "ReportInputs",
    "DISCLAIMER",
    "PROHIBITED_PHRASES",
    "validate_language",
]
