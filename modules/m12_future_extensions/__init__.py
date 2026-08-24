"""
M12 — Future extensions (quarantined legacy code).
==================================================

.. warning::

   **Nothing in this package is part of the validated current experiment.**

This package holds code from the previous NeuroGenesis research direction — a
self-evolving Digital Twin with 24-month longitudinal atrophy forecasting. That
direction assumed true longitudinal MRI sequences (OASIS-3), which the current
study does not have and does not assume. The code is retained per Section 21 so
that the work is not lost and so a future longitudinal extension has a starting
point, but it is quarantined for three concrete reasons:

1. **It requires data that does not exist here.** The forecasting and
   digital-twin state-update logic needs repeated visits per subject. OASIS-1 is
   cross-sectional.
2. **Its outputs were partly fabricated.** The archived NeuroProp-X engine
   computed vulnerability from hard-coded normative volumes and magic constants
   with no learnable parameters; the archived clinical report consumed
   hard-coded speech-assessment scores that OASIS-1 does not contain. Those
   numbers must never reach a results table.
3. **Its claims exceed what cross-sectional data supports.** "24-month atrophy
   forecast" and "counterfactual therapy simulation" are not validated by
   anything in the current study.

Contents
--------

``digital_twin/``
    Legacy per-subject state tracking and intervention simulation.
``temporal_longitudinal/``
    Legacy closed-form atrophy extrapolation, previously mislabelled a
    "Temporal Graph Transformer". Note that the *active* Stage-TGT in
    :mod:`modules.m07_stage_tgt` is an unrelated, genuine transformer that
    operates on ordered stage representations and requires no longitudinal data.
``oasis3/``
    Legacy OASIS-3 longitudinal manager. OASIS-3 is not assumed available.
``_legacy_backend/``
    Verbatim archive of the superseded ``backend/modules`` tree, kept for
    reference. The active equivalents live in ``modules/m01``..``m11``.

Access
------

Legacy classes are not re-exported. Loading one requires an explicit
acknowledgement, so a legacy result cannot enter the pipeline by an accidental
import::

    from modules.m12_future_extensions import load_legacy

    Twin = load_legacy("digital_twin", acknowledge_not_validated=True)
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any, Dict

#: Marker read by the report generator and dashboard. Any artifact traceable to
#: this package must be rendered under an explicit "not validated" banner.
IS_VALIDATED = False

#: Registry of loadable legacy components: alias -> (module path, attribute).
_LEGACY_REGISTRY: Dict[str, tuple] = {
    "digital_twin": (
        "modules.m12_future_extensions.digital_twin.legacy_digital_twin",
        "PatientDigitalTwin",
    ),
    "temporal_transformer": (
        "modules.m12_future_extensions.temporal_longitudinal."
        "legacy_temporal_transformer",
        "TemporalGraphTransformer",
    ),
    "oasis3_manager": (
        "modules.m12_future_extensions.oasis3.legacy_oasis3_manager",
        "OASIS3LongitudinalManager",
    ),
    "legacy_neuropropx": (
        "modules.m12_future_extensions._legacy_backend.legacy_neuropropx_engine",
        "NeuroPropXEngine",
    ),
}

_QUARANTINE_NOTICE = (
    "This component belongs to the superseded longitudinal / Digital Twin "
    "research direction. It is NOT part of the validated current experiment. "
    "Its outputs must not appear in any results table, figure or clinical "
    "report without an explicit 'not validated, future extension' label."
)


def available() -> list:
    """Return the aliases that :func:`load_legacy` accepts."""
    return sorted(_LEGACY_REGISTRY)


def load_legacy(alias: str, acknowledge_not_validated: bool = False) -> Any:
    """Load a quarantined legacy class by alias.

    Args:
        alias: One of :func:`available`.
        acknowledge_not_validated: Must be ``True``. The flag exists so that
            pulling legacy code into a run is a deliberate, greppable act rather
            than an ordinary import.

    Returns:
        The legacy class.

    Raises:
        PermissionError: If ``acknowledge_not_validated`` is not ``True``.
        KeyError: If the alias is unknown.
        ImportError: If the legacy module's own dependencies are unavailable.
    """
    if not acknowledge_not_validated:
        raise PermissionError(
            f"Refusing to load legacy component {alias!r}. {_QUARANTINE_NOTICE} "
            "Pass acknowledge_not_validated=True if you intend this."
        )
    if alias not in _LEGACY_REGISTRY:
        raise KeyError(
            f"Unknown legacy component {alias!r}. Available: {available()}"
        )

    module_path, attribute = _LEGACY_REGISTRY[alias]
    warnings.warn(
        f"Loading quarantined legacy component {alias!r}. {_QUARANTINE_NOTICE}",
        UserWarning,
        stacklevel=2,
    )
    module = importlib.import_module(module_path)
    return getattr(module, attribute)


__all__ = ["IS_VALIDATED", "available", "load_legacy"]
