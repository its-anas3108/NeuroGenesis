"""
JSON-safe serialisation of scientific values.
=============================================

Real neuroimaging headers hand back NumPy scalars and arrays, not Python
built-ins: an Analyze header's dimensions come through as ``int64``, voxel sizes
as ``float32``, the affine as an ``ndarray``. ``json.dumps`` rejects all of
them.

This matters more than it looks. The failure only appears once *real* data is
used — synthetic fixtures built from Python literals serialise fine — so it
surfaces at the worst moment, part-way through a long run, after the expensive
work is already done. :func:`json_safe` is applied at every artifact write so a
manifest can never fail to save because of a value's dtype.

Precision is preserved: NumPy integers become Python ``int``, floats become
Python ``float``, arrays become nested lists. Nothing is rounded or truncated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def json_safe(value: Any) -> Any:
    """Recursively convert a value into something ``json.dumps`` accepts.

    Args:
        value: Any Python, NumPy or path-like value.

    Returns:
        An equivalent structure of JSON-native types.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, float):
        # `np.float64` subclasses Python `float`, so this branch catches NumPy
        # doubles too and must therefore handle non-finite values itself.
        # NaN and infinity are not valid JSON: `json.dumps` emits bare `NaN`
        # and `Infinity`, which strict parsers reject. They become null, which
        # preserves the fact that the value was undefined.
        return value if np.isfinite(value) else None
    if isinstance(value, int):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        result = float(value)
        return result if np.isfinite(result) else None
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "tolist"):
        try:
            return json_safe(value.tolist())
        except Exception:  # pragma: no cover - defensive
            return str(value)
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:  # pragma: no cover - defensive
            return str(value)
    return str(value)


def json_default(value: Any) -> Any:
    """``default=`` hook for :func:`json.dumps`."""
    return json_safe(value)


def dump_json(payload: Any, path: Path, indent: int = 2) -> Path:
    """Write ``payload`` to ``path`` as JSON, coercing scientific types.

    Args:
        payload: The object to serialise.
        path: Destination file. Parent directories are created.
        indent: Indentation level.

    Returns:
        The written path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), indent=indent), encoding="utf-8"
    )
    return path


__all__ = ["json_safe", "json_default", "dump_json"]
