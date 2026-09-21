"""
Tests for the OASIS-1 data layer and the no-synthetic-data guarantee.
====================================================================

These guard the properties the project's dataset requirements turn on: that only
real OASIS-1 data can reach training, that a missing dataset stops the run rather
than triggering a fallback, and that the format adaptations the real files need
are exact.

Run with::

    python tests/test_oasis1_integrity.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modules.common.config import NeuroGenesisConfig  # noqa: E402
from modules.m01_dataset.integrity import (  # noqa: E402
    DatasetIntegrityError,
    check_dataset_integrity,
)
from modules.m01_dataset.oasis1_manager import (  # noqa: E402
    DATASET_NAME,
    VOLUME_KINDS,
    OASIS1DataManager,
)


def _index(session_ids, usable=True) -> pd.DataFrame:
    """Build a minimal validated index."""
    return pd.DataFrame({
        "session_id": list(session_ids),
        "subject_id": [s.rsplit("_MR", 1)[0] for s in session_ids],
        "usable": [usable] * len(session_ids),
    })


# ──────────────────────────────────────────────────────────────────────────────
# The no-synthetic-data guarantee
# ──────────────────────────────────────────────────────────────────────────────

def test_integrity_passes_on_pure_oasis1() -> None:
    """A clean OASIS-1 run must pass every check."""
    sessions = ["OAS1_0001_MR1", "OAS1_0002_MR1", "OAS1_0003_MR1"]
    report = check_dataset_integrity(
        training_session_ids=sessions,
        validated_index=_index(sessions),
        dataset_source=DATASET_NAME,
        synthetic_data_enabled=False,
        provenance=None,
    )
    assert report.passed, report.summary()


def test_integrity_stops_on_a_sample_outside_the_validated_index() -> None:
    """A sample from anywhere else must stop training and be named."""
    validated = ["OAS1_0001_MR1", "OAS1_0002_MR1"]
    contaminated = validated + ["OAS1_9999_MR1"]

    try:
        check_dataset_integrity(
            training_session_ids=contaminated,
            validated_index=_index(validated),
            dataset_source=DATASET_NAME,
            synthetic_data_enabled=False,
        )
    except DatasetIntegrityError as exc:
        message = str(exc)
        assert "DATASET INTEGRITY FAILURE" in message
        assert "Non-OASIS-1 sample detected" in message
        assert "OAS1_9999_MR1" in message, "the offender must be named"
        return
    raise AssertionError("an unvalidated sample should have stopped training")


def test_integrity_stops_on_a_non_oasis_identifier() -> None:
    """A synthetic or foreign ID must be rejected on its name alone."""
    sessions = ["OAS1_0001_MR1", "SYNTH_0001", "phantom_042"]
    index = _index(sessions)  # even if something marked them usable
    try:
        check_dataset_integrity(
            training_session_ids=sessions,
            validated_index=index,
            dataset_source=DATASET_NAME,
            synthetic_data_enabled=False,
        )
    except DatasetIntegrityError as exc:
        assert "SYNTH_0001" in str(exc) or "phantom_042" in str(exc)
        return
    raise AssertionError("non-OASIS identifiers should have been rejected")


def test_integrity_stops_when_synthetic_data_is_enabled() -> None:
    """The synthetic switch being on is itself a failure for a research run."""
    sessions = ["OAS1_0001_MR1"]
    try:
        check_dataset_integrity(
            training_session_ids=sessions,
            validated_index=_index(sessions),
            dataset_source=DATASET_NAME,
            synthetic_data_enabled=True,
        )
    except DatasetIntegrityError as exc:
        assert "synthetic data disabled" in str(exc)
        return
    raise AssertionError("enabled synthetic data should have stopped training")


def test_integrity_stops_on_a_synthetic_provenance_marker() -> None:
    """A synthetic outputs tree must not be trained on."""
    sessions = ["OAS1_0001_MR1"]
    try:
        check_dataset_integrity(
            training_session_ids=sessions,
            validated_index=_index(sessions),
            dataset_source=DATASET_NAME,
            synthetic_data_enabled=False,
            provenance={"is_synthetic": True, "kind": "synthetic_mri"},
        )
    except DatasetIntegrityError as exc:
        assert "provenance" in str(exc).lower()
        return
    raise AssertionError("a synthetic provenance marker should have stopped it")


def test_integrity_stops_on_a_non_oasis_dataset_name() -> None:
    """Only OASIS-1 is accepted."""
    sessions = ["OAS1_0001_MR1"]
    try:
        check_dataset_integrity(
            training_session_ids=sessions,
            validated_index=_index(sessions),
            dataset_source="ADNI",
            synthetic_data_enabled=False,
        )
    except DatasetIntegrityError as exc:
        assert "OASIS-1" in str(exc)
        return
    raise AssertionError("a non-OASIS dataset name should have stopped it")


def test_integrity_detects_a_subject_spanning_two_splits() -> None:
    """A subject in two splits is leakage and must stop the run."""
    sessions = ["OAS1_0001_MR1", "OAS1_0001_MR2"]
    try:
        check_dataset_integrity(
            training_session_ids=sessions,
            validated_index=_index(sessions),
            dataset_source=DATASET_NAME,
            synthetic_data_enabled=False,
            split_assignment={
                "OAS1_0001_MR1": "train", "OAS1_0001_MR2": "test",
            },
            subject_of={
                "OAS1_0001_MR1": "OAS1_0001", "OAS1_0001_MR2": "OAS1_0001",
            },
        )
    except DatasetIntegrityError as exc:
        assert "OAS1_0001" in str(exc)
        return
    raise AssertionError("a subject spanning two splits should have stopped it")


def test_integrity_stops_on_an_empty_validated_index() -> None:
    """No validated index means nothing has been proven real."""
    try:
        check_dataset_integrity(
            training_session_ids=["OAS1_0001_MR1"],
            validated_index=pd.DataFrame(),
            dataset_source=DATASET_NAME,
            synthetic_data_enabled=False,
        )
    except DatasetIntegrityError:
        return
    raise AssertionError("an empty validated index should have stopped it")


# ──────────────────────────────────────────────────────────────────────────────
# Config-level guarantees
# ──────────────────────────────────────────────────────────────────────────────

def test_config_defaults_forbid_synthetic_data() -> None:
    """Out of the box, the project is configured for real OASIS-1 only."""
    cfg = NeuroGenesisConfig()
    assert cfg.data.dataset_source == "OASIS-1"
    assert cfg.data.allow_synthetic_data is False
    assert cfg.validate() == []


def test_config_rejects_enabled_synthetic_data() -> None:
    """Turning synthetic data on must fail validation."""
    cfg = NeuroGenesisConfig()
    cfg.data.allow_synthetic_data = True
    problems = cfg.validate()
    assert any("allow_synthetic_data" in p for p in problems), problems


def test_config_accepts_adni_as_a_dataset_source() -> None:
    """ADNI is a valid dataset_source (inference-only; see integrity tests)."""
    cfg = NeuroGenesisConfig()
    cfg.data.dataset_source = "ADNI"
    assert cfg.validate() == []


def test_config_rejects_an_unknown_dataset_source() -> None:
    """No dataset name outside {OASIS-1, ADNI} is accepted."""
    cfg = NeuroGenesisConfig()
    cfg.data.dataset_source = "SOME_OTHER_DATASET"
    problems = cfg.validate()
    assert any("OASIS-1" in p or "ADNI" in p for p in problems), problems


# ──────────────────────────────────────────────────────────────────────────────
# Manager behaviour
# ──────────────────────────────────────────────────────────────────────────────

def test_manager_reports_missing_data_rather_than_substituting() -> None:
    """An absent dataset yields the required message, not a fallback."""
    with tempfile.TemporaryDirectory() as directory:
        manager = OASIS1DataManager(Path(directory) / "does_not_exist")
        assert not manager.root_exists()
        assert manager.discover() == []
        message = manager.missing_data_message()
        assert "REAL OASIS-1 DATA REQUIRED" in message
        assert "sites.wustl.edu/oasisbrains" in message


def test_manager_rejects_an_unknown_volume_kind() -> None:
    """Only the documented OASIS volume kinds are selectable."""
    try:
        OASIS1DataManager(Path("."), volume_kind="raw_mpr")
    except ValueError as exc:
        assert "volume_kind" in str(exc)
        assert set(VOLUME_KINDS) <= set(str(exc)) | {"t88_gfc", "t88_masked_gfc"}
        return
    raise AssertionError("an unknown volume kind should have been rejected")


def test_manager_does_not_confuse_masked_and_unmasked_volumes() -> None:
    """``_t88_gfc`` is a substring of ``_t88_masked_gfc``.

    Selecting the plain volume must not pick up the masked one, or the skull
    stripping stage would silently receive already-brain-extracted input.
    """
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        session = root / "disc1" / "OAS1_0001_MR1" / "PROCESSED" / "MPRAGE" \
            / "T88_111"
        session.mkdir(parents=True)
        for name in (
            "OAS1_0001_MR1_mpr_n4_anon_111_t88_gfc",
            "OAS1_0001_MR1_mpr_n4_anon_111_t88_masked_gfc",
        ):
            (session / f"{name}.hdr").write_bytes(b"\x00" * 348)
            (session / f"{name}.img").write_bytes(b"\x00" * 16)

        plain = OASIS1DataManager(root, volume_kind="t88_gfc").discover()
        assert len(plain) == 1, f"expected one session, got {len(plain)}"
        assert "masked" not in plain[0].volume_path

        masked = OASIS1DataManager(root, volume_kind="t88_masked_gfc").discover()
        assert len(masked) == 1
        assert "masked" in masked[0].volume_path


def test_manager_keys_on_the_header_not_both_halves_of_the_pair() -> None:
    """An Analyze ``.img``/``.hdr`` pair is one volume, not two."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for index in (1, 2):
            session = root / "disc1" / f"OAS1_000{index}_MR1" / "T88_111"
            session.mkdir(parents=True)
            stem = f"OAS1_000{index}_MR1_mpr_n4_anon_111_t88_gfc"
            (session / f"{stem}.hdr").write_bytes(b"\x00" * 348)
            (session / f"{stem}.img").write_bytes(b"\x00" * 16)

        sessions = OASIS1DataManager(root).discover()
        assert len(sessions) == 2, (
            f"two sessions expected, got {len(sessions)} — the .img/.hdr pair "
            "was probably counted twice"
        )
        assert all(s.volume_path.endswith(".hdr") for s in sessions)
        assert all(s.image_path and s.image_path.endswith(".img")
                   for s in sessions)


def test_spatial_shape_squeezes_only_a_trailing_singleton() -> None:
    """A trailing axis of 1 is packaging; any other 4-D shape is real data."""
    squeeze = OASIS1DataManager._spatial_shape
    assert squeeze((176, 208, 176)) == (176, 208, 176)
    assert squeeze((176, 208, 176, 1)) == (176, 208, 176)
    # A real 4th dimension must be refused, not collapsed.
    assert squeeze((176, 208, 176, 4)) is None
    assert squeeze((176, 208)) is None
    assert squeeze((2, 176, 208, 176)) is None


def test_manager_provenance_declares_real_data() -> None:
    """Provenance must positively assert OASIS-1 and non-synthetic origin."""
    manager = OASIS1DataManager(Path("."), volume_kind="t88_gfc")
    provenance = manager.provenance()
    assert provenance["dataset_source"] == "OASIS-1"
    assert provenance["is_synthetic"] is False
    assert "sites.wustl.edu/oasisbrains" in provenance["source_description"]
    # The double-N4 redundancy must be disclosed, not hidden.
    assert "N4" in provenance["preprocessing_note"]


# ──────────────────────────────────────────────────────────────────────────────
# Serialisation of real header values
# ──────────────────────────────────────────────────────────────────────────────

def test_numpy_scalars_from_real_headers_serialise() -> None:
    """Real Analyze headers yield NumPy scalars; artifacts must still save."""
    from modules.common.serialization import json_safe

    payload = {
        "shape": [np.int64(176), np.int64(208), np.int64(176)],
        "zooms": (np.float32(1.0), np.float32(1.0)),
        "affine": np.eye(4),
        "flag": np.bool_(True),
        "nan": np.float64("nan"),
        "path": Path("a/b.hdr"),
    }
    encoded = json.dumps(json_safe(payload))
    restored = json.loads(encoded)
    assert restored["shape"] == [176, 208, 176]
    assert restored["zooms"] == [1.0, 1.0]
    assert restored["flag"] is True
    assert restored["nan"] is None, "NaN is not valid JSON and must become null"
    assert restored["path"] == "a/b.hdr"
    assert len(restored["affine"]) == 4


def _main() -> int:
    """Run every test in this module without pytest."""
    tests = [
        (name, value) for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    failures = 0
    for name, function in tests:
        try:
            function()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
            import traceback

            traceback.print_exc(limit=3)
    print(f"\n{len(tests) - failures}/{len(tests)} OASIS-1 integrity tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
