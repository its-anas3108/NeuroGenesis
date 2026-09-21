"""
Invariant tests for the properties that must not silently regress.
=================================================================

These are not coverage tests. Each one guards a property whose violation would
produce plausible-looking but wrong results — the failure mode that is hardest to
notice by reading output.

Run with::

    python -m pytest tests/test_invariants.py -v
    # or without pytest:
    python tests/test_invariants.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modules.common.config import NeuroGenesisConfig  # noqa: E402
from modules.common.roi_constants import (  # noqa: E402
    N_ROI,
    ROI_ORDER,
    STAGE_ORDER,
    validate_roi_order,
)
from modules.common.seeds import set_all_seeds  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Leakage control (Section 23)
# ──────────────────────────────────────────────────────────────────────────────

def _synthetic_cohort(n_subjects: int = 60, sessions_per: int = 1
                      ) -> pd.DataFrame:
    """Build a cohort with a controllable number of sessions per subject."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_subjects):
        subject = f"S{i:04d}"
        stage = STAGE_ORDER[min(i % 3, 2)]
        for session in range(1, sessions_per + 1):
            rows.append({
                "session_id": f"{subject}_MR{session}",
                "subject_id": subject,
                "stage": stage,
                "label": STAGE_ORDER.index(stage),
                "eTIV": float(rng.normal(1400, 90)),
            })
    return pd.DataFrame(rows)


def test_split_never_shares_a_subject_across_splits() -> None:
    """A subject's sessions must land entirely in one split."""
    from modules.m01_dataset import make_subject_split

    # Two sessions per subject is exactly the case that breaks a row-wise split.
    cohort = _synthetic_cohort(n_subjects=60, sessions_per=2)
    for seed in range(8):
        manifest = make_subject_split(cohort, seed=seed)
        assert not manifest.verify_disjoint(), manifest.verify_disjoint()

        assignment = manifest.assignment()
        for subject, group in cohort.groupby("subject_id"):
            splits = {assignment.get(str(s)) for s in group["session_id"]}
            splits.discard(None)
            assert len(splits) <= 1, (
                f"subject {subject} spans splits {splits} at seed {seed}"
            )


def test_every_split_contains_every_class() -> None:
    """Stratification must place all three stages in all three splits."""
    from modules.m01_dataset import repeated_subject_splits

    cohort = _synthetic_cohort(n_subjects=60)
    for manifest in repeated_subject_splits(cohort, n_repeats=8):
        for split in ("train", "val", "test"):
            counts = manifest.session_counts[split]
            for stage in STAGE_ORDER:
                assert counts[stage] > 0, (
                    f"{split} has no {stage} at repeat {manifest.repeat}"
                )


def test_scaler_statistics_come_only_from_the_training_split() -> None:
    """Changing a test-split value must not change any fitted statistic."""
    from modules.m04_feature_extraction import FEATURE_ORDER, MorphometricScaler

    rng = np.random.default_rng(1)
    rows = []
    for i in range(30):
        for roi in ROI_ORDER:
            row = {"session_id": f"S{i:04d}_MR1", "roi_name": roi}
            for feature in FEATURE_ORDER:
                row[feature] = float(rng.normal(10, 2))
            rows.append(row)
    features = pd.DataFrame(rows)
    train = [f"S{i:04d}_MR1" for i in range(20)]

    baseline = MorphometricScaler().fit(features, train_session_ids=train)

    # Corrupt a held-out session by orders of magnitude.
    corrupted = features.copy()
    mask = corrupted["session_id"] == "S0025_MR1"
    corrupted.loc[mask, "brain_volume_mm3"] = 1e9

    after = MorphometricScaler().fit(corrupted, train_session_ids=train)

    for key, stats in baseline.stats.items():
        assert stats.center == after.stats[key].center, (
            f"{key} centre changed after altering a held-out session"
        )
        assert stats.scale == after.stats[key].scale, (
            f"{key} scale changed after altering a held-out session"
        )
    assert baseline.atrophy_reference == after.atrophy_reference


def test_class_weights_come_only_from_the_supplied_labels() -> None:
    """Weights must reflect the labels passed in, not a global distribution."""
    from modules.m01_dataset.labels import class_weights

    balanced = class_weights(np.array([0, 0, 1, 1, 2, 2]))
    assert np.allclose(balanced, np.ones(3)), balanced

    skewed = class_weights(np.array([0] * 90 + [1] * 9 + [2]))
    assert skewed[2] > skewed[1] > skewed[0], skewed

    # An absent class gets zero weight, never an infinite one.
    missing = class_weights(np.array([0, 0, 1, 1]))
    assert missing[2] == 0.0, missing
    assert np.isfinite(missing).all()


# ──────────────────────────────────────────────────────────────────────────────
# Honesty invariants (Section 28)
# ──────────────────────────────────────────────────────────────────────────────

def test_undefined_metrics_are_none_not_zero() -> None:
    """A class the model never predicted has undefined, not zero, precision."""
    from modules.training.metrics import compute_metrics

    metrics = compute_metrics([0, 0, 1, 1, 2], [0, 0, 1, 1, 1])
    ad = metrics.per_class[STAGE_ORDER.index("AD")]
    assert ad.n_predicted == 0
    assert ad.precision is None, "precision must be None, not 0.0"
    assert ad.recall == 0.0, "recall is defined and genuinely 0"
    assert ad.f1 == 0.0, "F1 is defined and genuinely 0"
    assert any("undefined" in note for note in metrics.notes)

    # A class absent from the labels has undefined recall.
    absent = compute_metrics([0, 0, 1, 1], [0, 1, 1, 0])
    assert absent.per_class[STAGE_ORDER.index("AD")].recall is None


def test_bounded_metric_confidence_intervals_stay_in_range() -> None:
    """A normal-approximation CI must be clamped to the metric's range."""
    from modules.training.metrics import aggregate_metrics, compute_metrics

    runs = [compute_metrics([0, 1, 2] * 4, [0, 1, 2] * 4) for _ in range(3)]
    runs[0] = compute_metrics([0, 1, 2] * 4, [0, 1, 1] + [0, 1, 2] * 3)
    aggregated = aggregate_metrics(runs)
    for name, cell in aggregated.items():
        if cell["ci_high"] is None:
            continue
        assert cell["ci_high"] <= 1.0 + 1e-12, f"{name} CI exceeds 1"
        assert cell["ci_low"] >= -1e-12, f"{name} CI below 0"


def test_report_rejects_prohibited_clinical_language() -> None:
    """Assertions must be rejected; negated disclaimers must be allowed."""
    from modules.m11_report.report import DISCLAIMER, validate_language

    assert validate_language(DISCLAIMER) == []
    assert validate_language(
        "This is not a clinically validated conversion probability."
    ) == []
    assert validate_language("The subject will develop AD.")
    assert validate_language("The conversion probability is 0.66.")
    assert validate_language("We recommend treatment with donepezil.")
    # A negation in a previous sentence must not license the next assertion.
    assert validate_language(
        "This is not a diagnosis. The conversion probability is 0.66."
    )


def test_figure_titles_reject_longitudinal_language() -> None:
    """Cross-sectional figures must not be labelled as progression."""
    from modules.m10_results.figures import _check_title

    assert _check_title("Stage-wise morphometric differences")
    for bad in ("24-month atrophy progression", "Longitudinal trajectory",
                "Predicted future atrophy", "Change over time"):
        try:
            _check_title(bad)
        except ValueError:
            continue
        raise AssertionError(f"title {bad!r} should have been rejected")


def test_attribution_never_claims_shap_without_shap() -> None:
    """A permutation attribution must never be labelled as SHAP."""
    from modules.m08_xai import FeatureAttributor, shap_available

    rng = np.random.default_rng(2)
    background = rng.normal(size=(6, N_ROI, 4))

    def predict(flat: np.ndarray) -> np.ndarray:
        rows = np.asarray(flat).reshape(-1, N_ROI * 4)
        logits = rows[:, :3]
        exponentiated = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exponentiated / exponentiated.sum(axis=1, keepdims=True)

    attributor = FeatureAttributor(
        predict_fn=predict, background=background,
        feature_names=["a", "b", "c", "d"], roi_names=list(ROI_ORDER),
    )
    result = attributor.explain(background[0], "MCI", n_background=3)
    if not shap_available():
        assert result.method == "permutation"
        assert result.is_shap is False
        assert any("NOT Shapley" in n or "not installed" in n
                   for n in result.notes)

    forced = attributor.explain(background[0], "MCI", n_background=3,
                                force_permutation=True)
    assert forced.method == "permutation"
    assert forced.is_shap is False


def test_statistics_reports_no_pvalue_for_tiny_groups() -> None:
    """A group below the minimum must yield no p-value at all."""
    from modules.common.config import StatsConfig
    from modules.m09_statistics import compare_groups

    result = compare_groups(
        np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0]),
        "Broca_Area", "volume", "CN", "AD", StatsConfig(min_group_n=5),
    )
    assert result.test == "insufficient_data"
    assert result.p_value is None
    assert result.note and "minimum" in result.note


def test_fdr_adjustment_is_monotone_and_preserves_gaps() -> None:
    """BH must be non-decreasing in rank and keep untested cells as None."""
    from modules.m09_statistics import benjamini_hochberg

    raw = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, None, 0.9]
    adjusted = benjamini_hochberg(raw)
    assert adjusted[6] is None, "an untested cell must stay None"

    pairs = sorted(
        (p, a) for p, a in zip(raw, adjusted) if p is not None
    )
    values = [a for _, a in pairs]
    assert values == sorted(values), f"not monotone: {values}"
    assert all(0.0 <= v <= 1.0 for v in values)


# ──────────────────────────────────────────────────────────────────────────────
# Structural invariants
# ──────────────────────────────────────────────────────────────────────────────

def test_roi_order_mismatch_is_rejected() -> None:
    """A permuted ROI order must raise, not silently transpose axes."""
    validate_roi_order(list(ROI_ORDER))
    try:
        validate_roi_order([ROI_ORDER[2]] + list(ROI_ORDER[1:]))
    except ValueError:
        return
    raise AssertionError("a permuted ROI order should have been rejected")


def test_checkpoint_refuses_mismatched_roi_order() -> None:
    """Loading a checkpoint whose ROI order differs must raise."""
    from modules.m04_feature_extraction import FEATURE_ORDER
    from modules.model import NeuroGenesisModel, build_model

    set_all_seeds(0)
    model = build_model(len(FEATURE_ORDER), NeuroGenesisConfig(), "A5",
                        list(FEATURE_ORDER))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "m.pt"
        model.save_checkpoint(path)

        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["roi_order"] = list(reversed(ROI_ORDER))
        torch.save(payload, path)
        try:
            NeuroGenesisModel.load_checkpoint(path)
        except ValueError as exc:
            assert "ROI order" in str(exc)
            return
    raise AssertionError("a mismatched ROI order should have been rejected")


def test_ablation_variants_preserve_downstream_shapes() -> None:
    """Every rung must emit identically shaped logits."""
    from modules.m04_feature_extraction import FEATURE_ORDER
    from modules.model import ABLATION_SPECS, build_model

    set_all_seeds(0)
    cfg = NeuroGenesisConfig()
    morph = torch.randn(2, N_ROI, len(FEATURE_ORDER))

    for variant, spec in ABLATION_SPECS.items():
        model = build_model(len(FEATURE_ORDER), cfg, variant,
                            list(FEATURE_ORDER))
        out = model(morph)
        assert out.logits.shape == (2, len(STAGE_ORDER)), (
            f"{variant} emitted {tuple(out.logits.shape)}"
        )
        assert torch.allclose(
            out.classification.probabilities.sum(dim=1), torch.ones(2),
            atol=1e-5,
        ), f"{variant} probabilities do not sum to 1"


def test_edge_gate_is_tied_to_anp_across_the_ladder() -> None:
    """Gating without ANP, or ANP without gating, would be uninterpretable."""
    from modules.model import ABLATION_SPECS

    for variant, spec in ABLATION_SPECS.items():
        expected = spec.use_anp and spec.graph_encoder == "saeg_gatv2"
        assert spec.use_edge_gate == expected, (
            f"{variant}: edge gate {spec.use_edge_gate} but ANP-derived "
            f"expectation is {expected}"
        )


def test_gradients_reach_every_trainable_parameter() -> None:
    """A dead parameter means a component is not being trained at all."""
    from modules.m04_feature_extraction import FEATURE_ORDER
    from modules.model import build_model
    from modules.training.losses import NeuroGenesisLoss

    set_all_seeds(0)
    cfg = NeuroGenesisConfig()
    model = build_model(len(FEATURE_ORDER), cfg, "A7", list(FEATURE_ORDER))
    criterion = NeuroGenesisLoss(
        cfg.loss, class_weights=np.ones(len(STAGE_ORDER)),
        d_model=model.fusion.out_dim,
    )
    out = model(torch.randn(3, N_ROI, len(FEATURE_ORDER)))
    criterion(out, torch.tensor([0, 1, 2])).total.backward()

    dead = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and (parameter.grad is None or not torch.isfinite(parameter.grad).all())
    ]
    assert not dead, f"parameters without a finite gradient: {dead}"


def test_stage_transition_propensity_is_undefined_at_the_last_stage() -> None:
    """There is no stage beyond AD, so no number may be emitted."""
    from modules.m07_stage_tgt import PropensityHead, StageTransformer

    set_all_seeds(0)
    transformer = StageTransformer(d_model=64)
    head = PropensityHead(d_model=transformer.width)
    z = torch.randn(4, 64)
    out = transformer(z)

    for index, stage in enumerate(STAGE_ORDER):
        result = head(
            out.z_t, z, out.prototype_output.prototypes,
            torch.full((4,), index, dtype=torch.long),
        )
        report = result.report(0)
        if stage == STAGE_ORDER[-1]:
            assert report["stage_transition_propensity"] is None, (
                "AD must report an undefined transition"
            )
            assert report["transition_target_stage"] is None
        else:
            assert report["stage_transition_propensity"] is not None
            assert report["transition_target_stage"] == STAGE_ORDER[index + 1]


def test_ad_propensity_is_bounded_and_anchored_to_the_prototypes() -> None:
    """0 on the CN prototype, 1 on the AD prototype, 0.5 at the midpoint."""
    from modules.m07_stage_tgt import PropensityHead

    set_all_seeds(0)
    prototypes = torch.randn(len(STAGE_ORDER), 32)
    cn = prototypes[STAGE_ORDER.index("CN")].unsqueeze(0)
    ad = prototypes[STAGE_ORDER.index("AD")].unsqueeze(0)

    assert abs(float(PropensityHead.relative_ad_proximity(cn, prototypes))) < 1e-5
    assert abs(
        float(PropensityHead.relative_ad_proximity(ad, prototypes)) - 1.0
    ) < 1e-5
    midpoint = float(
        PropensityHead.relative_ad_proximity((cn + ad) / 2, prototypes)
    )
    assert abs(midpoint - 0.5) < 1e-5

    arbitrary = PropensityHead.relative_ad_proximity(
        torch.randn(20, 32) * 10, prototypes
    )
    assert bool((arbitrary >= 0).all() and (arbitrary <= 1).all())


def test_legacy_code_cannot_be_loaded_without_acknowledgement() -> None:
    """Quarantined legacy components must require an explicit opt-in."""
    from modules.m12_future_extensions import IS_VALIDATED, available, load_legacy

    assert IS_VALIDATED is False
    for alias in available():
        try:
            load_legacy(alias)
        except PermissionError:
            continue
        except ImportError:
            # A legacy module whose own dependencies are missing still proves
            # the guard was not the thing that let it through.
            raise AssertionError(
                f"{alias} was loaded without acknowledgement"
            )
        raise AssertionError(f"{alias} loaded without acknowledgement")


def test_config_rejects_unknown_keys() -> None:
    """A typo in a config file must raise, not leave a silent default."""
    try:
        NeuroGenesisConfig.from_dict({"data": {"val_fracton": 0.2}})
    except ValueError as exc:
        assert "unknown key" in str(exc).lower()
        return
    raise AssertionError("an unknown config key should have been rejected")


def test_config_roundtrips_through_json() -> None:
    """A saved config must reload to exactly the same values."""
    cfg = NeuroGenesisConfig()
    with tempfile.TemporaryDirectory() as directory:
        path = cfg.to_file(Path(directory) / "config.json")
        reloaded = NeuroGenesisConfig.from_file(path)
    assert reloaded.to_dict() == cfg.to_dict()
    assert reloaded.preprocess.target_shape == (128, 128, 128)
    assert isinstance(reloaded.preprocess.target_shape, tuple)


def test_run_state_cannot_report_a_stage_that_never_ran() -> None:
    """The tracker must not allow a fabricated completed status."""
    from modules.common.run_state import RunStateTracker, StageStatus

    with tempfile.TemporaryDirectory() as directory:
        tracker = RunStateTracker(Path(directory))
        assert tracker.status("M12", "S1") == StageStatus.NOT_STARTED

        with tracker.stage("M12", "S1"):
            pass
        assert tracker.status("M12", "S1") == StageStatus.COMPLETED

        try:
            with tracker.stage("M14", "S1"):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert tracker.status("M14", "S1") == StageStatus.FAILED
        assert tracker.status("M13", "S1") == StageStatus.NOT_STARTED

        try:
            tracker.get("M99")
        except KeyError:
            return
    raise AssertionError("an unknown module code should have been rejected")


def test_patch_tensor_refuses_to_zero_fill_a_missing_roi() -> None:
    """A missing region must raise, never become a blank labelled volume."""
    from modules.m03_segmentation import save_patch_tensor

    patches = {roi: np.zeros((48, 48, 48), np.float32) for roi in ROI_ORDER}
    with tempfile.TemporaryDirectory() as directory:
        path, tensor = save_patch_tensor(patches, Path(directory), "S1")
        assert tensor.shape == (N_ROI, 48, 48, 48)

        del patches[ROI_ORDER[2]]
        try:
            save_patch_tensor(patches, Path(directory), "S2")
        except KeyError:
            return
    raise AssertionError("a missing ROI patch should have been rejected")


def test_synthetic_provenance_is_detected_from_either_generator() -> None:
    """Phantom MRI must be detected even though its outputs look real.

    The smoke-artifact generator marks the outputs tree, so it is easy to spot.
    The phantom-MRI generator marks only the *dataset*, and its artifacts come
    out of the real imaging pipeline, so an outputs tree built from it is
    indistinguishable from a genuine run unless the dataset marker is followed.
    """
    from modules.common.provenance import (
        PROVENANCE_MARKER,
        SMOKE_MARKER,
        SYNTHETIC_MRI_MARKER,
        detect_provenance,
        stamp_outputs,
    )

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        outputs, dataset = root / "outputs", root / "data"
        outputs.mkdir()
        dataset.mkdir()

        # Unmarked on both sides: real.
        assert not detect_provenance(outputs, dataset).is_synthetic

        # Marked dataset only: still synthetic, and named correctly.
        (dataset / SYNTHETIC_MRI_MARKER).write_text(
            json.dumps({"is_synthetic_mri": True, "warning": "phantom"}),
            encoding="utf-8",
        )
        provenance = detect_provenance(outputs, dataset)
        assert provenance.is_synthetic
        assert provenance.kind == "synthetic_mri"
        assert "PHANTOM" in provenance.banner

        # Stamping copies it into the outputs tree, so a later consumer that
        # only sees the outputs directory still finds it.
        stamped = stamp_outputs(outputs, dataset)
        assert stamped is not None and (outputs / PROVENANCE_MARKER).exists()
        assert detect_provenance(outputs).kind == "synthetic_mri"

    with tempfile.TemporaryDirectory() as directory:
        outputs = Path(directory)
        (outputs / SMOKE_MARKER).write_text(
            json.dumps({"is_smoke_test": True, "warning": "patches"}),
            encoding="utf-8",
        )
        provenance = detect_provenance(outputs)
        assert provenance.kind == "synthetic_patches"
        assert "SMOKE-TEST" in provenance.banner

    with tempfile.TemporaryDirectory() as directory:
        # An unparseable marker must never be upgraded to "real".
        outputs = Path(directory)
        (outputs / SMOKE_MARKER).write_text("{ not json", encoding="utf-8")
        assert detect_provenance(outputs).is_synthetic


def test_report_names_the_correct_synthetic_source() -> None:
    """A phantom-MRI report must not claim its data came from fake patches."""
    from modules.m11_report.report import ReportGenerator, ReportInputs

    generator = ReportGenerator(Path(tempfile.gettempdir()))

    phantom = generator.build_markdown(ReportInputs(
        subject_id="S1",
        smoke_marker={"kind": "synthetic_mri",
                      "banner": "SYNTHETIC PHANTOM MRI - NOT A RESEARCH RESULT"},
    ))
    assert "PHANTOM MRI" in phantom
    assert "make_synthetic_mri.py" in phantom
    assert "make_smoke_artifacts.py" not in phantom

    patches = generator.build_markdown(ReportInputs(
        subject_id="S2",
        smoke_marker={"kind": "synthetic_patches",
                      "banner": "SYNTHETIC SMOKE-TEST DATA - NOT A RESEARCH RESULT"},
    ))
    assert "make_smoke_artifacts.py" in patches
    assert "make_synthetic_mri.py" not in patches

    real = generator.build_markdown(ReportInputs(subject_id="S3"))
    assert "SYNTHETIC" not in real
    assert "OASIS-1 (real)" in real


def test_missing_cdr_is_not_reported_as_an_unmapped_value() -> None:
    """An unassessed session is an exclusion, not a data-quality problem.

    Regression test. ``Series.map`` does not preserve ``None``: on a float
    column pandas converts it to ``float('nan')``, so an identity check against
    ``None`` misclassified all 201 unassessed OASIS-1 sessions as carrying
    unmapped CDR values and emitted a misleading warning.
    """
    from modules.m01_dataset import map_labels

    frame = pd.DataFrame({
        "ID": [f"OAS1_{i:04d}_MR1" for i in range(6)],
        "CDR": [0.0, 0.5, 1.0, np.nan, np.nan, np.nan],
        "Age": [70, 72, 74, 25, 28, 31],
    })
    _, report = map_labels(frame)
    assert report.n_missing_cdr == 3, report.n_missing_cdr
    assert report.n_unmapped_cdr == 0, report.n_unmapped_cdr
    assert report.unmapped_cdr_values == [], report.unmapped_cdr_values
    # The small-sample warning is expected on a 6-row fixture; only the
    # unmapped-value warning must be absent.
    assert not any("outside the configured mapping" in w
                   for w in report.warnings), report.warnings

    # A genuinely unmapped value must still be caught and named.
    frame.loc[2, "CDR"] = 3.5
    _, strict = map_labels(
        frame, cdr_to_stage={"0.0": "CN", "0.5": "MCI", "1.0": "AD"}
    )
    assert strict.n_unmapped_cdr == 1
    assert strict.unmapped_cdr_values == [3.5]
    assert any("outside the configured mapping" in w for w in strict.warnings)


def test_real_oasis_label_mapping_matches_the_metadata() -> None:
    """The documented CN/MCI/AD counts must match the shipped CSV."""
    csv = _ROOT / "dataset" / "oasis_cross-sectional.csv"
    if not csv.exists():
        return  # nothing to assert against

    from modules.m01_dataset import map_labels

    labeled, report = map_labels(pd.read_csv(csv))
    assert report.stage_counts == {"CN": 135, "MCI": 70, "AD": 30}, (
        f"label mapping drifted: {report.stage_counts}"
    )
    assert report.n_missing_cdr == 201
    assert len(labeled) == 235


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

            traceback.print_exc(limit=4)
    print(f"\n{len(tests) - failures}/{len(tests)} invariant tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
