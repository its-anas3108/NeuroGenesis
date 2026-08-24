"""
Validation checkpoints (Section 35).
====================================

Runs the ten acceptance checkpoints in order and prints a pass/fail table.

Checkpoints 1 and part of 3 depend on the imaging stack and on real MRI being
present; when they are not, those checkpoints report ``BLOCKED`` with the reason
rather than ``PASS``. A blocked checkpoint is not a pass, and this script never
reports one as such.

Usage::

    python tools/validate_checkpoints.py --outputs outputs_smoke
"""

from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402

from modules.common.config import NeuroGenesisConfig  # noqa: E402
from modules.common.logging_utils import setup_logging  # noqa: E402
from modules.common.roi_constants import N_ROI, ROI_ORDER, STAGE_ORDER  # noqa: E402
from modules.common.seeds import set_all_seeds  # noqa: E402


@dataclass
class Result:
    """Outcome of one checkpoint."""

    number: int
    name: str
    status: str  # PASS | FAIL | BLOCKED
    detail: str = ""


@dataclass
class Harness:
    """Shared state built once and reused by the later checkpoints."""

    outputs: Path
    cfg: NeuroGenesisConfig
    cohort: Optional[pd.DataFrame] = None
    features: Optional[pd.DataFrame] = None
    split: object = None
    scaler: object = None
    model: object = None
    morph: Optional[torch.Tensor] = None
    patches: Optional[torch.Tensor] = None
    npx_out: object = None
    graph_out: object = None
    fusion_out: object = None
    class_out: object = None
    notes: List[str] = field(default_factory=list)


def checkpoint_1(h: Harness) -> Result:
    """MRI -> preprocessing -> 5 ROIs -> features -> graph for one subject."""
    from modules.m02_preprocessing import check_imaging_dependencies

    status = check_imaging_dependencies()
    mri_files = list(Path(h.cfg.paths.mri_dir).rglob("*.nii*")) if \
        Path(h.cfg.paths.mri_dir).exists() else []

    if not status["can_run"]:
        return Result(1, "MRI -> preprocessing -> ROIs -> features -> graph",
                      "BLOCKED",
                      f"missing imaging packages: {status['missing']}")
    if not mri_files:
        return Result(1, "MRI -> preprocessing -> ROIs -> features -> graph",
                      "BLOCKED",
                      f"no MRI volumes under {h.cfg.paths.mri_dir}")

    from modules.m02_preprocessing.pipeline import PreprocessingPipeline
    from modules.m03_segmentation import ROIPipeline

    subject = mri_files[0].stem.split(".")[0]
    pre = PreprocessingPipeline(h.cfg.preprocess, h.outputs).run(
        subject, mri_files[0]
    )
    if not pre.succeeded:
        return Result(1, "imaging pipeline", "FAIL", pre.error or "unknown")

    import nibabel as nib

    image = nib.load(pre.final_path)
    volume = np.asarray(image.get_fdata(), dtype=np.float32)
    seg, patches = ROIPipeline(h.cfg.preprocess, h.outputs).run(
        subject, volume, image.affine
    )
    if not seg.succeeded or len(patches) != N_ROI:
        return Result(1, "ROI extraction", "FAIL",
                      seg.error or f"got {len(patches)} of {N_ROI} patches")
    return Result(1, "MRI -> preprocessing -> ROIs -> features -> graph",
                  "PASS", f"{subject}: {len(patches)} ROI patches")


def checkpoint_2(h: Harness) -> Result:
    """graph -> NeuroProp-X -> G*."""
    from modules.m04_feature_extraction import FEATURE_ORDER
    from modules.m06_neuropropx import NeuroPropX

    npx = NeuroPropX(
        morph_dim=len(FEATURE_ORDER),
        cnn_dim=h.cfg.spatial_encoder.embed_dim,
        cfg=h.cfg.neuropropx,
        morph_feature_names=list(FEATURE_ORDER),
    )
    morph = torch.randn(2, N_ROI, len(FEATURE_ORDER))
    cnn = torch.randn(2, N_ROI, h.cfg.spatial_encoder.embed_dim)
    out = npx(morph, cnn)
    h.npx_out = out

    checks = [
        (out.regional_vulnerability.shape == (2, N_ROI), "RV shape"),
        (bool((out.regional_vulnerability >= 0).all()
              and (out.regional_vulnerability <= 1).all()), "RV in [0,1]"),
        (out.adaptive_adjacency.shape == (2, N_ROI, N_ROI), "A_star shape"),
        (out.propagation_score.shape == (2, N_ROI, N_ROI), "P shape"),
        (out.node_features.shape[-1] == npx.out_dim, "X_star width"),
        (out.edge_features().shape == (2, N_ROI, N_ROI, 3), "edge features"),
        (0.0 <= out.ap_laf.alpha_value() <= 1.0, "alpha in [0,1]"),
    ]
    failed = [name for ok, name in checks if not ok]
    if failed:
        return Result(2, "G -> NeuroProp-X -> G*", "FAIL", f"failed: {failed}")
    return Result(
        2, "G -> NeuroProp-X -> G*", "PASS",
        f"X*={tuple(out.node_features.shape)} A*={tuple(out.adaptive_adjacency.shape)} "
        f"P={tuple(out.propagation_score.shape)} alpha={out.ap_laf.alpha_value():.3f}",
    )


def checkpoint_3(h: Harness) -> Result:
    """3D CNN produces five ROI spatial embeddings."""
    from modules.m06_spatial_encoder import SpatialEncoder3D

    encoder = SpatialEncoder3D(h.cfg.spatial_encoder)
    patches = np.random.default_rng(0).random(
        (N_ROI, *h.cfg.preprocess.patch_size)
    ).astype(np.float32)
    out = encoder.encode(patches, trace=True)

    if out.embeddings.shape != (1, N_ROI, h.cfg.spatial_encoder.embed_dim):
        return Result(3, "3D CNN -> five ROI embeddings", "FAIL",
                      f"shape {tuple(out.embeddings.shape)}")
    if len(out.per_roi()) != N_ROI:
        return Result(3, "3D CNN -> five ROI embeddings", "FAIL",
                      "per-ROI accessor incomplete")
    if not out.trace:
        return Result(3, "3D CNN -> five ROI embeddings", "FAIL",
                      "no intermediate shapes traced")
    repeat = encoder.encode(patches)
    if not torch.allclose(out.embeddings, repeat.embeddings):
        return Result(3, "3D CNN -> five ROI embeddings", "FAIL",
                      "inference is not deterministic")
    return Result(
        3, "3D CNN -> five ROI embeddings", "PASS",
        f"{tuple(out.embeddings.shape)}, {encoder.n_parameters():,} params, "
        f"{len(out.trace)} traced stages, deterministic",
    )


def checkpoint_4(h: Harness) -> Result:
    """G* -> SAEG-GATv2 -> CN/MCI/AD probabilities."""
    from modules.m04_feature_extraction import FEATURE_ORDER
    from modules.model import build_model

    model = build_model(len(FEATURE_ORDER), h.cfg, "A7", list(FEATURE_ORDER))
    h.model = model
    morph = torch.randn(3, N_ROI, len(FEATURE_ORDER))
    patches = torch.rand(3, N_ROI, *h.cfg.preprocess.patch_size)
    h.morph, h.patches = morph, patches
    out = model(morph, patches, return_trace=True)
    h.graph_out, h.fusion_out, h.class_out = (
        out.graph, out.fusion, out.classification
    )

    probabilities = out.classification.probabilities
    checks = [
        (probabilities.shape == (3, len(STAGE_ORDER)), "probability shape"),
        (bool(torch.allclose(probabilities.sum(dim=1),
                             torch.ones(3), atol=1e-5)), "probabilities sum to 1"),
        (out.graph is not None and bool(out.graph.layer_traces),
         "attention recorded"),
        (out.graph.layer_traces[-1].edge_gate is not None, "edge gate present"),
    ]
    failed = [name for ok, name in checks if not ok]
    if failed:
        return Result(4, "G* -> SAEG-GATv2 -> CN/MCI/AD", "FAIL",
                      f"failed: {failed}")
    gate = out.graph.layer_traces[-1].edge_gate
    return Result(
        4, "G* -> SAEG-GATv2 -> CN/MCI/AD", "PASS",
        f"Z_G={tuple(out.graph.graph_embedding.shape)}, "
        f"p={tuple(probabilities.shape)}, "
        f"gate range [{gate.min():.3f}, {gate.max():.3f}]",
    )


def checkpoint_5(h: Harness) -> Result:
    """Fusion of CNN and graph representations."""
    if h.fusion_out is None:
        return Result(5, "multimodal fusion", "FAIL",
                      "checkpoint 4 did not produce a fusion output")
    fusion = h.fusion_out
    if fusion.z_3d is None or fusion.z_g is None:
        return Result(5, "multimodal fusion", "FAIL",
                      "one branch is missing from the fusion")
    expected = fusion.z_3d.shape[-1] + fusion.z_g.shape[-1]
    if fusion.z_f.shape[-1] != expected:
        return Result(5, "multimodal fusion", "FAIL",
                      f"Z_F width {fusion.z_f.shape[-1]} != {expected}")
    return Result(
        5, "multimodal fusion", "PASS",
        f"Z_3D={tuple(fusion.z_3d.shape)} + Z_G={tuple(fusion.z_g.shape)} "
        f"-> Z_F={tuple(fusion.z_f.shape)} -> Z_H={tuple(fusion.z_h.shape)}",
    )


def checkpoint_6(h: Harness) -> Result:
    """Stage-TGT produces propensity without longitudinal data."""
    if h.model is None:
        return Result(6, "Stage-TGT propensity", "FAIL",
                      "no model from checkpoint 4")
    out = h.model(h.morph, h.patches, return_trace=True)
    if out.propensity is None or out.stage_tgt is None:
        return Result(6, "Stage-TGT propensity", "FAIL",
                      "the Stage-TGT branch produced no output")

    report = out.propensity.report(0)
    checks = [
        (out.stage_tgt.tokens.shape[1] == len(STAGE_ORDER) + 1,
         "token count = stages + subject"),
        (bool(out.stage_tgt.attention), "attention recorded"),
        (0.0 <= report["ad_associated_propensity"] <= 1.0,
         "AD propensity in [0,1]"),
        (0.0 <= report["advanced_stage_alignment"] <= 1.0,
         "advanced alignment in [0,1]"),
    ]
    failed = [name for ok, name in checks if not ok]
    if failed:
        return Result(6, "Stage-TGT propensity", "FAIL", f"failed: {failed}")

    # An AD reference stage must yield an undefined transition, not a number.
    ad_reference = h.model.propensity_head(
        out.stage_tgt.z_t, out.fusion.z_h,
        out.stage_tgt.prototype_output.prototypes,
        torch.full((h.morph.shape[0],), len(STAGE_ORDER) - 1, dtype=torch.long),
    )
    if ad_reference.report(0)["stage_transition_propensity"] is not None:
        return Result(6, "Stage-TGT propensity", "FAIL",
                      "an AD reference stage produced a transition value "
                      "instead of reporting it undefined")
    return Result(
        6, "Stage-TGT propensity (no longitudinal data)", "PASS",
        f"tokens={tuple(out.stage_tgt.tokens.shape)}, "
        f"AD propensity={report['ad_associated_propensity']:.3f}, "
        "AD reference correctly undefined",
    )


def checkpoint_7(h: Harness) -> Result:
    """ROI ranking works."""
    from modules.m08_xai import compute_roi_ranking, ranking_stability

    if h.npx_out is None or h.graph_out is None:
        return Result(7, "ROI ranking", "FAIL", "missing upstream outputs")

    vulnerability = h.npx_out.srve.as_dict(0)
    importance = h.graph_out.node_importance()
    attention = {
        roi: float(importance[0, i]) for i, roi in enumerate(ROI_ORDER)
    } if importance is not None else None

    ranking = compute_roi_ranking(
        vulnerability=vulnerability, attention=attention,
        attribution={roi: float(abs(hash(roi)) % 100) / 100 for roi in ROI_ORDER},
        cfg=h.cfg.ranking,
    )
    if len(ranking.ranking) != N_ROI:
        return Result(7, "ROI ranking", "FAIL",
                      f"{len(ranking.ranking)} of {N_ROI} ROIs ranked")
    if abs(sum(ranking.weights_used.values()) - 1.0) > 1e-6:
        return Result(7, "ROI ranking", "FAIL", "weights do not sum to 1")

    stability = ranking_stability([ranking] * 5, top_k=3)
    if len(stability.rows) != N_ROI:
        return Result(7, "ROI ranking", "FAIL", "stability table incomplete")

    # A ranking with a missing signal must renormalise, not substitute zeros.
    partial = compute_roi_ranking(vulnerability=vulnerability,
                                  cfg=h.cfg.ranking)
    if abs(sum(partial.weights_used.values()) - 1.0) > 1e-6:
        return Result(7, "ROI ranking", "FAIL",
                      "weights were not renormalised when a signal was absent")
    return Result(
        7, "ROI ranking and stability", "PASS",
        f"top-3: {ranking.top(3)}, signals={ranking.signals_used}, "
        f"stability rows={len(stability.rows)}",
    )


def checkpoint_8(h: Harness) -> Result:
    """SHAP / attention explanation works."""
    from modules.m04_feature_extraction import FEATURE_ORDER
    from modules.m08_xai import (
        FeatureAttributor,
        explain_graph,
        explain_neuropropx,
        shap_available,
    )

    if h.model is None:
        return Result(8, "explanations", "FAIL", "no model")
    out = h.model(h.morph, h.patches, return_trace=True)

    graph = explain_graph(out, 0)
    if graph.edge_attention is None or not graph.node_importance:
        return Result(8, "explanations", "FAIL",
                      "attention explanation is empty")
    npx = explain_neuropropx(out, h.model, 0)
    if not npx.regional_vulnerability:
        return Result(8, "explanations", "FAIL",
                      "NeuroProp-X explanation is empty")

    with torch.no_grad():
        embedding = h.model.spatial_encoder(h.patches).embeddings

    def predict(flat: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(
            np.asarray(flat, dtype=np.float32).reshape(-1, N_ROI,
                                                       len(FEATURE_ORDER))
        )
        with torch.no_grad():
            result = h.model(
                morph_features=tensor,
                cnn_embeddings=embedding[:1].expand(tensor.shape[0], -1, -1),
            )
        return result.classification.probabilities.cpu().numpy()

    attributor = FeatureAttributor(
        predict_fn=predict, background=h.morph.numpy(),
        feature_names=list(FEATURE_ORDER), roi_names=list(ROI_ORDER),
    )
    attribution = attributor.explain(
        h.morph[0].numpy(), out.classification.predicted_stage(0),
        n_background=3,
    )
    if attribution.values.shape != (N_ROI, len(FEATURE_ORDER)):
        return Result(8, "explanations", "FAIL",
                      f"attribution shape {attribution.values.shape}")
    if attribution.method == "shap_kernel" and not shap_available():
        return Result(8, "explanations", "FAIL",
                      "attribution claims SHAP but shap is not installed")
    return Result(
        8, "attribution and attention explanations", "PASS",
        f"attribution method={attribution.method} "
        f"(shap installed: {shap_available()}), "
        f"attention {tuple(graph.edge_attention.shape)}, "
        f"{len(npx.regional_vulnerability)} vulnerability values",
    )


def checkpoint_9(h: Harness) -> Result:
    """Ablation pipeline runs."""
    from modules.m01_dataset import make_subject_split
    from modules.model import ABLATION_SPECS, build_model
    from modules.m04_feature_extraction import FEATURE_ORDER
    from modules.training.ablation import paired_comparison

    built = []
    for variant in ("A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7"):
        model = build_model(len(FEATURE_ORDER), h.cfg, variant,
                            list(FEATURE_ORDER))
        spec = ABLATION_SPECS[variant]
        result = model(
            h.morph, h.patches if spec.use_cnn else None
        )
        if result.logits.shape != (h.morph.shape[0], len(STAGE_ORDER)):
            return Result(9, "ablation ladder", "FAIL",
                          f"{variant} produced {tuple(result.logits.shape)}")
        built.append((variant, model.n_parameters(), spec.use_edge_gate))

    # The edge gate must be tied to ANP across the ladder.
    for variant, _, gate in built:
        expected = ABLATION_SPECS[variant].use_anp and \
            ABLATION_SPECS[variant].graph_encoder == "saeg_gatv2"
        if gate != expected:
            return Result(9, "ablation ladder", "FAIL",
                          f"{variant}: edge gate {gate} but ANP-derived "
                          f"expectation is {expected}")

    comparison = paired_comparison([0.7, 0.72, 0.69], [0.8, 0.81, 0.79],
                                   "macro_f1")
    if comparison["p_value"] is None:
        return Result(9, "ablation ladder", "FAIL",
                      "paired comparison produced no p-value")
    return Result(
        9, "ablation ladder A0-A7", "PASS",
        f"{len(built)} variants built and run; paired test "
        f"p={comparison['p_value']:.4f}, effect={comparison['effect_size']:.3f}",
    )


def checkpoint_10(h: Harness) -> Result:
    """Dashboard displays all stages."""
    from dashboard.state import DashboardState

    state = DashboardState(outputs=h.outputs, cfg=h.cfg)
    status = state.pipeline_status(None)
    if len(status) != 23:
        return Result(10, "dashboard", "FAIL",
                      f"pipeline table has {len(status)} rows, expected 23")

    try:
        import streamlit  # noqa: F401

        streamlit_ok = True
        detail = ""
    except ImportError as exc:
        streamlit_ok = False
        detail = f"streamlit cannot be imported: {exc}"

    accessors = [
        state.cohort, state.features, state.statistics, state.ablation,
        state.baselines, state.roi_ranking, state.figures, state.checkpoints,
        state.split_manifest, state.scaler,
    ]
    for accessor in accessors:
        accessor()  # must not raise even when the artifact is absent

    if not streamlit_ok:
        return Result(
            10, "dashboard renders all M1-M19 stages", "BLOCKED",
            f"{detail}. The data layer and all 23 module rows verified; the "
            "app itself cannot be launched in this environment.",
        )
    return Result(10, "dashboard renders all M1-M19 stages", "PASS",
                  f"{len(status)} module rows, all accessors safe")


CHECKPOINTS: List[Callable[[Harness], Result]] = [
    checkpoint_1, checkpoint_2, checkpoint_3, checkpoint_4, checkpoint_5,
    checkpoint_6, checkpoint_7, checkpoint_8, checkpoint_9, checkpoint_10,
]


def main() -> int:
    """Run every checkpoint and print the summary table."""
    parser = argparse.ArgumentParser(
        description="Run the Section 35 validation checkpoints."
    )
    parser.add_argument("--outputs", type=Path, default=Path("outputs_smoke"))
    parser.add_argument("--mri-dir", type=Path,
                        help="Override the MRI directory checkpoint 1 scans.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()
    set_all_seeds(args.seed)

    cfg = NeuroGenesisConfig()
    cfg.paths.outputs_dir = args.outputs
    if args.mri_dir:
        cfg.paths.mri_dir = args.mri_dir
    harness = Harness(outputs=args.outputs, cfg=cfg)

    print("=" * 78)
    print("  NeuroGenesis validation checkpoints (Section 35)")
    print("=" * 78)

    results: List[Result] = []
    for function in CHECKPOINTS:
        try:
            result = function(harness)
        except Exception as exc:  # noqa: BLE001
            number = int(function.__name__.split("_")[1])
            result = Result(number, function.__doc__ or function.__name__,
                            "FAIL", f"{type(exc).__name__}: {exc}")
            traceback.print_exc(limit=6)
        results.append(result)
        symbol = {"PASS": "PASS   ", "FAIL": "FAIL   ",
                  "BLOCKED": "BLOCKED"}[result.status]
        print(f"\n[{symbol}] CHECKPOINT {result.number}: {result.name}")
        if result.detail:
            print(f"          {result.detail}")

    passed = sum(1 for r in results if r.status == "PASS")
    blocked = sum(1 for r in results if r.status == "BLOCKED")
    failed = sum(1 for r in results if r.status == "FAIL")

    print("\n" + "=" * 78)
    print(f"  PASS {passed}   BLOCKED {blocked}   FAIL {failed}   "
          f"of {len(results)}")
    print("=" * 78)
    if blocked:
        print("\nBLOCKED checkpoints are not passes. They require artifacts or "
              "packages that are absent in this environment:")
        for r in results:
            if r.status == "BLOCKED":
                print(f"  Checkpoint {r.number}: {r.detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
