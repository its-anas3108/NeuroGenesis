"""
Single-subject inference (Section 31).
=====================================

Runs one subject through the trained model and produces the machine-readable
record the design specifies, plus the XAI signals and the report inputs.

.. code-block:: text

    cached ROI patches + morphometric features
        -> 3D CNN -> NeuroProp-X -> SAEG-GATv2 -> fusion
        -> current stage -> Stage-TGT -> ROI ranking -> XAI -> report

Every artifact is loaded, never recomputed
------------------------------------------

The scaler, the model checkpoint and the split manifest are all **loaded from
disk**, not re-derived. Re-fitting a scaler at inference time would standardise a
new subject against its own statistics, producing an all-zero feature vector for
a single subject and silently different values for a batch. Loading the fitted
scaler is what makes a prediction reproducible months later.

The attribution background set is drawn from the **training** split recorded in
the loaded manifest, so explaining a test subject never consults test-set
statistics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from modules.common.config import NeuroGenesisConfig
from modules.common.logging_utils import get_logger
from modules.common.roi_constants import ROI_ORDER, STAGE_ORDER
from modules.common.run_state import RunStateTracker
from modules.m04_feature_extraction import FEATURE_ORDER, MorphometricScaler
from modules.m06_spatial_encoder.patch_dataset import patch_tensor_path
from modules.m08_xai import (
    FeatureAttributor,
    cnn_occlusion_importance,
    compute_roi_ranking,
    explain_graph,
    explain_neuropropx,
)
from modules.m11_report.report import ReportGenerator, ReportInputs
from modules.model import ModelOutput, NeuroGenesisModel

logger = get_logger(__name__)


@dataclass
class InferenceResult:
    """Everything one subject's inference produces."""

    subject_id: str
    record: Dict[str, Any] = field(default_factory=dict)
    output: Optional[ModelOutput] = None
    report_inputs: Optional[ReportInputs] = None
    warnings: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        """Return the machine-readable record as indented JSON."""
        return json.dumps(self.record, indent=2)


class InferencePipeline:
    """Run inference and explanation for individual subjects.

    Args:
        model: A trained model, loaded from a checkpoint.
        scaler: The fitted scaler used during training.
        cohort: Cohort table, for labels and eTIV.
        features_raw: **Unstandardised** long-format feature table. The pipeline
            applies the loaded scaler itself, so a caller cannot accidentally
            pass features standardised by a different scaler.
        outputs_root: Root outputs directory holding cached patch tensors.
        cfg: Framework configuration.
        train_session_ids: Training sessions, used as the attribution background.
        smoke_marker: Smoke-test marker, propagated into the report.
    """

    def __init__(
        self,
        model: NeuroGenesisModel,
        scaler: MorphometricScaler,
        cohort: pd.DataFrame,
        features_raw: pd.DataFrame,
        outputs_root: Path,
        cfg: Optional[NeuroGenesisConfig] = None,
        train_session_ids: Optional[List[str]] = None,
        smoke_marker: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model.eval()
        self.scaler = scaler
        self.cohort = cohort
        self.outputs_root = Path(outputs_root)
        self.cfg = cfg or NeuroGenesisConfig()
        self.train_session_ids = list(train_session_ids or [])
        self.smoke_marker = smoke_marker

        if not scaler.fitted:
            raise RuntimeError(
                "InferencePipeline requires a fitted scaler loaded from the "
                "training run. Fitting one here would standardise the subject "
                "against its own statistics."
            )
        self.features = scaler.transform(scaler.add_atrophy_index(features_raw))
        self.features_raw = features_raw
        self._attributor: Optional[FeatureAttributor] = None
        self._current_patches: Optional[torch.Tensor] = None
        #: The current subject's CNN embedding, computed once per subject.
        self._current_embedding: Optional[torch.Tensor] = None

    # ── Inputs ────────────────────────────────────────────────────────────

    def load_subject(self, subject_id: str
                     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[str]]:
        """Load one subject's standardised features and ROI patches.

        Returns:
            ``(morph, patches, warnings)``. ``patches`` is ``None`` when the
            variant has no CNN branch or the tensor is absent.

        Raises:
            KeyError: If the subject has no feature rows.
            FileNotFoundError: If the CNN branch is active but no patch tensor
                exists — inference on a zero-filled patch would be meaningless.
        """
        warnings: List[str] = []
        array, missing = self.scaler.to_tensor_array(self.features, [subject_id])
        if missing:
            warnings.append(
                f"{len(missing)} feature cell(s) were imputed with the training "
                f"median: {missing[:3]}"
            )
        if not (self.features["session_id"].astype(str) == subject_id).any():
            raise KeyError(
                f"No feature rows found for {subject_id}. Run the "
                "feature-extraction stage first."
            )
        morph = torch.from_numpy(array)

        patches = None
        if self.model.spec.use_cnn:
            path = patch_tensor_path(self.outputs_root, subject_id)
            if not path.exists():
                raise FileNotFoundError(
                    f"The model variant uses the 3D CNN branch but no ROI patch "
                    f"tensor exists at {path}. Run the preprocessing and ROI "
                    "patch stages first."
                )
            volume = np.load(path)
            patches = torch.from_numpy(
                np.ascontiguousarray(volume, dtype=np.float32)
            ).unsqueeze(0)
        return morph, patches, warnings

    def _background(self) -> Optional[np.ndarray]:
        """Build the attribution background from the training split."""
        if not self.train_session_ids:
            return None
        present = [
            s for s in self.train_session_ids
            if (self.features["session_id"].astype(str) == s).any()
        ]
        if not present:
            return None
        array, _ = self.scaler.to_tensor_array(self.features, present)
        return array

    def attributor(self) -> Optional[FeatureAttributor]:
        """Return a lazily built feature attributor, or ``None`` if impossible."""
        if self._attributor is not None:
            return self._attributor
        background = self._background()
        if background is None:
            return None

        n_roi, n_feat = background.shape[1], background.shape[2]

        def predict(flat: np.ndarray) -> np.ndarray:
            """Predict class probabilities from flattened morphometric input.

            The CNN branch is held fixed while the morphometric input varies,
            which is what isolates the morphometric contribution; varying both
            would attribute the CNN's behaviour to morphometric features.

            Because it is fixed, the CNN embedding is computed **once per
            subject** in :meth:`run` and reused here. Re-running the encoder for
            every perturbed row would repeat the same 3D convolution hundreds of
            times and, at the default fan-out, allocate tens of gigabytes.
            """
            tensor = torch.from_numpy(
                np.asarray(flat, dtype=np.float32).reshape(-1, n_roi, n_feat)
            )
            batch = tensor.shape[0]
            embeddings = None
            if self.model.spec.use_cnn:
                if self._current_embedding is None:
                    raise RuntimeError(
                        "The CNN branch is active but no cached embedding is "
                        "available. InferencePipeline.run() computes it; call "
                        "that before requesting attribution."
                    )
                embeddings = self._current_embedding.expand(batch, -1, -1)
            with torch.no_grad():
                out = self.model(
                    morph_features=tensor, cnn_embeddings=embeddings
                )
            return out.classification.probabilities.cpu().numpy()

        self._attributor = FeatureAttributor(
            predict_fn=predict,
            background=background,
            feature_names=list(FEATURE_ORDER),
            roi_names=list(ROI_ORDER),
        )
        return self._attributor

    # ── Inference ─────────────────────────────────────────────────────────

    def run(
        self,
        subject_id: str,
        explain: bool = True,
        occlusion: bool = True,
        tracker: Optional[RunStateTracker] = None,
    ) -> InferenceResult:
        """Run inference and explanation for one subject.

        Args:
            subject_id: Session identifier.
            explain: Compute attribution, attention and ROI ranking.
            occlusion: Additionally compute the 3D CNN occlusion attribution.
                Costs ``N_ROI + 1`` forward passes.
            tracker: Optional run-state tracker to record M12-M16 and M19.

        Returns:
            An :class:`InferenceResult`.
        """
        result = InferenceResult(subject_id=subject_id)
        morph, patches, warnings = self.load_subject(subject_id)
        result.warnings.extend(warnings)
        self._current_patches = patches

        # Compute the subject's CNN embedding once. Attribution holds the
        # spatial branch fixed, so this is both the correct semantics and the
        # difference between hundreds of 3D convolutions and one.
        self._current_embedding = None
        if patches is not None and self.model.spatial_encoder is not None:
            with torch.no_grad():
                self._current_embedding = self.model.spatial_encoder(
                    patches
                ).embeddings

        def stage(code: str):
            """Return a stage context manager, or a no-op when untracked."""
            if tracker is None:
                from contextlib import nullcontext

                return nullcontext(None)
            return tracker.stage(code, subject_id)

        with torch.no_grad():
            output = self.model(
                morph_features=morph, patches=patches, return_trace=True
            )
        result.output = output

        # M10 and M11.x execute inside the single forward pass above, so their
        # state is recorded from the produced tensors rather than by wrapping
        # separate calls. Recording them is not cosmetic: leaving them
        # NOT_STARTED would tell the dashboard that NeuroProp-X never ran.
        with stage("M9") as record:
            if record is not None:
                if output.spatial is not None:
                    record.record_metric(
                        "embed_dim", int(output.spatial.embeddings.shape[-1])
                    )
                    record.record_metric(
                        "layer_trace",
                        [t.name for t in output.spatial.trace],
                    )
                else:
                    record.note(
                        "This model variant has no 3D CNN branch."
                    )

        with stage("M10") as record:
            if record is not None:
                record.record_metric("n_nodes", len(ROI_ORDER))
                record.record_metric("roi_order", list(ROI_ORDER))
                record.record_metric(
                    "node_input_dim", int(morph.shape[-1])
                    + (int(output.spatial.embeddings.shape[-1])
                       if output.spatial is not None else 0),
                )

        if output.neuropropx is not None:
            npx = output.neuropropx
            with stage("M11") as record:
                if record is not None:
                    record.record_metric(
                        "X_star_dim", int(npx.node_features.shape[-1])
                    )
                    record.record_metric(
                        "edge_feature_dim", int(npx.edge_features().shape[-1])
                    )
                    record.record_metric(
                        "active_components", npx.active_components
                    )
            with stage("M11.1") as record:
                if record is not None:
                    record.record_metric(
                        "regional_vulnerability", npx.srve.as_dict(0)
                    )
            with stage("M11.2") as record:
                if record is not None:
                    record.record_metric("alpha", npx.ap_laf.alpha_value())
            with stage("M11.3") as record:
                if record is not None:
                    record.record_metric("beta", npx.anp.beta)
                    record.record_metric(
                        "propagation_range",
                        [float(npx.propagation_score.min()),
                         float(npx.propagation_score.max())],
                    )
            with stage("M11.4") as record:
                if record is not None:
                    record.record_metric(
                        "component_slices",
                        {k: list(v)
                         for k, v in npx.sagr.component_slices.items()},
                    )
        elif tracker is not None:
            reason = "This model variant does not include NeuroProp-X."
            for code in ("M11", "M11.1", "M11.2", "M11.3", "M11.4"):
                tracker.mark_skipped(code, reason, subject_id)

        with stage("M12") as record:
            if record is not None:
                record.record_metric(
                    "graph_embedding_dim",
                    int(output.graph.graph_embedding.shape[-1])
                    if output.graph is not None else 0,
                )
                if output.graph is not None and output.graph.layer_traces:
                    trace = output.graph.layer_traces[-1]
                    record.record_metric(
                        "attention_shape", list(trace.attention.shape)
                    )
                    record.record_metric(
                        "edge_gate_active", trace.edge_gate is not None
                    )

        with stage("M13") as record:
            if record is not None:
                record.record_metric("z_h_dim", int(output.z_h.shape[-1]))

        row = self.cohort[self.cohort["session_id"].astype(str) == subject_id]
        true_stage = str(row.iloc[0]["stage"]) if not row.empty else None

        record_dict = output.subject_record(0, subject_id)
        record_dict["reference_stage"] = true_stage
        record_dict["is_synthetic"] = bool(self.smoke_marker)

        graph_expl = explain_graph(output, 0)
        npx_expl = explain_neuropropx(output, self.model, 0)

        with stage("M14") as record:
            propensity = (
                output.propensity.report(0) if output.propensity is not None
                else None
            )
            if record is not None and propensity:
                record.record_metric(
                    "ad_associated_propensity",
                    propensity["ad_associated_propensity"],
                )

        attribution = None
        attribution_per_roi: Optional[Dict[str, float]] = None
        occlusion_result = None

        if explain:
            with stage("M16") as record:
                attributor = self.attributor()
                if attributor is not None and output.classification is not None:
                    target = output.classification.predicted_stage(0)
                    try:
                        attribution = attributor.explain(
                            morph[0].numpy(), target, n_background=10
                        )
                        attribution_per_roi = attribution.per_roi()
                    except (ValueError, RuntimeError) as exc:
                        result.warnings.append(
                            f"Feature attribution failed: {exc}"
                        )
                else:
                    result.warnings.append(
                        "Feature attribution unavailable: no training-split "
                        "background set was supplied."
                    )

                if occlusion and patches is not None:
                    def predict_patches(volume: np.ndarray) -> np.ndarray:
                        tensor = torch.from_numpy(
                            np.asarray(volume, dtype=np.float32)
                        )
                        batch = tensor.shape[0]
                        with torch.no_grad():
                            out = self.model(
                                morph_features=morph.expand(batch, -1, -1),
                                patches=tensor,
                            )
                        return out.classification.probabilities.cpu().numpy()

                    try:
                        occlusion_result = cnn_occlusion_importance(
                            predict_patches,
                            patches[0].numpy(),
                            int(output.classification.predictions[0]),
                        )
                    except (ValueError, RuntimeError) as exc:
                        result.warnings.append(f"Occlusion analysis failed: {exc}")

                if record is not None:
                    record.record_metric(
                        "attribution_method",
                        attribution.method if attribution else "unavailable",
                    )

            with stage("M15") as record:
                ranking = compute_roi_ranking(
                    vulnerability=npx_expl.regional_vulnerability or None,
                    attention=graph_expl.node_importance or None,
                    attribution=attribution_per_roi,
                    cnn_importance=(
                        occlusion_result["importance"] if occlusion_result
                        else None
                    ),
                    cfg=self.cfg.ranking,
                    group=true_stage,
                )
                record_dict["roi_ranking"] = ranking.ranking
                if record is not None:
                    record.record_metric("signals_used", ranking.signals_used)
        else:
            ranking = compute_roi_ranking(
                vulnerability=npx_expl.regional_vulnerability or None,
                attention=graph_expl.node_importance or None,
                cfg=self.cfg.ranking, group=true_stage,
            )
            record_dict["roi_ranking"] = ranking.ranking

        record_dict["important_edges"] = graph_expl.top_edges(5)
        record_dict["explanations"] = {
            "graph": graph_expl.to_dict(),
            "neuropropx": npx_expl.to_dict(),
            "attribution": attribution.to_dict() if attribution else None,
            "cnn_occlusion": occlusion_result,
        }
        result.record = record_dict

        result.report_inputs = ReportInputs(
            subject_id=subject_id,
            current_stage=output.classification.predicted_stage(0),
            class_probabilities=output.classification.probability_dict(0),
            confidence=output.classification.confidence(0),
            true_stage=true_stage,
            regional_vulnerability=npx_expl.regional_vulnerability or None,
            vulnerability_ranking=npx_expl.vulnerability_ranking or None,
            adaptive_adjacency=npx_expl.adaptive_adjacency,
            propagation_matrix=npx_expl.propagation_matrix,
            alpha=npx_expl.alpha,
            top_pathways=npx_expl.top_pathways or None,
            propensity=propensity,
            stage_geometry=(
                self.model.stage_tgt.prototypes.prototype_geometry()
                if self.model.stage_tgt is not None else None
            ),
            roi_ranking=ranking.ranking or None,
            ranking_weights=ranking.weights_used or None,
            attribution_method=attribution.method if attribution else None,
            top_features=attribution.top_features(10) if attribution else None,
            attribution_notes=attribution.notes if attribution else None,
            node_importance=graph_expl.node_importance or None,
            top_edges=graph_expl.top_edges(5) or None,
            model_summary=self.model.summary(),
            config_snapshot=self.cfg.to_dict(),
            experiment_id=self.cfg.paths.experiment_id,
            smoke_marker=self.smoke_marker,
        )
        return result

    def generate_report(
        self,
        result: InferenceResult,
        out_dir: Optional[Path] = None,
        extra: Optional[Dict[str, Any]] = None,
        tracker: Optional[RunStateTracker] = None,
    ) -> Dict[str, Path]:
        """Write the subject report from an inference result.

        Args:
            result: The inference result.
            out_dir: Report directory. Defaults to ``outputs/reports``.
            extra: Optional cohort-level context to merge into the report inputs
                (statistics summary, ablation table, figures).
            tracker: Optional run-state tracker, to record M19.

        Returns:
            Mapping of format -> written path.

        Raises:
            RuntimeError: If the result carries no report inputs.
        """
        if result.report_inputs is None:
            raise RuntimeError(
                "This InferenceResult has no report inputs; run() must be "
                "called before generate_report()."
            )
        for key, value in (extra or {}).items():
            if hasattr(result.report_inputs, key):
                setattr(result.report_inputs, key, value)

        generator = ReportGenerator(
            Path(out_dir) if out_dir else self.outputs_root / "reports"
        )
        if tracker is not None:
            with tracker.stage("M19", result.subject_id) as record:
                written = generator.generate(result.report_inputs)
                for name, path in written.items():
                    record.record_artifact(name, path)
            return written
        return generator.generate(result.report_inputs)


def load_inference_pipeline(
    outputs_root: Path,
    checkpoint: Path,
    cfg: Optional[NeuroGenesisConfig] = None,
    scaler_path: Optional[Path] = None,
    split_path: Optional[Path] = None,
    smoke_marker: Optional[Dict[str, Any]] = None,
) -> InferencePipeline:
    """Assemble an inference pipeline from persisted artifacts.

    Args:
        outputs_root: Root outputs directory.
        checkpoint: Model checkpoint path.
        cfg: Framework configuration.
        scaler_path: Fitted scaler JSON. Defaults to
            ``outputs/scalers/morphometric_scaler.json``.
        split_path: Split manifest JSON, used for the attribution background.
        smoke_marker: Smoke-test marker.

    Returns:
        A ready :class:`InferencePipeline`.

    Raises:
        FileNotFoundError: If a required artifact is missing. The message names
            the mode that produces it.
    """
    outputs_root = Path(outputs_root)
    cfg = cfg or NeuroGenesisConfig()

    cohort_path = outputs_root / "patient" / "cohort.csv"
    features_path = outputs_root / "features" / "morphometric_features.csv"
    scaler_path = Path(scaler_path) if scaler_path else \
        outputs_root / "scalers" / "morphometric_scaler.json"

    for path, produced_by in (
        (cohort_path, "--mode preprocess"),
        (features_path, "--mode preprocess"),
        (Path(checkpoint), "--mode train_full"),
        (scaler_path, "--mode train_full"),
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Required artifact not found: {path}. It is produced by "
                f"`python run.py {produced_by}`."
            )

    train_sessions: List[str] = []
    if split_path is None:
        split_path = outputs_root / "splits" / "split.json"
    if Path(split_path).exists():
        from modules.m01_dataset.splits import SplitManifest

        train_sessions = SplitManifest.load(Path(split_path)).train_sessions
    else:
        logger.warning(
            "No split manifest at %s; feature attribution will be unavailable "
            "because there is no training-split background set.", split_path,
        )

    return InferencePipeline(
        model=NeuroGenesisModel.load_checkpoint(Path(checkpoint)),
        scaler=MorphometricScaler.load(scaler_path),
        cohort=pd.read_csv(cohort_path),
        features_raw=pd.read_csv(features_path),
        outputs_root=outputs_root,
        cfg=cfg,
        train_session_ids=train_sessions,
        smoke_marker=smoke_marker,
    )


__all__ = [
    "InferenceResult",
    "InferencePipeline",
    "load_inference_pipeline",
]
