"""
The assembled NeuroGenesis model and the ablation ladder.
=========================================================

Composes every component into one ``nn.Module``:

.. code-block:: text

    ROI patches (5,48,48,48)      morphometric features (5,F)
            |                                |
      3D CNN encoder                         |
            |                                |
       E_3D (5,128) ----+---------------------+
                        |
                   NeuroProp-X  ->  G* = (V, X*, A*, P)
                        |
                   SAEG-GATv2  ->  Z_G
                        |
     Z_3D ------- Multimodal Fusion -------  Z_H
                        |
             +----------+-----------+
             |                      |
      Stage classifier         Stage-TGT
             |                      |
       P(CN/MCI/AD)          propensity scores

Ablation ladder (Section 17)
----------------------------

:data:`ABLATION_SPECS` defines A0-A7 as :class:`ModelSpec` instances. Every rung
shares the same fusion head, classifier and training procedure, so a measured
difference between rungs is attributable to the component that changed.

===  ==========================================================================
A0   Morphometry only (graph-free MLP)
A1   Morphometry + anatomical prior (fixed propagation, no attention)
A2   Morphometry + standard GAT (static attention)
A3   Morphometry + anatomical prior + learned attention (AP-LAF)
A4   A3 + SRVE
A5   A4 + ANP
A6   3D CNN + graph baseline (GATv2, no NeuroProp-X)
A7   Full: 3D CNN + NeuroProp-X + SAEG-GATv2
===  ==========================================================================

Two deliberate couplings in the ladder:

* **The edge gate turns on exactly when ANP does** (A5 onward). The gate's input
  is the NeuroProp-X edge feature whose third channel is ``P``; enabling the
  gate before ANP exists would gate on a duplicate of ``A*`` and enabling ANP
  without the gate would compute ``P`` and then ignore it. Either would make
  the A4->A5 increment uninterpretable.
* **Stage-TGT is independent of the ladder.** It attaches to ``Z_H`` and does not
  change the classifier's input, so "full model with and without Stage-TGT" is a
  separate, orthogonal comparison rather than another rung.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from modules.common.config import NeuroGenesisConfig
from modules.common.logging_utils import get_logger
from modules.common.roi_constants import N_ROI, N_STAGE, ROI_ORDER, STAGE_ORDER
from modules.m06_graph_learning.classifier import (
    ClassificationOutput,
    StageClassifier,
)
from modules.m06_graph_learning.fusion import (
    FusionOutput,
    MultimodalFusion,
    SpatialBranchProjection,
)
from modules.m06_graph_learning.gat_baseline import GATBaseline, GATv2Baseline
from modules.m06_graph_learning.prior_propagation import (
    MorphometryMLP,
    PriorPropagationEncoder,
)
from modules.m06_graph_learning.saeg_gatv2 import GraphEncoderOutput, SAEGGATv2
from modules.m06_neuropropx import (
    NeuroPropX,
    NeuroPropXConfigFlags,
    NeuroPropXOutput,
)
from modules.m06_spatial_encoder.cnn3d import EncoderOutput, SpatialEncoder3D
from modules.m07_stage_tgt import (
    PropensityHead,
    PropensityOutput,
    StageTransformer,
    StageTransformerOutput,
)

logger = get_logger(__name__)


@dataclass
class ModelSpec:
    """Declarative description of one model variant.

    Attributes:
        name: Short identifier, e.g. ``"A7"``.
        description: Human-readable label used in tables.
        use_cnn: Include the 3D CNN spatial branch.
        graph_encoder: ``"none"`` | ``"prior"`` | ``"gat"`` | ``"gatv2"`` |
            ``"saeg_gatv2"``.
        use_neuropropx: Route node features through NeuroProp-X. Required for
            ``"saeg_gatv2"``, since that encoder consumes ``X*`` and ``E_ij``.
        use_srve: Enable SRVE inside NeuroProp-X.
        use_learned_attention: Enable the AP-LAF learned-attention branch.
        use_anp: Enable ANP. Also switches on the SAEG-GATv2 edge gate.
        use_centrality: Include SAGR centrality features.
        use_stage_tgt: Attach the Stage-TGT branch.
    """

    name: str
    description: str
    use_cnn: bool = True
    graph_encoder: str = "saeg_gatv2"
    use_neuropropx: bool = True
    use_srve: bool = True
    use_learned_attention: bool = True
    use_anp: bool = True
    use_centrality: bool = True
    use_stage_tgt: bool = True

    VALID_ENCODERS = ("none", "prior", "gat", "gatv2", "saeg_gatv2")

    def __post_init__(self) -> None:
        if self.graph_encoder not in self.VALID_ENCODERS:
            raise ValueError(
                f"graph_encoder must be one of {self.VALID_ENCODERS}, got "
                f"{self.graph_encoder!r}"
            )
        if self.graph_encoder == "saeg_gatv2" and not self.use_neuropropx:
            raise ValueError(
                "graph_encoder='saeg_gatv2' requires use_neuropropx=True: the "
                "encoder consumes the NeuroProp-X node features X* and edge "
                "features E_ij."
            )

    @property
    def use_edge_gate(self) -> bool:
        """Whether the SAEG-GATv2 edge gate is active.

        Tied to ANP: the gate's third input channel is ``P``, so gating without
        ANP would gate on a duplicate of ``A*``, and running ANP without the
        gate would compute ``P`` and discard it.
        """
        return self.use_anp and self.graph_encoder == "saeg_gatv2"

    def neuropropx_flags(self) -> NeuroPropXConfigFlags:
        """Translate the spec into NeuroProp-X component flags."""
        return NeuroPropXConfigFlags(
            use_srve=self.use_srve,
            use_learned_attention=self.use_learned_attention,
            use_anp=self.use_anp,
            use_cnn=self.use_cnn,
            use_centrality=self.use_centrality,
        )

    def as_dict(self) -> Dict[str, Any]:
        """Return the spec as a plain dict, including derived fields."""
        payload = asdict(self)
        payload["use_edge_gate"] = self.use_edge_gate
        return payload


#: The A0-A7 ablation ladder plus the Stage-TGT contrast.
ABLATION_SPECS: Dict[str, ModelSpec] = {
    "A0": ModelSpec(
        name="A0", description="Morphometry only",
        use_cnn=False, graph_encoder="none", use_neuropropx=False,
        use_srve=False, use_learned_attention=False, use_anp=False,
        use_centrality=False, use_stage_tgt=False,
    ),
    "A1": ModelSpec(
        name="A1", description="Morphometry + Anatomical Prior",
        use_cnn=False, graph_encoder="prior", use_neuropropx=False,
        use_srve=False, use_learned_attention=False, use_anp=False,
        use_centrality=False, use_stage_tgt=False,
    ),
    "A2": ModelSpec(
        name="A2", description="Morphometry + Standard GAT",
        use_cnn=False, graph_encoder="gat", use_neuropropx=False,
        use_srve=False, use_learned_attention=False, use_anp=False,
        use_centrality=False, use_stage_tgt=False,
    ),
    "A3": ModelSpec(
        name="A3",
        description="Morphometry + Anatomical Prior + Learned Attention",
        use_cnn=False, graph_encoder="saeg_gatv2", use_neuropropx=True,
        use_srve=False, use_learned_attention=True, use_anp=False,
        use_centrality=False, use_stage_tgt=False,
    ),
    "A4": ModelSpec(
        name="A4", description="A3 + SRVE",
        use_cnn=False, graph_encoder="saeg_gatv2", use_neuropropx=True,
        use_srve=True, use_learned_attention=True, use_anp=False,
        use_centrality=False, use_stage_tgt=False,
    ),
    "A5": ModelSpec(
        name="A5", description="A4 + ANP",
        use_cnn=False, graph_encoder="saeg_gatv2", use_neuropropx=True,
        use_srve=True, use_learned_attention=True, use_anp=True,
        use_centrality=True, use_stage_tgt=False,
    ),
    "A6": ModelSpec(
        name="A6", description="3D CNN + graph baseline (GATv2, no NeuroProp-X)",
        use_cnn=True, graph_encoder="gatv2", use_neuropropx=False,
        use_srve=False, use_learned_attention=False, use_anp=False,
        use_centrality=False, use_stage_tgt=False,
    ),
    "A7": ModelSpec(
        name="A7",
        description="Full NeuroProp-X + SAEG-GATv2 + 3D CNN",
        use_cnn=True, graph_encoder="saeg_gatv2", use_neuropropx=True,
        use_srve=True, use_learned_attention=True, use_anp=True,
        use_centrality=True, use_stage_tgt=True,
    ),
    "A7_no_tgt": ModelSpec(
        name="A7_no_tgt", description="Full model without Stage-TGT",
        use_cnn=True, graph_encoder="saeg_gatv2", use_neuropropx=True,
        use_srve=True, use_learned_attention=True, use_anp=True,
        use_centrality=True, use_stage_tgt=False,
    ),
}

#: Section 18 baselines, expressed in the same spec language.
BASELINE_SPECS: Dict[str, ModelSpec] = {
    "morph_mlp": ABLATION_SPECS["A0"],
    "morph_prior_graph": ABLATION_SPECS["A1"],
    "morph_gat": ABLATION_SPECS["A2"],
    "morph_gatv2": ModelSpec(
        name="morph_gatv2", description="Morphometry + GATv2",
        use_cnn=False, graph_encoder="gatv2", use_neuropropx=False,
        use_srve=False, use_learned_attention=False, use_anp=False,
        use_centrality=False, use_stage_tgt=False,
    ),
    "proposed": ABLATION_SPECS["A7"],
}


@dataclass
class ModelOutput:
    """Everything one forward pass produces, for training and for inspection."""

    classification: ClassificationOutput
    fusion: FusionOutput
    graph: Optional[GraphEncoderOutput] = None
    neuropropx: Optional[NeuroPropXOutput] = None
    spatial: Optional[EncoderOutput] = None
    stage_tgt: Optional[StageTransformerOutput] = None
    propensity: Optional[PropensityOutput] = None

    @property
    def logits(self) -> torch.Tensor:
        """``(B, N_STAGE)`` classification logits."""
        return self.classification.logits

    @property
    def z_h(self) -> torch.Tensor:
        """``(B, D)`` shared representation."""
        return self.fusion.z_h

    def subject_record(self, batch_index: int = 0, subject_id: Optional[str] = None
                       ) -> Dict[str, Any]:
        """Assemble the machine-readable single-subject result (Section 31).

        Fields whose component was not present in this model variant are
        reported as ``None`` rather than being filled with a placeholder, so a
        consumer can always tell "not computed" from "computed as zero".
        """
        record: Dict[str, Any] = {
            "subject_id": subject_id,
            "current_stage": self.classification.predicted_stage(batch_index),
            "class_probabilities": self.classification.probability_dict(batch_index),
            "confidence": self.classification.confidence(batch_index),
            "ad_associated_propensity": None,
            "stage_transition_propensity": None,
            "regional_vulnerability": None,
            "roi_ranking": None,
            "important_edges": None,
            "explanations": {},
        }

        if self.propensity is not None:
            prop = self.propensity.report(batch_index)
            record["ad_associated_propensity"] = prop["ad_associated_propensity"]
            record["stage_transition_propensity"] = \
                prop["stage_transition_propensity"]
            record["stage_propensity_detail"] = prop

        if self.neuropropx is not None:
            record["regional_vulnerability"] = \
                self.neuropropx.srve.as_dict(batch_index)
            record["explanations"]["neuropropx"] = \
                self.neuropropx.metadata(batch_index)

        if self.graph is not None and self.graph.layer_traces:
            record["important_edges"] = self.graph.top_edges(batch_index, k=5)

        return record


class NeuroGenesisModel(nn.Module):
    """The full model, configured by a :class:`ModelSpec`.

    Args:
        morph_dim: Per-ROI morphometric feature width.
        cfg: Framework configuration.
        spec: Model variant. Defaults to the full model (A7).
        morph_feature_names: Feature names, forwarded to SRVE so learned weights
            can be displayed against real feature names.
        n_roi: Number of ROIs.
    """

    def __init__(
        self,
        morph_dim: int,
        cfg: Optional[NeuroGenesisConfig] = None,
        spec: Optional[ModelSpec] = None,
        morph_feature_names: Optional[List[str]] = None,
        n_roi: int = N_ROI,
    ) -> None:
        super().__init__()
        self.cfg = cfg or NeuroGenesisConfig()
        self.spec = spec or ABLATION_SPECS["A7"]
        self.morph_dim = morph_dim
        self.morph_feature_names = list(morph_feature_names) if \
            morph_feature_names else [f"morph_{i}" for i in range(morph_dim)]
        self.n_roi = n_roi

        # ── Spatial branch (M9) ───────────────────────────────────────────
        if self.spec.use_cnn:
            self.spatial_encoder = SpatialEncoder3D(
                self.cfg.spatial_encoder, n_roi=n_roi
            )
            cnn_dim = self.cfg.spatial_encoder.embed_dim
            self.spatial_projection = SpatialBranchProjection(
                n_roi=n_roi,
                embed_dim=cnn_dim,
                out_dim=self.cfg.fusion.hidden_dim,
                dropout=self.cfg.fusion.dropout,
            )
            spatial_dim = self.spatial_projection.out_dim
        else:
            self.spatial_encoder = None
            self.spatial_projection = None
            cnn_dim = 0
            spatial_dim = 0
        self.cnn_dim = cnn_dim

        # ── NeuroProp-X (M11) ─────────────────────────────────────────────
        if self.spec.use_neuropropx:
            self.neuropropx = NeuroPropX(
                morph_dim=morph_dim,
                cnn_dim=cnn_dim,
                cfg=self.cfg.neuropropx,
                flags=self.spec.neuropropx_flags(),
                morph_feature_names=self.morph_feature_names,
                n_roi=n_roi,
            )
            node_dim = self.neuropropx.out_dim
            edge_dim = self.neuropropx.edge_feature_dim
        else:
            self.neuropropx = None
            # Without NeuroProp-X the node representation is the plain
            # concatenation H_i = [X_i_morph || E_i_3D].
            node_dim = morph_dim + cnn_dim
            edge_dim = 0
        self.node_dim = node_dim

        # ── Graph encoder (M12) ───────────────────────────────────────────
        gl = self.cfg.graph_learning
        if self.spec.graph_encoder == "saeg_gatv2":
            gl_spec = type(gl)(**{**asdict(gl),
                                  "use_edge_gate": self.spec.use_edge_gate})
            self.graph_encoder = SAEGGATv2(
                in_dim=node_dim, edge_dim=edge_dim, cfg=gl_spec, n_roi=n_roi
            )
        elif self.spec.graph_encoder == "gat":
            self.graph_encoder = GATBaseline(node_dim, cfg=gl, n_roi=n_roi)
        elif self.spec.graph_encoder == "gatv2":
            self.graph_encoder = GATv2Baseline(node_dim, cfg=gl, n_roi=n_roi)
        elif self.spec.graph_encoder == "prior":
            self.graph_encoder = PriorPropagationEncoder(
                node_dim, cfg=gl, n_roi=n_roi
            )
        else:
            self.graph_encoder = MorphometryMLP(
                in_dim=node_dim,
                hidden_dim=gl.hidden_dim,
                out_dim=gl.hidden_dim * 2,
                dropout=gl.dropout,
                n_roi=n_roi,
            )
        graph_dim = self.graph_encoder.out_dim

        # ── Fusion (M13) and classifier ───────────────────────────────────
        self.fusion = MultimodalFusion(
            spatial_dim=spatial_dim, graph_dim=graph_dim, cfg=self.cfg.fusion
        )
        self.classifier = StageClassifier(self.fusion.out_dim, n_classes=N_STAGE)

        # ── Stage-TGT (M14) ───────────────────────────────────────────────
        if self.spec.use_stage_tgt and self.cfg.stage_tgt.enabled:
            self.stage_tgt = StageTransformer(
                d_model=self.fusion.out_dim, cfg=self.cfg.stage_tgt
            )
            self.propensity_head = PropensityHead(
                d_model=self.stage_tgt.width, dropout=self.cfg.stage_tgt.dropout
            )
        else:
            self.stage_tgt = None
            self.propensity_head = None

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        morph_features: torch.Tensor,
        patches: Optional[torch.Tensor] = None,
        cnn_embeddings: Optional[torch.Tensor] = None,
        return_trace: bool = False,
    ) -> ModelOutput:
        """Run the full pipeline.

        Args:
            morph_features: ``(B, N, morph_dim)`` normalized morphometric
                features.
            patches: ``(B, N, D, H, W)`` ROI patches. Required when the spatial
                branch is enabled unless ``cnn_embeddings`` is supplied.
            cnn_embeddings: ``(B, N, cnn_dim)`` precomputed CNN embeddings, used
                instead of running the encoder. Lets the graph stage train
                against cached embeddings.
            return_trace: Record attention, gates and intermediate shapes.

        Returns:
            A :class:`ModelOutput`.

        Raises:
            ValueError: If the spatial branch is enabled but neither patches nor
                embeddings were provided.
        """
        if morph_features.dim() == 2:
            morph_features = morph_features.unsqueeze(0)

        spatial_out: Optional[EncoderOutput] = None
        emb: Optional[torch.Tensor] = None
        z_3d: Optional[torch.Tensor] = None

        if self.spec.use_cnn:
            if cnn_embeddings is not None:
                emb = cnn_embeddings
                if emb.dim() == 2:
                    emb = emb.unsqueeze(0)
            elif patches is not None:
                spatial_out = self.spatial_encoder(patches, trace=return_trace)
                emb = spatial_out.embeddings
            else:
                raise ValueError(
                    "The spatial branch is enabled but neither `patches` nor "
                    "`cnn_embeddings` was provided."
                )
            z_3d = self.spatial_projection(emb)

        npx_out: Optional[NeuroPropXOutput] = None
        if self.neuropropx is not None:
            npx_out = self.neuropropx(
                morph_features, emb, keep_features=return_trace
            )
            node_x = npx_out.node_features
            adj = npx_out.adaptive_adjacency
            edge = npx_out.edge_features()
        else:
            node_x = torch.cat([morph_features, emb], dim=-1) \
                if emb is not None else morph_features
            adj = None
            edge = None

        graph_out = self.graph_encoder(
            node_x, adj, edge, return_trace=return_trace
        )
        fusion_out = self.fusion(z_3d, graph_out.graph_embedding)
        class_out = self.classifier(fusion_out.z_h)

        tgt_out: Optional[StageTransformerOutput] = None
        prop_out: Optional[PropensityOutput] = None
        if self.stage_tgt is not None:
            tgt_out = self.stage_tgt(
                fusion_out.z_h, return_attention=return_trace
            )
            prop_out = self.propensity_head(
                tgt_out.z_t,
                fusion_out.z_h,
                tgt_out.prototype_output.prototypes,
                class_out.predictions,
            )

        return ModelOutput(
            classification=class_out,
            fusion=fusion_out,
            graph=graph_out,
            neuropropx=npx_out,
            spatial=spatial_out,
            stage_tgt=tgt_out,
            propensity=prop_out,
        )

    # ── Introspection and checkpointing ───────────────────────────────────

    def n_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def component_parameters(self) -> Dict[str, int]:
        """Return trainable parameter counts per component."""
        parts = {
            "spatial_encoder": self.spatial_encoder,
            "spatial_projection": self.spatial_projection,
            "neuropropx": self.neuropropx,
            "graph_encoder": self.graph_encoder,
            "fusion": self.fusion,
            "classifier": self.classifier,
            "stage_tgt": self.stage_tgt,
            "propensity_head": self.propensity_head,
        }
        return {
            name: (sum(p.numel() for p in mod.parameters() if p.requires_grad)
                   if mod is not None else 0)
            for name, mod in parts.items()
        }

    def summary(self) -> Dict[str, Any]:
        """Return a complete description of the configured model."""
        out: Dict[str, Any] = {
            "spec": self.spec.as_dict(),
            "dimensions": {
                "morph_dim": self.morph_dim,
                "cnn_dim": self.cnn_dim,
                "node_dim": self.node_dim,
                "graph_embedding_dim": self.graph_encoder.out_dim,
                "z_h_dim": self.fusion.out_dim,
                "n_roi": self.n_roi,
                "n_classes": N_STAGE,
            },
            "n_parameters": self.n_parameters(),
            "component_parameters": self.component_parameters(),
            "roi_order": list(ROI_ORDER),
            "stage_order": list(STAGE_ORDER),
        }
        if self.spatial_encoder is not None:
            out["spatial_encoder"] = {
                "shared": self.cfg.spatial_encoder.shared_encoder,
                "embed_dim": self.cfg.spatial_encoder.embed_dim,
            }
        if self.neuropropx is not None:
            out["neuropropx"] = self.neuropropx.summary()
        out["graph_encoder"] = self.graph_encoder.summary()
        out["fusion"] = self.fusion.summary()
        out["classifier"] = self.classifier.summary()
        if self.stage_tgt is not None:
            out["stage_tgt"] = self.stage_tgt.summary()
            out["propensity_head"] = self.propensity_head.summary()
        return out

    def save_checkpoint(self, path: Path, extra: Optional[Dict[str, Any]] = None
                        ) -> Path:
        """Save weights plus everything needed to rebuild the model exactly."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "module": "NeuroGenesisModel",
            "state_dict": self.state_dict(),
            "config": self.cfg.to_dict(),
            "spec": asdict(self.spec),
            "morph_dim": self.morph_dim,
            "morph_feature_names": self.morph_feature_names,
            "n_roi": self.n_roi,
            "roi_order": list(ROI_ORDER),
            "stage_order": list(STAGE_ORDER),
        }
        if extra:
            payload["extra"] = extra
        torch.save(payload, path)
        logger.info("Model checkpoint saved: %s (%d parameters)",
                    path, self.n_parameters())
        return path

    @classmethod
    def load_checkpoint(cls, path: Path, map_location: str = "cpu"
                        ) -> "NeuroGenesisModel":
        """Rebuild a model from a checkpoint.

        Raises:
            ValueError: If the checkpoint's ROI or stage ordering differs from
                the current constants, which would silently permute every
                ROI-indexed and class-indexed axis.
        """
        payload = torch.load(Path(path), map_location=map_location,
                             weights_only=False)
        if list(payload.get("roi_order", ROI_ORDER)) != list(ROI_ORDER):
            raise ValueError(
                "Checkpoint ROI order differs from the current ROI_ORDER.\n"
                f"  checkpoint: {payload.get('roi_order')}\n"
                f"  current   : {list(ROI_ORDER)}"
            )
        if list(payload.get("stage_order", STAGE_ORDER)) != list(STAGE_ORDER):
            raise ValueError(
                "Checkpoint stage order differs from the current STAGE_ORDER.\n"
                f"  checkpoint: {payload.get('stage_order')}\n"
                f"  current   : {list(STAGE_ORDER)}"
            )
        cfg = NeuroGenesisConfig.from_dict(payload["config"])
        spec = ModelSpec(**payload["spec"])
        model = cls(
            morph_dim=payload["morph_dim"],
            cfg=cfg,
            spec=spec,
            morph_feature_names=payload.get("morph_feature_names"),
            n_roi=payload.get("n_roi", N_ROI),
        )
        model.load_state_dict(payload["state_dict"])
        model.eval()
        logger.info("Model loaded: %s (spec=%s)", path, spec.name)
        return model


def build_model(
    morph_dim: int,
    cfg: Optional[NeuroGenesisConfig] = None,
    variant: str = "A7",
    morph_feature_names: Optional[List[str]] = None,
) -> NeuroGenesisModel:
    """Build a model from a named variant in :data:`ABLATION_SPECS`.

    Args:
        morph_dim: Per-ROI morphometric feature width.
        cfg: Framework configuration.
        variant: Key of :data:`ABLATION_SPECS` or :data:`BASELINE_SPECS`.
        morph_feature_names: Feature names for SRVE weight display.

    Returns:
        A configured :class:`NeuroGenesisModel`.

    Raises:
        KeyError: If the variant is unknown.
    """
    if variant in ABLATION_SPECS:
        spec = ABLATION_SPECS[variant]
    elif variant in BASELINE_SPECS:
        spec = BASELINE_SPECS[variant]
    else:
        raise KeyError(
            f"Unknown model variant {variant!r}. Ablations: "
            f"{sorted(ABLATION_SPECS)}; baselines: {sorted(BASELINE_SPECS)}"
        )
    return NeuroGenesisModel(
        morph_dim=morph_dim, cfg=cfg, spec=spec,
        morph_feature_names=morph_feature_names,
    )


__all__ = [
    "ModelSpec",
    "ABLATION_SPECS",
    "BASELINE_SPECS",
    "ModelOutput",
    "NeuroGenesisModel",
    "build_model",
]
