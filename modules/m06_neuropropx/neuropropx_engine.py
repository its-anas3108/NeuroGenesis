"""
M11 — NeuroProp-X engine (Section 8).
=====================================

Composes the four stages into the framework's primary algorithmic contribution:

.. code-block:: text

    G = (V, X_morph || E_3D, A_prior)
              |
          [ M11.1 ]  SRVE    -> RV        stage-relevant regional vulnerability
              |
          [ M11.2 ]  AP-LAF  -> A_star    anatomical prior fused with learned attention
              |
          [ M11.3 ]  ANP     -> P         vulnerability-aware propagation representation
              |
          [ M11.4 ]  SAGR    -> X_star    enriched node representation
              |
    G* = (V, X_star, A_star, P)

Ablation switches
-----------------

:class:`NeuroPropXConfigFlags` exposes one boolean per component, and disabling a
component swaps in a shape-preserving stand-in (``ConstantVulnerability``,
``IdentityPropagation``, prior-only AP-LAF) rather than removing a tensor. Every
ablation therefore keeps identical downstream shapes, so a measured difference
is attributable to the component's *information* rather than to a change in
model capacity or input width. This is what makes the A0-A7 comparison in
Section 17 a fair test.

Claim discipline (Section 33)
-----------------------------

NeuroProp-X is presented as the *proposed* framework. Its components build on
established ideas — logistic scoring, attention-based adjacency learning,
learned edge features, feature concatenation — and none of those is claimed as
new. What is proposed is the specific SRVE + AP-LAF + ANP + SAGR formulation,
its integration of local 3D ROI representation with a disease-aware graph
representation, and its ablation-based validation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from modules.common.config import NeuroPropXConfig
from modules.common.logging_utils import get_logger
from modules.common.roi_constants import N_ROI, ROI_ORDER
from modules.m06_neuropropx.anp import ANP, IdentityPropagation
from modules.m06_neuropropx.ap_laf import APLAF
from modules.m06_neuropropx.sagr import SAGR
from modules.m06_neuropropx.srve import SRVE, ConstantVulnerability
from modules.m06_neuropropx.types import NeuroPropXOutput
from modules.common.serialization import json_safe

logger = get_logger(__name__)


@dataclass
class NeuroPropXConfigFlags:
    """Per-component enable flags used by the ablation study.

    Attributes:
        use_srve: Enable SRVE. When ``False``, every ROI receives a constant
            neutral vulnerability of 0.5.
        use_learned_attention: Enable the AP-LAF learned-attention branch. When
            ``False``, ``A_star`` reduces to the row-normalised anatomical prior.
        use_anp: Enable ANP. When ``False``, the propagation channel carries
            ``A_star`` unchanged.
        use_structural: Enable the AP-LAF structural-covariance operand,
            a per-subject correlation between ROI morphometric profiles.
            When ``False``, ``A_star`` mixes only the prior and attention.
        use_cnn: Include the 3D CNN embedding block in ``X_star``.
        use_centrality: Include the centrality block in ``X_star``.
    """

    use_srve: bool = True
    use_learned_attention: bool = True
    use_structural: bool = True
    use_anp: bool = True
    use_cnn: bool = True
    use_centrality: bool = True

    def as_dict(self) -> Dict[str, bool]:
        """Return the flags as a plain dict."""
        return asdict(self)


class NeuroPropX(nn.Module):
    """The NeuroProp-X framework: SRVE -> AP-LAF -> ANP -> SAGR.

    Args:
        morph_dim: Width of the per-ROI morphometric feature vector.
        cnn_dim: Width of the per-ROI 3D CNN embedding. Ignored when
            ``flags.use_cnn`` is ``False``.
        cfg: NeuroProp-X hyper-parameters.
        flags: Component enable flags for ablation.
        morph_feature_names: Optional morphometric feature names, forwarded to
            SRVE so the dashboard can label the learned weights.
        n_roi: Number of nodes.

    Shape:
        ``morph (B, N, morph_dim)``, ``cnn (B, N, cnn_dim)`` ->
        ``X_star (B, N, F*)``, ``A_star (B, N, N)``, ``P (B, N, N)``.
    """

    def __init__(
        self,
        morph_dim: int,
        cnn_dim: int,
        cfg: Optional[NeuroPropXConfig] = None,
        flags: Optional[NeuroPropXConfigFlags] = None,
        morph_feature_names: Optional[List[str]] = None,
        n_roi: int = N_ROI,
    ) -> None:
        super().__init__()
        self.cfg = cfg or NeuroPropXConfig()
        self.flags = flags or NeuroPropXConfigFlags()
        self.n_roi = n_roi
        self.morph_dim = morph_dim
        self.cnn_dim = cnn_dim if self.flags.use_cnn else 0

        #: Width of the node representation ``H`` that SRVE and AP-LAF consume.
        self.input_dim = self.morph_dim + self.cnn_dim
        if self.input_dim == 0:
            raise ValueError(
                "NeuroProp-X requires a non-empty node representation: both "
                "morph_dim and the effective cnn_dim are zero."
            )

        srve_names = None
        if morph_feature_names and self.cnn_dim == 0:
            srve_names = list(morph_feature_names)
        elif morph_feature_names:
            srve_names = list(morph_feature_names) + [
                f"cnn_{i}" for i in range(self.cnn_dim)
            ]

        # M11.1
        if self.flags.use_srve:
            self.srve: nn.Module = SRVE(
                in_dim=self.input_dim,
                n_roi=n_roi,
                per_roi_bias=True,
                feature_names=srve_names,
            )
        else:
            self.srve = ConstantVulnerability(n_roi=n_roi)

        # M11.2
        self.ap_laf = APLAF(
            node_dim=self.input_dim,
            hidden_dim=self.cfg.attention_hidden,
            alpha_logit_init=self.cfg.alpha_logit_init,
            learn_alpha=self.cfg.learn_alpha,
            learn_attention=self.flags.use_learned_attention,
            use_structural=self.flags.use_structural,
            structural_temperature=self.cfg.structural_temperature,
            negative_slope=0.2,
            n_roi=n_roi,
        )

        # M11.3
        if self.flags.use_anp:
            self.anp: nn.Module = ANP(
                use_topology=self.cfg.anp_use_topology, n_roi=n_roi
            )
        else:
            self.anp = IdentityPropagation(n_roi=n_roi)

        # M11.4
        use_centrality = self.flags.use_centrality and self.cfg.sagr_use_centrality
        self.sagr = SAGR(
            morph_dim=self.morph_dim,
            cnn_dim=self.cnn_dim,
            use_centrality=use_centrality,
            n_roi=n_roi,
        )

    @property
    def out_dim(self) -> int:
        """Width ``F*`` of the enriched node representation ``X_star``."""
        return self.sagr.out_dim

    @property
    def edge_feature_dim(self) -> int:
        """Width of the per-edge feature ``E_ij`` consumed by SAEG-GATv2."""
        return 3  # [A_prior_ij, A_att_ij, P_ij]

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        morph_features: Optional[torch.Tensor],
        cnn_embeddings: Optional[torch.Tensor] = None,
        keep_features: bool = False,
    ) -> NeuroPropXOutput:
        """Transform the ordinary graph ``G`` into the enhanced graph ``G*``.

        Args:
            morph_features: ``(B, N, morph_dim)`` normalized morphometric
                features, or ``None`` when ``morph_dim == 0``.
            cnn_embeddings: ``(B, N, cnn_dim)`` 3D CNN embeddings, or ``None``
                when the CNN branch is disabled.
            keep_features: Retain SRVE's input features in the output for
                dashboard traceability.

        Returns:
            A :class:`~modules.m06_neuropropx.types.NeuroPropXOutput`.

        Raises:
            ValueError: If the supplied blocks contradict the configured widths.
        """
        blocks: List[torch.Tensor] = []

        if self.morph_dim:
            if morph_features is None:
                raise ValueError(
                    f"morph_dim={self.morph_dim} but morph_features is None."
                )
            m = morph_features.unsqueeze(0) if morph_features.dim() == 2 \
                else morph_features
            if m.shape[-1] != self.morph_dim:
                raise ValueError(
                    f"morph_features width {m.shape[-1]} != morph_dim "
                    f"{self.morph_dim}"
                )
            blocks.append(m)
        else:
            m = None

        if self.cnn_dim:
            if cnn_embeddings is None:
                raise ValueError(
                    f"cnn_dim={self.cnn_dim} but cnn_embeddings is None. Pass "
                    "flags.use_cnn=False to run without the spatial branch."
                )
            c = cnn_embeddings.unsqueeze(0) if cnn_embeddings.dim() == 2 \
                else cnn_embeddings
            if c.shape[-1] != self.cnn_dim:
                raise ValueError(
                    f"cnn_embeddings width {c.shape[-1]} != cnn_dim "
                    f"{self.cnn_dim}"
                )
            blocks.append(c)
        else:
            c = None

        # H_i = [X_i_morph || E_i_3D]  (Section 7)
        h = torch.cat(blocks, dim=-1) if len(blocks) > 1 else blocks[0]

        srve_out = self.srve(h, keep_features=keep_features)
        # The structural operand is deliberately computed from the
        # morphometric block alone, not from `h`. Mixing the CNN embedding
        # into a 'structural similarity' would make the term a similarity of
        # learned representations, which is neither interpretable nor the
        # structural-covariance construct it is meant to be.
        aplaf_out = self.ap_laf(h, structural_source=m if m is not None else h)
        anp_out = self.anp(
            srve_out.regional_vulnerability, aplaf_out.adaptive_adjacency
        )
        sagr_out = self.sagr(
            morph_features=m,
            cnn_embeddings=c,
            regional_vulnerability=srve_out.regional_vulnerability,
            a_star=aplaf_out.adaptive_adjacency,
        )

        return NeuroPropXOutput(
            srve=srve_out,
            ap_laf=aplaf_out,
            anp=anp_out,
            sagr=sagr_out,
            active_components=self.flags.as_dict(),
        )

    # ── Introspection ─────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """Return a full description of the configured framework."""
        return {
            "name": "NeuroProp-X",
            "full_name": "Stage-Aware Regional Vulnerability and Adaptive "
                         "Graph Propagation",
            "status": "proposed framework",
            "stages": {
                "M11.1_SRVE": {
                    "full_name": "Stage-Aware Regional Vulnerability Estimation",
                    "active": self.flags.use_srve,
                    "formula": "RV_i = sigmoid(w_v^T xhat_i + b_v)",
                    "learned_weights": self.srve.learned_weights(),
                    "learned_bias": self.srve.learned_bias(),
                },
                "M11.2_AP_LAF": {
                    "full_name": "Anatomical-Prior and Learned-Attention Fusion",
                    "active": True,
                    **self.ap_laf.summary(),
                },
                "M11.3_ANP": {
                    "full_name": "Adaptive Neurodegeneration Propagation",
                    "active": self.flags.use_anp,
                    **self.anp.summary(),
                },
                "M11.4_SAGR": {
                    "full_name": "Stage-Aware Graph Representation",
                    "active": True,
                    **self.sagr.summary(),
                },
            },
            "dimensions": {
                "morph_dim": self.morph_dim,
                "cnn_dim": self.cnn_dim,
                "node_input_dim": self.input_dim,
                "node_output_dim": self.out_dim,
                "edge_feature_dim": self.edge_feature_dim,
                "n_nodes": self.n_roi,
            },
            "flags": self.flags.as_dict(),
            "n_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
            "novelty_statement": (
                "The proposed contribution is the specific SRVE + AP-LAF + ANP "
                "+ SAGR formulation, its integration of local 3D ROI "
                "representation with a disease-aware graph representation, and "
                "its ablation-based validation. The underlying techniques "
                "(logistic scoring, attention-based adjacency learning, learned "
                "edge features) are established and are not claimed as new."
            ),
            "interpretation_note": NeuroPropXOutput.disclaimer(),
        }

    def extra_repr(self) -> str:
        """Torch module repr."""
        return (
            f"morph_dim={self.morph_dim}, cnn_dim={self.cnn_dim}, "
            f"out_dim={self.out_dim}, flags={self.flags.as_dict()}"
        )


def save_neuropropx_output(
    output: NeuroPropXOutput,
    out_root: Path,
    subject_id: str,
    batch_index: int = 0,
) -> Dict[str, Path]:
    """Persist ``G*`` and all its components (Section 8.4, Section 25).

    Writes into the ``outputs/neuropropx/`` subtree:

    * ``vulnerability/``       — ``RV`` as ``.npy`` and ``.csv``
    * ``adaptive_adjacency/``  — ``A_prior``, ``A_att``, ``A_star``
    * ``propagation/``         — ``P``
    * ``enriched_graph/``      — ``X_star``, edge features, metadata JSON

    Args:
        output: The NeuroProp-X output.
        out_root: Root ``outputs/`` directory.
        subject_id: Subject/session identifier.
        batch_index: Which subject in the batch to export.

    Returns:
        Mapping of logical name -> written path.
    """
    import pandas as pd

    arrays = output.to_numpy(batch_index)
    meta = output.metadata(batch_index)
    written: Dict[str, Path] = {}

    def target(sub: str, name: str) -> Path:
        d = Path(out_root) / "neuropropx" / sub / subject_id
        d.mkdir(parents=True, exist_ok=True)
        return d / name

    # Vulnerability
    p = target("vulnerability", f"{subject_id}_regional_vulnerability.npy")
    np.save(p, arrays["regional_vulnerability"])
    written["vulnerability_npy"] = p

    rv_df = pd.DataFrame({
        "roi": ROI_ORDER,
        "regional_vulnerability": arrays["regional_vulnerability"],
    }).sort_values("regional_vulnerability", ascending=False)
    rv_df.insert(0, "rank", range(1, len(rv_df) + 1))
    p = target("vulnerability", f"{subject_id}_regional_vulnerability.csv")
    rv_df.to_csv(p, index=False)
    written["vulnerability_csv"] = p

    # Adjacency matrices
    for key, fname in (
        ("prior_adjacency", "A_prior"),
        ("attention_adjacency", "A_att"),
        ("adaptive_adjacency", "A_star"),
    ):
        p = target("adaptive_adjacency", f"{subject_id}_{fname}.npy")
        np.save(p, arrays[key])
        written[f"{fname}_npy"] = p
        p = target("adaptive_adjacency", f"{subject_id}_{fname}.csv")
        pd.DataFrame(arrays[key], index=ROI_ORDER, columns=ROI_ORDER).to_csv(p)
        written[f"{fname}_csv"] = p

    # Propagation
    p = target("propagation", f"{subject_id}_propagation_score.npy")
    np.save(p, arrays["propagation_score"])
    written["propagation_npy"] = p
    p = target("propagation", f"{subject_id}_propagation_score.csv")
    pd.DataFrame(
        arrays["propagation_score"], index=ROI_ORDER, columns=ROI_ORDER
    ).to_csv(p)
    written["propagation_csv"] = p

    # Enriched graph
    p = target("enriched_graph", f"{subject_id}_X_star.npy")
    np.save(p, arrays["node_features"])
    written["X_star_npy"] = p
    p = target("enriched_graph", f"{subject_id}_edge_features.npy")
    np.save(p, arrays["edge_features"])
    written["edge_features_npy"] = p

    p = target("enriched_graph", f"{subject_id}_G_star.json")
    p.write_text(json.dumps(json_safe(meta), indent=2), encoding="utf-8")
    written["metadata_json"] = p

    logger.info("NeuroProp-X output saved for %s (%d artifacts)",
                subject_id, len(written))
    return written


__all__ = [
    "NeuroPropXConfigFlags",
    "NeuroPropX",
    "save_neuropropx_output",
]
