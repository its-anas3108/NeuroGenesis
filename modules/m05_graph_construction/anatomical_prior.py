"""
Anatomical prior adjacency ``A_prior`` (Section 7).
==================================================

Provenance
----------

Section 7 forbids arbitrarily inventing edge weights and requires the
anatomical-prior source to be clearly defined and documented. This is that
documentation.

The 13 directed edges below are taken **unchanged** from the pre-existing
``graph.graph_builder.ANATOMICAL_EDGES`` table, which this refactor preserves.
Each edge names the white-matter pathway it represents:

* **Superior longitudinal / arcuate fasciculus (SLF/AF)** — the dorsal stream
  linking Broca's area and Wernicke's area; the core speech loop.
* **Short arcuate fibres** — Broca <-> insula, the articulatory relay.
* **Extreme capsule system** — ventral-stream connections involving the insula
  and superior temporal cortex.
* **Intra-STG fibres** — short association fibres within the superior temporal
  gyrus, connecting STG to the posterior STG territory of Wernicke's area.
* **Within-IFG fibres** — short association fibres inside the inferior frontal
  gyrus, connecting the IFG node to the Broca subregion it contains.
* **Inferior fronto-occipital fasciculus (IFOF)** — long ventral association
  pathway giving IFG its temporal and insular connections.

.. important::

   **What the weights are, and what they are not.** The source file's comment
   claimed the values were "normalised DTI fractional anisotropy values from
   literature", but supplied no citation, and no DTI data exists anywhere in
   this repository. Presenting them as measured FA would be unsupportable.

   They are therefore documented here for what they demonstrably are:
   **expert-assigned ordinal weights in [0, 1] encoding the relative strength
   and directness of established speech-network pathways** — highest for short
   intra-gyral fibres (IFG->Broca 0.95, STG->Wernicke 0.90), high for the
   arcuate speech loop (0.85), moderate for the insular relay (0.60-0.75), and
   lowest for the long ventral IFOF route (0.50).

   Two consequences follow, and both are deliberate design responses:

   1. The prior is a *prior*, not ground truth. This is precisely why AP-LAF
      (Section 8.2) learns a data-driven attention adjacency ``A_att`` and fuses
      it with ``A_prior`` under a learnable ``alpha``, rather than trusting the
      prior outright.
   2. Ablation A1 (prior only) versus A3 (prior + learned attention) measures
      empirically how much the prior actually contributes, so its value is
      tested rather than assumed.

   If measured tractography becomes available, replace
   :data:`ANATOMICAL_EDGES` and record the source in
   :data:`PRIOR_PROVENANCE`; nothing else needs to change.

Asymmetry
---------

The prior is **directed and asymmetric** — e.g. ``STG -> Wernicke`` is 0.90
while ``Wernicke -> STG`` is 0.80 — reflecting the predominant direction of
information flow in the speech network. Only 13 of the 20 possible ordered
pairs carry an edge; the remainder are zero, and the learned ``A_att`` is what
allows a genuinely informative but anatomically indirect pathway to be
recovered.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from modules.common.logging_utils import get_logger
from modules.common.roi_constants import N_ROI, ROI_INDEX, ROI_ORDER

logger = get_logger(__name__)

#: ``(source, target, weight, tract)``. Preserved verbatim from
#: ``graph.graph_builder.ANATOMICAL_EDGES``.
ANATOMICAL_EDGES: List[Tuple[str, str, float, str]] = [
    # Superior longitudinal / arcuate fasciculus — core speech loop
    ("Broca_Area", "Wernicke_Area", 0.85, "Superior_Longitudinal_Fasciculus"),
    ("Wernicke_Area", "Broca_Area", 0.85, "Superior_Longitudinal_Fasciculus"),
    # Insular connections — articulatory relay
    ("Broca_Area", "Insula", 0.75, "Short_Arcuate_Fibres"),
    ("Insula", "Broca_Area", 0.70, "Short_Arcuate_Fibres"),
    ("Insula", "Wernicke_Area", 0.65, "Extreme_Capsule_System"),
    ("Wernicke_Area", "Insula", 0.60, "Extreme_Capsule_System"),
    # STG connections — auditory input route
    ("Superior_Temporal_Gyrus", "Wernicke_Area", 0.90, "Intra_STG_Fibres"),
    ("Wernicke_Area", "Superior_Temporal_Gyrus", 0.80, "Intra_STG_Fibres"),
    ("Superior_Temporal_Gyrus", "Insula", 0.55, "Extreme_Capsule_System"),
    # IFG connections — frontal syntactic network
    ("Inferior_Frontal_Gyrus", "Broca_Area", 0.95, "Within_IFG_Fibres"),
    ("Broca_Area", "Inferior_Frontal_Gyrus", 0.90, "Within_IFG_Fibres"),
    ("Inferior_Frontal_Gyrus", "Insula", 0.65, "Inferior_Fronto_Occipital_Fasciculus"),
    ("Inferior_Frontal_Gyrus", "Superior_Temporal_Gyrus", 0.50,
     "Inferior_Fronto_Occipital_Fasciculus"),
]

#: Machine-readable provenance record, embedded in every saved graph artifact
#: and rendered in the dashboard AP-LAF panel and the report.
PRIOR_PROVENANCE: Dict[str, Any] = {
    "source": "graph.graph_builder.ANATOMICAL_EDGES (preserved from the "
              "pre-refactor implementation)",
    "weight_semantics": "Expert-assigned ordinal weights in [0, 1] encoding the "
                        "relative strength and directness of established "
                        "speech-network white-matter pathways.",
    "is_measured_dti": False,
    "measurement_caveat": "These are NOT measured DTI fractional anisotropy "
                          "values. No diffusion data exists in this study. The "
                          "pre-refactor source comment claimed FA provenance "
                          "but supplied no citation; that claim is not carried "
                          "forward.",
    "directed": True,
    "symmetric": False,
    "n_edges": len(ANATOMICAL_EDGES),
    "n_possible_ordered_pairs": N_ROI * (N_ROI - 1),
    "tracts": sorted({tract for *_, tract in ANATOMICAL_EDGES}),
    "tested_by_ablation": "A1 (prior only) vs A3 (prior + learned attention) "
                          "quantifies the prior's empirical contribution.",
}


def build_prior_matrix(
    include_self_loops: bool = False,
    self_loop_weight: float = 1.0,
    normalize: bool = False,
) -> np.ndarray:
    """Build the dense ``A_prior`` matrix in canonical :data:`ROI_ORDER`.

    Args:
        include_self_loops: Place ``self_loop_weight`` on the diagonal. Graph
            attention layers need self-connections so a node can attend to
            itself; the *displayed* prior keeps a zero diagonal so the
            dashboard heatmap shows anatomy rather than an artificial diagonal.
        self_loop_weight: Diagonal value when self-loops are included.
        normalize: Row-normalise to sum 1 (excluding any self-loop scaling
            asymmetry). Off by default: normalisation destroys the absolute
            weight scale that makes the prior interpretable next to ``A_att``.

    Returns:
        Float32 array of shape ``(5, 5)`` where entry ``[i, j]`` is the prior
        weight of the directed edge ROI ``i`` -> ROI ``j``.

    Raises:
        KeyError: If an edge names an ROI outside :data:`ROI_ORDER`.
    """
    A = np.zeros((N_ROI, N_ROI), dtype=np.float32)
    for src, dst, weight, tract in ANATOMICAL_EDGES:
        if src not in ROI_INDEX or dst not in ROI_INDEX:
            raise KeyError(
                f"Anatomical edge ({src} -> {dst}, tract={tract}) references an "
                f"ROI outside ROI_ORDER {ROI_ORDER}"
            )
        A[ROI_INDEX[src], ROI_INDEX[dst]] = float(weight)

    if include_self_loops:
        np.fill_diagonal(A, float(self_loop_weight))

    if normalize:
        row_sum = A.sum(axis=1, keepdims=True)
        # A node with no outgoing prior edge would divide by zero; leave its row
        # at zero rather than manufacturing uniform connectivity.
        A = np.divide(A, row_sum, out=np.zeros_like(A), where=row_sum > 0)

    return A


def prior_mask(include_self_loops: bool = True) -> np.ndarray:
    """Return a boolean mask of positions the prior considers connected.

    Used as the attention support: positions outside the mask are masked out of
    the softmax so a node never attends across an anatomically absent pathway.
    Self-loops are included by default because a node must always be able to
    attend to itself.

    Args:
        include_self_loops: Mark the diagonal as connected.

    Returns:
        Boolean array of shape ``(5, 5)``.
    """
    mask = build_prior_matrix(include_self_loops=False) > 0
    if include_self_loops:
        np.fill_diagonal(mask, True)
    return mask


def edge_table() -> List[Dict[str, Any]]:
    """Return the prior as a list of row dicts for tables and the dashboard."""
    from modules.common.roi_constants import roi_short

    return [
        {
            "source": src,
            "target": dst,
            "source_short": roi_short(src),
            "target_short": roi_short(dst),
            "weight": float(weight),
            "tract": tract,
        }
        for src, dst, weight, tract in ANATOMICAL_EDGES
    ]


def tract_of(source: str, target: str) -> Optional[str]:
    """Return the tract name for a directed ROI pair, or ``None`` if unconnected."""
    for src, dst, _, tract in ANATOMICAL_EDGES:
        if src == source and dst == target:
            return tract
    return None


def describe_prior() -> str:
    """Return a multi-line human-readable description of the prior."""
    A = build_prior_matrix()
    density = float((A > 0).sum()) / (N_ROI * (N_ROI - 1))
    lines = [
        f"A_prior: {N_ROI}x{N_ROI} directed, asymmetric",
        f"  edges          : {len(ANATOMICAL_EDGES)} of "
        f"{N_ROI * (N_ROI - 1)} possible ordered pairs (density {density:.2f})",
        f"  weight range   : [{A[A > 0].min():.2f}, {A.max():.2f}]",
        f"  tracts         : {len(PRIOR_PROVENANCE['tracts'])}",
        f"  measured DTI   : {PRIOR_PROVENANCE['is_measured_dti']}",
        f"  semantics      : {PRIOR_PROVENANCE['weight_semantics']}",
    ]
    return "\n".join(lines)


__all__ = [
    "ANATOMICAL_EDGES",
    "PRIOR_PROVENANCE",
    "build_prior_matrix",
    "prior_mask",
    "edge_table",
    "tract_of",
    "describe_prior",
]
