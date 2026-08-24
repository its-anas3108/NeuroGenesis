"""
M11.3 — ANP: Adaptive Neurodegeneration Propagation (Section 8.3).
=================================================================

Builds a vulnerability-aware relational representation over the adaptive graph:

.. math::

    P_{ij} = \\sigma\\!\\left(
        \\beta_1 RV_i + \\beta_2 RV_j + \\beta_3 A^*_{ij}
        + \\beta_4 T_{ij} + b
    \\right)

with learned coefficients :math:`\\beta_{1..4}` and bias :math:`b`. The optional
topology term :math:`T_{ij}` is described below.

.. important::

   **What ``P`` is.** A learned edge-level representation that combines how
   discriminative each endpoint region is with how strongly the adaptive graph
   connects them. It is consumed as an edge feature by SAEG-GATv2 and surfaced
   as pathway importance.

   **What ``P`` is not.** It is not a biological disease-spread probability, not
   a transmission rate, and not a validated measure of pathology propagation.
   Nothing in a cross-sectional dataset could establish any of those. The
   variable is deliberately named ``propagation_score`` throughout the codebase
   so that a reader of the code cannot mistake it for a spread probability, and
   :meth:`ANP.summary` returns the wording required by Section 54.

Asymmetry
---------

:math:`\\beta_1` and :math:`\\beta_2` are **separate** coefficients for the
source and target endpoint. A single shared coefficient would force
:math:`P_{ij}` to be symmetric in the endpoint-vulnerability term, which would
discard the directionality that the anatomical prior and the attention branch
both express. Keeping them separate lets the model learn whether source-region
or target-region vulnerability is the more informative signal.

The topology term
-----------------

:math:`T_{ij}` is the product of the source's out-strength and the target's
in-strength under :math:`A^*`, normalised by the largest such product in the
subject's graph:

.. math::

    T_{ij} = \\frac{s^{\\text{out}}_i \\cdot s^{\\text{in}}_j}
                   {\\max_{kl}\\left(s^{\\text{out}}_k \\cdot s^{\\text{in}}_l\\right)}

This gives the model access to a hub-to-hub signal — an edge between two highly
connected regions is structurally different from an edge between two peripheral
ones — that the pairwise terms alone cannot express. Per-subject normalisation
keeps the term in ``[0, 1]`` and comparable across subjects.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from modules.common.logging_utils import get_logger
from modules.common.roi_constants import N_ROI
from modules.m06_neuropropx.types import ANPOutput

logger = get_logger(__name__)


class ANP(nn.Module):
    """Adaptive Neurodegeneration Propagation.

    Args:
        use_topology: Include the topology term :math:`T_{ij}`.
        n_roi: Number of nodes.
        init_scale: Initial magnitude of the beta coefficients. Small positive
            values start the module near ``sigmoid(0) = 0.5`` for every edge, so
            the propagation matrix begins uninformative and is shaped by
            training rather than by initialisation.

    Shape:
        ``RV`` of ``(B, N)`` and ``A_star`` of ``(B, N, N)`` -> ``P`` of
        ``(B, N, N)``.
    """

    def __init__(
        self,
        use_topology: bool = True,
        n_roi: int = N_ROI,
        init_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.use_topology = use_topology
        self.n_roi = n_roi

        self.beta1 = nn.Parameter(torch.tensor(init_scale))  # source RV
        self.beta2 = nn.Parameter(torch.tensor(init_scale))  # target RV
        self.beta3 = nn.Parameter(torch.tensor(init_scale))  # adaptive weight
        if use_topology:
            self.beta4 = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("beta4", torch.tensor(0.0), persistent=True)
        self.bias = nn.Parameter(torch.tensor(0.0))

    # ── Topology ──────────────────────────────────────────────────────────

    @staticmethod
    def topology_term(a_star: torch.Tensor, eps: float = 1e-12
                      ) -> torch.Tensor:
        """Compute the normalised hub-to-hub topology term ``T``.

        Args:
            a_star: ``(B, N, N)`` adaptive adjacency.
            eps: Numerical floor for the per-subject normaliser.

        Returns:
            ``(B, N, N)`` tensor in ``[0, 1]``.
        """
        out_strength = a_star.sum(dim=2)  # (B, N) row sums
        in_strength = a_star.sum(dim=1)   # (B, N) column sums
        product = out_strength.unsqueeze(2) * in_strength.unsqueeze(1)
        # Per-subject max so the term is comparable across subjects with
        # different overall connectivity mass.
        denom = product.amax(dim=(1, 2), keepdim=True).clamp_min(eps)
        return product / denom

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, regional_vulnerability: torch.Tensor,
                a_star: torch.Tensor) -> ANPOutput:
        """Compute the propagation representation ``P``.

        Args:
            regional_vulnerability: ``(B, N)`` ``RV`` from SRVE.
            a_star: ``(B, N, N)`` adaptive adjacency from AP-LAF.

        Returns:
            An :class:`~modules.m06_neuropropx.types.ANPOutput`.

        Raises:
            ValueError: If the shapes are inconsistent with each other or with
                ``n_roi``.
        """
        rv = regional_vulnerability
        if rv.dim() == 1:
            rv = rv.unsqueeze(0)
        if a_star.dim() == 2:
            a_star = a_star.unsqueeze(0)
        if rv.dim() != 2 or a_star.dim() != 3:
            raise ValueError(
                "Expected RV of shape (B, N) and A_star of shape (B, N, N); got "
                f"{tuple(rv.shape)} and {tuple(a_star.shape)}"
            )
        b, n = rv.shape
        if a_star.shape[0] != b or a_star.shape[1:] != (n, n):
            raise ValueError(
                f"RV shape {tuple(rv.shape)} is inconsistent with A_star shape "
                f"{tuple(a_star.shape)}"
            )
        if n != self.n_roi:
            raise ValueError(
                f"Node axis is {n} but ANP was built for {self.n_roi} nodes."
            )

        rv_source = rv.unsqueeze(2).expand(b, n, n)  # RV_i broadcast over j
        rv_target = rv.unsqueeze(1).expand(b, n, n)  # RV_j broadcast over i

        logits = (
            self.beta1 * rv_source
            + self.beta2 * rv_target
            + self.beta3 * a_star
            + self.bias
        )

        topo: Optional[torch.Tensor] = None
        if self.use_topology:
            topo = self.topology_term(a_star)
            logits = logits + self.beta4 * topo

        return ANPOutput(
            propagation_score=torch.sigmoid(logits),
            logits=logits,
            topology_term=topo,
            beta=self.coefficients(),
        )

    # ── Introspection ─────────────────────────────────────────────────────

    def coefficients(self) -> Dict[str, float]:
        """Return the learned coefficients as plain floats."""
        return {
            "beta1_source_vulnerability": float(self.beta1.detach().cpu()),
            "beta2_target_vulnerability": float(self.beta2.detach().cpu()),
            "beta3_adaptive_adjacency": float(self.beta3.detach().cpu()),
            "beta4_topology": float(self.beta4.detach().cpu()),
            "bias": float(self.bias.detach().cpu()),
        }

    def explain_edge(
        self,
        regional_vulnerability: torch.Tensor,
        a_star: torch.Tensor,
        source_index: int,
        target_index: int,
        batch_index: int = 0,
    ) -> Dict[str, object]:
        """Decompose one edge's propagation score into its additive terms.

        Section 54 requires that selecting an edge in the dashboard shows
        ``RV_i``, ``RV_j``, ``A_star_ij`` and the resulting score. Because the
        model is additive before the sigmoid, this decomposition is exact.

        Args:
            regional_vulnerability: ``(B, N)`` ``RV``.
            a_star: ``(B, N, N)`` adaptive adjacency.
            source_index: Source ROI index.
            target_index: Target ROI index.
            batch_index: Subject within the batch.

        Returns:
            A dict of the individual terms, the logit and the final score.
        """
        from modules.common.roi_constants import ROI_ORDER

        with torch.no_grad():
            out = self.forward(regional_vulnerability, a_star)
            rv = regional_vulnerability
            if rv.dim() == 1:
                rv = rv.unsqueeze(0)
            a = a_star if a_star.dim() == 3 else a_star.unsqueeze(0)

            rv_i = float(rv[batch_index, source_index])
            rv_j = float(rv[batch_index, target_index])
            a_ij = float(a[batch_index, source_index, target_index])
            coef = self.coefficients()

            terms = {
                "beta1 * RV_source": coef["beta1_source_vulnerability"] * rv_i,
                "beta2 * RV_target": coef["beta2_target_vulnerability"] * rv_j,
                "beta3 * A_star": coef["beta3_adaptive_adjacency"] * a_ij,
                "bias": coef["bias"],
            }
            if self.use_topology and out.topology_term is not None:
                t_ij = float(out.topology_term[batch_index, source_index,
                                               target_index])
                terms["beta4 * topology"] = coef["beta4_topology"] * t_ij
            else:
                t_ij = None

            return {
                "source": ROI_ORDER[source_index],
                "target": ROI_ORDER[target_index],
                "RV_source": rv_i,
                "RV_target": rv_j,
                "A_star": a_ij,
                "topology": t_ij,
                "coefficients": coef,
                "terms": terms,
                "logit": float(out.logits[batch_index, source_index,
                                          target_index]),
                "propagation_score": float(
                    out.propagation_score[batch_index, source_index, target_index]
                ),
                "formula": (
                    "P_ij = sigmoid(beta1*RV_i + beta2*RV_j + beta3*A_star_ij"
                    + (" + beta4*T_ij" if self.use_topology else "")
                    + " + b)"
                ),
                "interpretation": self.interpretation_note(),
            }

    @staticmethod
    def interpretation_note() -> str:
        """Return the mandatory labelling for ``P`` (Section 54)."""
        return (
            "Model-derived propagation representation. This is a learned "
            "edge-level score combining endpoint regional vulnerability with "
            "adaptive graph connectivity. It is not a biological "
            "disease-spread probability."
        )

    def summary(self) -> Dict[str, object]:
        """Return a description of the module for the dashboard."""
        return {
            "formula": (
                "P_ij = sigmoid(beta1*RV_i + beta2*RV_j + beta3*A_star_ij"
                + (" + beta4*T_ij" if self.use_topology else "")
                + " + b)"
            ),
            "coefficients": self.coefficients(),
            "topology_active": self.use_topology,
            "topology_definition": (
                "T_ij = (out-strength_i * in-strength_j) normalised by the "
                "per-subject maximum over all node pairs."
            ),
            "label": self.interpretation_note(),
            "n_nodes": self.n_roi,
        }

    def extra_repr(self) -> str:
        """Torch module repr."""
        return f"n_roi={self.n_roi}, use_topology={self.use_topology}"


class IdentityPropagation(nn.Module):
    """ANP replacement that passes the adaptive adjacency through unchanged.

    Used by ablations A0-A4, where ANP must be absent but the edge-feature
    tensor consumed by SAEG-GATv2 must keep its width so the comparison isolates
    ANP's contribution rather than an architectural change.
    """

    def __init__(self, n_roi: int = N_ROI) -> None:
        super().__init__()
        self.n_roi = n_roi
        self.use_topology = False

    def forward(self, regional_vulnerability: torch.Tensor,
                a_star: torch.Tensor) -> ANPOutput:
        """Return ``A_star`` as the propagation representation."""
        if a_star.dim() == 2:
            a_star = a_star.unsqueeze(0)
        return ANPOutput(
            propagation_score=a_star,
            logits=a_star,
            topology_term=None,
            beta=None,
        )

    def coefficients(self) -> Dict[str, float]:
        """No coefficients exist in the ablated variant."""
        return {}

    def summary(self) -> Dict[str, object]:
        """Description for the dashboard."""
        return {
            "formula": "P = A_star (ANP ablated)",
            "coefficients": {},
            "topology_active": False,
            "label": "ANP disabled for this ablation configuration.",
            "n_nodes": self.n_roi,
        }

    @staticmethod
    def interpretation_note() -> str:
        """Ablated-variant note."""
        return "ANP disabled; the propagation channel carries A_star unchanged."

    def extra_repr(self) -> str:
        """Torch module repr."""
        return f"n_roi={self.n_roi} (ANP ablated)"


__all__ = ["ANP", "IdentityPropagation"]
