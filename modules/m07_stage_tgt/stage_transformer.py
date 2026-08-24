"""
M14 — Stage-Temporal Graph Transformer encoder (Section 12).
===========================================================

What this is, and what it deliberately is not
---------------------------------------------

OASIS-1 is cross-sectional. There are no repeated visits to model, so this
module **does not** forecast future MRI, does not simulate future scans, and
does not estimate a clinically validated MCI-to-AD conversion probability.

What it *does* is treat the ordered stage vocabulary CN -> MCI -> AD as the
sequence axis. The transformer receives four tokens:

.. code-block:: text

    [ c_CN ,  c_MCI ,  c_AD ,  Z_H ]
       |        |        |      |
       +--------+--------+      +--- current subject representation
       ordered stage tokens
       (+ stage positional embeddings)

and applies multi-head self-attention over them. The subject token can therefore
attend to each stage prototype, and each prototype to the subject, producing a
contextualised output ``Z_T`` that encodes *where in the ordered stage geometry
this subject's representation sits*.

The "temporal" in the name refers to this ordered stage axis — clinical severity
order — not to observed time. Positional embeddings over the three stage tokens
are what make the axis ordered rather than a bag of three labels; without them
the encoder would be permutation-invariant across stages and could not represent
"between MCI and AD" at all.

Terminology (Sections 12, 32)
-----------------------------

The outputs are **stage-transition propensity**, **advanced-stage alignment** and
**AD-associated propensity**. They are never to be described as conversion
probabilities, guaranteed future diagnoses, or biological predictions of onset.
:meth:`StageTransformer.disclaimer` returns the required wording.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from modules.common.config import StageTGTConfig
from modules.common.logging_utils import get_logger
from modules.common.roi_constants import N_STAGE, STAGE_ORDER
from modules.m07_stage_tgt.stage_prototypes import PrototypeOutput, StagePrototypes

logger = get_logger(__name__)

#: Index of the subject token within the sequence (it follows the stage tokens).
SUBJECT_TOKEN_INDEX = N_STAGE


@dataclass
class StageTransformerOutput:
    """Result of the Stage-TGT encoder."""

    #: ``(B, d_model)`` contextualised subject representation ``Z_T``.
    z_t: torch.Tensor
    #: ``(B, N_STAGE + 1, d_model)`` full token sequence output.
    tokens: torch.Tensor
    #: Prototype distances and similarities.
    prototype_output: PrototypeOutput
    #: Per-layer attention, ``(B, heads, L, L)`` each, when tracing was on.
    attention: List[torch.Tensor] = field(default_factory=list)

    def subject_to_stage_attention(self, batch_index: int = 0
                                   ) -> Optional[Dict[str, float]]:
        """Return how much the subject token attends to each stage token.

        Averaged over heads and layers. This is the most directly interpretable
        quantity the transformer produces: it says which stage representations
        the model consulted when contextualising this subject.

        Returns:
            ``{stage: attention}``, or ``None`` if no attention was recorded.
        """
        if not self.attention:
            return None
        stacked = torch.stack([a[batch_index] for a in self.attention])
        # (layers, heads, L, L) -> mean over layers and heads -> (L, L)
        mean = stacked.mean(dim=(0, 1))
        row = mean[SUBJECT_TOKEN_INDEX]
        return {stage: float(row[i]) for i, stage in enumerate(STAGE_ORDER)}


class _TracedEncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer that can return its attention map.

    ``torch.nn.TransformerEncoderLayer`` does not expose attention weights, and
    Section 58 requires the dashboard to display them, so the layer is written
    out explicitly. Pre-norm is used because it trains more stably at the very
    small batch sizes this study runs at.
    """

    def __init__(self, d_model: int, n_heads: int, ff_dim: int,
                 dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """Run the layer, optionally returning per-head attention weights."""
        h = self.norm1(x)
        attn_out, weights = self.attn(
            h, h, h,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x, weights


class StageTransformer(nn.Module):
    """Stage-Temporal Graph Transformer over ordered stage tokens.

    Args:
        d_model: Width of the shared representation ``Z_H``. Stage prototypes
            and tokens live in the same width.
        cfg: Stage-TGT hyper-parameters.
        n_stages: Number of ordered stages.
    """

    def __init__(
        self,
        d_model: int,
        cfg: Optional[StageTGTConfig] = None,
        n_stages: int = N_STAGE,
    ) -> None:
        super().__init__()
        self.cfg = cfg or StageTGTConfig()
        self.n_stages = n_stages
        self.d_model = d_model

        # Prototypes live in Z_H's space so distances are commensurate with the
        # classifier's decision geometry.
        #
        # The softmax temperature is deliberately NOT learned here. No loss term
        # depends on the prototype similarities `q_s` — the prototypes are
        # trained by the distance-based L_proto term and the propensity head has
        # its own learned alignment layer — so a trainable temperature would be
        # a parameter with no gradient path. It is fixed at sqrt(d_model), which
        # is the scale at which squared distances grow with dimensionality, and
        # serves purely to keep the reported `q_s` off the saturated ends of the
        # softmax.
        self.prototypes = StagePrototypes(
            d_model=d_model, n_stages=n_stages, learn_temperature=False
        )

        # Project into the transformer width. When they already match this is an
        # identity-shaped linear, kept for uniformity of the code path.
        self.input_proj = (
            nn.Identity() if d_model == self.cfg.d_model
            else nn.Linear(d_model, self.cfg.d_model)
        )
        self.width = self.cfg.d_model

        # Positional embeddings over [CN, MCI, AD, subject]. These are what make
        # the stage axis ordered rather than an unordered set.
        self.position = nn.Parameter(torch.zeros(n_stages + 1, self.width))
        nn.init.normal_(self.position, std=0.02)
        # A learned type embedding distinguishes "this is a stage prototype"
        # from "this is a subject".
        self.token_type = nn.Parameter(torch.zeros(2, self.width))
        nn.init.normal_(self.token_type, std=0.02)

        if self.cfg.d_model % self.cfg.n_heads != 0:
            raise ValueError(
                f"stage_tgt.d_model ({self.cfg.d_model}) must be divisible by "
                f"n_heads ({self.cfg.n_heads})"
            )

        self.layers = nn.ModuleList([
            _TracedEncoderLayer(
                d_model=self.width,
                n_heads=self.cfg.n_heads,
                ff_dim=self.cfg.ff_dim,
                dropout=self.cfg.dropout,
            )
            for _ in range(self.cfg.n_layers)
        ])
        self.norm_out = nn.LayerNorm(self.width)

    def forward(self, z_h: torch.Tensor,
                return_attention: bool = False) -> StageTransformerOutput:
        """Contextualise a subject representation against the stage prototypes.

        Args:
            z_h: ``(B, d_model)`` shared representation.
            return_attention: Record per-layer attention weights.

        Returns:
            A :class:`StageTransformerOutput`.

        Raises:
            ValueError: On a width mismatch.
        """
        if z_h.dim() == 1:
            z_h = z_h.unsqueeze(0)
        if z_h.shape[-1] != self.d_model:
            raise ValueError(
                f"Z_H width {z_h.shape[-1]} != Stage-TGT d_model {self.d_model}"
            )
        b = z_h.shape[0]

        proto_out = self.prototypes(z_h)

        stage_tokens = self.input_proj(
            proto_out.prototypes.unsqueeze(0).expand(b, self.n_stages, self.d_model)
        )
        subject_token = self.input_proj(z_h).unsqueeze(1)

        stage_tokens = stage_tokens + self.token_type[0]
        subject_token = subject_token + self.token_type[1]

        seq = torch.cat([stage_tokens, subject_token], dim=1)
        seq = seq + self.position.unsqueeze(0)

        attention: List[torch.Tensor] = []
        for layer in self.layers:
            seq, weights = layer(seq, return_attention=return_attention)
            if weights is not None:
                attention.append(weights.detach())

        seq = self.norm_out(seq)
        z_t = seq[:, SUBJECT_TOKEN_INDEX, :]

        return StageTransformerOutput(
            z_t=z_t,
            tokens=seq,
            prototype_output=proto_out,
            attention=attention,
        )

    @staticmethod
    def disclaimer() -> str:
        """Return the mandatory wording for every Stage-TGT output."""
        return (
            "Stage-transition propensity is a model-derived measure of how "
            "closely the subject's current structural representation aligns with "
            "a more advanced disease-stage representation. It is derived from a "
            "single cross-sectional scan. It is not a clinically validated "
            "conversion probability, not a prediction of future diagnosis, and "
            "not a biological estimate of disease onset."
        )

    def summary(self) -> Dict[str, Any]:
        """Return a description of the encoder."""
        return {
            "name": "Stage-TGT",
            "full_name": "Stage-Temporal Graph Transformer",
            "sequence_axis": "ordered disease stages CN -> MCI -> AD "
                             "(clinical severity order, not observed time)",
            "tokens": [*STAGE_ORDER, "subject (Z_H)"],
            "sequence_length": self.n_stages + 1,
            "d_model": self.width,
            "n_heads": self.cfg.n_heads,
            "n_layers": self.cfg.n_layers,
            "ff_dim": self.cfg.ff_dim,
            "positional_embeddings": "learned, over the ordered stage tokens; "
                                     "these make the stage axis ordered rather "
                                     "than a permutation-invariant set",
            "requires_longitudinal_data": False,
            "does_not_do": [
                "forecast future MRI",
                "simulate future scans",
                "estimate validated MCI-to-AD conversion probability",
            ],
            "prototypes": self.prototypes.summary(),
            "n_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
            "disclaimer": self.disclaimer(),
        }

    def extra_repr(self) -> str:
        """Torch module repr."""
        return (f"d_model={self.d_model}, width={self.width}, "
                f"layers={self.cfg.n_layers}, heads={self.cfg.n_heads}")


__all__ = [
    "SUBJECT_TOKEN_INDEX",
    "StageTransformerOutput",
    "StageTransformer",
]
