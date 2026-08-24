"""
M9 — Lightweight 3D CNN ROI patch encoder (Section 5).
======================================================

Learns local 3D spatial/morphological structure inside each 48x48x48 speech-ROI
patch — the patterns that hand-crafted morphometric features (volume, thickness,
entropy) cannot express.

Architecture, per Section 5::

    input  (1, 48, 48, 48)
      Conv3d -> BatchNorm3d -> ReLU -> MaxPool3d   ->  (16, 24, 24, 24)
      Conv3d -> BatchNorm3d -> ReLU -> MaxPool3d   ->  (32, 12, 12, 12)
      Conv3d -> BatchNorm3d -> ReLU                ->  (64, 12, 12, 12)
      GlobalAvgPool3d                              ->  (64,)
      Dropout -> Linear                            ->  (128,)   = E_i_3D

Deliberately small. There are only five patches per subject and, on the OASIS-1
labelled cohort, 154 training subjects — roughly 250k parameters is already
generous, and a deeper 3D network would memorise the training set before
learning anything transferable.

By default a **single encoder is shared** across all five ROIs
(``shared_encoder=True``). Per-ROI encoders quintuple the parameter count for
the same amount of data; sharing also lets the encoder pool morphological
evidence across regions. A learned per-ROI embedding vector is added to the
projection so the shared trunk can still specialise its output per region.

Every forward pass can emit a :class:`LayerTrace` list recording real
intermediate tensor shapes, because Section 49 requires the dashboard to show
intermediate shapes rather than only the final embedding.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.common.config import SpatialEncoderConfig
from modules.common.logging_utils import get_logger
from modules.common.roi_constants import N_ROI, ROI_ORDER

logger = get_logger(__name__)


@dataclass
class LayerTrace:
    """Recorded shape and cost of one stage of the encoder."""

    name: str
    kind: str
    output_shape: Tuple[int, ...]
    n_params: int = 0
    #: Number of elements in the output tensor for a single sample.
    n_elements: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "name": self.name,
            "kind": self.kind,
            "output_shape": list(self.output_shape),
            "n_params": self.n_params,
            "n_elements": self.n_elements,
        }


@dataclass
class EncoderOutput:
    """Result of encoding a batch of ROI tensors."""

    #: ``(B, N_ROI, embed_dim)`` per-ROI spatial embeddings ``E_i_3D``.
    embeddings: torch.Tensor
    #: ``(B, N_ROI * embed_dim)`` flattened subject-level spatial representation.
    flat: torch.Tensor
    #: Intermediate shapes for one sample, when tracing was requested.
    trace: List[LayerTrace] = field(default_factory=list)

    def per_roi(self) -> Dict[str, torch.Tensor]:
        """Return ``{roi_name: (B, embed_dim)}`` for the canonical ROI order."""
        return {name: self.embeddings[:, i, :] for i, name in enumerate(ROI_ORDER)}


class ROIPatchEncoder(nn.Module):
    """Three-block 3D CNN mapping one ROI patch to an embedding.

    Args:
        cfg: Encoder configuration. ``cfg.channels`` gives the three block
            widths and ``cfg.embed_dim`` the output size.

    Shape:
        input ``(B, 1, D, H, W)`` -> output ``(B, embed_dim)``.
    """

    def __init__(self, cfg: SpatialEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c1, c2, c3 = cfg.channels
        k = cfg.kernel_size
        pad = k // 2

        self.block1 = nn.Sequential(
            nn.Conv3d(1, c1, kernel_size=k, padding=pad, bias=False),
            nn.BatchNorm3d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(c1, c2, kernel_size=k, padding=pad, bias=False),
            nn.BatchNorm3d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv3d(c2, c3, kernel_size=k, padding=pad, bias=False),
            nn.BatchNorm3d(c3),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.project = nn.Linear(c3, cfg.embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming-initialise convolutions; unit-initialise norms."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of single-channel patches to embeddings."""
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        # Global average pooling over the three spatial axes.
        h = h.mean(dim=(2, 3, 4))
        h = self.dropout(h)
        return self.project(h)

    def forward_with_trace(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[LayerTrace]]:
        """Encode and record real intermediate tensor shapes.

        Args:
            x: ``(B, 1, D, H, W)`` patch batch.

        Returns:
            ``(embeddings, trace)``. Shapes in the trace omit the batch axis so
            they describe one sample, which is what the dashboard displays.
        """
        trace: List[LayerTrace] = [
            LayerTrace("input", "input", tuple(x.shape[1:]), 0,
                       int(np.prod(x.shape[1:])))
        ]

        def record(name: str, kind: str, tensor: torch.Tensor,
                   module: Optional[nn.Module]) -> None:
            n_params = (
                sum(p.numel() for p in module.parameters()) if module else 0
            )
            shape = tuple(tensor.shape[1:])
            trace.append(
                LayerTrace(name, kind, shape, n_params, int(np.prod(shape)))
            )

        h = self.block1(x)
        record("block1 (Conv3d-BN-ReLU-MaxPool)", "conv_block", h, self.block1)
        h = self.block2(h)
        record("block2 (Conv3d-BN-ReLU-MaxPool)", "conv_block", h, self.block2)
        h = self.block3(h)
        record("block3 (Conv3d-BN-ReLU)", "conv_block", h, self.block3)
        h = h.mean(dim=(2, 3, 4))
        record("global_avg_pool", "pool", h, None)
        h = self.dropout(h)
        out = self.project(h)
        record("projection (Linear)", "linear", out, self.project)
        return out, trace


class SpatialEncoder3D(nn.Module):
    """Encode a subject's five ROI patches into five spatial embeddings.

    Args:
        cfg: Encoder configuration. When ``cfg.shared_encoder`` is ``False`` a
            separate :class:`ROIPatchEncoder` is instantiated per ROI.
        n_roi: Number of ROIs; defaults to the canonical five.

    Shape:
        input ``(B, n_roi, D, H, W)`` -> output ``(B, n_roi, embed_dim)``.
    """

    def __init__(self, cfg: Optional[SpatialEncoderConfig] = None,
                 n_roi: int = N_ROI) -> None:
        super().__init__()
        self.cfg = cfg or SpatialEncoderConfig()
        self.n_roi = n_roi
        self.embed_dim = self.cfg.embed_dim

        if self.cfg.shared_encoder:
            self.encoder = ROIPatchEncoder(self.cfg)
            self.encoders = None
            # Lets one shared trunk still produce region-specific embeddings.
            self.roi_embedding = nn.Parameter(
                torch.zeros(n_roi, self.cfg.embed_dim)
            )
            nn.init.normal_(self.roi_embedding, std=0.02)
        else:
            self.encoder = None
            self.encoders = nn.ModuleList(
                [ROIPatchEncoder(self.cfg) for _ in range(n_roi)]
            )
            self.roi_embedding = None

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, patches: torch.Tensor,
                trace: bool = False) -> EncoderOutput:
        """Encode a batch of ROI patch tensors.

        Args:
            patches: ``(B, n_roi, D, H, W)``. A single unbatched
                ``(n_roi, D, H, W)`` tensor is accepted and promoted to batch
                size 1, because single-subject inference is a first-class mode.
            trace: Record intermediate shapes (adds one extra traced forward
                pass over the first sample only).

        Returns:
            An :class:`EncoderOutput`.

        Raises:
            ValueError: If the ROI axis does not match ``n_roi``, or the tensor
                is not 4-D/5-D. A silent mismatch here would misalign every
                downstream ROI-indexed axis.
        """
        if patches.dim() == 4:
            patches = patches.unsqueeze(0)
        if patches.dim() != 5:
            raise ValueError(
                "Expected patches of shape (B, n_roi, D, H, W) or "
                f"(n_roi, D, H, W); got {tuple(patches.shape)}"
            )
        b, n, d, h, w = patches.shape
        if n != self.n_roi:
            raise ValueError(
                f"ROI axis is {n} but the encoder was built for {self.n_roi} "
                f"ROIs ({ROI_ORDER}). Patch tensors must follow ROI_ORDER."
            )

        traces: List[LayerTrace] = []

        if self.cfg.shared_encoder:
            # Fold the ROI axis into the batch so one trunk sees all patches.
            flat = patches.reshape(b * n, 1, d, h, w)
            if trace:
                _, traces = self.encoder.forward_with_trace(flat[:1])
            emb = self.encoder(flat).reshape(b, n, self.embed_dim)
            emb = emb + self.roi_embedding.unsqueeze(0)
        else:
            outs = []
            for i in range(n):
                patch_i = patches[:, i:i + 1]
                if trace and i == 0:
                    _, traces = self.encoders[i].forward_with_trace(patch_i[:1])
                outs.append(self.encoders[i](patch_i))
            emb = torch.stack(outs, dim=1)

        return EncoderOutput(
            embeddings=emb,
            flat=emb.reshape(b, n * self.embed_dim),
            trace=traces,
        )

    # ── Inference helper ──────────────────────────────────────────────────

    @torch.no_grad()
    def encode(self, patches: np.ndarray | torch.Tensor,
               device: str = "cpu", trace: bool = False) -> EncoderOutput:
        """Deterministic inference: eval mode, no grad, no dropout.

        Batch-norm running statistics are used rather than batch statistics, so
        a single subject encoded alone gets the same embedding as when encoded
        inside a larger batch. Encoding in train mode would make the embedding
        depend on which other subjects happened to share the batch.

        Args:
            patches: ``(n_roi, D, H, W)`` or ``(B, n_roi, D, H, W)``.
            device: Torch device string.
            trace: Also return intermediate shapes.

        Returns:
            An :class:`EncoderOutput` on the CPU.
        """
        was_training = self.training
        self.eval()
        try:
            if isinstance(patches, np.ndarray):
                tensor = torch.from_numpy(np.ascontiguousarray(patches)).float()
            else:
                tensor = patches.float()
            tensor = tensor.to(device)
            self.to(device)
            out = self.forward(tensor, trace=trace)
            return EncoderOutput(
                embeddings=out.embeddings.detach().cpu(),
                flat=out.flat.detach().cpu(),
                trace=out.trace,
            )
        finally:
            if was_training:
                self.train()

    # ── Introspection ─────────────────────────────────────────────────────

    def n_parameters(self, trainable_only: bool = True) -> int:
        """Count parameters."""
        params = self.parameters()
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def layer_shapes(
        self, patch_size: Sequence[int] = (48, 48, 48)
    ) -> List[Dict[str, Any]]:
        """Return intermediate shapes for a given patch size, without training.

        Runs one traced forward pass over a zero tensor in eval mode. Cheap
        (a single 48-cubed patch) and gives the dashboard real measured shapes
        instead of hand-computed ones that could drift from the code.
        """
        d, h, w = patch_size
        dummy = torch.zeros(1, self.n_roi, d, h, w)
        out = self.encode(dummy, trace=True)
        return [t.to_dict() for t in out.trace]

    # ── Checkpointing ─────────────────────────────────────────────────────

    def save_checkpoint(self, path: Path, extra: Optional[Dict[str, Any]] = None
                        ) -> Path:
        """Save weights together with the config needed to rebuild the module.

        Storing the config inside the checkpoint means a checkpoint can never be
        loaded into a mismatched architecture without the mismatch being
        detected.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "state_dict": self.state_dict(),
            "config": asdict(self.cfg),
            "n_roi": self.n_roi,
            "roi_order": list(ROI_ORDER),
            "module": "SpatialEncoder3D",
        }
        if extra:
            payload["extra"] = extra
        torch.save(payload, path)
        logger.info("Spatial encoder checkpoint saved: %s (%d params)",
                    path, self.n_parameters())
        return path

    @classmethod
    def load_checkpoint(cls, path: Path, map_location: str = "cpu"
                        ) -> "SpatialEncoder3D":
        """Rebuild an encoder from a checkpoint written by :meth:`save_checkpoint`.

        Raises:
            ValueError: If the checkpoint's ROI order differs from the current
                :data:`ROI_ORDER`, which would silently transpose every ROI
                embedding.
        """
        payload = torch.load(Path(path), map_location=map_location,
                             weights_only=False)
        saved_order = payload.get("roi_order")
        if saved_order and list(saved_order) != list(ROI_ORDER):
            raise ValueError(
                "Checkpoint ROI order does not match the current ROI_ORDER.\n"
                f"  checkpoint: {saved_order}\n  current   : {list(ROI_ORDER)}"
            )
        cfg = SpatialEncoderConfig(**payload["config"])
        model = cls(cfg, n_roi=payload.get("n_roi", N_ROI))
        model.load_state_dict(payload["state_dict"])
        model.eval()
        logger.info("Spatial encoder loaded: %s", path)
        return model

    def describe(self) -> str:
        """Return a human-readable architecture summary."""
        shapes = self.layer_shapes()
        lines = [
            f"SpatialEncoder3D  (shared={self.cfg.shared_encoder}, "
            f"embed_dim={self.embed_dim}, ROIs={self.n_roi})",
            f"  trainable parameters: {self.n_parameters():,}",
            "  per-patch stages:",
        ]
        for s in shapes:
            lines.append(
                f"    {s['name']:<34s} -> {tuple(s['output_shape'])!s:<20s} "
                f"params={s['n_params']:,}"
            )
        return "\n".join(lines)


def save_embeddings(
    output: EncoderOutput,
    out_dir: Path,
    subject_id: str,
) -> Dict[str, Path]:
    """Persist per-ROI embeddings as ``.npy`` plus a JSON summary.

    Args:
        output: Result of :meth:`SpatialEncoder3D.encode` for one subject
            (batch size 1).
        out_dir: Destination directory.
        subject_id: Subject/session identifier used in filenames.

    Returns:
        Mapping of logical name -> written path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emb = output.embeddings.detach().cpu().numpy()
    if emb.ndim == 3:
        emb = emb[0]

    npy_path = out_dir / f"{subject_id}_cnn_embeddings.npy"
    np.save(npy_path, emb)

    summary = {
        "subject_id": subject_id,
        "roi_order": list(ROI_ORDER),
        "shape": list(emb.shape),
        "embed_dim": int(emb.shape[-1]),
        "per_roi_statistics": {
            roi: {
                "mean": float(emb[i].mean()),
                "std": float(emb[i].std()),
                "min": float(emb[i].min()),
                "max": float(emb[i].max()),
                "l2_norm": float(np.linalg.norm(emb[i])),
            }
            for i, roi in enumerate(ROI_ORDER)
        },
        "layer_trace": [t.to_dict() for t in output.trace],
    }
    json_path = out_dir / f"{subject_id}_cnn_embeddings.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {"embeddings_npy": npy_path, "summary_json": json_path}


__all__ = [
    "LayerTrace",
    "EncoderOutput",
    "ROIPatchEncoder",
    "SpatialEncoder3D",
    "save_embeddings",
]
