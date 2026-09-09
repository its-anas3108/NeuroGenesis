"""
Typed configuration for the NeuroGenesis framework.
===================================================

All tunable parameters live here as nested dataclasses, are serialisable to and
from YAML/JSON, and are stamped into every experiment directory so that any run
can be reproduced from its saved config alone (Section 24).

Usage::

    from modules.common.config import NeuroGenesisConfig

    cfg = NeuroGenesisConfig()                      # defaults
    cfg = NeuroGenesisConfig.from_file("config.yaml")
    cfg.to_file(cfg.paths.experiment_dir / "config.yaml")

Nothing in the framework is permitted to hard-code a path, a hyper-parameter or
a threshold: if a number affects a result, it belongs in this file.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

try:  # PyYAML is optional; JSON is always available.
    import yaml

    _HAS_YAML = True
except ImportError:  # pragma: no cover
    _HAS_YAML = False


T = TypeVar("T")


# ──────────────────────────────────────────────────────────────────────────────
# Sections
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PathsConfig:
    """Filesystem layout. All paths are resolved relative to ``project_root``."""

    project_root: Path = Path(".")
    #: Directory scanned recursively for OASIS-1 T1 MRI volumes.
    mri_dir: Path = Path("dataset/OASIS")
    #: OASIS-1 cross-sectional metadata table (contains the CDR labels).
    metadata_csv: Path = Path("dataset/oasis_cross-sectional.csv")
    #: Root of the extracted real OASIS-1 dataset (Section 19). Set by
    #: configuration or --oasis1-root; never hard-coded to a local path.
    #: The official source is
    #: https://sites.wustl.edu/oasisbrains/home/oasis-1/ , but the
    #: experiment uses only locally supplied files and never downloads.
    oasis1_root: Optional[Path] = None
    #: Nilearn atlas cache.
    atlas_dir: Path = Path("dataset/nilearn_data")
    #: Root of all generated artifacts.
    outputs_dir: Path = Path("outputs")
    #: Experiment-scoped subdirectory name; resolved under ``outputs_dir``.
    experiment_id: str = "default"

    @property
    def experiment_dir(self) -> Path:
        """Directory holding this experiment's checkpoints, splits and metrics."""
        return self.outputs_dir / "experiments" / self.experiment_id


@dataclass
class DataConfig:
    """Cohort definition, label mapping and splitting."""

    #: The only dataset this project accepts for research execution.
    dataset_source: str = "OASIS-1"
    #: Which OASIS-1 volume to consume. ``t88_gfc`` is atlas-registered
    #: with the skull present, so the pipeline's own skull-stripping
    #: stage still runs; ``t88_masked_gfc`` is pre-stripped by OASIS and
    #: would make that stage a no-op.
    oasis1_volume_kind: str = "t88_gfc"
    #: Hard switch. Research runs must keep this False. When False the
    #: integrity guard refuses to start training on any sample that is
    #: not in the validated OASIS-1 index.
    allow_synthetic_data: bool = False
    #: Read voxel data during dataset validation, not just headers.
    #: Slower, but the only way to detect a truncated or all-zero
    #: volume.
    deep_validation: bool = True

    #: CDR value -> stage label. OASIS-1 encodes CDR as {0, 0.5, 1, 2}.
    #: Keys are strings because YAML/JSON cannot use floats as mapping keys.
    cdr_to_stage: Dict[str, str] = field(
        default_factory=lambda: {"0.0": "CN", "0.5": "MCI", "1.0": "AD", "2.0": "AD"}
    )
    #: How to treat sessions with a missing CDR. In OASIS-1 these are 201 young
    #: subjects who were never clinically assessed. ``"exclude"`` (default) keeps
    #: the cohort clinically meaningful; ``"cn"`` would inject a severe age
    #: confound and is provided only for explicit sensitivity analysis.
    missing_cdr_policy: str = "exclude"
    #: Optional lower age bound. Set to 60 for a strictly age-matched cohort.
    min_age: Optional[int] = None
    #: Optional upper age bound.
    max_age: Optional[int] = None
    #: Fraction of *subjects* (never sessions) held out.
    val_fraction: float = 0.15
    test_fraction: float = 0.20
    #: How evaluation partitions are drawn.
    #: ``"folds"`` (default) runs subject-wise stratified k-fold cross-validation:
    #: every subject is tested exactly once per pass, so the mean is an
    #: estimate over the whole cohort rather than over a resample of it.
    #: ``"repeats"`` keeps the older scheme of independent random draws, where
    #: a subject may be tested many times or never.
    split_scheme: str = "folds"
    #: Folds when ``split_scheme == "folds"``. 5 gives a 20% test fold and,
    #: at AD n=30, 6 held-out AD subjects per fold.
    n_folds: int = 5
    #: Number of repeated stratified splits used for Table 9 / ablation CIs
    #: when ``split_scheme == "repeats"``.
    n_repeats: int = 10


@dataclass
class PreprocessConfig:
    """MRI preprocessing chain (wraps the preserved ``preprocessing/`` package)."""

    target_shape: Tuple[int, int, int] = (128, 128, 128)
    interpolator: str = "linear"
    gaussian_sigma: float = 1.0
    aniso_iterations: int = 10
    aniso_conductance: float = 3.0
    clahe_clip_limit: float = 0.03
    morph_radius: int = 4
    min_quality_score: float = 50.0
    skip_failed_qc: bool = False
    gm_threshold: float = 0.30
    atlas_name: str = "cort-maxprob-thr25-2mm"
    patch_size: Tuple[int, int, int] = (48, 48, 48)
    context_pad: int = 4
    save_nifti: bool = True
    save_figures: bool = True
    max_subjects: Optional[int] = None


@dataclass
class SpatialEncoderConfig:
    """Lightweight 3D CNN ROI patch encoder (Section 5)."""

    #: Channel widths of the three convolutional blocks.
    channels: Tuple[int, int, int] = (16, 32, 64)
    kernel_size: int = 3
    #: Dimension of the per-ROI spatial embedding ``E_i_3D``.
    embed_dim: int = 128
    dropout: float = 0.2
    #: Share one encoder across all five ROIs (True) or learn a separate
    #: encoder per ROI (False). Sharing is the default: with only five patches
    #: per subject, per-ROI encoders overfit immediately.
    shared_encoder: bool = True


@dataclass
class NeuroPropXConfig:
    """NeuroProp-X: SRVE, AP-LAF, ANP, SAGR (Section 8)."""

    #: Initial value of the AP-LAF mixing logit ``a``, where
    #: ``alpha = sigmoid(a)``. 0.0 -> alpha = 0.5, i.e. an unbiased start that
    #: weights the anatomical prior and learned attention equally.
    alpha_logit_init: float = 0.0
    #: Whether ``alpha`` is learned. Fixing it is used by ablation A1/A3.
    learn_alpha: bool = True
    #: Hidden width of the AP-LAF learned-attention scorer.
    attention_hidden: int = 64
    #: Softmax temperature for the AP-LAF structural-covariance operand.
    #: Lower values concentrate adjacency mass on the most correlated
    #: neighbours; 0.5 keeps the row distribution informative without
    #: collapsing onto a single edge.
    structural_temperature: float = 0.5
    #: Include the optional topology term in the ANP propagation score.
    anp_use_topology: bool = True
    #: Append graph centrality features to the SAGR node representation.
    sagr_use_centrality: bool = True


@dataclass
class GraphLearningConfig:
    """SAEG-GATv2 and the GAT / GATv2 baselines (Section 9)."""

    hidden_dim: int = 64
    heads: int = 4
    n_layers: int = 2
    dropout: float = 0.2
    negative_slope: float = 0.2
    #: Graph readout: ``"mean_max"`` concatenates mean and max pooling;
    #: ``"attention"`` uses a learned attention-pooling vector.
    readout: str = "mean_max"
    #: Enable the NeuroProp-X edge gate ``g_ij``. Disabled by ablation A3.
    use_edge_gate: bool = True


@dataclass
class FusionConfig:
    """Multimodal fusion of the 3D CNN and graph branches (Section 10)."""

    hidden_dim: int = 128
    out_dim: int = 64
    dropout: float = 0.3


@dataclass
class StageTGTConfig:
    """Stage-Temporal Graph Transformer (Section 12)."""

    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    ff_dim: int = 128
    dropout: float = 0.1
    #: Include the Stage-TGT branch at all. Disabled by the optional ablation.
    enabled: bool = True


@dataclass
class LossConfig:
    """Loss weighting (Section 13)."""

    #: Weight of the prototype-consistency term ``L_proto``.
    #:
    #: Both prototype weights are large relative to the classification term
    #: because ``L_proto`` and ``L_order`` divide squared distances by
    #: ``d_model`` to stay transferable across representation widths. That
    #: normalisation shrinks the raw distance by a factor of 64 at the default
    #: width, so the small weights tried first (0.10 / 0.05) left the
    #: prototypes effectively untrained: a measured run ended with class
    #: centroids ~8.5 away from their own prototypes while the prototypes were
    #: only ~5.1 apart, and the CN < MCI < AD ordering was not respected. These
    #: values make the terms bind. Retune on validation only.
    lambda_proto: float = 1.00
    #: Weight of the ordinal stage-order term ``L_order``.
    lambda_order: float = 0.50
    #: Weight of the Stage-TGT alignment term ``L_align``.
    #:
    #: This term is an addition to the three-term loss of the research design,
    #: and it is required rather than optional. The Stage-TGT alignment head is
    #: the only consumer of the transformer output ``Z_T``; with no loss on it,
    #: the transformer would receive no gradient and every stage-alignment and
    #: stage-transition-propensity number reported by the dashboard and the
    #: report would come from randomly initialised weights. Supervising it with
    #: cross-entropy on the same stage labels is what makes the Stage-TGT branch
    #: trained rather than decorative. Set to 0.0 only to demonstrate that
    #: failure mode deliberately.
    lambda_align: float = 0.30
    #: Margin used by the ordinal stage-order loss.
    order_margin: float = 0.50
    #: Imbalance strategy: ``"class_weighted"`` | ``"none"``.
    imbalance: str = "class_weighted"
    label_smoothing: float = 0.0


@dataclass
class AugmentationConfig:
    """Image augmentation of REAL OASIS-1 volumes only (Section 14).

    Augmentation transforms real scans; it never creates subjects and
    never creates labels. A transformed volume inherits the label of the
    real subject it came from and is counted as that subject for
    splitting, so an augmented copy can never appear in a different
    split from its source.

    Defaults are conservative and augmentation is **off** unless enabled
    explicitly, because anatomically unrealistic transforms would be a
    silent confound.
    """

    enabled: bool = False
    #: Maximum rotation in degrees about each axis.
    max_rotation_degrees: float = 5.0
    #: Maximum translation in voxels along each axis.
    max_translation_voxels: float = 3.0
    #: Multiplicative intensity scaling range, as a fraction.
    intensity_scale: float = 0.05
    #: Probability that any given training sample is augmented.
    probability: float = 0.5


@dataclass
class TrainConfig:
    """Optimisation and checkpointing."""

    epochs: int = 120
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    #: Metric used for model selection on the validation split. Balanced
    #: accuracy is the default because AD is a 30-session minority class and
    #: plain accuracy would select a CN-biased model.
    monitor: str = "balanced_accuracy"
    early_stopping_patience: int = 25
    #: When two epochs tie on the monitored metric, prefer the one with the
    #: better-trained stage geometry (lower validation ``L_order``).
    #:
    #: This exists because the two objectives converge at very different rates.
    #: Classification can saturate the monitored metric within a handful of
    #: epochs while the prototype geometry is still being shaped -- a measured
    #: run selected epoch 4 on a perfect validation balanced accuracy, at which
    #: point ``L_order`` was still 1.88 and the CN < MCI < AD prototype ordering
    #: was violated, leaving the propensity scores barely separated. Because the
    #: tie-break only applies when the monitored metric is *equal* (within
    #: ``tie_break_tolerance``), it never trades classification performance for
    #: geometry.
    tie_break_on_geometry: bool = True
    #: Absolute tolerance within which two monitored values count as tied.
    tie_break_tolerance: float = 1e-6
    grad_clip: float = 1.0
    device: str = "auto"
    num_workers: int = 0


@dataclass
class RankingConfig:
    """Unified ROI importance score (Section 14)."""

    #: Weight on NeuroProp-X regional vulnerability ``RV_i``.
    lambda_vulnerability: float = 0.40
    #: Weight on SAEG-GATv2 node attention.
    lambda_attention: float = 0.30
    #: Weight on SHAP feature contribution aggregated per ROI.
    lambda_shap: float = 0.30


@dataclass
class StatsConfig:
    """Stage-wise statistical analysis (Section 16)."""

    alpha: float = 0.05
    #: Multiple-comparison correction across all ROI x feature x contrast tests.
    fdr_method: str = "benjamini_hochberg"
    #: Normality screen used to choose between a parametric and rank-based test.
    normality_test: str = "shapiro"
    normality_alpha: float = 0.05
    #: Minimum group size below which no test is attempted and the cell is
    #: reported as insufficient data rather than as a p-value.
    min_group_n: int = 5


@dataclass
class ReproConfig:
    """Reproducibility controls (Section 24)."""

    seed: int = 42
    deterministic: bool = True
    #: Disable cuDNN autotuning; required for bit-reproducible GPU runs.
    cudnn_benchmark: bool = False


@dataclass
class NeuroGenesisConfig:
    """Root configuration object."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    spatial_encoder: SpatialEncoderConfig = field(default_factory=SpatialEncoderConfig)
    neuropropx: NeuroPropXConfig = field(default_factory=NeuroPropXConfig)
    graph_learning: GraphLearningConfig = field(default_factory=GraphLearningConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    stage_tgt: StageTGTConfig = field(default_factory=StageTGTConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    augmentation: AugmentationConfig = field(
        default_factory=AugmentationConfig
    )
    ranking: RankingConfig = field(default_factory=RankingConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    repro: ReproConfig = field(default_factory=ReproConfig)

    # ── Serialisation ─────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain, JSON/YAML-safe nested dictionary."""
        return _to_plain(self)

    def to_file(self, path: Path) -> Path:
        """Write the config to ``path``; format inferred from the suffix."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        if path.suffix in (".yaml", ".yml"):
            if not _HAS_YAML:
                raise RuntimeError(
                    "PyYAML is not installed — write a .json config instead."
                )
            path.write_text(
                yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
        else:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "NeuroGenesisConfig":
        """Build a config from a nested dictionary, validating section names.

        Unknown keys raise rather than being silently dropped: a typo in a
        config file must not quietly leave a default in place.
        """
        return _from_plain(cls, payload, path="config")

    @classmethod
    def from_file(cls, path: Path) -> "NeuroGenesisConfig":
        """Load a config from a ``.yaml``, ``.yml`` or ``.json`` file."""
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix in (".yaml", ".yml"):
            if not _HAS_YAML:
                raise RuntimeError("PyYAML is not installed — cannot read YAML.")
            payload = yaml.safe_load(text)
        else:
            payload = json.loads(text)
        return cls.from_dict(payload or {})

    # ── Validation ────────────────────────────────────────────────────────

    def validate(self) -> List[str]:
        """Check internal consistency.

        Returns:
            A list of human-readable problems. Empty means the config is
            self-consistent. Callers decide whether to warn or abort.
        """
        problems: List[str] = []

        if self.data.dataset_source != "OASIS-1":
            problems.append(
                "data.dataset_source must be 'OASIS-1'; this project "
                f"accepts no other dataset, got "
                f"{self.data.dataset_source!r}"
            )
        if self.data.allow_synthetic_data:
            problems.append(
                "data.allow_synthetic_data is True. Research execution "
                "requires it to be False; synthetic data must never "
                "enter the experiment."
            )
        if self.data.oasis1_volume_kind not in ("t88_gfc", "t88_masked_gfc"):
            problems.append(
                "data.oasis1_volume_kind must be 't88_gfc' or "
                f"'t88_masked_gfc', got "
                f"{self.data.oasis1_volume_kind!r}"
            )
        if self.data.split_scheme not in ("folds", "repeats"):
            problems.append(
                "data.split_scheme must be 'folds' or 'repeats', got "
                f"{self.data.split_scheme!r}"
            )
        if self.data.split_scheme == "folds" and self.data.n_folds < 2:
            problems.append(
                f"data.n_folds must be at least 2, got {self.data.n_folds}"
            )
        if self.data.missing_cdr_policy not in ("exclude", "cn"):
            problems.append(
                f"data.missing_cdr_policy must be 'exclude' or 'cn', "
                f"got {self.data.missing_cdr_policy!r}"
            )
        total_held_out = self.data.val_fraction + self.data.test_fraction
        if not 0.0 < total_held_out < 1.0:
            problems.append(
                f"data.val_fraction + data.test_fraction must lie in (0, 1), "
                f"got {total_held_out}"
            )
        if self.graph_learning.readout not in ("mean_max", "attention"):
            problems.append(
                "graph_learning.readout must be 'mean_max' or 'attention', "
                f"got {self.graph_learning.readout!r}"
            )
        if self.graph_learning.hidden_dim % self.graph_learning.heads != 0:
            problems.append(
                f"graph_learning.hidden_dim ({self.graph_learning.hidden_dim}) must be "
                f"divisible by heads ({self.graph_learning.heads})"
            )
        if self.stage_tgt.d_model % self.stage_tgt.n_heads != 0:
            problems.append(
                f"stage_tgt.d_model ({self.stage_tgt.d_model}) must be divisible by "
                f"n_heads ({self.stage_tgt.n_heads})"
            )
        if self.loss.imbalance not in ("class_weighted", "none"):
            problems.append(
                "loss.imbalance must be 'class_weighted' or 'none', "
                f"got {self.loss.imbalance!r}"
            )
        weight_sum = (
            self.ranking.lambda_vulnerability
            + self.ranking.lambda_attention
            + self.ranking.lambda_shap
        )
        if abs(weight_sum - 1.0) > 1e-6:
            problems.append(
                f"ranking lambdas must sum to 1.0, got {weight_sum:.6f}"
            )
        stages = set(self.data.cdr_to_stage.values())
        if not stages.issubset({"CN", "MCI", "AD"}):
            problems.append(
                f"data.cdr_to_stage maps to unknown stages: {sorted(stages - {'CN','MCI','AD'})}"
            )
        return problems


# ──────────────────────────────────────────────────────────────────────────────
# Nested dataclass <-> plain dict conversion
# ──────────────────────────────────────────────────────────────────────────────

def _to_plain(obj: Any) -> Any:
    """Recursively convert dataclasses, Paths and tuples to JSON-safe values."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, tuple):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, list):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    return obj


def _from_plain(cls: Type[T], payload: Any, path: str) -> T:
    """Rebuild a (possibly nested) dataclass from a plain dictionary."""
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected a mapping, got {type(payload).__name__}")

    known = {f.name: f for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(payload) - set(known)
    if unknown:
        raise ValueError(
            f"{path}: unknown key(s) {sorted(unknown)}. "
            f"Valid keys: {sorted(known)}"
        )

    kwargs: Dict[str, Any] = {}
    for name, f in known.items():
        if name not in payload:
            continue
        raw = payload[name]
        kwargs[name] = _coerce(f.type, raw, f"{path}.{name}")
    return cls(**kwargs)  # type: ignore[call-arg]


def _coerce(annotation: Any, raw: Any, path: str) -> Any:
    """Coerce a raw config value to the annotated type where it matters.

    Only the conversions that actually bite in practice are handled: nested
    dataclasses, ``Path``, and tuple-typed shapes (YAML always yields lists,
    but ``(128, 128, 128)`` must stay a tuple so it can be hashed and compared).
    """
    # Nested dataclasses arrive as strings under `from __future__ import
    # annotations`, so resolve against this module's namespace.
    if isinstance(annotation, str):
        annotation = _SECTION_TYPES.get(annotation, annotation)

    if is_dataclass(annotation) and isinstance(annotation, type):
        return _from_plain(annotation, raw, path)

    text = annotation if isinstance(annotation, str) else str(annotation)

    if "Path" in text and raw is not None:
        return Path(raw)
    if "Tuple" in text and isinstance(raw, list):
        return tuple(raw)
    return raw


#: Name -> type map so string annotations can be resolved without ``typing
#: .get_type_hints``, which would fail on ``Optional`` forward refs under some
#: Python versions.
_SECTION_TYPES: Dict[str, Any] = {
    "PathsConfig": PathsConfig,
    "DataConfig": DataConfig,
    "PreprocessConfig": PreprocessConfig,
    "SpatialEncoderConfig": SpatialEncoderConfig,
    "NeuroPropXConfig": NeuroPropXConfig,
    "GraphLearningConfig": GraphLearningConfig,
    "FusionConfig": FusionConfig,
    "StageTGTConfig": StageTGTConfig,
    "LossConfig": LossConfig,
    "TrainConfig": TrainConfig,
    "RankingConfig": RankingConfig,
    "StatsConfig": StatsConfig,
    "AugmentationConfig": AugmentationConfig,
    "ReproConfig": ReproConfig,
}


def default_config() -> NeuroGenesisConfig:
    """Return a fresh default configuration."""
    return NeuroGenesisConfig()


__all__ = [
    "NeuroGenesisConfig",
    "PathsConfig",
    "DataConfig",
    "PreprocessConfig",
    "SpatialEncoderConfig",
    "NeuroPropXConfig",
    "GraphLearningConfig",
    "FusionConfig",
    "StageTGTConfig",
    "LossConfig",
    "TrainConfig",
    "AugmentationConfig",
    "RankingConfig",
    "StatsConfig",
    "ReproConfig",
    "default_config",
]
