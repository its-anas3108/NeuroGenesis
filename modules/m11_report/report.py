"""
M19 — Subject report generation (Section 20).
=============================================

Assembles a per-subject report from real computed values, in Markdown, HTML and
JSON. The fifteen sections of Section 20 are all present, and a section whose
inputs are absent renders as an explicit "not available" block rather than being
silently dropped.

Language discipline (Sections 32, 33)
-------------------------------------

Every quantity is described as model-derived. The report never states a
diagnosis, never predicts future degeneration, and never recommends treatment.
:func:`validate_language` scans the assembled text for prohibited phrasing and
raises before anything is written, so a future edit cannot reintroduce
"guaranteed conversion" or "will develop AD" into a generated document.

Provenance
----------

If the outputs tree is marked as synthetic smoke-test data, the report carries a
prominent banner and every metric section is prefixed accordingly. A report that
could be mistaken for a clinical document about a real patient is the single
worst failure mode available to this codebase, so provenance is stated in the
title, the header and the footer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from modules.common.logging_utils import get_logger
from modules.common.roi_constants import (
    ROI_METADATA,
    ROI_ORDER,
    STAGE_DESCRIPTION,
    STAGE_ORDER,
    roi_short,
)
from modules.common.serialization import json_safe

logger = get_logger(__name__)

#: Phrasings that must never appear in a generated report.
PROHIBITED_PHRASES: List[str] = [
    "will develop",
    "will definitely",
    "guaranteed",
    "confirmed diagnosis",
    "diagnosed with",
    "certain to",
    "definitely degenerate",
    "prognosis is",
    "we recommend treatment",
    "should be prescribed",
    "treatment recommendation",
    "conversion probability",
    "predicted conversion",
    "24-month",
    "future atrophy",
    "will progress",
]

#: The mandatory closing disclaimer.
DISCLAIMER = (
    "These results are model-derived associations and stage-propensity "
    "estimates produced by a research system from a single cross-sectional "
    "structural MRI scan. They are not a confirmed clinical diagnosis, not a "
    "validated prediction of future disease conversion, and not a basis for "
    "any clinical decision. Regional vulnerability, propagation scores and "
    "stage-transition propensity describe how informative each region and "
    "pathway is for the model's current discrimination between CN, MCI and AD; "
    "they are not measurements of biological degeneration or disease spread. "
    "This system has not been clinically validated."
)


#: Cues that turn a prohibited phrase into a legitimate denial of it. A
#: disclaimer must be able to say "this is not a validated conversion
#: probability" without tripping the guard that exists to stop the report from
#: *asserting* one.
NEGATION_CUES: List[str] = [
    "not ", "never ", "no ", "cannot ", "can not ", "does not ", "do not ",
    "is not", "are not", "isn't", "aren't", "without ", "rather than ",
    "instead of ", "must not ", "should not ", "forbid", "avoid ",
]

#: How many characters before a match are inspected for a negation cue. Long
#: enough to cover "this is not a clinically validated conversion probability",
#: short enough that a negation in a previous sentence does not license an
#: assertion in this one.
NEGATION_WINDOW = 60


def _is_negated(lowered: str, position: int) -> bool:
    """Return whether the match at ``position`` sits inside a negation."""
    start = max(0, position - NEGATION_WINDOW)
    window = lowered[start:position]
    # A sentence boundary ends the negation's scope.
    for boundary in (". ", "! ", "? ", "\n\n"):
        cut = window.rfind(boundary)
        if cut != -1:
            window = window[cut + len(boundary):]
    return any(cue in window for cue in NEGATION_CUES)


def _provenance_label(marker: Optional[Dict[str, Any]]) -> str:
    """Return a short human label for the data provenance."""
    if not marker:
        return "OASIS-1 (real)"
    return {
        "synthetic_mri": "SYNTHETIC phantom MRI",
        "synthetic_patches": "SYNTHETIC ROI patches (smoke test)",
    }.get(marker.get("kind", ""), "SYNTHETIC")


def validate_language(text: str) -> List[str]:
    """Scan report text for prohibited clinical phrasing.

    A prohibited phrase is permitted when it appears inside a negation, so that
    the mandatory disclaimers can name what the system does *not* claim. Every
    other occurrence is a violation.

    Args:
        text: The assembled report body.

    Returns:
        A list of ``"phrase (context)"`` strings for offending occurrences.
        Empty means the text is compliant.
    """
    lowered = text.lower()
    violations: List[str] = []
    for phrase in PROHIBITED_PHRASES:
        start = 0
        while True:
            position = lowered.find(phrase, start)
            if position == -1:
                break
            if not _is_negated(lowered, position):
                context = text[max(0, position - 45):position + len(phrase) + 25]
                violations.append(
                    f"{phrase!r} in: ...{context.strip()}..."
                )
                break
            start = position + len(phrase)
    return violations


@dataclass
class ReportInputs:
    """Everything a subject report can draw on. Any field may be ``None``."""

    subject_id: str
    #: Predicted stage and class probabilities.
    current_stage: Optional[str] = None
    class_probabilities: Optional[Dict[str, float]] = None
    confidence: Optional[Dict[str, Any]] = None
    #: Ground-truth stage, when known (evaluation context, not inference).
    true_stage: Optional[str] = None
    #: NeuroProp-X explanation.
    regional_vulnerability: Optional[Dict[str, float]] = None
    vulnerability_ranking: Optional[List[Dict[str, Any]]] = None
    adaptive_adjacency: Optional[np.ndarray] = None
    propagation_matrix: Optional[np.ndarray] = None
    alpha: Optional[float] = None
    top_pathways: Optional[List[Dict[str, Any]]] = None
    #: Stage-TGT propensity report.
    propensity: Optional[Dict[str, Any]] = None
    stage_geometry: Optional[Dict[str, Any]] = None
    #: ROI ranking.
    roi_ranking: Optional[List[Dict[str, Any]]] = None
    ranking_weights: Optional[Dict[str, float]] = None
    #: Attribution.
    attribution_method: Optional[str] = None
    top_features: Optional[List[Dict[str, Any]]] = None
    attribution_notes: Optional[List[str]] = None
    #: Attention explanation.
    node_importance: Optional[Dict[str, float]] = None
    top_edges: Optional[List[Dict[str, Any]]] = None
    #: Cohort-level context.
    statistics_summary: Optional[Dict[str, Any]] = None
    significant_findings: Optional[List[Dict[str, Any]]] = None
    ablation_table: Optional[pd.DataFrame] = None
    ablation_caveats: Optional[List[str]] = None
    #: Dataset provenance (Sections 20, 24). Every report must state the
    #: dataset it was produced from.
    dataset_provenance: Optional[Dict[str, Any]] = None
    #: Model provenance.
    model_summary: Optional[Dict[str, Any]] = None
    config_snapshot: Optional[Dict[str, Any]] = None
    experiment_id: Optional[str] = None
    checkpoint_path: Optional[str] = None
    #: Figures to embed, name -> path.
    figures: Dict[str, Path] = field(default_factory=dict)
    #: Smoke-test marker; when present the report is stamped as synthetic.
    smoke_marker: Optional[Dict[str, Any]] = None
    #: Extra limitations to append.
    limitations: List[str] = field(default_factory=list)


class ReportGenerator:
    """Render a subject report in Markdown, HTML and JSON.

    Args:
        out_dir: Destination directory.
    """

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = Path(out_dir)

    # ── Section builders ──────────────────────────────────────────────────

    @staticmethod
    def _fmt(value: Optional[float], digits: int = 4) -> str:
        """Format a float, or ``n/a`` when absent."""
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return "n/a"
        return f"{value:.{digits}f}"

    def _header(self, data: ReportInputs) -> List[str]:
        """Title, provenance banner and identification."""
        lines: List[str] = []
        if data.smoke_marker:
            # The wording must match which generator produced the data. The
            # phantom-MRI case needs naming precisely: its artifacts come out of
            # the real imaging pipeline, so calling them "fabricated patches"
            # would understate how convincingly real they look.
            kind = data.smoke_marker.get("kind", "synthetic_patches")
            banner = data.smoke_marker.get(
                "banner", "SYNTHETIC DATA - NOT A RESEARCH RESULT"
            )
            if kind == "synthetic_mri":
                detail = (
                    "> **This report was generated from synthetic phantom MRI, "
                    "not from a real scan.** The volumes behind every number "
                    "below are parametric ellipsoids produced by "
                    "`tools/make_synthetic_mri.py`. They were processed by the "
                    "real preprocessing, atlas-registration and "
                    "feature-extraction chain, so these artifacts look exactly "
                    "like those of a genuine run, which is precisely why this "
                    "banner exists. Nothing in this document describes a real "
                    "person or a real finding."
                )
            else:
                detail = (
                    "> **This report was generated from synthetic artifacts, "
                    "not from real MRI.** The ROI patches behind every number "
                    "below are parametric blobs produced by "
                    "`tools/make_smoke_artifacts.py`; the imaging pipeline was "
                    "skipped entirely. Nothing in this document describes a "
                    "real person or a real finding."
                )
            lines += [f"> # {banner}", "> ", detail, ""]
        lines += [
            "# NeuroGenesis subject report",
            "",
            "Stage-aware speech-network analysis of structural MRI: CN / MCI / "
            "AD classification, regional vulnerability, and stage-transition "
            "propensity.",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Subject ID | `{data.subject_id}` |",
            f"| Generated | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |",
            f"| Experiment | `{data.experiment_id or 'not recorded'}` |",
            f"| Dataset | "
            f"{(data.dataset_provenance or {}).get('dataset_source', 'OASIS-1')} |",
            f"| Dataset source | "
            f"{(data.dataset_provenance or {}).get('source_description', 'Washington University / OASIS')} |",
            f"| Data provenance | {_provenance_label(data.smoke_marker)} |",
            "",
        ]
        return lines

    def _section_prediction(self, data: ReportInputs) -> List[str]:
        """Sections 2-4: stage prediction, probabilities and confidence."""
        lines = ["## 1. Current stage classification", ""]
        if not data.current_stage or not data.class_probabilities:
            lines += [
                "_Not available — requires a trained model._", "",
            ]
            return lines

        lines += [
            f"**Model-assigned stage: {data.current_stage}** "
            f"({STAGE_DESCRIPTION.get(data.current_stage, '')})",
            "",
            "| Stage | Probability |",
            "|---|---|",
        ]
        for stage in STAGE_ORDER:
            probability = data.class_probabilities.get(stage)
            marker = "  **<-- assigned**" if stage == data.current_stage else ""
            lines.append(f"| {stage} | {self._fmt(probability)}{marker} |")
        lines.append("")

        if data.true_stage:
            agreement = "agrees with" if data.true_stage == data.current_stage \
                else "differs from"
            lines += [
                f"Reference CDR-derived stage for this session: "
                f"**{data.true_stage}**. The model's assignment {agreement} it.",
                "",
            ]

        if data.confidence:
            lines += [
                "### Model confidence",
                "",
                f"- Top-class probability: "
                f"{self._fmt(data.confidence.get('top_probability'))}",
                f"- Margin over runner-up: "
                f"{self._fmt(data.confidence.get('margin'))}",
                f"- Normalised predictive entropy: "
                f"{self._fmt(data.confidence.get('normalized_entropy'))} "
                "(0 = fully confident, 1 = uniform)",
                "",
                f"_{data.confidence.get('note', '')}_",
                "",
            ]
        return lines

    def _section_vulnerability(self, data: ReportInputs) -> List[str]:
        """Section 5: NeuroProp-X regional vulnerability."""
        lines = ["## 2. NeuroProp-X regional vulnerability", ""]
        if not data.vulnerability_ranking:
            lines += ["_Not available — this model variant has no NeuroProp-X._",
                      ""]
            return lines

        lines += [
            "Model-derived vulnerability `RV_i = sigmoid(w_v^T xhat_i + b_v)`, "
            "measuring how informative each speech-related region is for the "
            "current CN/MCI/AD discrimination.",
            "",
            "| Rank | Region | Vulnerability | Function |",
            "|---|---|---|---|",
        ]
        for entry in data.vulnerability_ranking:
            roi = entry["roi"]
            lines.append(
                f"| {entry['rank']} | {roi_short(roi)} | "
                f"{self._fmt(entry['vulnerability'])} | "
                f"{ROI_METADATA.get(roi, {}).get('function', '')} |"
            )
        lines += [
            "",
            "> Vulnerability is a model-derived discriminative score. It is not "
            "a probability that the region will degenerate.",
            "",
        ]
        if data.alpha is not None:
            lines += [
                f"AP-LAF anatomical-prior weight `alpha` = "
                f"{self._fmt(data.alpha)} — the share of the adaptive "
                "adjacency's mass contributed by the anatomical prior, with the "
                "remainder from learned attention.",
                "",
            ]
        return lines

    def _section_pathways(self, data: ReportInputs) -> List[str]:
        """Sections 7-8: adaptive graph and informative pathways."""
        lines = ["## 3. Inter-regional pathways", ""]
        if not data.top_pathways and not data.top_edges:
            lines += ["_Not available — requires a graph encoder._", ""]
            return lines

        if data.top_edges:
            lines += [
                "### Most attended pathways (SAEG-GATv2 gated attention)",
                "",
                "| Pathway | Attention | Edge gate | Anatomical tract |",
                "|---|---|---|---|",
            ]
            for edge in data.top_edges[:5]:
                lines.append(
                    f"| {edge.get('pathway', '')} | "
                    f"{self._fmt(edge.get('attention'))} | "
                    f"{self._fmt(edge.get('gate'))} | "
                    f"{edge.get('anatomical_tract') or 'no prior edge'} |"
                )
            lines.append("")

        if data.top_pathways:
            lines += [
                "### Highest propagation-representation pathways (NeuroProp-X ANP)",
                "",
                "| Pathway | Propagation score |",
                "|---|---|",
            ]
            for path in data.top_pathways[:5]:
                lines.append(
                    f"| {path['source_short']} -> {path['target_short']} | "
                    f"{self._fmt(path['propagation_score'])} |"
                )
            lines += [
                "",
                "> Propagation scores are a learned edge-level representation "
                "combining endpoint vulnerability with adaptive connectivity. "
                "They are not biological disease-spread probabilities.",
                "",
            ]
        return lines

    def _section_propensity(self, data: ReportInputs) -> List[str]:
        """Section 9: stage-transition propensity."""
        lines = ["## 4. Stage-transition propensity", ""]
        if not data.propensity:
            lines += [
                "_Not available — this model variant has no Stage-TGT branch._",
                "",
            ]
            return lines

        propensity = data.propensity
        transition = propensity.get("stage_transition_propensity")
        lines += [
            "| Quantity | Value |",
            "|---|---|",
            f"| Reference stage | {propensity.get('reference_stage', 'n/a')} |",
            f"| Advanced-stage alignment | "
            f"{self._fmt(propensity.get('advanced_stage_alignment'))} |",
            f"| AD-associated propensity | "
            f"{self._fmt(propensity.get('ad_associated_propensity'))} |",
            f"| Stage-transition propensity | "
            f"{self._fmt(transition) if transition is not None else 'undefined'} |",
            f"| Transition target | "
            f"{propensity.get('transition_target_stage') or 'none'} |",
            "",
            f"_{propensity.get('transition_note', '')}_",
            "",
        ]
        alignment = propensity.get("stage_alignment") or {}
        if alignment:
            lines += ["Stage alignment distribution:", ""]
            for stage in STAGE_ORDER:
                lines.append(
                    f"- {stage}: {self._fmt(alignment.get(stage))}"
                )
            lines.append("")

        if data.stage_geometry is not None:
            ordered = data.stage_geometry.get("ordering_respected")
            lines += [
                "### Stage geometry quality",
                "",
                f"- CN < MCI < AD prototype ordering respected: **{ordered}**",
            ]
            if ordered is False:
                lines.append(
                    "- The learned prototypes do not respect the stage "
                    "ordering at this checkpoint, so the propensity values "
                    "above reflect under-trained geometry and should not be "
                    "interpreted as a finding."
                )
            lines.append("")

        lines += [
            "> " + (propensity.get("disclaimer") or ""),
            "",
        ]
        return lines

    def _section_ranking(self, data: ReportInputs) -> List[str]:
        """Section 6: unified ROI ranking."""
        lines = ["## 5. Unified ROI ranking", ""]
        if not data.roi_ranking:
            lines += ["_Not available — requires ROI importance signals._", ""]
            return lines

        if data.ranking_weights:
            weights = ", ".join(
                f"{k} = {v:.2f}" for k, v in sorted(data.ranking_weights.items())
            )
            lines += [f"Signal weights actually applied: {weights}.", ""]

        columns = [
            c for c in ("vulnerability_normalized", "attention_normalized",
                        "attribution_normalized", "cnn_occlusion_normalized")
            if any(c in entry for entry in data.roi_ranking)
        ]
        header = "| Rank | Region | Combined score |" + "".join(
            f" {c.replace('_normalized', '')} |" for c in columns
        )
        lines += [header, "|---|---|---|" + "---|" * len(columns)]
        for entry in data.roi_ranking:
            row = (f"| {entry['rank']} | {entry.get('roi_short', '')} | "
                   f"{self._fmt(entry.get('score'))} |")
            for column in columns:
                row += f" {self._fmt(entry.get(column))} |"
            lines.append(row)
        lines.append("")
        return lines

    def _section_xai(self, data: ReportInputs) -> List[str]:
        """Sections 10-11: attribution and attention explanation."""
        lines = ["## 6. Explainable AI", ""]
        if not data.top_features and not data.node_importance:
            lines += ["_Not available — requires a trained model._", ""]
            return lines

        if data.top_features:
            method = data.attribution_method or "unknown"
            label = "SHAP (Kernel)" if method == "shap_kernel" \
                else "permutation (single-feature replacement)"
            lines += [
                f"### Feature attribution — method: **{label}**",
                "",
                "| Region | Feature | Attribution | Direction |",
                "|---|---|---|---|",
            ]
            for entry in data.top_features[:10]:
                lines.append(
                    f"| {entry.get('roi_short', '')} | {entry.get('feature', '')} "
                    f"| {self._fmt(entry.get('value'))} | "
                    f"{entry.get('direction', '')} |"
                )
            lines.append("")
            if method != "shap_kernel":
                lines += [
                    "> These are **not** Shapley values. The `shap` package was "
                    "unavailable or failed, so exact single-feature replacement "
                    "attribution was used. It measures the marginal effect of "
                    "replacing one feature with its training-split reference "
                    "values and does not decompose additively.",
                    "",
                ]
            for note in (data.attribution_notes or []):
                lines += [f"- {note}", ""]

        if data.node_importance:
            lines += [
                "### Node importance (incoming attention mass)",
                "",
                "| Region | Incoming attention |",
                "|---|---|",
            ]
            for roi, value in sorted(data.node_importance.items(),
                                     key=lambda kv: kv[1], reverse=True):
                lines.append(f"| {roi_short(roi)} | {self._fmt(value)} |")
            lines.append("")
        return lines

    def _section_statistics(self, data: ReportInputs) -> List[str]:
        """Section 12: cohort statistical context."""
        lines = ["## 7. Stage-wise statistical context (cohort level)", ""]
        if not data.statistics_summary:
            lines += ["_Not available — statistics have not been run._", ""]
            return lines

        summary = data.statistics_summary
        lines += [
            f"- Group sizes: {summary.get('group_sizes')}",
            f"- Tests run: {summary.get('n_tests_run')} "
            f"(skipped: {summary.get('n_tests_skipped')})",
            f"- FDR-significant: {summary.get('n_significant_fdr')}",
            f"- Multiple-comparison correction: {summary.get('fdr_method')}, "
            "applied across all ROI x feature x contrast tests as one family",
            "",
        ]
        if data.significant_findings:
            lines += [
                "Largest FDR-significant stage differences:",
                "",
                "| Region | Feature | Contrast | FDR p | Effect size | Trend |",
                "|---|---|---|---|---|---|",
            ]
            for finding in data.significant_findings[:8]:
                lines.append(
                    f"| {finding.get('roi_short')} | {finding.get('feature')} | "
                    f"{finding.get('contrast')} | "
                    f"{finding.get('p_adjusted'):.2e} | "
                    f"{self._fmt(finding.get('effect_size'), 3)} "
                    f"({finding.get('effect_size_name')}) | "
                    f"{finding.get('trend')} |"
                )
            lines.append("")
        for note in (summary.get("notes") or []):
            lines += [f"> {note}", ""]
        return lines

    def _section_ablation(self, data: ReportInputs) -> List[str]:
        """Section 13: ablation results."""
        lines = ["## 8. Ablation results (cohort level)", ""]
        if data.ablation_table is None or data.ablation_table.empty:
            lines += ["_Not available — the ablation study has not been run._",
                      ""]
            return lines
        lines += [data.ablation_table.to_markdown(index=False), ""]
        for caveat in (data.ablation_caveats or []):
            lines += [f"> {caveat}", ""]
        return lines

    def _section_model(self, data: ReportInputs) -> List[str]:
        """Section 14: model version and configuration."""
        lines = ["## 9. Model version and configuration", ""]
        if not data.model_summary:
            lines += ["_Not recorded._", ""]
            return lines

        summary = data.model_summary
        spec = summary.get("spec", {})
        dimensions = summary.get("dimensions", {})
        lines += [
            "| Field | Value |",
            "|---|---|",
            f"| Variant | `{spec.get('name', 'n/a')}` — "
            f"{spec.get('description', '')} |",
            f"| Graph encoder | {spec.get('graph_encoder', 'n/a')} |",
            f"| NeuroProp-X | {spec.get('use_neuropropx', 'n/a')} "
            f"(SRVE={spec.get('use_srve')}, "
            f"AP-LAF attention={spec.get('use_learned_attention')}, "
            f"ANP={spec.get('use_anp')}) |",
            f"| Edge gate | {spec.get('use_edge_gate', 'n/a')} |",
            f"| 3D CNN branch | {spec.get('use_cnn', 'n/a')} |",
            f"| Stage-TGT | {spec.get('use_stage_tgt', 'n/a')} |",
            f"| Trainable parameters | {summary.get('n_parameters', 'n/a'):,} |"
            if isinstance(summary.get("n_parameters"), int) else
            "| Trainable parameters | n/a |",
            f"| Node feature width | {dimensions.get('node_dim', 'n/a')} |",
            f"| Shared representation width | {dimensions.get('z_h_dim', 'n/a')} |",
            f"| Checkpoint | `{data.checkpoint_path or 'not recorded'}` |",
            "",
        ]
        components = summary.get("component_parameters") or {}
        if components:
            lines += ["Parameters by component:", ""]
            for name, count in components.items():
                lines.append(f"- {name}: {count:,}")
            lines.append("")
        return lines

    def _section_dataset(self, data: ReportInputs) -> List[str]:
        """State the dataset this result came from (Sections 20, 24)."""
        lines = ["## 10. Dataset", ""]
        provenance = data.dataset_provenance or {}
        if not provenance:
            lines += [
                "_Dataset provenance was not recorded for this run._", "",
            ]
            return lines
        lines += [
            "| Field | Value |",
            "|---|---|",
            f"| Dataset | **{provenance.get('dataset_source', 'OASIS-1')}** |",
            f"| Source | {provenance.get('source_description', '')} |",
            f"| Volume | {provenance.get('volume_description', '')} |",
            f"| Synthetic data | {'YES' if provenance.get('is_synthetic') else 'DISABLED'} |",
            "",
        ]
        note = provenance.get("preprocessing_note")
        if note:
            lines += [f"> {note}", ""]
        return lines

    def _section_limitations(self, data: ReportInputs) -> List[str]:
        """Section 15: limitations and disclaimer."""
        lines = ["## 10. Limitations", ""]
        limitations = [
            "The dataset is **OASIS-1 cross-sectional** (Washington "
            "University / OASIS). Findings apply to that cohort and have not "
            "been replicated on any independent dataset.",
            "The study is **cross-sectional**. Every subject contributes one "
            "scan, so no within-subject change is measured and no statement "
            "about the future can be supported.",
            "The MCI class is derived from CDR = 0.5, documented in OASIS-1 as "
            "'very mild dementia'. It is a conventional cross-sectional stand-in "
            "for MCI, not an independent clinical MCI diagnosis.",
            "The labelled OASIS-1 cohort contains 30 AD sessions. Point "
            "estimates from a single split are unreliable at this size, which is "
            "why every comparison is reported over repeated splits with "
            "confidence intervals.",
            "The anatomical prior uses expert-assigned ordinal tract weights, "
            "not measured diffusion tractography. Its contribution is tested "
            "empirically by ablations A1 and A3 rather than assumed.",
            "Grey-matter volume is an intensity-threshold proxy, and cortical "
            "thickness is a volume-to-surface proxy. Neither is a tissue-class "
            "segmentation or a FreeSurfer measurement.",
            "Confidence indicators come from softmax outputs and are not "
            "calibrated probabilities of correctness.",
            "The system has not been clinically validated and is not a medical "
            "device.",
        ]
        limitations += list(data.limitations)
        if data.smoke_marker:
            limitations.insert(
                0,
                "**This particular report was generated from "
                + ("synthetic phantom MRI processed by the real imaging "
                   "pipeline"
                   if data.smoke_marker.get("kind") == "synthetic_mri"
                   else "fabricated ROI patches, with imaging skipped")
                + ".** Nothing in it describes a real subject, and no number "
                "in it is a research result.",
            )
        for item in limitations:
            lines.append(f"- {item}")
        lines += ["", "## Disclaimer", "", f"> {DISCLAIMER}", ""]
        return lines

    # ── Assembly ──────────────────────────────────────────────────────────

    def build_markdown(self, data: ReportInputs) -> str:
        """Assemble the full Markdown report.

        Raises:
            ValueError: If the assembled text contains prohibited clinical
                phrasing.
        """
        blocks: List[str] = []
        blocks += self._header(data)
        blocks += self._section_prediction(data)
        blocks += self._section_vulnerability(data)
        blocks += self._section_pathways(data)
        blocks += self._section_propensity(data)
        blocks += self._section_ranking(data)
        blocks += self._section_xai(data)
        blocks += self._section_statistics(data)
        blocks += self._section_ablation(data)
        blocks += self._section_model(data)
        blocks += self._section_dataset(data)

        if data.figures:
            blocks += ["## 11. Figures", ""]
            for name, path in data.figures.items():
                blocks.append(f"- **{name}**: `{Path(path).as_posix()}`")
            blocks.append("")

        blocks += self._section_limitations(data)

        text = "\n".join(blocks)
        violations = validate_language(text)
        if violations:
            raise ValueError(
                "The generated report contains prohibited clinical phrasing: "
                f"{violations}. Section 32 forbids this language; fix the text "
                "template rather than relaxing the check."
            )
        return text

    def build_json(self, data: ReportInputs) -> Dict[str, Any]:
        """Assemble the machine-readable report (Section 31 shape)."""
        def matrix(a: Optional[np.ndarray]) -> Optional[List[List[float]]]:
            return np.asarray(a).tolist() if a is not None else None

        return {
            "subject_id": data.subject_id,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "experiment_id": data.experiment_id,
            "data_provenance": (
                data.smoke_marker.get("kind", "synthetic")
                if data.smoke_marker else "oasis1"
            ),
            "is_synthetic": bool(data.smoke_marker),
            "current_stage": data.current_stage,
            "class_probabilities": data.class_probabilities,
            "confidence": data.confidence,
            "reference_stage": data.true_stage,
            "ad_associated_propensity": (
                (data.propensity or {}).get("ad_associated_propensity")
            ),
            "stage_transition_propensity": (
                (data.propensity or {}).get("stage_transition_propensity")
            ),
            "stage_propensity_detail": data.propensity,
            "stage_geometry": data.stage_geometry,
            "regional_vulnerability": data.regional_vulnerability,
            "roi_ranking": data.roi_ranking,
            "important_edges": data.top_edges,
            "top_pathways": data.top_pathways,
            "adaptive_adjacency": matrix(data.adaptive_adjacency),
            "propagation_matrix": matrix(data.propagation_matrix),
            "alpha": data.alpha,
            "explanations": {
                "attribution_method": data.attribution_method,
                "top_features": data.top_features,
                "attribution_notes": data.attribution_notes,
                "node_importance": data.node_importance,
            },
            "statistics_summary": data.statistics_summary,
            "dataset": (data.dataset_provenance or {}).get(
                "dataset_source", "OASIS-1"
            ),
            "dataset_provenance": data.dataset_provenance,
            "model": data.model_summary,
            "config": data.config_snapshot,
            "checkpoint": data.checkpoint_path,
            "figures": {k: Path(v).as_posix() for k, v in data.figures.items()},
            "roi_order": list(ROI_ORDER),
            "stage_order": list(STAGE_ORDER),
            "disclaimer": DISCLAIMER,
        }

    def build_html(self, markdown_text: str, data: ReportInputs) -> str:
        """Wrap the Markdown body in a minimal self-contained HTML page.

        The Markdown is embedded verbatim inside a ``<pre>`` block rather than
        converted, because no Markdown renderer is a guaranteed dependency and a
        half-working converter would silently mangle the tables that carry the
        numbers.
        """
        banner = ""
        if data.smoke_marker:
            text = data.smoke_marker.get(
                "banner", "SYNTHETIC DATA - NOT A RESEARCH RESULT"
            )
            banner = (
                f'<div class="banner">{text} &mdash; '
                "NOT ABOUT A REAL SUBJECT</div>"
            )
        escaped = (
            markdown_text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NeuroGenesis report - {data.subject_id}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 15px/1.6 -apple-system, "Segoe UI", Roboto, sans-serif;
          max-width: 60rem; margin: 2rem auto; padding: 0 1.25rem;
          background: #ffffff; color: #1b1f24; }}
  .banner {{ background: #fdecea; border: 1px solid #b3261e; color: #b3261e;
             padding: .75rem 1rem; border-radius: .5rem; font-weight: 700;
             text-align: center; margin-bottom: 1.5rem; }}
  pre {{ white-space: pre-wrap; word-wrap: break-word; font: 13px/1.55
         ui-monospace, "SF Mono", Menlo, Consolas, monospace;
         background: #f6f8fa; border: 1px solid #d9dde3; border-radius: .5rem;
         padding: 1.25rem; }}
  footer {{ margin-top: 2rem; color: #6a737d; font-size: 12px; }}
  @media (prefers-color-scheme: dark) {{
    body {{ background: #0d1117; color: #e6edf3; }}
    pre {{ background: #161b22; border-color: #30363d; }}
    footer {{ color: #8b949e; }}
  }}
</style>
</head>
<body>
{banner}
<pre>{escaped}</pre>
<footer>{DISCLAIMER}</footer>
</body>
</html>
"""

    def generate(self, data: ReportInputs) -> Dict[str, Path]:
        """Build and write the report in all three formats.

        Args:
            data: The report inputs.

        Returns:
            Mapping of format name -> written path.
        """
        subject_dir = self.out_dir / data.subject_id
        subject_dir.mkdir(parents=True, exist_ok=True)

        markdown_text = self.build_markdown(data)
        written: Dict[str, Path] = {}

        path = subject_dir / f"{data.subject_id}_report.md"
        path.write_text(markdown_text, encoding="utf-8")
        written["markdown"] = path

        path = subject_dir / f"{data.subject_id}_report.html"
        path.write_text(self.build_html(markdown_text, data), encoding="utf-8")
        written["html"] = path

        path = subject_dir / f"{data.subject_id}_report.json"
        path.write_text(json.dumps(json_safe(self.build_json(data)), indent=2),
                        encoding="utf-8")
        written["json"] = path

        logger.info("Report generated for %s: %s", data.subject_id, subject_dir)
        return written


__all__ = [
    "PROHIBITED_PHRASES",
    "NEGATION_CUES",
    "DISCLAIMER",
    "validate_language",
    "ReportInputs",
    "ReportGenerator",
]
