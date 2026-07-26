"""
NeuroGenesis — MRI Artifact Detector (Novel QC Module)
========================================================
Module: preprocessing/artifact_detector.py
Author: NeuroGenesis Research Team
Phase : 1 (Preprocessing Pipeline)

Novel Contribution:
    Automated MRI quality control prior to any preprocessing. Computes a
    composite quality score (0–100) by detecting:

    1. Motion Artifacts     — Inter-slice intensity variance anomalies
    2. Signal Dropout       — Slices with abnormally low mean signal
    3. Gibbs Ringing        — High-frequency oscillations near sharp edges
    4. SNR Estimation       — Signal-to-noise ratio across the volume
    5. Saturation Detection — Clipped intensity regions

    Scans below the quality threshold are flagged in the pipeline log and
    can optionally be skipped automatically.

Usage:
    detector = ArtifactDetector(output_dir=Path("outputs/processed"))
    report   = detector.run_qc(data, patient_id="OAS1_0001")
    print(report.summary())
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, sobel

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data class for structured QC results
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class QCReport:
    """
    Structured quality-control result for a single MRI volume.

    Attributes:
        patient_id         : Subject identifier.
        quality_score      : Composite quality score 0–100 (higher = better).
        motion_score       : Sub-score for motion artifact presence (0–100).
        dropout_score      : Sub-score for signal dropout severity (0–100).
        ringing_score      : Sub-score for Gibbs ringing strength (0–100).
        snr_db             : Estimated signal-to-noise ratio in decibels.
        saturation_fraction: Fraction of voxels at max intensity (clipped).
        n_dropout_slices   : Number of axial slices flagged as dropped-out.
        passed             : Whether the scan meets minimum quality threshold.
        flags              : List of human-readable warning messages.
        raw_metrics        : Full dictionary of intermediate metrics.
    """
    patient_id:          str
    quality_score:       float
    motion_score:        float
    dropout_score:       float
    ringing_score:       float
    snr_db:              float
    saturation_fraction: float
    n_dropout_slices:    int
    passed:              bool
    flags:               List[str] = field(default_factory=list)
    raw_metrics:         Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a human-readable one-line summary of the QC report."""
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return (
            f"[QC] {self.patient_id} │ {status} │ "
            f"Score={self.quality_score:.1f}/100 │ "
            f"SNR={self.snr_db:.1f} dB │ "
            f"Dropout slices={self.n_dropout_slices} │ "
            f"Flags={len(self.flags)}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Main detector class
# ──────────────────────────────────────────────────────────────────────────────

class ArtifactDetector:
    """
    Automated MRI artifact detection and quality control module.

    Implements five complementary artifact detectors whose sub-scores are
    combined into a single composite quality score. Each detector is designed
    to be independent so that future detectors can be added without modifying
    the scoring logic.

    Args:
        output_dir          : Directory for saving QC figures.
        min_quality_score   : Scans below this threshold are marked as failed.
        dropout_z_threshold : Z-score threshold for slice-level dropout detection.
        ringing_percentile  : Percentile cutoff for Gibbs ringing energy.
        motion_var_percentile: Percentile cutoff for motion variance anomaly.
    """

    def __init__(
        self,
        output_dir: Path,
        min_quality_score: float = 50.0,
        dropout_z_threshold: float = 2.5,
        ringing_percentile: float = 98.0,
        motion_var_percentile: float = 95.0,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.min_quality_score = min_quality_score
        self.dropout_z_threshold = dropout_z_threshold
        self.ringing_percentile = ringing_percentile
        self.motion_var_percentile = motion_var_percentile

        logger.info(
            f"[ArtifactDetector] Initialised │ min_score={min_quality_score} │ "
            f"output={self.output_dir}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Public — main entry point
    # ──────────────────────────────────────────────────────────────────────────

    def run_qc(
        self,
        data: np.ndarray,
        patient_id: str,
        save_figure: bool = True,
    ) -> QCReport:
        """
        Run the full quality control pipeline on a 3-D MRI volume.

        Executes all five artifact detectors, aggregates their sub-scores into
        a composite quality score, and optionally saves a QC figure.

        Args:
            data       : Float32 MRI array of shape (X, Y, Z).
            patient_id : Subject identifier for logging and file naming.
            save_figure: If *True*, save a QC summary figure to *output_dir*.

        Returns:
            :class:`QCReport` containing all quality metrics and flags.
        """
        logger.info(f"[ArtifactDetector] Running QC on [{patient_id}]")

        raw_metrics: Dict[str, Any] = {}
        flags: List[str] = []

        # ── 1. Motion artifact detection ──────────────────────────────────
        motion_score, motion_meta = self._detect_motion(data)
        raw_metrics["motion"] = motion_meta
        if motion_score < 70:
            flags.append(f"Motion artifacts suspected (score={motion_score:.1f})")

        # ── 2. Signal dropout detection ───────────────────────────────────
        dropout_score, n_dropout, dropout_meta = self._detect_dropout(data)
        raw_metrics["dropout"] = dropout_meta
        if n_dropout > 0:
            flags.append(f"{n_dropout} axial slice(s) with signal dropout")

        # ── 3. Gibbs ringing detection ────────────────────────────────────
        ringing_score, ringing_meta = self._detect_ringing(data)
        raw_metrics["ringing"] = ringing_meta
        if ringing_score < 70:
            flags.append(f"Gibbs ringing detected (score={ringing_score:.1f})")

        # ── 4. SNR estimation ─────────────────────────────────────────────
        snr_db, snr_meta = self._estimate_snr(data)
        raw_metrics["snr"] = snr_meta
        if snr_db < 15.0:
            flags.append(f"Low SNR: {snr_db:.1f} dB (threshold=15 dB)")

        # ── 5. Saturation detection ───────────────────────────────────────
        sat_fraction, sat_meta = self._detect_saturation(data)
        raw_metrics["saturation"] = sat_meta
        if sat_fraction > 0.005:
            flags.append(
                f"Intensity saturation in {sat_fraction*100:.2f}% of voxels"
            )

        # ── Composite score ───────────────────────────────────────────────
        # Weights reflect clinical importance of each artifact type
        weights = {"motion": 0.35, "dropout": 0.25, "ringing": 0.20, "snr": 0.15, "sat": 0.05}
        snr_score = min(100.0, max(0.0, (snr_db / 30.0) * 100.0))
        sat_score = max(0.0, 100.0 - sat_fraction * 10000)

        quality_score = (
            weights["motion"]  * motion_score +
            weights["dropout"] * dropout_score +
            weights["ringing"] * ringing_score +
            weights["snr"]     * snr_score +
            weights["sat"]     * sat_score
        )
        quality_score = round(float(np.clip(quality_score, 0.0, 100.0)), 2)
        passed = quality_score >= self.min_quality_score

        report = QCReport(
            patient_id=patient_id,
            quality_score=quality_score,
            motion_score=float(motion_score),
            dropout_score=float(dropout_score),
            ringing_score=float(ringing_score),
            snr_db=float(snr_db),
            saturation_fraction=float(sat_fraction),
            n_dropout_slices=int(n_dropout),
            passed=passed,
            flags=flags,
            raw_metrics=raw_metrics,
        )

        status_str = "✓ PASS" if passed else "✗ FAIL"
        logger.info(
            f"[ArtifactDetector] {status_str} │ {patient_id} │ "
            f"QC Score={quality_score:.1f}/100"
        )

        if flags:
            for flag in flags:
                logger.warning(f"[ArtifactDetector]   ⚠ {flag}")

        if save_figure:
            self._save_qc_figure(data, report)

        return report

    # ──────────────────────────────────────────────────────────────────────────
    # Artifact detectors (private)
    # ──────────────────────────────────────────────────────────────────────────

    def _detect_motion(
        self, data: np.ndarray
    ) -> tuple[float, Dict[str, Any]]:
        """
        Detect motion artifacts via inter-slice intensity variance analysis.

        Motion during acquisition causes abrupt intensity discontinuities
        between consecutive axial slices. We compute the variance of mean
        slice intensities, then flag volumes where the top-percentile variance
        exceeds the median by a large factor.

        Args:
            data : 3-D MRI array (X, Y, Z).

        Returns:
            Tuple of (score 0-100, metadata dict).
        """
        # Mean intensity per axial slice
        slice_means = np.array([data[:, :, z].mean() for z in range(data.shape[2])])

        # First-order differences → motion spikes produce large jumps
        diffs = np.abs(np.diff(slice_means))
        diff_median = float(np.median(diffs)) + 1e-6
        diff_max = float(np.max(diffs))

        # Ratio of max jump to median jump — higher = more motion
        motion_ratio = diff_max / diff_median

        # Convert to score: ratio of 1 → 100, ratio of 20+ → 0
        motion_score = float(np.clip(100.0 - (motion_ratio - 1.0) * 5.0, 0.0, 100.0))

        meta = {
            "slice_mean_variance": float(np.var(slice_means)),
            "max_diff": diff_max,
            "median_diff": float(np.median(diffs)),
            "motion_ratio": motion_ratio,
        }
        return motion_score, meta

    def _detect_dropout(
        self, data: np.ndarray
    ) -> tuple[float, int, Dict[str, Any]]:
        """
        Identify axial slices with anomalously low signal (dropout).

        Signal dropout occurs when RF pulses interact with metallic implants
        or due to susceptibility artefacts. Affected slices have mean intensity
        below (global_mean − z_threshold × global_std).

        Args:
            data : 3-D MRI array (X, Y, Z).

        Returns:
            Tuple of (score 0-100, n_dropout_slices, metadata dict).
        """
        slice_means = np.array([data[:, :, z].mean() for z in range(data.shape[2])])
        global_mean = float(np.mean(slice_means))
        global_std = float(np.std(slice_means)) + 1e-6

        z_scores = (slice_means - global_mean) / global_std
        dropout_mask = z_scores < -self.dropout_z_threshold
        n_dropout = int(np.sum(dropout_mask))
        dropout_fraction = n_dropout / len(slice_means)

        # Score: 0 dropout → 100, heavy dropout → 0
        dropout_score = float(np.clip(100.0 - dropout_fraction * 500.0, 0.0, 100.0))

        meta = {
            "n_dropout_slices": n_dropout,
            "dropout_fraction": dropout_fraction,
            "z_threshold": self.dropout_z_threshold,
            "dropped_slice_indices": list(np.where(dropout_mask)[0]),
        }
        return dropout_score, n_dropout, meta

    def _detect_ringing(
        self, data: np.ndarray
    ) -> tuple[float, Dict[str, Any]]:
        """
        Detect Gibbs ringing artefacts using high-frequency edge energy.

        Gibbs ringing manifests as oscillating stripes near sharp intensity
        transitions. We estimate ringing energy by comparing high-frequency
        content near Sobel-detected edges against the global signal energy.

        Args:
            data : 3-D MRI array (X, Y, Z).

        Returns:
            Tuple of (score 0-100, metadata dict).
        """
        # Use central 1/3 of slices for efficiency
        nz = data.shape[2]
        z_start, z_end = nz // 3, 2 * nz // 3
        sample = data[:, :, z_start:z_end].astype(np.float32)

        # Edge detection via Sobel filter
        smoothed = gaussian_filter(sample, sigma=1.0)
        edge_x = sobel(smoothed, axis=0)
        edge_y = sobel(smoothed, axis=1)
        edge_magnitude = np.hypot(edge_x, edge_y)

        # High-frequency content (signal - smoothed)
        hf_content = np.abs(sample - smoothed)

        # Ringing metric: ratio of HF energy near edges to total edge energy
        edge_threshold = np.percentile(edge_magnitude, self.ringing_percentile)
        near_edge = edge_magnitude > edge_threshold
        hf_near_edge = float(np.mean(hf_content[near_edge])) if near_edge.any() else 0.0
        hf_global = float(np.mean(hf_content)) + 1e-6

        ringing_ratio = hf_near_edge / hf_global

        # Higher ratio → more ringing → lower score
        ringing_score = float(np.clip(100.0 / (1.0 + ringing_ratio * 2.0), 0.0, 100.0))

        meta = {
            "hf_near_edge_mean": hf_near_edge,
            "hf_global_mean": hf_global,
            "ringing_ratio": ringing_ratio,
        }
        return ringing_score, meta

    def _estimate_snr(
        self, data: np.ndarray
    ) -> tuple[float, Dict[str, Any]]:
        """
        Estimate signal-to-noise ratio (dB) using brain/background split.

        Signal region = voxels above 15th percentile of non-zero intensities.
        Noise region  = voxels below 5th percentile of all intensities.

        Args:
            data : 3-D MRI array (X, Y, Z).

        Returns:
            Tuple of (snr_db, metadata dict).
        """
        nonzero = data[data > 0]
        if len(nonzero) == 0:
            return 0.0, {"error": "all-zero volume"}

        signal_threshold = float(np.percentile(nonzero, 15))
        noise_threshold = float(np.percentile(data, 5))

        signal_vals = data[data > signal_threshold]
        noise_vals = data[data <= noise_threshold]

        signal_mean = float(np.mean(signal_vals)) if len(signal_vals) else 1e-6
        noise_std = float(np.std(noise_vals)) if len(noise_vals) else 1e-6

        snr = signal_mean / (noise_std + 1e-10)
        snr_db = 20.0 * np.log10(snr + 1e-10)

        meta = {
            "signal_mean": signal_mean,
            "noise_std": noise_std,
            "snr_linear": float(snr),
            "snr_db": float(snr_db),
        }
        return float(snr_db), meta

    def _detect_saturation(
        self, data: np.ndarray
    ) -> tuple[float, Dict[str, Any]]:
        """
        Detect intensity saturation (clipped voxels at maximum value).

        Args:
            data : 3-D MRI array (X, Y, Z).

        Returns:
            Tuple of (saturation_fraction, metadata dict).
        """
        max_val = float(data.max())
        saturated = np.sum(data >= max_val * 0.999)
        sat_fraction = saturated / data.size

        meta = {
            "max_intensity": max_val,
            "saturated_voxels": int(saturated),
            "saturation_fraction": float(sat_fraction),
        }
        return float(sat_fraction), meta

    # ──────────────────────────────────────────────────────────────────────────
    # Visualisation
    # ──────────────────────────────────────────────────────────────────────────

    def _save_qc_figure(self, data: np.ndarray, report: QCReport) -> str:
        """
        Save a publication-quality QC dashboard figure.

        Args:
            data   : Original MRI volume.
            report : Completed :class:`QCReport`.

        Returns:
            Path to saved PNG.
        """
        fig = plt.figure(figsize=(16, 10), facecolor="#0d1117")
        fig.suptitle(
            f"NeuroGenesis │ MRI Quality Control Report │ {report.patient_id}",
            color="white", fontsize=14, fontweight="bold", y=0.98
        )

        # ── Layout: 2 rows ─────────────────────────────────────────────────
        gs = plt.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

        # Axial slice
        ax_slice = fig.add_subplot(gs[0, :2])
        cz = data.shape[2] // 2
        ax_slice.imshow(np.rot90(data[:, :, cz]), cmap="bone", aspect="auto")
        ax_slice.set_title("Central Axial Slice", color="#58a6ff", fontsize=11)
        ax_slice.axis("off")

        # Slice-mean intensity profile
        ax_profile = fig.add_subplot(gs[0, 2:])
        slice_means = [data[:, :, z].mean() for z in range(data.shape[2])]
        ax_profile.plot(slice_means, color="#58a6ff", linewidth=1.5)
        ax_profile.set_facecolor("#161b22")
        ax_profile.tick_params(colors="white")
        ax_profile.set_xlabel("Axial Slice Index", color="#8b949e", fontsize=9)
        ax_profile.set_ylabel("Mean Intensity", color="#8b949e", fontsize=9)
        ax_profile.set_title("Slice Intensity Profile", color="#58a6ff", fontsize=11)
        for spine in ax_profile.spines.values():
            spine.set_edgecolor("#30363d")

        # Score bars
        ax_scores = fig.add_subplot(gs[1, :2])
        score_names = ["Motion", "Dropout", "Ringing", "Overall"]
        score_vals = [report.motion_score, report.dropout_score,
                      report.ringing_score, report.quality_score]
        colors = ["#58a6ff" if s >= 70 else "#f97583" if s < 50 else "#e3b341"
                  for s in score_vals]
        bars = ax_scores.barh(score_names, score_vals, color=colors, height=0.5)
        ax_scores.set_xlim(0, 100)
        ax_scores.axvline(x=self.min_quality_score, color="#f97583",
                          linestyle="--", linewidth=1, alpha=0.7)
        ax_scores.set_facecolor("#161b22")
        ax_scores.tick_params(colors="white")
        for spine in ax_scores.spines.values():
            spine.set_edgecolor("#30363d")
        ax_scores.set_title("Quality Sub-scores", color="#58a6ff", fontsize=11)
        for bar, val in zip(bars, score_vals):
            ax_scores.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                           f"{val:.1f}", va="center", color="white", fontsize=9)

        # Report text
        ax_text = fig.add_subplot(gs[1, 2:])
        ax_text.set_facecolor("#161b22")
        ax_text.axis("off")
        status_color = "#3fb950" if report.passed else "#f97583"
        status_text = "PASS ✓" if report.passed else "FAIL ✗"

        report_lines = [
            f"Status     : {status_text}",
            f"QC Score   : {report.quality_score:.1f} / 100",
            f"SNR        : {report.snr_db:.1f} dB",
            f"Dropout    : {report.n_dropout_slices} slice(s)",
            f"Saturation : {report.saturation_fraction*100:.3f}%",
            "─" * 30,
        ] + [f"⚠ {f}" for f in report.flags] if report.flags else [
            f"Status     : {status_text}",
            f"QC Score   : {report.quality_score:.1f} / 100",
            f"SNR        : {report.snr_db:.1f} dB",
            f"Dropout    : {report.n_dropout_slices} slice(s)",
            f"Saturation : {report.saturation_fraction*100:.3f}%",
            "─" * 30,
            "No flags — scan is clean.",
        ]

        ax_text.text(
            0.05, 0.95, "\n".join(report_lines),
            transform=ax_text.transAxes,
            color="white", fontsize=9, fontfamily="monospace",
            verticalalignment="top",
        )
        ax_text.text(
            0.05, 0.05, status_text,
            transform=ax_text.transAxes,
            color=status_color, fontsize=20, fontweight="bold",
        )

        save_path = self.output_dir / f"{report.patient_id}_qc_report.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight",
                    facecolor="#0d1117", edgecolor="none")
        plt.close(fig)
        logger.info(f"[ArtifactDetector] QC figure saved → {save_path}")
        return str(save_path)
