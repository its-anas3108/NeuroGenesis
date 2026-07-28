"""
NeuroGenesis — Publication-Quality Visualiser
===============================================
Module: visualization/visualize.py
Author: NeuroGenesis Research Team
Phase : 1 (Preprocessing Pipeline)

Generates a comprehensive set of publication-quality figures summarising
the complete NeuroGenesis Phase 1 pipeline for a single subject.

Figures produced:
    1. Pipeline summary (6-panel overview: original → processed → ROIs →
       features → graph → tensor mosaic)
    2. Preprocessing stage comparison (all 5 stages in one figure)
    3. ROI panel (all 5 speech regions side-by-side)
    4. Feature bar charts (brain volume, GM volume, mean intensity per ROI)
    5. Connectivity graph summary with adjacency matrix side-by-side

Design system:
    - Dark theme (#0d1117 background, GitHub dark palette)
    - All fonts: DejaVu Sans (system-safe) or Inter if available
    - Consistent NeuroGenesis colour palette
    - All figures at 300 DPI, saved as PNG

Usage:
    visualizer = NeuroGenesisVisualizer(output_dir=Path("outputs"))
    visualizer.generate_pipeline_summary(
        patient_id   = "OAS1_0001",
        original     = raw_data,
        preprocessed = final_data,
        masks        = roi_masks_dict,
        feature_df   = features_df,
        graph        = graph_obj,
    )
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd

# ── Try to register Inter font (optional) ─────────────────────────────────
try:
    from matplotlib import font_manager
    font_manager.fontManager.addfont("/System/Library/Fonts/Supplemental/Futura.ttc")
except Exception:
    pass

logger = logging.getLogger(__name__)

# ── Design tokens ─────────────────────────────────────────────────────────────
BG_COLOR     = "#0d1117"   # Canvas background
PANEL_COLOR  = "#161b22"   # Panel background
BORDER_COLOR = "#30363d"   # Borders
TEXT_WHITE   = "#e6edf3"   # Primary text
TEXT_DIM     = "#8b949e"   # Secondary text
ACCENT_BLUE  = "#388bfd"   # Primary accent
ACCENT_GREEN = "#3fb950"   # Success / processed
ACCENT_AMBER = "#e3b341"   # Warning / features
ACCENT_RED   = "#ff7b72"   # Danger / error

ROI_COLORS = {
    "Broca_Area":               "#388bfd",
    "Wernicke_Area":            "#3fb950",
    "Insula":                   "#e3b341",
    "Inferior_Frontal_Gyrus":   "#ff7b72",
    "Superior_Temporal_Gyrus":  "#bc8cff",
}


class NeuroGenesisVisualizer:
    """
    Central visualisation hub for the NeuroGenesis Phase 1 pipeline.

    All figures share a consistent dark theme and 300 DPI resolution.
    Individual figure methods can be called independently, or the
    comprehensive ``generate_pipeline_summary`` can be used for a
    complete end-to-end visual report.

    Args:
        output_dir : Root directory under which all figures are saved.
                     Sub-figures are saved to appropriate sub-directories.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Configure matplotlib defaults
        plt.rcParams.update({
            "figure.facecolor":  BG_COLOR,
            "axes.facecolor":    PANEL_COLOR,
            "axes.edgecolor":    BORDER_COLOR,
            "axes.labelcolor":   TEXT_DIM,
            "xtick.color":       TEXT_DIM,
            "ytick.color":       TEXT_DIM,
            "text.color":        TEXT_WHITE,
            "grid.color":        BORDER_COLOR,
            "grid.alpha":        0.5,
        })

        logger.info(f"[NeuroGenesisVisualizer] Initialised │ output={self.output_dir}")

    # ──────────────────────────────────────────────────────────────────────────
    # Public — comprehensive pipeline summary
    # ──────────────────────────────────────────────────────────────────────────

    def generate_pipeline_summary(
        self,
        patient_id:   str,
        original:     np.ndarray,
        preprocessed: np.ndarray,
        masked:       Optional[np.ndarray] = None,
        masks:        Optional[Dict[str, np.ndarray]] = None,
        feature_df:   Optional[pd.DataFrame] = None,
        graph:        Optional[nx.DiGraph] = None,
        patches:      Optional[Dict[str, np.ndarray]] = None,
    ) -> str:
        """
        Generate the master pipeline summary figure (12-panel layout).

        This single figure encapsulates the complete Phase 1 pipeline:
        Original MRI → QC → Preprocessing → Skull Strip → ROI Extraction →
        ROI Patches → Feature Table → Connectivity Graph

        Args:
            patient_id   : Subject identifier.
            original     : Raw MRI float32 array (X, Y, Z).
            preprocessed : Final preprocessed float32 array.
            masked       : Skull-stripped float32 array (optional).
            masks        : Dict of ROI binary masks (optional).
            feature_df   : Feature DataFrame from FeatureExtractor (optional).
            graph        : NetworkX DiGraph from BrainConnectivityGraph (optional).
            patches      : Dict of ROI patch arrays (optional).

        Returns:
            Path to saved summary PNG.
        """
        logger.info(f"[Visualizer] Generating pipeline summary for [{patient_id}]")

        fig = plt.figure(figsize=(24, 20), facecolor=BG_COLOR)
        fig.suptitle(
            f"NeuroGenesis Phase 1 ─ Full Pipeline Summary ─ Patient: {patient_id}",
            color=TEXT_WHITE, fontsize=18, fontweight="bold",
            y=0.98, fontfamily="monospace"
        )

        gs_main = gridspec.GridSpec(
            3, 4, figure=fig,
            hspace=0.45, wspace=0.30,
            top=0.94, bottom=0.04, left=0.04, right=0.96
        )

        # ── Row 0: Tri-plane views ─────────────────────────────────────────
        cz = original.shape[2] // 2

        # Panel 0,0 — Original MRI axial
        ax00 = fig.add_subplot(gs_main[0, 0])
        self._draw_slice(ax00, np.rot90(original[:, :, cz]),
                         "Original MRI (Axial)", "bone", ACCENT_BLUE)

        # Panel 0,1 — Preprocessed
        ax01 = fig.add_subplot(gs_main[0, 1])
        self._draw_slice(ax01, np.rot90(preprocessed[:, :, cz]),
                         "Preprocessed MRI", "bone", ACCENT_GREEN)

        # Panel 0,2 — Skull stripped
        ax02 = fig.add_subplot(gs_main[0, 2])
        if masked is not None:
            self._draw_slice(ax02, np.rot90(masked[:, :, cz]),
                             "Skull Stripped", "bone", ACCENT_AMBER)
        else:
            self._draw_placeholder(ax02, "Skull Strip\n(not available)")

        # Panel 0,3 — All ROIs combined
        ax03 = fig.add_subplot(gs_main[0, 3])
        if masks:
            self._draw_roi_overlay(ax03, original, masks, cz,
                                   "All Speech ROIs (Axial)")
        else:
            self._draw_placeholder(ax03, "ROI Masks\n(not available)")

        # ── Row 1: ROI patches + feature bar ──────────────────────────────
        # Panel 1,0–1,2: ROI patch mosaic (3 of 5)
        roi_names = list(masks.keys()) if masks else []
        roi_show = roi_names[:3] if roi_names else []
        for col, roi_name in enumerate(roi_show):
            ax = fig.add_subplot(gs_main[1, col])
            if patches and roi_name in patches:
                p = patches[roi_name]
                cz_p = p.shape[2] // 2
                color = ROI_COLORS.get(roi_name, ACCENT_BLUE)
                self._draw_slice(ax, np.rot90(p[:, :, cz_p]),
                                 roi_name.replace("_", " "),
                                 "plasma", color)
            else:
                self._draw_placeholder(ax, roi_name.replace("_", "\n"))

        # Panel 1,3: remaining ROI patches or feature preview
        ax13 = fig.add_subplot(gs_main[1, 3])
        if patches and len(roi_names) > 3:
            # Show 4th patch
            rn4 = roi_names[3]
            p4 = patches[rn4]
            cz4 = p4.shape[2] // 2
            color4 = ROI_COLORS.get(rn4, ACCENT_BLUE)
            self._draw_slice(ax13, np.rot90(p4[:, :, cz4]),
                             rn4.replace("_", " "), "plasma", color4)
        else:
            self._draw_placeholder(ax13, "ROI Patch 4\n(5th omitted\nfor space)")

        # ── Row 2: Feature chart + Graph ──────────────────────────────────
        # Panel 2,0–1: Feature bar chart
        ax20 = fig.add_subplot(gs_main[2, :2])
        if feature_df is not None and not feature_df.empty:
            self._draw_feature_bar(ax20, feature_df)
        else:
            self._draw_placeholder(ax20, "Feature Table\n(not available)")

        # Panel 2,2–3: Graph
        ax22 = fig.add_subplot(gs_main[2, 2:])
        if graph is not None:
            self._draw_graph_inline(ax22, graph)
        else:
            self._draw_placeholder(ax22, "Connectivity Graph\n(not available)")

        # ── Footer ────────────────────────────────────────────────────────
        fig.text(
            0.5, 0.01,
            "NeuroGenesis Phase 1 │ Speech Atrophy Preprocessing Pipeline │ "
            "ROI-Centric Processing │ Ready for NeuroProp-X",
            ha="center", color=TEXT_DIM, fontsize=9, fontfamily="monospace"
        )

        save_path = self.output_dir / f"{patient_id}_pipeline_summary.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight",
                    facecolor=BG_COLOR, edgecolor="none")
        plt.close(fig)
        logger.info(f"[Visualizer] Pipeline summary saved → {save_path}")
        return str(save_path)

    # ──────────────────────────────────────────────────────────────────────────
    # Public — individual stage figures
    # ──────────────────────────────────────────────────────────────────────────

    def plot_preprocessing_pipeline(
        self,
        stages:     Dict[str, np.ndarray],
        patient_id: str,
    ) -> str:
        """
        Plot all preprocessing stages in a single horizontal figure.

        Args:
            stages     : OrderedDict of stage_name → volume array.
            patient_id : Subject identifier.

        Returns:
            Path to saved PNG.
        """
        n = len(stages)
        fig, axes = plt.subplots(1, n, figsize=(n * 5, 5), facecolor=BG_COLOR)
        if n == 1:
            axes = [axes]

        fig.suptitle(
            f"NeuroGenesis │ Preprocessing Pipeline │ {patient_id}",
            color=TEXT_WHITE, fontsize=14, fontweight="bold"
        )

        colors = [ACCENT_BLUE, "#58a6ff", "#d2a8ff", ACCENT_AMBER,
                  ACCENT_GREEN, "#ff7b72"]

        for ax, (name, vol), color in zip(axes, stages.items(), colors * 2):
            cz = vol.shape[2] // 2 if vol.ndim == 3 else 0
            sl = np.rot90(vol[:, :, cz]) if vol.ndim == 3 else vol
            self._draw_slice(ax, sl, name.replace("_", " "), "bone", color)

        plt.tight_layout()
        path = self.output_dir / f"{patient_id}_preprocessing_pipeline.png"
        fig.savefig(path, dpi=300, bbox_inches="tight",
                    facecolor=BG_COLOR, edgecolor="none")
        plt.close(fig)
        logger.info(f"[Visualizer] Preprocessing pipeline figure → {path}")
        return str(path)

    def plot_roi_panel(
        self,
        original:   np.ndarray,
        masks:      Dict[str, np.ndarray],
        patient_id: str,
    ) -> str:
        """
        Plot all 5 speech ROIs as individual overlay panels.

        Args:
            original   : Background MRI array.
            masks      : Dict of ROI binary masks.
            patient_id : Subject identifier.

        Returns:
            Path to saved PNG.
        """
        roi_names = list(masks.keys())
        n = len(roi_names)
        fig, axes = plt.subplots(1, n, figsize=(n * 4, 4), facecolor=BG_COLOR)
        if n == 1:
            axes = [axes]

        fig.suptitle(
            f"NeuroGenesis │ Speech ROI Panel │ {patient_id}",
            color=TEXT_WHITE, fontsize=14, fontweight="bold"
        )

        cz = original.shape[2] // 2
        sl_mri = np.rot90(original[:, :, cz])

        for ax, roi_name in zip(axes, roi_names):
            color_hex = ROI_COLORS.get(roi_name, ACCENT_BLUE)
            rgb = matplotlib.colors.to_rgb(color_hex)

            ax.imshow(sl_mri, cmap="bone", aspect="auto")
            mask_sl = np.rot90(masks[roi_name][:, :, cz])
            if mask_sl.any():
                overlay = np.zeros((*mask_sl.shape, 4), dtype=np.float32)
                overlay[mask_sl > 0] = [*rgb, 0.65]
                ax.imshow(overlay, aspect="auto")

            ax.set_title(roi_name.replace("_", "\n"),
                         color=color_hex, fontsize=9, fontweight="bold")
            ax.axis("off")

        plt.tight_layout()
        path = self.output_dir / f"{patient_id}_roi_panel.png"
        fig.savefig(path, dpi=300, bbox_inches="tight",
                    facecolor=BG_COLOR, edgecolor="none")
        plt.close(fig)
        logger.info(f"[Visualizer] ROI panel figure → {path}")
        return str(path)

    def plot_feature_table_figure(
        self,
        feature_df: pd.DataFrame,
        patient_id: str,
    ) -> str:
        """
        Plot a visual feature table showing numeric values per ROI.

        Args:
            feature_df : Feature DataFrame from FeatureExtractor.
            patient_id : Subject identifier.

        Returns:
            Path to saved PNG.
        """
        numeric_cols = [
            "voxel_count", "brain_volume_mm3", "gm_volume_mm3",
            "mean_intensity", "std_intensity", "surface_area_vox",
        ]
        cols = [c for c in numeric_cols if c in feature_df.columns]
        if not cols:
            logger.warning("[Visualizer] No numeric columns for feature table figure.")
            return ""

        fig, axes = plt.subplots(2, 3, figsize=(16, 8), facecolor=BG_COLOR)
        axes = axes.flatten()

        fig.suptitle(
            f"NeuroGenesis │ ROI Feature Comparison │ {patient_id}",
            color=TEXT_WHITE, fontsize=14, fontweight="bold"
        )

        roi_col = "roi_name" if "roi_name" in feature_df.columns else feature_df.index
        roi_names = feature_df[roi_col].tolist() if "roi_name" in feature_df.columns \
            else feature_df.index.tolist()
        colors = [ROI_COLORS.get(r, ACCENT_BLUE) for r in roi_names]
        short_names = [r.replace("_", "\n") for r in roi_names]

        for ax, col_name in zip(axes, cols):
            vals = feature_df[col_name].tolist()
            bars = ax.bar(short_names, vals, color=colors, alpha=0.85)
            ax.set_title(col_name.replace("_", " ").title(),
                         color=ACCENT_BLUE, fontsize=10)
            ax.tick_params(labelsize=7)
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f"{val:.2f}", ha="center", va="bottom",
                    color=TEXT_WHITE, fontsize=7
                )
            for spine in ax.spines.values():
                spine.set_edgecolor(BORDER_COLOR)

        # Hide unused
        for ax in axes[len(cols):]:
            ax.set_visible(False)

        plt.tight_layout()
        path = self.output_dir / f"{patient_id}_feature_table.png"
        fig.savefig(path, dpi=300, bbox_inches="tight",
                    facecolor=BG_COLOR, edgecolor="none")
        plt.close(fig)
        logger.info(f"[Visualizer] Feature table figure → {path}")
        return str(path)

    # ──────────────────────────────────────────────────────────────────────────
    # Private — drawing primitives
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_slice(
        self,
        ax:     plt.Axes,
        sl:     np.ndarray,
        title:  str,
        cmap:   str,
        color:  str,
    ) -> None:
        """Draw a single MRI slice on an axes with styled title."""
        im = ax.imshow(sl, cmap=cmap, aspect="auto", interpolation="bilinear")
        ax.set_title(title, color=color, fontsize=9, fontweight="bold", pad=6)
        ax.axis("off")
        ax.set_facecolor(PANEL_COLOR)

    def _draw_placeholder(self, ax: plt.Axes, text: str) -> None:
        """Draw a placeholder panel when data is not available."""
        ax.set_facecolor(PANEL_COLOR)
        ax.axis("off")
        ax.text(0.5, 0.5, text, transform=ax.transAxes,
                ha="center", va="center", color=TEXT_DIM, fontsize=9,
                multialignment="center")
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER_COLOR)

    def _draw_roi_overlay(
        self,
        ax:       plt.Axes,
        data:     np.ndarray,
        masks:    Dict[str, np.ndarray],
        cz:       int,
        title:    str,
    ) -> None:
        """Draw all ROI masks overlaid on one axial slice."""
        import matplotlib
        sl = np.rot90(data[:, :, cz])
        ax.imshow(sl, cmap="bone", aspect="auto")

        for roi_name, mask in masks.items():
            rgb = matplotlib.colors.to_rgb(ROI_COLORS.get(roi_name, ACCENT_BLUE))
            sl_mask = np.rot90(mask[:, :, cz])
            if sl_mask.any():
                overlay = np.zeros((*sl_mask.shape, 4), dtype=np.float32)
                overlay[sl_mask > 0] = [*rgb, 0.55]
                ax.imshow(overlay, aspect="auto")

        ax.set_title(title, color=ACCENT_AMBER, fontsize=9, fontweight="bold")
        ax.axis("off")

    def _draw_feature_bar(
        self,
        ax:         plt.Axes,
        feature_df: pd.DataFrame,
    ) -> None:
        """Draw a grouped feature bar chart on axes."""
        if "roi_name" not in feature_df.columns:
            self._draw_placeholder(ax, "Feature data\nunavailable")
            return

        rois = feature_df["roi_name"].tolist()
        colors = [ROI_COLORS.get(r, ACCENT_BLUE) for r in rois]
        short_rois = [r.replace("_", " ") for r in rois]

        col = "brain_volume_mm3" if "brain_volume_mm3" in feature_df.columns else \
              "voxel_count"
        vals = feature_df[col].tolist()

        bars = ax.bar(short_rois, vals, color=colors, alpha=0.85)
        ax.set_title("ROI Brain Volume (mm³)", color=ACCENT_BLUE, fontsize=10)
        ax.tick_params(labelsize=7, colors=TEXT_DIM)
        ax.set_xlabel("ROI", color=TEXT_DIM, fontsize=8)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f"{val:.0f}", ha="center", va="bottom",
                    color=TEXT_WHITE, fontsize=7)

        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER_COLOR)

    def _draw_graph_inline(
        self,
        ax:    plt.Axes,
        graph: nx.DiGraph,
    ) -> None:
        """Draw the speech connectivity graph inline on given axes."""
        ax.set_facecolor(PANEL_COLOR)
        ax.axis("off")
        ax.set_aspect("equal")

        nodes = list(graph.nodes())
        n = len(nodes)
        angles = [2 * np.pi * i / n for i in range(n)]
        pos = {node: (np.cos(a), np.sin(a)) for node, a in zip(nodes, angles)}

        ax.set_xlim(-1.7, 1.7)
        ax.set_ylim(-1.7, 1.7)

        weights = [graph[u][v].get("weight", 1.0) for u, v in graph.edges()]
        max_w = max(weights) if weights else 1.0

        for (u, v), w in zip(graph.edges(), weights):
            x0, y0 = pos.get(u, (0, 0))
            x1, y1 = pos.get(v, (0, 0))
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(
                            arrowstyle="-|>",
                            color=plt.cm.plasma(w / max_w),
                            lw=0.5 + 2.0 * (w / max_w),
                            alpha=0.5 + 0.4 * (w / max_w),
                            connectionstyle="arc3,rad=0.15",
                            mutation_scale=10,
                        ))

        for node in nodes:
            x, y = pos.get(node, (0, 0))
            color = ROI_COLORS.get(node, ACCENT_BLUE)
            circle = plt.Circle((x, y), 0.18, color=color, zorder=5, alpha=0.9)
            ax.add_patch(circle)
            short = node.split("_")[0]
            ax.text(x, y, short, ha="center", va="center",
                    color="white", fontsize=6, fontweight="bold", zorder=6)

        ax.set_title("Connectivity Graph", color=ACCENT_BLUE,
                     fontsize=10, fontweight="bold")

    # ──────────────────────────────────────────────────────────────────────────
    # Steps 7, 8, 9 — Anatomical ROI Overlays & Multi-Plane Views
    # ──────────────────────────────────────────────────────────────────────────

    def plot_individual_roi_overlays(
        self,
        data:       np.ndarray,
        masks:      Dict[str, np.ndarray],
        patient_id: str,
        subj_dir:   Optional[Path] = None,
    ) -> Dict[str, str]:
        """
        Step 7: Generate separate publication-quality labelled images for each ROI.

        Outputs:
            broca_area_overlay.png
            wernicke_area_overlay.png
            insula_overlay.png
            inferior_frontal_gyrus_overlay.png
            superior_temporal_gyrus_overlay.png
        """
        import matplotlib
        out_dir = Path(subj_dir) if subj_dir else self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        colors_map = {
            "Broca_Area":               "#ff4d4d",  # Red
            "Wernicke_Area":            "#3399ff",  # Blue
            "Insula":                   "#33cc33",  # Green
            "Inferior_Frontal_Gyrus":   "#ff9933",  # Orange
            "Superior_Temporal_Gyrus":  "#cc66ff",  # Purple
        }

        filename_map = {
            "Broca_Area":               "broca_area_overlay.png",
            "Wernicke_Area":            "wernicke_area_overlay.png",
            "Insula":                   "insula_overlay.png",
            "Inferior_Frontal_Gyrus":   "inferior_frontal_gyrus_overlay.png",
            "Superior_Temporal_Gyrus":  "superior_temporal_gyrus_overlay.png",
        }

        saved_paths = {}

        for roi_name, mask in masks.items():
            if roi_name not in filename_map:
                continue

            fname = filename_map[roi_name]
            hex_col = colors_map.get(roi_name, ACCENT_BLUE)
            rgb = matplotlib.colors.to_rgb(hex_col)

            # Find slice with maximum ROI area
            if mask.sum() > 0:
                slice_sums = mask.sum(axis=(0, 1))
                cz = int(np.argmax(slice_sums))
            else:
                cz = data.shape[2] // 2

            fig, ax = plt.subplots(figsize=(8, 8), facecolor=BG_COLOR)
            mri_sl = np.rot90(data[:, :, cz])
            mask_sl = np.rot90(mask[:, :, cz])

            ax.imshow(mri_sl, cmap="bone", aspect="auto")

            if mask_sl.any():
                overlay = np.zeros((*mask_sl.shape, 4), dtype=np.float32)
                overlay[mask_sl > 0] = [*rgb, 0.6]
                ax.imshow(overlay, aspect="auto")
                # Contour boundary
                ax.contour(mask_sl > 0, colors=[hex_col], linewidths=1.5)

            display_title = f"{patient_id} │ {roi_name.replace('_', ' ')}\nAxial Slice {cz} │ Orientation: Axial"
            ax.set_title(display_title, color=hex_col, fontsize=13, fontweight="bold", pad=12)
            ax.axis("off")

            # Annotation box
            n_vox = int(mask.sum())
            annot_text = f"ROI: {roi_name.replace('_', ' ')}\nVoxels: {n_vox}\nSlice: {cz} (Axial)" if n_vox > 0 else f"ROI: {roi_name}\nNo Voxels Detected"
            ax.text(0.03, 0.03, annot_text, transform=ax.transAxes,
                    color="white", fontsize=9, fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor=PANEL_COLOR, edgecolor=hex_col, alpha=0.85))

            plt.tight_layout()
            out_path = out_dir / fname
            fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR, edgecolor="none")
            plt.close(fig)

            saved_paths[roi_name] = str(out_path)
            logger.info(f"[Visualizer] Individual ROI overlay saved → {out_path}")

        return saved_paths

    def plot_combined_speech_network_overlay(
        self,
        data:       np.ndarray,
        masks:      Dict[str, np.ndarray],
        patient_id: str,
        subj_dir:   Optional[Path] = None,
    ) -> str:
        """
        Step 8: Generate speech_network_overlay.png combining all 5 ROIs simultaneously.

        Colours:
            Broca → Red
            Wernicke → Blue
            Insula → Green
            IFG → Orange
            STG → Purple
        """
        import matplotlib
        out_dir = Path(subj_dir) if subj_dir else self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        colors_map = {
            "Broca_Area":               "#ff4d4d",  # Red
            "Wernicke_Area":            "#3399ff",  # Blue
            "Insula":                   "#33cc33",  # Green
            "Inferior_Frontal_Gyrus":   "#ff9933",  # Orange
            "Superior_Temporal_Gyrus":  "#cc66ff",  # Purple
        }

        # Select axial slice with maximum total speech ROI voxels
        total_mask = sum(masks.values()) if masks else np.zeros_like(data)
        if total_mask.sum() > 0:
            cz = int(np.argmax(total_mask.sum(axis=(0, 1))))
        else:
            cz = data.shape[2] // 2

        fig, ax = plt.subplots(figsize=(10, 10), facecolor=BG_COLOR)
        mri_sl = np.rot90(data[:, :, cz])
        ax.imshow(mri_sl, cmap="bone", aspect="auto")

        legend_patches = []
        for roi_name, mask in masks.items():
            hex_col = colors_map.get(roi_name, ACCENT_BLUE)
            rgb = matplotlib.colors.to_rgb(hex_col)
            mask_sl = np.rot90(mask[:, :, cz])

            if mask_sl.any():
                overlay = np.zeros((*mask_sl.shape, 4), dtype=np.float32)
                overlay[mask_sl > 0] = [*rgb, 0.55]
                ax.imshow(overlay, aspect="auto")
                ax.contour(mask_sl > 0, colors=[hex_col], linewidths=1.2)

            patch = mpatches.Patch(color=hex_col, label=f"{roi_name.replace('_', ' ')} ({int(mask.sum())} vox)")
            legend_patches.append(patch)

        ax.set_title(
            f"NeuroGenesis │ Complete Speech Network Overlay\nPatient: {patient_id} │ Axial Slice {cz}",
            color="white", fontsize=14, fontweight="bold", pad=12
        )
        ax.axis("off")
        ax.legend(handles=legend_patches, loc="upper right", facecolor=PANEL_COLOR,
                  edgecolor=BORDER_COLOR, labelcolor=TEXT_WHITE, fontsize=9)

        plt.tight_layout()
        out_path = out_dir / "speech_network_overlay.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR, edgecolor="none")
        plt.close(fig)

        logger.info(f"[Visualizer] Combined speech network overlay → {out_path}")
        return str(out_path)

    def plot_speech_network_multiview(
        self,
        data:       np.ndarray,
        masks:      Dict[str, np.ndarray],
        patient_id: str,
        subj_dir:   Optional[Path] = None,
    ) -> str:
        """
        Step 9: Generate speech_network_multiview.png displaying Axial, Coronal, and Sagittal views.
        """
        import matplotlib
        out_dir = Path(subj_dir) if subj_dir else self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        colors_map = {
            "Broca_Area":               "#ff4d4d",  # Red
            "Wernicke_Area":            "#3399ff",  # Blue
            "Insula":                   "#33cc33",  # Green
            "Inferior_Frontal_Gyrus":   "#ff9933",  # Orange
            "Superior_Temporal_Gyrus":  "#cc66ff",  # Purple
        }

        cx, cy, cz = [s // 2 for s in data.shape]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG_COLOR)

        plane_data = [
            (np.rot90(data[cx, :, :]), f"Sagittal View (x={cx})", 0, cx),
            (np.rot90(data[:, cy, :]), f"Coronal View (y={cy})", 1, cy),
            (np.rot90(data[:, :, cz]), f"Axial View (z={cz})", 2, cz),
        ]

        for ax, (sl_mri, title, plane_axis, idx) in zip(axes, plane_data):
            ax.imshow(sl_mri, cmap="bone", aspect="auto")

            for roi_name, mask in masks.items():
                hex_col = colors_map.get(roi_name, ACCENT_BLUE)
                rgb = matplotlib.colors.to_rgb(hex_col)

                if plane_axis == 0:
                    sl_mask = np.rot90(mask[idx, :, :])
                elif plane_axis == 1:
                    sl_mask = np.rot90(mask[:, idx, :])
                else:
                    sl_mask = np.rot90(mask[:, :, idx])

                if sl_mask.any():
                    overlay = np.zeros((*sl_mask.shape, 4), dtype=np.float32)
                    overlay[sl_mask > 0] = [*rgb, 0.55]
                    ax.imshow(overlay, aspect="auto")

            ax.set_title(title, color=ACCENT_BLUE, fontsize=11, fontweight="bold", pad=8)
            ax.axis("off")

        fig.suptitle(
            f"NeuroGenesis │ Multi-View Speech Network Visualisation │ Patient: {patient_id}",
            color="white", fontsize=14, fontweight="bold", y=1.02
        )
        plt.tight_layout()
        out_path = out_dir / "speech_network_multiview.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR, edgecolor="none")
        plt.close(fig)

        logger.info(f"[Visualizer] Multi-view speech network saved → {out_path}")
        return str(out_path)
