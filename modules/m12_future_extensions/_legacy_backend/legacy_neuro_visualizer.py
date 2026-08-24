"""
NeuroGenesis — Module 7: IEEE Journal-Grade End-to-End Visualizer
==================================================================
File   : backend/modules/07_visualization/neuro_visualizer.py
Purpose: Complete publication-quality figure generation pipeline for NeuroGenesis.
         Generates composite multi-stage figures covering:
         Original MRI → Processed MRI → Speech ROIs → Morphological Features →
         Brain Graph → Disease State Matrix → Propagation Readiness Heatmap.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from visualization.visualize import NeuroGenesisVisualizer

logger = logging.getLogger(__name__)

# Dark theme palette for IEEE Journal figures
STYLE_CONFIG = {
    "bg_color": "#0D1117",
    "card_color": "#161B22",
    "text_color": "#F0F6FC",
    "accent_teal": "#2A9D8F",
    "accent_red": "#E63946",
    "accent_gold": "#E9C46A",
    "accent_blue": "#1D3557"
}

class EndToEndNeuroVisualizer:
    """
    IEEE Journal-Grade Publication Visualizer
    Generates multi-panel pipeline composite figures and exportable plots.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir) / "visualization"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.legacy_visualizer = NeuroGenesisVisualizer(output_dir=self.output_dir)

    def generate_full_pipeline_figure(
        self,
        subject_id: str,
        original_mri: np.ndarray,
        processed_mri: np.ndarray,
        speech_masks: Dict[str, np.ndarray],
        features_df: pd.DataFrame,
        brain_graph: nx.DiGraph,
        disease_state_df: pd.DataFrame,
        readiness_df: pd.DataFrame
    ) -> Path:
        """
        Generate complete 7-stage master figure for paper publication.
        """
        logger.info(f"--- Generating Master Publication Figure for {subject_id} ---")
        fig = plt.figure(figsize=(20, 12), facecolor=STYLE_CONFIG["bg_color"])
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

        # 1. Original MRI
        ax1 = fig.add_subplot(gs[0, 0], facecolor=STYLE_CONFIG["card_color"])
        mid_z = original_mri.shape[2] // 2
        ax1.imshow(np.rot90(original_mri[:, :, mid_z]), cmap="gray")
        ax1.set_title("1. Original T1w MRI", color=STYLE_CONFIG["text_color"], fontsize=11, fontweight="bold")
        ax1.axis("off")

        # 2. Processed MRI
        ax2 = fig.add_subplot(gs[0, 1], facecolor=STYLE_CONFIG["card_color"])
        p_mid_z = processed_mri.shape[2] // 2
        ax2.imshow(np.rot90(processed_mri[:, :, p_mid_z]), cmap="bone")
        ax2.set_title("2. Preprocessed & Stripped", color=STYLE_CONFIG["text_color"], fontsize=11, fontweight="bold")
        ax2.axis("off")

        # 3. Speech ROIs Overlay
        ax3 = fig.add_subplot(gs[0, 2:], facecolor=STYLE_CONFIG["card_color"])
        ax3.imshow(np.rot90(processed_mri[:, :, p_mid_z]), cmap="gray", alpha=0.7)
        colors = ["#E63946", "#1D3557", "#2A9D8F", "#F4A261", "#9C27B0"]
        for idx, (roi_name, mask) in enumerate(speech_masks.items()):
            if mask.shape[2] > p_mid_z and np.sum(mask[:, :, p_mid_z]) > 0:
                ax3.contour(np.rot90(mask[:, :, p_mid_z]), levels=[0.5], colors=[colors[idx % len(colors)]], linewidths=1.5)
        ax3.set_title("3. Speech Region Segmentation Masks", color=STYLE_CONFIG["text_color"], fontsize=11, fontweight="bold")
        ax3.axis("off")

        # 4. Morphological Features Bar Chart
        ax4 = fig.add_subplot(gs[1, :2], facecolor=STYLE_CONFIG["card_color"])
        rois = features_df["roi_name"].values
        vols = features_df["regional_volume_mm3"].values
        ax4.barh(rois, vols, color=STYLE_CONFIG["accent_teal"])
        ax4.set_title("4. Morphological Regional Volumes (mm³)", color=STYLE_CONFIG["text_color"], fontsize=11, fontweight="bold")
        ax4.tick_params(colors=STYLE_CONFIG["text_color"])
        for spine in ax4.spines.values():
            spine.set_color(STYLE_CONFIG["card_color"])

        # 5. Brain Graph G=(V,E)
        ax5 = fig.add_subplot(gs[1, 2:], facecolor=STYLE_CONFIG["card_color"])
        pos = nx.circular_layout(brain_graph)
        nx.draw_networkx_nodes(brain_graph, pos, node_size=600, node_color=STYLE_CONFIG["accent_teal"], ax=ax5)
        nx.draw_networkx_labels(brain_graph, pos, font_color="white", font_size=8, ax=ax5)
        nx.draw_networkx_edges(brain_graph, pos, edge_color=STYLE_CONFIG["accent_blue"], width=2, ax=ax5)
        ax5.set_title("5. Speech Network Brain Graph G=(V,E)", color=STYLE_CONFIG["text_color"], fontsize=11, fontweight="bold")
        ax5.axis("off")

        # 6. Disease State Matrix Heatmap
        ax6 = fig.add_subplot(gs[2, :2], facecolor=STYLE_CONFIG["card_color"])
        im6 = ax6.imshow(disease_state_df.values, cmap="YlOrRd", aspect="auto")
        ax6.set_xticks(range(len(disease_state_df.columns)))
        ax6.set_xticklabels(disease_state_df.columns, rotation=30, ha="right", color=STYLE_CONFIG["text_color"], fontsize=8)
        ax6.set_yticks(range(len(disease_state_df.index)))
        ax6.set_yticklabels(disease_state_df.index, color=STYLE_CONFIG["text_color"], fontsize=8)
        ax6.set_title("6. NeuroProp-X Disease State Matrix S", color=STYLE_CONFIG["text_color"], fontsize=11, fontweight="bold")

        # 7. Propagation Readiness Heatmap
        ax7 = fig.add_subplot(gs[2, 2:], facecolor=STYLE_CONFIG["card_color"])
        im7 = ax7.imshow(readiness_df.values, cmap="magma", aspect="auto")
        ax7.set_xticks(range(len(readiness_df.columns)))
        ax7.set_xticklabels(readiness_df.columns, rotation=30, ha="right", color=STYLE_CONFIG["text_color"], fontsize=8)
        ax7.set_yticks(range(len(readiness_df.index)))
        ax7.set_yticklabels(readiness_df.index, color=STYLE_CONFIG["text_color"], fontsize=8)
        ax7.set_title("7. Adaptive Propagation Readiness Matrix R_ij", color=STYLE_CONFIG["text_color"], fontsize=11, fontweight="bold")

        out_path = self.output_dir / f"{subject_id}_master_pipeline_summary.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        logger.info(f"Saved master pipeline figure for {subject_id} to {out_path}")
        return out_path
