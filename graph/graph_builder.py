"""
NeuroGenesis — Brain Connectivity Graph Builder
=================================================
Module: graph/graph_builder.py
Author: NeuroGenesis Research Team
Phase : 1 (Preprocessing Pipeline)

Constructs a directed, weighted graph of known anatomical speech-network
connectivity using NetworkX. The graph encodes:

    Nodes : Five speech-related ROIs with morphometric attributes
    Edges : Known anatomical white-matter tracts with directional weights

Anatomical Connectivity Basis (from DTI/tractography literature):
    ┌─────────────────────────────────────────────────────────┐
    │  Broca ←──── Superior Longitudinal Fasciculus ───────► │
    │    │                                                  Wernicke │
    │    ▼                                                     │    │
    │  Insula ◄──── Extreme Capsule Fibre System ────────►  STG │
    │    │                                                     │    │
    │    └──────────────► IFG ◄────── IFOF ──────────────────┘    │
    └─────────────────────────────────────────────────────────────┘

Edge Weights:
    - Based on published DTI tract strength / FA values (normalised 0–1)
    - Higher weight = stronger anatomical connection
    - Can be updated in Phase 2 with subject-specific DTI connectivity

Outputs:
    - NetworkX directed graph (returned and used in-memory)
    - Publication-quality graph figure (outputs/graphs/)
    - Adjacency matrix CSV (outputs/graphs/)

NeuroProp-X Hook:
    get_graph_data() → (nx.DiGraph, adjacency_matrix_np)

Usage:
    builder = BrainConnectivityGraph(output_dir=Path("outputs/graphs"))
    graph   = builder.build(roi_features=feature_dict, patient_id="OAS1_0001")
    builder.visualize(patient_id="OAS1_0001")
    builder.save_adjacency_matrix(patient_id="OAS1_0001")
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Anatomical speech network definition
# ──────────────────────────────────────────────────────────────────────────────

#: Ordered list of speech ROI node identifiers
SPEECH_NODES: List[str] = [
    "Broca_Area",
    "Wernicke_Area",
    "Insula",
    "Inferior_Frontal_Gyrus",
    "Superior_Temporal_Gyrus",
]

#: Node display labels and metadata
NODE_METADATA: Dict[str, Dict[str, Any]] = {
    "Broca_Area": {
        "label":       "Broca\n(IFG BA44/45)",
        "function":    "Speech production, phonological encoding",
        "hemisphere":  "Left",
        "brodmann":    "44, 45",
        "color":       "#388bfd",
    },
    "Wernicke_Area": {
        "label":       "Wernicke\n(pSTG BA22)",
        "function":    "Speech comprehension, auditory word recognition",
        "hemisphere":  "Left",
        "brodmann":    "22",
        "color":       "#3fb950",
    },
    "Insula": {
        "label":       "Insula\n(Insular Cortex)",
        "function":    "Articulatory planning, phonological awareness",
        "hemisphere":  "Bilateral",
        "brodmann":    "13, 14",
        "color":       "#e3b341",
    },
    "Inferior_Frontal_Gyrus": {
        "label":       "IFG\n(Frontal Operculum)",
        "function":    "Syntactic processing, working memory for language",
        "hemisphere":  "Left",
        "brodmann":    "44, 45, 47",
        "color":       "#ff7b72",
    },
    "Superior_Temporal_Gyrus": {
        "label":       "STG\n(Planum Temporale)",
        "function":    "Auditory-verbal processing, spectrotemporal analysis",
        "hemisphere":  "Bilateral",
        "brodmann":    "22, 41, 42",
        "color":       "#d2a8ff",
    },
}

#: Anatomical edges: (source, target, weight, tract_name)
#: Weights based on normalised DTI fractional anisotropy values from literature
ANATOMICAL_EDGES: List[Tuple[str, str, float, str]] = [
    # Superior Longitudinal Fasciculus (SLF/AF) — core speech loop
    ("Broca_Area",            "Wernicke_Area",          0.85, "Superior_Longitudinal_Fasciculus"),
    ("Wernicke_Area",         "Broca_Area",             0.85, "Superior_Longitudinal_Fasciculus"),

    # Insula connections — articulatory relay
    ("Broca_Area",            "Insula",                 0.75, "Short_Arcuate_Fibres"),
    ("Insula",                "Broca_Area",             0.70, "Short_Arcuate_Fibres"),
    ("Insula",                "Wernicke_Area",          0.65, "Extreme_Capsule_System"),
    ("Wernicke_Area",         "Insula",                 0.60, "Extreme_Capsule_System"),

    # STG connections — auditory input route
    ("Superior_Temporal_Gyrus", "Wernicke_Area",        0.90, "Intra_STG_Fibres"),
    ("Wernicke_Area",         "Superior_Temporal_Gyrus",0.80, "Intra_STG_Fibres"),
    ("Superior_Temporal_Gyrus", "Insula",               0.55, "Extreme_Capsule_System"),

    # IFG connections — frontal syntactic network
    ("Inferior_Frontal_Gyrus",  "Broca_Area",           0.95, "Within_IFG_Fibres"),
    ("Broca_Area",            "Inferior_Frontal_Gyrus", 0.90, "Within_IFG_Fibres"),
    ("Inferior_Frontal_Gyrus",  "Insula",               0.65, "Inferior_Fronto_Occipital_Fasciculus"),
    ("Inferior_Frontal_Gyrus",  "Superior_Temporal_Gyrus", 0.50, "Inferior_Fronto_Occipital_Fasciculus"),
]


class BrainConnectivityGraph:
    """
    Construct and visualise the speech brain connectivity graph.

    Builds a NetworkX DiGraph with five speech-region nodes and twelve
    anatomically-grounded weighted directed edges. Designed to serve as the
    graph-structured input to the NeuroProp-X graph encoder in Phase 2.

    Args:
        output_dir : Directory for saving graph figures and adjacency matrix.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._graphs: Dict[str, nx.DiGraph] = {}

        logger.info(f"[BrainConnectivityGraph] Initialised │ output={self.output_dir}")

    # ──────────────────────────────────────────────────────────────────────────
    # Public
    # ──────────────────────────────────────────────────────────────────────────

    def build(
        self,
        patient_id:   str,
        roi_features: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> nx.DiGraph:
        """
        Build the speech connectivity graph for one subject.

        Nodes are annotated with anatomical metadata and (optionally) subject-
        specific morphometric features extracted by FeatureExtractor. Edges
        carry anatomical weight and tract name attributes.

        Args:
            patient_id   : Subject identifier.
            roi_features : Optional dict mapping roi_name → feature dict.
                           If provided, volume/intensity stats are added as
                           node attributes for NeuroProp-X conditioning.

        Returns:
            NetworkX DiGraph with annotated nodes and weighted edges.
        """
        logger.info(f"[BrainConnectivityGraph] Building graph for [{patient_id}]")
        G = nx.DiGraph(patient_id=patient_id)

        # ── Add nodes ─────────────────────────────────────────────────────
        for node_id in SPEECH_NODES:
            attrs = dict(NODE_METADATA[node_id])  # copy metadata

            # Attach subject-specific features if available
            if roi_features and node_id in roi_features:
                for k, v in roi_features[node_id].items():
                    attrs[f"feat_{k}"] = v

            G.add_node(node_id, **attrs)

        # ── Add edges ─────────────────────────────────────────────────────
        for src, dst, weight, tract in ANATOMICAL_EDGES:
            G.add_edge(src, dst, weight=weight, tract=tract)

        self._graphs[patient_id] = G

        logger.info(
            f"[BrainConnectivityGraph] ✓ Graph built │ "
            f"nodes={G.number_of_nodes()} │ edges={G.number_of_edges()}"
        )
        return G

    def visualize(
        self,
        patient_id: str,
        save:       bool = True,
    ) -> Optional[str]:
        """
        Generate a publication-quality graph visualisation.

        Uses a manually specified circular layout so that nodes are always
        placed in the same relative positions for multi-subject comparison.

        Args:
            patient_id : Subject whose graph to visualise.
            save       : Save figure to *output_dir*.

        Returns:
            Path to saved figure, or *None* if graph not built.
        """
        if patient_id not in self._graphs:
            logger.error(f"[BrainConnectivityGraph] No graph found for {patient_id}.")
            return None

        G = self._graphs[patient_id]

        # ── Manual layout — circular, consistent across subjects ───────────
        n = len(SPEECH_NODES)
        angles = [2 * np.pi * i / n for i in range(n)]
        pos = {
            node: (np.cos(a) * 2.2, np.sin(a) * 2.2)
            for node, a in zip(SPEECH_NODES, angles)
        }

        fig, ax = plt.subplots(figsize=(14, 12), facecolor="#0d1117")
        ax.set_facecolor("#0d1117")
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)

        fig.suptitle(
            f"NeuroGenesis │ Speech Brain Connectivity Graph │ {patient_id}",
            color="white", fontsize=15, fontweight="bold", y=0.98
        )

        # ── Draw edges ────────────────────────────────────────────────────
        weights = [G[u][v]["weight"] for u, v in G.edges()]
        max_w   = max(weights) if weights else 1.0

        for (u, v), w in zip(G.edges(), weights):
            x_src, y_src = pos[u]
            x_dst, y_dst = pos[v]

            # Curved arrow using matplotlib FancyArrowPatch
            lw = 0.5 + 3.0 * (w / max_w)
            alpha = 0.4 + 0.5 * (w / max_w)
            color_val = plt.cm.plasma(w / max_w)

            ax.annotate(
                "",
                xy=(x_dst, y_dst),
                xytext=(x_src, y_src),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color_val,
                    lw=lw,
                    alpha=alpha,
                    connectionstyle="arc3,rad=0.12",
                    mutation_scale=15,
                ),
            )

            # Edge weight label at midpoint
            mx = (x_src + x_dst) / 2
            my = (y_src + y_dst) / 2
            ax.text(mx, my, f"{w:.2f}", color="#c9d1d9", fontsize=7,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="#21262d",
                              ec="#30363d", alpha=0.8))

        # ── Draw nodes — size & saturation driven by per-subject health scores ──
        for node_id in SPEECH_NODES:
            x, y = pos[node_id]
            base_color = NODE_METADATA[node_id]["color"]
            label = NODE_METADATA[node_id]["label"]

            node_data = G.nodes.get(node_id, {})
            h_score = float(node_data.get("health_score", 75.0))
            vol = float(node_data.get("feat_brain_volume_mm3", node_data.get("volume", 25000.0)))

            # Node radius scales with regional volume (bigger = more tissue preserved)
            vol_norm = np.clip(vol / 35000.0, 0.3, 1.0)
            radius = 0.30 + 0.28 * vol_norm

            # Health score drives alpha (dimmer = more atrophied)
            alpha = float(np.clip(0.45 + 0.55 * (h_score / 100.0), 0.45, 1.0))

            # Ring width scales with health score
            ring_lw = 1.0 + 3.0 * (h_score / 100.0)

            circle = plt.Circle((x, y), radius, color=base_color, alpha=alpha,
                                 zorder=5, linewidth=2, fill=True)
            ax.add_patch(circle)

            ring = plt.Circle((x, y), radius + 0.10, color=base_color,
                               alpha=alpha * 0.4, zorder=4, fill=False,
                               linewidth=ring_lw)
            ax.add_patch(ring)

            ax.text(x, y, label, ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold",
                    zorder=6, multialignment="center")

            # Health score badge below node
            badge_color = "#22c55e" if h_score >= 75 else ("#f59e0b" if h_score >= 50 else "#ef4444")
            ax.text(x, y - radius - 0.18, f"H={h_score:.1f}",
                    ha="center", va="top", color=badge_color,
                    fontsize=7, fontweight="bold", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.12", fc="#0d1117", ec=badge_color, alpha=0.85))


        # ── Colorbar (edge weight) ────────────────────────────────────────
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.plasma,
            norm=plt.Normalize(vmin=0, vmax=1)
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, shrink=0.6)
        cbar.set_label("Edge Weight\n(Tract Strength)", color="white", fontsize=9)
        cbar.ax.tick_params(colors="white")

        # ── Legend ────────────────────────────────────────────────────────
        legend_patches = [
            mpatches.Patch(color=NODE_METADATA[n]["color"], alpha=0.9,
                           label=NODE_METADATA[n]["label"].replace("\n", " "))
            for n in SPEECH_NODES
        ]
        ax.legend(handles=legend_patches, loc="lower left",
                  framealpha=0.2, labelcolor="white",
                  fontsize=8, title="Speech ROIs",
                  title_fontsize=9)

        save_path: Optional[str] = None
        if save:
            out_path = self.output_dir / f"{patient_id}_connectivity_graph.png"
            fig.savefig(out_path, dpi=300, bbox_inches="tight",
                        facecolor="#0d1117", edgecolor="none")
            save_path = str(out_path)
            logger.info(f"[BrainConnectivityGraph] Graph figure saved → {out_path}")

        plt.close(fig)
        return save_path

    def save_adjacency_matrix(self, patient_id: str) -> Dict[str, str]:
        """
        Save the graph adjacency matrix as CSV and a heatmap figure.

        Args:
            patient_id : Subject whose graph to serialise.

        Returns:
            Dict with keys 'csv' and 'heatmap' → file paths.
        """
        if patient_id not in self._graphs:
            logger.error(f"[BrainConnectivityGraph] No graph for {patient_id}.")
            return {}

        G = self._graphs[patient_id]
        nodes = list(G.nodes())
        n = len(nodes)
        adj = np.zeros((n, n), dtype=np.float32)

        node_idx = {node: i for i, node in enumerate(nodes)}
        for u, v, data in G.edges(data=True):
            adj[node_idx[u], node_idx[v]] = data.get("weight", 1.0)

        # CSV
        df = pd.DataFrame(adj, index=nodes, columns=nodes)
        csv_path = self.output_dir / f"{patient_id}_adjacency_matrix.csv"
        df.to_csv(csv_path)
        logger.info(f"[BrainConnectivityGraph] Adjacency matrix CSV → {csv_path}")

        # Heatmap figure
        heatmap_path = self._save_adjacency_heatmap(adj, nodes, patient_id)

        return {"csv": str(csv_path), "heatmap": heatmap_path}

    def save_graph_json(self, patient_id: str) -> str:
        """
        Serialise the graph to JSON (node-link format).

        Args:
            patient_id : Subject identifier.

        Returns:
            Path to saved JSON.
        """
        if patient_id not in self._graphs:
            logger.error(f"[BrainConnectivityGraph] No graph for {patient_id}.")
            return ""

        G = self._graphs[patient_id]
        data = nx.node_link_data(G)
        path = self.output_dir / f"{patient_id}_graph.json"
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2, default=str)
        logger.info(f"[BrainConnectivityGraph] Graph JSON saved → {path}")
        return str(path)

    # ──────────────────────────────────────────────────────────────────────────
    # NeuroProp-X integration hook
    # ──────────────────────────────────────────────────────────────────────────

    def get_graph_data(
        self, patient_id: str
    ) -> Optional[Tuple[nx.DiGraph, np.ndarray]]:
        """
        Return the graph and adjacency matrix for NeuroProp-X graph encoder.

        In Phase 2, NeuroProp-X's graph encoder (e.g. GCN or GAT) will consume
        these objects directly.

        Args:
            patient_id : Subject identifier.

        Returns:
            Tuple of (nx.DiGraph, float32 adjacency matrix ndarray), or *None*.

        .. note::
            This is the primary graph hand-off for NeuroProp-X Phase 2.
        """
        if patient_id not in self._graphs:
            logger.error(
                f"[BrainConnectivityGraph] No graph for '{patient_id}'. "
                "Call build() first."
            )
            return None

        G = self._graphs[patient_id]
        adj = nx.to_numpy_array(G, nodelist=SPEECH_NODES,
                                weight="weight").astype(np.float32)
        return G, adj

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _save_adjacency_heatmap(
        self,
        adj:        np.ndarray,
        nodes:      List[str],
        patient_id: str,
    ) -> str:
        """Save a publication-quality adjacency matrix heatmap."""
        fig, ax = plt.subplots(figsize=(8, 7), facecolor="#0d1117")
        im = ax.imshow(adj, cmap="YlOrRd", aspect="auto",
                       vmin=0, vmax=1)

        short_labels = [
            NODE_METADATA.get(n, {}).get("label", n).split("\n")[0]
            for n in nodes
        ]

        ax.set_xticks(range(len(nodes)))
        ax.set_yticks(range(len(nodes)))
        ax.set_xticklabels(short_labels, rotation=30, ha="right",
                           color="white", fontsize=9)
        ax.set_yticklabels(short_labels, color="white", fontsize=9)
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="white")

        # Annotate cells with weight values
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if adj[i, j] > 0:
                    ax.text(j, i, f"{adj[i,j]:.2f}",
                            ha="center", va="center",
                            color="black" if adj[i,j] > 0.5 else "white",
                            fontsize=8, fontweight="bold")

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Connection Strength", color="white", fontsize=10)
        cbar.ax.tick_params(colors="white")

        ax.set_title(
            f"Speech Network Adjacency Matrix │ {patient_id}",
            color="white", fontsize=12, fontweight="bold", pad=12
        )

        plt.tight_layout()
        path = self.output_dir / f"{patient_id}_adjacency_heatmap.png"
        fig.savefig(path, dpi=300, bbox_inches="tight",
                    facecolor="#0d1117", edgecolor="none")
        plt.close(fig)
        logger.info(f"[BrainConnectivityGraph] Adjacency heatmap → {path}")
        return str(path)
