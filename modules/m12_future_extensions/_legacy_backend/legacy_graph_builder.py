"""
NeuroGenesis — Module 5: Dynamic Brain Graph Construction
=========================================================
File   : backend/modules/05_graph_construction/graph_builder.py
Purpose: Construct dynamic structural connectivity brain graph G=(V,E) for speech ROIs.
         Nodes store morphometric features & health scores; edges store tract connectivity & distance.
         Exports GraphML, NetworkX JSON/Pickle, and NumPy adjacency matrices.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from graph.graph_builder import BrainConnectivityGraph

logger = logging.getLogger(__name__)

# Known Speech Network Anatomical Connectivity (Normalized 0.0 - 1.0) & Distances (mm)
ANATOMICAL_TRACTS = [
    ("Broca_Area", "Wernicke_Area", {"weight": 0.85, "distance_mm": 65.0, "tract": "Arcuate Fasciculus / SLF"}),
    ("Wernicke_Area", "Broca_Area", {"weight": 0.85, "distance_mm": 65.0, "tract": "Arcuate Fasciculus / SLF"}),
    ("Broca_Area", "Insula", {"weight": 0.75, "distance_mm": 25.0, "tract": "Short Association Fibers"}),
    ("Insula", "Broca_Area", {"weight": 0.75, "distance_mm": 25.0, "tract": "Short Association Fibers"}),
    ("Broca_Area", "Inferior_Frontal_Gyrus", {"weight": 0.90, "distance_mm": 15.0, "tract": "Local Intra-gyral"}),
    ("Inferior_Frontal_Gyrus", "Broca_Area", {"weight": 0.90, "distance_mm": 15.0, "tract": "Local Intra-gyral"}),
    ("Wernicke_Area", "Superior_Temporal_Gyrus", {"weight": 0.92, "distance_mm": 18.0, "tract": "Local Intra-gyral"}),
    ("Superior_Temporal_Gyrus", "Wernicke_Area", {"weight": 0.92, "distance_mm": 18.0, "tract": "Local Intra-gyral"}),
    ("Insula", "Wernicke_Area", {"weight": 0.65, "distance_mm": 45.0, "tract": "Extreme Capsule System"}),
    ("Wernicke_Area", "Insula", {"weight": 0.65, "distance_mm": 45.0, "tract": "Extreme Capsule System"}),
    ("Inferior_Frontal_Gyrus", "Superior_Temporal_Gyrus", {"weight": 0.70, "distance_mm": 70.0, "tract": "IFOF"}),
    ("Superior_Temporal_Gyrus", "Inferior_Frontal_Gyrus", {"weight": 0.70, "distance_mm": 70.0, "tract": "IFOF"}),
]

class DynamicBrainGraphBuilder:
    """
    Dynamic Brain Connectivity Graph Builder Module.
    Constructs, enriches, and exports structural speech network graphs.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir) / "graphs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.legacy_builder = BrainConnectivityGraph(output_dir=self.output_dir)

    def build_graph(
        self,
        features_df: pd.DataFrame,
        subject_id: str,
        timestamp: float = 0.0
    ) -> Tuple[nx.DiGraph, np.ndarray]:
        """
        Build NetworkX DiGraph G=(V,E) enriched with ROI features and structural edges.
        """
        logger.info(f"--- Constructing Speech Brain Graph for {subject_id} (t={timestamp}) ---")
        G = nx.DiGraph(subject_id=subject_id, timestamp=timestamp)

        # 1. Add Nodes with Features
        for _, row in features_df.iterrows():
            roi_name = str(row["roi_name"])
            vol = float(row.get("regional_volume_mm3", row.get("brain_volume_mm3", 0.0)))
            sa = float(row.get("surface_area_mm2", row.get("surface_area_vox", 0.0)))
            thick = float(row.get("cortical_thickness_mm", 2.5))
            if np.isnan(thick) or thick <= 0:
                thick = float(vol / (sa + 1e-5)) if sa > 0 else 2.5

            G.add_node(
                roi_name,
                region_name=roi_name,
                volume=vol,
                gm_volume=float(row.get("gm_volume_mm3", 0.0)),
                thickness=thick,
                surface_area=sa,
                atrophy_index=float(row.get("atrophy_index", vol / 1450.0)),
                mean_intensity=float(row.get("mean_intensity", 0.0)),
                entropy=float(row.get("voxel_entropy", row.get("entropy", 0.0))),
                health_score=100.0,            # Default placeholder; updated by NeuroProp-X
                speech_score=100.0            # Default placeholder; updated by NeuroProp-X
            )


        # 2. Add Structural Edges
        for u, v, data in ANATOMICAL_TRACTS:
            if u in G and v in G:
                G.add_edge(
                    u, v,
                    connectivity_weight=float(data["weight"]),
                    distance_mm=float(data["distance_mm"]),
                    tract=data["tract"],
                    readiness_score=0.0       # Placeholder; updated by NeuroProp-X
                )

        # 3. Export Formats
        roi_order = sorted(list(G.nodes()))
        adj_matrix = nx.to_numpy_array(G, nodelist=roi_order, weight="connectivity_weight")

        out_dir = self.output_dir / subject_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # Export GraphML
        nx.write_graphml(G, out_dir / f"{subject_id}_brain_graph.graphml")

        # Export JSON
        graph_data = nx.node_link_data(G)
        with open(out_dir / f"{subject_id}_brain_graph.json", "w") as f:
            json.dump(graph_data, f, indent=2)

        # Export NumPy Adjacency Matrix
        np.save(out_dir / f"{subject_id}_adjacency_matrix.npy", adj_matrix)
        pd.DataFrame(adj_matrix, index=roi_order, columns=roi_order).to_csv(out_dir / f"{subject_id}_adjacency_matrix.csv")

        # Generate Graph Plot
        self._visualize_graph(G, subject_id, out_dir)

        logger.info(f"Successfully constructed Brain Graph for {subject_id} with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

        return G, adj_matrix

    def _visualize_graph(self, G: nx.DiGraph, subject_id: str, out_dir: Path):
        """Draw publication-style circular graph visualization."""
        fig, ax = plt.subplots(figsize=(8, 8))
        pos = nx.circular_layout(G)

        # Node sizes based on volume
        node_sizes = [G.nodes[n].get("volume", 5000.0) / 10.0 for n in G.nodes()]

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="#2A9D8F", alpha=0.9, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", font_family="sans-serif", ax=ax)

        # Draw edges with varying width
        edge_widths = [G.edges[e].get("connectivity_weight", 0.5) * 3.0 for e in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color="#1D3557", alpha=0.6, arrowsize=15, ax=ax)

        ax.set_title(f"NeuroGenesis Speech Connectivity Graph G=(V,E) — {subject_id}", fontsize=12, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / f"{subject_id}_brain_graph.png", dpi=150)
        plt.close(fig)
