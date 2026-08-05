"""
NeuroGenesis — Module 6: NeuroProp-X Core Engine
=================================================
File   : backend/modules/06_neuropropx/engine.py
Purpose: CORE FRAMEWORK — Intelligent Disease Graph Evolution Framework.
         Constructs DRHE, Disease State Vectors (DSV node embeddings),
         Adaptive Disease Propagation Readiness Layer, Longitudinal Velocity/Acceleration
         architectural stubs, Progressive Graph Memory Buffer, and full output matrices.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Expected normative baseline volumes for speech regions in healthy control (mm³)
NORMATIVE_VOLUMES = {
    "Broca_Area": 12500.0,
    "Wernicke_Area": 11000.0,
    "Insula": 13500.0,
    "Inferior_Frontal_Gyrus": 18000.0,
    "Superior_Temporal_Gyrus": 16500.0
}

@dataclass
class GraphTimestampState:
    timestamp: float
    subject_id: str
    disease_state_matrix: np.ndarray
    adjacency_matrix: np.ndarray
    propagation_readiness_matrix: np.ndarray
    velocity_matrix: Optional[np.ndarray] = None
    acceleration_matrix: Optional[np.ndarray] = None

class ProgressiveGraphMemory:
    """
    Progressive Graph Memory Buffer (Module 6.6)
    Stores timestamped graph states: Graph_t0, Graph_t1, Graph_t2, ...
    """
    def __init__(self, subject_id: str):
        self.subject_id = subject_id
        self.buffer: List[GraphTimestampState] = []

    def push(self, state: GraphTimestampState):
        self.buffer.append(state)

    def get_latest(self) -> Optional[GraphTimestampState]:
        return self.buffer[-1] if self.buffer else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "timestamps_count": len(self.buffer),
            "timestamps": [s.timestamp for s in self.buffer]
        }

class NeuroPropXEngine:
    """
    NeuroProp-X Framework (Module 6)
    Intelligent Disease Graph Evolution & Embedding Generator
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir) / "neuropropx"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process(
        self,
        G: nx.DiGraph,
        features_df: pd.DataFrame,
        subject_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        previous_memory: Optional[ProgressiveGraphMemory] = None
    ) -> Dict[str, Any]:
        """
        Execute full NeuroProp-X engine processing pipeline.
        """
        logger.info(f"=== Running NeuroProp-X Core Engine for {subject_id} ===")
        roi_order = sorted(list(G.nodes()))
        num_rois = len(roi_order)

        # 6.1 Dynamic Regional Health Estimation (DRHE)
        health_scores: Dict[str, float] = {}
        for roi in roi_order:
            node_data = G.nodes[roi]
            vol = node_data.get("volume", 10000.0)
            norm_vol = NORMATIVE_VOLUMES.get(roi, 12000.0)
            vol_ratio = np.clip(vol / norm_vol, 0.2, 1.2)
            
            entropy = node_data.get("entropy", 3.0)
            entropy_factor = np.clip(1.0 - (entropy / 6.0), 0.5, 1.0)
            
            # Health Score H_i in range 0 - 100
            h_score = float(np.clip(vol_ratio * entropy_factor * 100.0, 0.0, 100.0))
            health_scores[roi] = h_score
            
            # Store in graph node
            G.nodes[roi]["health_score"] = h_score
            G.nodes[roi]["speech_score"] = float(np.clip(h_score * 0.95 + 5.0, 0.0, 100.0))

        # 6.2 Disease State Vector (DSV) & Node Embedding Matrix
        # Vector: [Health Score, Volume, Thickness, Surface Area, Atrophy Index, Texture Contrast, Connectivity Importance]
        dsv_matrix = np.zeros((num_rois, 7), dtype=np.float32)
        
        for idx, roi in enumerate(roi_order):
            nd = G.nodes[roi]
            conn_imp = float(sum(G.edges[e].get("connectivity_weight", 0.0) for e in G.out_edges(roi)))
            
            dsv = np.array([
                nd.get("health_score", 100.0),
                nd.get("volume", 0.0),
                nd.get("thickness", 2.5),
                nd.get("surface_area", 0.0),
                nd.get("atrophy_index", 0.0),
                nd.get("texture_contrast", 0.0),
                conn_imp
            ], dtype=np.float32)
            
            dsv_matrix[idx, :] = dsv
            G.nodes[roi]["dsv_vector"] = dsv.tolist()

        # 6.3 & 6.4 Disease Velocity & Acceleration Frameworks
        # If single scan (OASIS-1): Velocity = Unknown, Acceleration = Unknown
        if previous_memory and len(previous_memory.buffer) > 0:
            last_state = previous_memory.get_latest()
            dt = 1.0  # Time step
            velocity_matrix = (dsv_matrix - last_state.disease_state_matrix) / dt
            acceleration_matrix = np.zeros_like(dsv_matrix) if last_state.velocity_matrix is None else (velocity_matrix - last_state.velocity_matrix) / dt
            velocity_status = "Computed from longitudinal history"
        else:
            velocity_matrix = np.full_like(dsv_matrix, np.nan)
            acceleration_matrix = np.full_like(dsv_matrix, np.nan)
            velocity_status = "Velocity = Unknown (Single OASIS-1 scan)"

        # 6.5 Adaptive Disease Propagation Layer
        # Compute Propagation Readiness Score R_ij between regions i and j
        readiness_matrix = np.zeros((num_rois, num_rois), dtype=np.float32)
        
        for i, roi_i in enumerate(roi_order):
            h_i = health_scores[roi_i]
            vulnerability_i = (100.0 - h_i) / 100.0
            
            for j, roi_j in enumerate(roi_order):
                if i == j:
                    continue
                if G.has_edge(roi_i, roi_j):
                    edge_data = G.edges[roi_i, roi_j]
                    weight = edge_data.get("connectivity_weight", 0.5)
                    dist = edge_data.get("distance_mm", 50.0)
                    
                    # Readiness Score formula
                    r_ij = float((vulnerability_i * weight * 100.0) / (dist / 20.0 + 1.0))
                    readiness_matrix[i, j] = r_ij
                    G.edges[roi_i, roi_j]["readiness_score"] = r_ij

        # 6.6 Progressive Graph Memory Buffer
        memory = previous_memory if previous_memory else ProgressiveGraphMemory(subject_id=subject_id)
        adj_matrix = nx.to_numpy_array(G, nodelist=roi_order, weight="connectivity_weight")
        
        current_state = GraphTimestampState(
            timestamp=0.0,
            subject_id=subject_id,
            disease_state_matrix=dsv_matrix,
            adjacency_matrix=adj_matrix,
            propagation_readiness_matrix=readiness_matrix,
            velocity_matrix=velocity_matrix,
            acceleration_matrix=acceleration_matrix
        )
        memory.push(current_state)

        # 6.7 NeuroProp-X Output Export
        out_dir = self.output_dir / subject_id
        out_dir.mkdir(parents=True, exist_ok=True)

        dsv_df = pd.DataFrame(dsv_matrix, index=roi_order, columns=[
            "Health_Score", "Volume_mm3", "Thickness_mm", "Surface_Area_mm2",
            "Atrophy_Index", "Texture_Contrast", "Connectivity_Importance"
        ])
        dsv_df.to_csv(out_dir / f"{subject_id}_disease_state_matrix.csv")
        np.save(out_dir / f"{subject_id}_disease_state_matrix.npy", dsv_matrix)

        readiness_df = pd.DataFrame(readiness_matrix, index=roi_order, columns=roi_order)
        readiness_df.to_csv(out_dir / f"{subject_id}_propagation_readiness.csv")
        np.save(out_dir / f"{subject_id}_propagation_readiness.npy", readiness_matrix)

        # Node Embeddings (Same as DSV matrix for downstream GNN/Transformer)
        np.save(out_dir / f"{subject_id}_node_embeddings.npy", dsv_matrix)

        # Manifest JSON
        manifest = {
            "subject_id": subject_id,
            "velocity_status": velocity_status,
            "acceleration_status": velocity_status,
            "speech_rois": roi_order,
            "health_scores": health_scores,
            "dsv_shape": list(dsv_matrix.shape),
            "readiness_matrix_shape": list(readiness_matrix.shape),
            "graph_memory": memory.to_dict()
        }
        with open(out_dir / f"{subject_id}_neuropropx_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        # Generate Visual Heatmaps
        self._generate_neuropropx_heatmaps(subject_id, dsv_df, readiness_df, out_dir)

        logger.info(f"=== NeuroProp-X Execution Complete for {subject_id} ===")

        return {
            "disease_state_matrix": dsv_matrix,
            "disease_state_df": dsv_df,
            "propagation_readiness_matrix": readiness_matrix,
            "propagation_readiness_df": readiness_df,
            "health_scores": health_scores,
            "velocity_status": velocity_status,
            "graph_memory": memory,
            "manifest": manifest
        }

    def _generate_neuropropx_heatmaps(self, subject_id: str, dsv_df: pd.DataFrame, readiness_df: pd.DataFrame, out_dir: Path):
        """Generate publication-grade heatmaps for Disease State Matrix and Propagation Readiness."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"NeuroProp-X Engine Representation — {subject_id}", fontsize=14, fontweight="bold")

        # Heatmap 1: Disease State Matrix
        im1 = axes[0].imshow(dsv_df.values, cmap="YlOrRd", aspect="auto")
        axes[0].set_xticks(range(len(dsv_df.columns)))
        axes[0].set_xticklabels(dsv_df.columns, rotation=45, ha="right", fontsize=8)
        axes[0].set_yticks(range(len(dsv_df.index)))
        axes[0].set_yticklabels(dsv_df.index, fontsize=9)
        axes[0].set_title("Disease State Matrix S (Node Embeddings)")
        plt.colorbar(im1, ax=axes[0])

        # Heatmap 2: Propagation Readiness Matrix
        im2 = axes[1].imshow(readiness_df.values, cmap="magma", aspect="auto")
        axes[1].set_xticks(range(len(readiness_df.columns)))
        axes[1].set_xticklabels(readiness_df.columns, rotation=45, ha="right", fontsize=8)
        axes[1].set_yticks(range(len(readiness_df.index)))
        axes[1].set_yticklabels(readiness_df.index, fontsize=9)
        axes[1].set_title("Propagation Readiness Matrix R_ij")
        plt.colorbar(im2, ax=axes[1])

        plt.tight_layout()
        plt.savefig(out_dir / f"{subject_id}_neuropropx_heatmaps.png", dpi=150)
        plt.close(fig)
