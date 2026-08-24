"""
NeuroGenesis — Module 6: NeuroProp-X Core Engine
=================================================
File   : backend/modules/06_neuropropx/engine.py
Purpose: CORE FRAMEWORK — Dynamic Regional Vulnerability Estimation (DRVE),
         Adaptive Neurodegeneration Propagation Engine (ANPE),
         Temporal Disease Memory (TDM), and Progressive Risk Refinement (PRR).
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
    """Progressive Graph Memory Buffer (Module 6.6)"""
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
    NeuroProp-X Framework (Module 5 & 6)
    Dynamic Regional Vulnerability Estimation (DRVE), ANPE, TDM, and PRR.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir) / "neuropropx"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compute_drve(self, G: nx.DiGraph, speech_scores: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Module 5.1: Dynamic Regional Vulnerability Estimation (DRVE)"""
        vulnerability_scores = {}
        for roi in G.nodes():
            nd = G.nodes[roi]
            vol = nd.get("volume", 10000.0)
            norm_vol = NORMATIVE_VOLUMES.get(roi, 12000.0)
            vol_ratio = np.clip(vol / norm_vol, 0.2, 1.2)
            entropy = nd.get("entropy", 3.0)
            entropy_factor = np.clip(1.0 - (entropy / 6.0), 0.5, 1.0)
            
            speech_factor = (speech_scores.get("overall_speech_index", 85.0) / 100.0) if speech_scores else 1.0
            h_score = float(np.clip(vol_ratio * entropy_factor * speech_factor * 100.0, 0.0, 100.0))
            vuln_score = float(np.clip(1.0 - (h_score / 100.0), 0.0, 1.0))
            
            vulnerability_scores[roi] = vuln_score
            G.nodes[roi]["health_score"] = h_score
            G.nodes[roi]["vulnerability_score"] = vuln_score
        return vulnerability_scores

    def compute_anpe(self, G: nx.DiGraph, drve_scores: Dict[str, float]) -> np.ndarray:
        """Module 5.2: Adaptive Neurodegeneration Propagation Engine (ANPE)"""
        roi_order = sorted(list(G.nodes()))
        n = len(roi_order)
        p_matrix = np.zeros((n, n), dtype=np.float32)
        
        for i, r_i in enumerate(roi_order):
            v_i = drve_scores.get(r_i, 0.2)
            for j, r_j in enumerate(roi_order):
                if i == j:
                    continue
                if G.has_edge(r_i, r_j):
                    edge_w = G.edges[r_i, r_j].get("connectivity_weight", 0.5)
                    dist = G.edges[r_i, r_j].get("distance_mm", 50.0)
                    v_j = drve_scores.get(r_j, 0.2)
                    prob = (v_i * 0.6 + v_j * 0.4) * edge_w * (50.0 / (dist + 10.0))
                    p_matrix[i, j] = float(np.clip(prob, 0.0, 1.0))
        return p_matrix

    def compute_tdm(self, longitudinal_states: List[np.ndarray]) -> Dict[str, Any]:
        """Module 5.3: Temporal Disease Memory (TDM)"""
        if not longitudinal_states:
            return {"num_visits": 0, "progression_rate": 0.0}
        num_visits = len(longitudinal_states)
        if num_visits == 1:
            return {"num_visits": 1, "progression_rate": 0.0, "trajectory": [0.0]}
        deltas = [np.mean(longitudinal_states[k] - longitudinal_states[k-1]) for k in range(1, num_visits)]
        prog_rate = float(np.mean(deltas))
        return {
            "num_visits": num_visits,
            "progression_rate": round(prog_rate, 4),
            "deltas": [round(float(d), 4) for d in deltas]
        }

    def compute_prr(self, G: nx.DiGraph, anpe_matrix: np.ndarray) -> nx.DiGraph:
        """Module 5.4: Progressive Risk Refinement (PRR)"""
        G_refined = G.copy()
        roi_order = sorted(list(G.nodes()))
        for i, r_i in enumerate(roi_order):
            for j, r_j in enumerate(roi_order):
                if G_refined.has_edge(r_i, r_j):
                    G_refined.edges[r_i, r_j]["propagation_risk"] = float(anpe_matrix[i, j])
        return G_refined

    def process(
        self,
        G: nx.DiGraph,
        features_df: pd.DataFrame,
        subject_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        previous_memory: Optional[ProgressiveGraphMemory] = None
    ) -> Dict[str, Any]:
        """Execute full NeuroProp-X engine processing pipeline."""
        logger.info(f"=== Running NeuroProp-X Core Engine for {subject_id} ===")
        roi_order = sorted(list(G.nodes()))
        num_rois = len(roi_order)

        # 1. Health & Vulnerability Scores
        drve_scores = self.compute_drve(G)
        health_scores = {r: float(np.clip((1.0 - v) * 100.0, 0.0, 100.0)) for r, v in drve_scores.items()}

        # 2. Disease State Matrix (Node Embeddings)
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

        # 3. ANPE Propagation Matrix
        readiness_matrix = self.compute_anpe(G, drve_scores)

        # 4. Velocity & Memory Buffer
        velocity_matrix = np.full_like(dsv_matrix, np.nan)
        acceleration_matrix = np.full_like(dsv_matrix, np.nan)
        velocity_status = "Computed baseline state"

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

        # 5. Export Data & Heatmaps
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
        np.save(out_dir / f"{subject_id}_node_embeddings.npy", dsv_matrix)

        manifest = {
            "subject_id": subject_id,
            "velocity_status": velocity_status,
            "speech_rois": roi_order,
            "health_scores": health_scores,
            "vulnerability_scores": drve_scores,
            "dsv_shape": list(dsv_matrix.shape),
            "readiness_matrix_shape": list(readiness_matrix.shape),
            "graph_memory": memory.to_dict()
        }
        with open(out_dir / f"{subject_id}_neuropropx_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        self._generate_neuropropx_heatmaps(subject_id, dsv_df, readiness_df, out_dir)

        logger.info(f"=== NeuroProp-X Execution Complete for {subject_id} ===")

        return {
            "disease_state_matrix": dsv_matrix,
            "disease_state_df": dsv_df,
            "propagation_readiness_matrix": readiness_matrix,
            "propagation_readiness_df": readiness_df,
            "health_scores": health_scores,
            "drve_vulnerability": drve_scores,
            "velocity_status": velocity_status,
            "graph_memory": memory,
            "manifest": manifest
        }

    def _generate_neuropropx_heatmaps(self, subject_id: str, dsv_df: pd.DataFrame, readiness_df: pd.DataFrame, out_dir: Path):
        """Generate publication-grade heatmaps."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"NeuroProp-X Engine Representation — {subject_id}", fontsize=14, fontweight="bold")

        im1 = axes[0].imshow(dsv_df.values, cmap="YlOrRd", aspect="auto")
        axes[0].set_xticks(range(len(dsv_df.columns)))
        axes[0].set_xticklabels(dsv_df.columns, rotation=45, ha="right", fontsize=8)
        axes[0].set_yticks(range(len(dsv_df.index)))
        axes[0].set_yticklabels(dsv_df.index, fontsize=9)
        axes[0].set_title("Disease State Matrix S (Node Embeddings)")
        plt.colorbar(im1, ax=axes[0])

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
