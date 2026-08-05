"""
NeuroGenesis — Module 7: Patient Digital Twin Framework
========================================================
File   : backend/modules/07_digital_twin/digital_twin.py
Purpose: Creates a self-evolving virtual digital twin representation of the
         speech-related brain network per patient.
         Integrates longitudinal MRI scans, clinical speech assessment scores,
         and neuro-imaging biomarkers. Enables "What-If" intervention forecasting.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class DigitalTwinState:
    subject_id: str
    last_updated_visit: str
    age: float
    mmse: float
    cdr: float
    overall_speech_score: float
    regional_volumes: Dict[str, float]
    vulnerability_scores: Dict[str, float]
    network_connectivity_matrix: np.ndarray
    simulation_history: List[Dict[str, Any]] = field(default_factory=list)

class PatientDigitalTwin:
    """
    Patient Digital Twin Framework (Module 7).
    Maintains a continuous virtual state representation of the patient's speech network.
    """

    def __init__(self, subject_id: str, output_dir: Path):
        self.subject_id = subject_id
        self.output_dir = Path(output_dir) / "digital_twin" / subject_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state: Optional[DigitalTwinState] = None

    def initialize_or_update(
        self,
        visit_id: str,
        age: float,
        mmse: float,
        cdr: float,
        overall_speech_score: float,
        regional_volumes: Dict[str, float],
        vulnerability_scores: Dict[str, float],
        connectivity_matrix: np.ndarray
    ) -> DigitalTwinState:
        """
        Update Digital Twin with new longitudinal MRI scan / speech assessment data.
        """
        logger.info(f"=== Updating Digital Twin state for {self.subject_id} [Visit: {visit_id}] ===")
        
        self.state = DigitalTwinState(
            subject_id=self.subject_id,
            last_updated_visit=visit_id,
            age=age,
            mmse=mmse,
            cdr=cdr,
            overall_speech_score=overall_speech_score,
            regional_volumes=regional_volumes,
            vulnerability_scores=vulnerability_scores,
            network_connectivity_matrix=connectivity_matrix,
            simulation_history=[]
        )
        
        self._save_state()
        return self.state

    def run_intervention_simulation(
        self,
        therapy_efficacy_pct: float = 30.0,
        forecast_years: float = 2.0
    ) -> Dict[str, Any]:
        """
        Run a "What-If" Therapeutic Simulation.
        Simulates disease-modifying treatment slowing progressive atrophy.
        """
        if not self.state:
            raise ValueError(f"Digital Twin for {self.subject_id} is not initialized.")

        logger.info(f"Running What-If Simulation for {self.subject_id}: Therapy Efficacy={therapy_efficacy_pct}%, Horizon={forecast_years}yr")
        
        baseline_vols = self.state.regional_volumes
        vulns = self.state.vulnerability_scores

        # Un-treated vs Treated Volumetric Trajectories
        untreated_vols = {}
        treated_vols = {}
        saved_volume = {}

        efficacy_factor = 1.0 - (therapy_efficacy_pct / 100.0)

        for roi, v_base in baseline_vols.items():
            vuln = vulns.get(roi, 0.25)
            # Baseline natural decay rate
            natural_decay = (vuln * 4.5 + 1.0) * forecast_years
            
            v_untreated = max(v_base * (1.0 - natural_decay / 100.0), v_base * 0.5)
            v_treated = max(v_base * (1.0 - (natural_decay * efficacy_factor) / 100.0), v_base * 0.5)

            untreated_vols[roi] = round(v_untreated, 1)
            treated_vols[roi] = round(v_treated, 1)
            saved_volume[roi] = round(v_treated - v_untreated, 1)

        simulation_result = {
            "subject_id": self.subject_id,
            "therapy_efficacy_pct": therapy_efficacy_pct,
            "forecast_years": forecast_years,
            "baseline_volumes": baseline_vols,
            "untreated_forecast_volumes": untreated_vols,
            "treated_forecast_volumes": treated_vols,
            "volume_preserved_mm3": saved_volume,
            "total_brain_tissue_preserved_mm3": round(sum(saved_volume.values()), 1)
        }

        self.state.simulation_history.append(simulation_result)
        self._save_simulation(simulation_result)

        return simulation_result

    def _save_state(self):
        """Save Digital Twin state to JSON."""
        state_dict = {
            "subject_id": self.state.subject_id,
            "last_updated_visit": self.state.last_updated_visit,
            "age": self.state.age,
            "mmse": self.state.mmse,
            "cdr": self.state.cdr,
            "overall_speech_score": self.state.overall_speech_score,
            "regional_volumes": self.state.regional_volumes,
            "vulnerability_scores": self.state.vulnerability_scores
        }
        with open(self.output_dir / "digital_twin_state.json", "w") as f:
            json.dump(state_dict, f, indent=2)

    def _save_simulation(self, sim: Dict[str, Any]):
        """Save simulation run to file."""
        eff = int(sim["therapy_efficacy_pct"])
        with open(self.output_dir / f"simulation_therapy_{eff}pct.json", "w") as f:
            json.dump(sim, f, indent=2)
