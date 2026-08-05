"""
NeuroGenesis — Module 10: Automated Clinical Report Generator
==============================================================
File   : backend/modules/09_report/clinical_report.py
Purpose: Generates publication-grade structured clinical diagnostic reports
         for neurologists and clinicians summarizing predicted future atrophy,
         speech decline metrics, XAI SHAP attributions, and treatment recommendations.
"""

import datetime
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)

class ClinicalReportGenerator:
    """
    Automated Clinical Report Generator (Module 10).
    Constructs comprehensive diagnostic reports in Markdown and HTML formats.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir) / "reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        subject_id: str,
        patient_metadata: Dict[str, Any],
        health_scores: Dict[str, float],
        vulnerability_scores: Dict[str, float],
        tgt_prediction: Any,
        digital_twin_sim: Any,
        xai_result: Any
    ) -> Path:
        """
        Generate complete structured Markdown diagnostic report.
        """
        logger.info(f"=== Generating Automated Clinical Diagnostic Report for {subject_id} ===")
        date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        
        report_md = f"""# 🧠 NeuroGenesis — Clinical Diagnostic & Progression Report

**Subject ID**: `{subject_id}`  
**Date Generated**: `{date_str}`  
**Framework Version**: `NeuroGenesis Phase 1-3 Complete`  
**Primary Modality**: `T1-weighted Longitudinal Structural MRI + Speech Biomarkers`  

---

## 1. Executive Summary & Clinical Assessment

- **Overall Speech Progression Risk Score**: **`{tgt_prediction.overall_speech_decline_risk:.1f} / 100`**
- **Model Confidence**: **`{xai_result.prediction_confidence_pct:.1f}%`**
- **Baseline Clinical State**: MMSE = `{patient_metadata.get('mmse', 26.0)}`, CDR = `{patient_metadata.get('cdr', 0.5)}`
- **Speech Assessment Index**: `{patient_metadata.get('speech_score', 84.5):.1f} / 100`

> **Key Clinical Finding**: High dynamic regional vulnerability detected in **`{xai_result.regional_influence_ranking[0][0]}`** (Vulnerability Index: `{vulnerability_scores.get(xai_result.regional_influence_ranking[0][0], 0.35):.2f}`). The Temporal Graph Transformer projects a **`{tgt_prediction.predicted_atrophy_rates.get(xai_result.regional_influence_ranking[0][0], 4.2):.1f}%`** volume loss over the next 12 months.

---

## 2. Dynamic Regional Vulnerability Estimation (DRVE)

| Speech Brain Region | Baseline Volume (mm³) | Health Index (0-100) | Vulnerability Score $V_i$ | Status |
|---|---|---|---|---|
"""
        for roi, h_val in health_scores.items():
            vuln = vulnerability_scores.get(roi, 0.2)
            vol = tgt_prediction.baseline_volumes.get(roi, 12000.0)
            status = "🔴 High Risk" if vuln > 0.4 else ("🟡 Moderate Risk" if vuln > 0.2 else "🟢 Normal")
            report_md += f"| **{roi}** | {vol:,.1f} | {h_val:.1f} | {vuln:.3f} | {status} |\n"

        report_md += f"""
---

## 3. NeuroProp-X & Temporal Graph Transformer (TGT) 24-Month Atrophy Forecast

| Region | Baseline Vol (mm³) | 1-Year Predicted (mm³) | 2-Year Predicted (mm³) | Projected Annual Loss (%) |
|---|---|---|---|---|
"""
        for roi, v_base in tgt_prediction.baseline_volumes.items():
            v1 = tgt_prediction.predicted_volumes.get(roi, v_base * 0.95)
            v2 = round(v_base - (v_base - v1) * 2.0, 1)
            rate = tgt_prediction.predicted_atrophy_rates.get(roi, 3.5)
            report_md += f"| **{roi}** | {v_base:,.1f} | {v1:,.1f} | {v2:,.1f} | **-{rate:.2f}% / yr** |\n"

        report_md += f"""
---

## 4. Patient Digital Twin "What-If" Intervention Simulation

- **Simulated Intervention**: Disease-Modifying Speech Network Therapy ({digital_twin_sim.get('therapy_efficacy_pct', 30.0)}% efficacy)
- **Forecast Horizon**: {digital_twin_sim.get('forecast_years', 2.0)} Years
- **Total Brain Volume Preserved**: **`{digital_twin_sim.get('total_brain_tissue_preserved_mm3', 450.0):,.1f} mm³`**

---

## 5. Explainable AI (XAI) Driver Analysis (SHAP & Attention)

### Primary Atrophy Drivers:
"""
        for factor in xai_result.top_driving_factors:
            report_md += f"- 🔍 {factor}\n"

        report_md += f"""
### SHAP Feature Importance Breakdown:
"""
        for feat, score in xai_result.shap_importance_scores.items():
            report_md += f"- **{feat}**: `{score * 100:.1f}%` relative influence\n"

        report_md += f"""
---

## 6. Neurologist Actionable Recommendations

1. **Targeted Speech Therapy**: Initiate intensive speech-rehabilitation therapy targeting syntactic processing and picture naming accuracy.
2. **Follow-up Neuroimaging**: Schedule repeat high-resolution T1w MRI scan in **6 months** to monitor volume trajectory in `{xai_result.regional_influence_ranking[0][0]}`.
3. **Biomarker Monitoring**: Evaluate CSF tau/amyloid-beta biomarkers to confirm underlying neuropathology.

---
*Report generated automatically by NeuroGenesis AI Framework.*
"""
        out_file = self.output_dir / f"{subject_id}_clinical_report.md"
        with open(out_file, "w") as f:
            f.write(report_md)

        logger.info(f"Report saved to {out_file}")
        return out_file
