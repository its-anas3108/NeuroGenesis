"""
NeuroGenesis Dashboard Utilities
================================
File   : dashboard/utils.py
Purpose: UI styling, dark medical AI CSS, data loaders, and helper rendering components.
"""

import streamlit as st
from pathlib import Path
import json
import pandas as pd
import numpy as np

def apply_medical_ai_theme():
    """Apply dark glassmorphism medical AI theme to Streamlit."""
    st.markdown("""
    <style>
        /* Main background */
        .stApp {
            background-color: #0b0f19;
            color: #e2e8f0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #111827;
            border-right: 1px solid #1f2937;
        }

        /* Metric cards */
        div[data-testid="stMetricValue"] {
            font-size: 1.8rem;
            font-weight: 700;
            color: #38bdf8;
        }
        
        div[data-testid="stMetricLabel"] {
            font-weight: 600;
            color: #94a3b8;
        }

        /* Medical Header Card */
        .med-card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        }

        .med-title {
            font-size: 1.5rem;
            font-weight: 700;
            color: #38bdf8;
            margin-bottom: 8px;
        }

        .med-subtitle {
            font-size: 0.95rem;
            color: #94a3b8;
        }

        /* Status badge */
        .status-badge {
            background-color: #0369a1;
            color: #e0f2fe;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
        }
    </style>
    """, unsafe_allow_html=True)

def render_header(title: str, subtitle: str, badge: str = "NeuroGenesis Phase 1 (60%)"):
    """Render medical dashboard page header."""
    st.markdown(f"""
    <div class="med-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div class="med-title">{title}</div>
            <span class="status-badge">{badge}</span>
        </div>
        <div class="med-subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

def load_available_subjects(output_dir: Path) -> list:
    """Discover subjects with outputs in the outputs directory."""
    out_dir = Path(output_dir)
    subjects = []
    if out_dir.exists():
        for p in (out_dir / "neuropropx").glob("*"):
            if p.is_dir():
                subjects.append(p.name)
    if not subjects:
        # Fallback list for demo/testing
        subjects = ["OAS1_0001_MR1", "OAS1_0002_MR1", "OAS1_0003_MR1"]
    return sorted(subjects)
