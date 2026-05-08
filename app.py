"""
app.py  –  Eco-Resilience AI System
=====================================
Main Streamlit dashboard for environmental experts.

Run with:
    streamlit run app.py

Architecture:
    ┌──────────────────────────────────────────────────────────┐
    │  Sidebar  │  Data Input (CSV upload or manual sliders)   │
    │           │  + Global settings                           │
    ├───────────┴──────────────────────────────────────────────┤
    │  Tab 1 – Overview KPIs & Distribution Charts             │
    │  Tab 2 – Interactive Folium Map                          │
    │  Tab 3 – ML Insights (Anomalies & Recovery)              │
    │  Tab 4 – Data Table & Export                             │
    └──────────────────────────────────────────────────────────┘
"""

# ── Standard library & third-party imports ────────────────────────────────
import sys, os
# Make sure sibling packages resolve when running from project root
sys.path.insert(0, os.path.dirname(__file__))

import io
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from streamlit_folium import st_folium

# ── Project modules ────────────────────────────────────────────────────────
from utils.data_generator   import generate_dummy_data, get_sample_csv_string
from utils.indices          import (
    apply_all_indices, classify_rri, classify_runoff,
    LAND_USE_RUNOFF, LAND_USE_DEGRADATION,
)
from utils.map_builder      import build_resilience_map
from models.anomaly_detection  import run_anomaly_detection, anomaly_summary
from models.recovery_predictor import train_recovery_model, predict_recovery


# ══════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG & GLOBAL STYLING
# ══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Eco-Resilience AI System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Global font & background ── */
  html, body, [class*="css"]  { font-family: 'Segoe UI', Arial, sans-serif; }
  .block-container             { padding-top: 1.5rem; padding-bottom: 2rem; }

  /* ── Hero banner ── */
  .hero-banner {
      background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
      border-radius: 14px;
      padding: 28px 36px;
      margin-bottom: 24px;
      color: #ecf0f1;
  }
  .hero-banner h1 { margin: 0; font-size: 2rem; font-weight: 800; letter-spacing: -0.5px; }
  .hero-banner p  { margin: 6px 0 0; font-size: 1rem; color: #a9cce3; }

  /* ── KPI cards ── */
  .kpi-card {
      background: #ffffff;
      border-radius: 12px;
      padding: 20px 24px;
      box-shadow: 0 2px 12px rgba(0,0,0,0.08);
      border-left: 5px solid #27ae60;
      margin-bottom: 16px;
  }
  .kpi-card.warn  { border-left-color: #e67e22; }
  .kpi-card.alert { border-left-color: #e74c3c; }
  .kpi-card.info  { border-left-color: #2980b9; }

  .kpi-label { font-size: 0.78rem; color: #7f8c8d; font-weight: 600;
               text-transform: uppercase; letter-spacing: .6px; }
  .kpi-value { font-size: 2.0rem; font-weight: 800; color: #2c3e50; line-height: 1.1; }
  .kpi-sub   { font-size: 0.82rem; color: #95a5a6; margin-top: 2px; }

  /* ── Section headers ── */
  .section-title {
      font-size: 1.1rem; font-weight: 700; color: #2c3e50;
      border-bottom: 2px solid #eaecee; padding-bottom: 6px; margin-bottom: 16px;
  }

  /* ── Anomaly badge ── */
  .anomaly-badge {
      display: inline-block;
      background: #fdecea; color: #c0392b;
      border: 1px solid #f1948a; border-radius: 6px;
      padding: 2px 8px; font-size: 0.78rem; font-weight: 700;
  }
  .normal-badge {
      display: inline-block;
      background: #eafaf1; color: #196f3d;
      border: 1px solid #82e0aa; border-radius: 6px;
      padding: 2px 8px; font-size: 0.78rem; font-weight: 700;
  }

  /* ── Sidebar overrides ── */
  [data-testid="stSidebar"] { background: #1a2332; }
  [data-testid="stSidebar"] * { color: #ecf0f1 !important; }
  [data-testid="stSidebar"] .stSlider > label { font-size: 0.82rem !important; }
  [data-testid="stSidebar"] hr { border-color: #2c3e50 !important; }

  /* ── Tab styling ── */
  .stTabs [data-baseweb="tab-list"]  { gap: 6px; }
  .stTabs [data-baseweb="tab"]       { border-radius: 8px 8px 0 0; padding: 8px 20px; }

  /* ── Divider ── */
  .eco-divider { border: none; border-top: 1px solid #eaecee; margin: 20px 0; }

  /* ── Formula box ── */
  .formula-box {
      background: #f8f9fa; border: 1px solid #dee2e6;
      border-radius: 8px; padding: 14px 18px;
      font-family: 'Courier New', monospace; font-size: 0.85rem;
      color: #2c3e50; margin: 10px 0;
  }

  /* Streamlit metric tweaks */
  [data-testid="stMetricValue"] { font-size: 1.6rem !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_and_process(raw_df: pd.DataFrame, contamination: float) -> pd.DataFrame:
    """
    Full pipeline: indices → anomaly detection → recovery prediction.
    Result is cached so map/charts don't recompute on every widget change.
    """
    df = apply_all_indices(raw_df)

    # Anomaly detection
    is_anom, anom_score = run_anomaly_detection(df, contamination=contamination)
    df["is_anomaly"]    = is_anom.values
    df["anomaly_score"] = anom_score.values

    # Recovery prediction
    model_bundle             = train_recovery_model(df)
    df["recovery_potential"] = predict_recovery(model_bundle, df)

    # Human-readable classifications
    df["rri_class"]    = df["rri"].apply(classify_rri)
    df["runoff_class"] = df["runoff_risk"].apply(classify_runoff)

    return df, model_bundle


def kpi_card(label: str, value: str, sub: str = "", level: str = "ok") -> str:
    css_class = {"ok": "kpi-card", "warn": "kpi-card warn",
                 "alert": "kpi-card alert", "info": "kpi-card info"}[level]
    return f"""
    <div class="{css_class}">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>
    """


def gauge_chart(value: float, title: str, low_good: bool = False) -> go.Figure:
    """
    Render a Plotly gauge for a single index value.
    low_good=True  → green sector is at low end (e.g. runoff risk).
    low_good=False → green sector is at high end (e.g. RRI).
    """
    if low_good:
        colors = ["#27ae60", "#f39c12", "#e74c3c"]
    else:
        colors = ["#e74c3c", "#f39c12", "#27ae60"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value, 3),
        number={"font": {"size": 28, "color": "#2c3e50"}, "valueformat": ".3f"},
        title={"text": title, "font": {"size": 13, "color": "#7f8c8d"}},
        gauge={
            "axis":     {"range": [0, 1], "tickwidth": 1, "tickcolor": "#7f8c8d"},
            "bar":      {"color": "#2980b9", "thickness": 0.28},
            "bgcolor":  "white",
            "steps": [
                {"range": [0.00, 0.33], "color": colors[0]},
                {"range": [0.33, 0.66], "color": colors[1]},
                {"range": [0.66, 1.00], "color": colors[2]},
            ],
            "threshold": {
                "line": {"color": "#2c3e50", "width": 3},
                "thickness": 0.80,
                "value": value,
            },
        },
    ))
    fig.update_layout(
        height=210, margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  SIDEBAR – DATA INPUT
# ══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:16px 0 8px">
      <div style="font-size:2.5rem">🌍</div>
      <div style="font-size:1.05rem;font-weight:800;color:#ecf0f1;letter-spacing:.5px">
        Eco-Resilience AI
      </div>
      <div style="font-size:0.75rem;color:#7f8c8d;margin-top:2px">
        Environmental Intelligence Platform
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # ── Data source selection ──────────────────────────────────────────────
    st.markdown("**📂 Data Source**")
    data_source = st.radio(
        "Choose input method",
        ["🔬 Use Demo Dataset", "📤 Upload CSV", "✏ Manual Entry"],
        label_visibility="collapsed",
    )

    raw_df = None  # will be set below

    # ── Option A: Demo dataset ─────────────────────────────────────────────
    if "Demo" in data_source:
        n_pts = st.slider("Number of sample points", 20, 120, 60, step=10)
        raw_df = generate_dummy_data(n_samples=n_pts, seed=7)
        st.success(f"✅ Demo dataset: {n_pts} observation points")

        # Download template CSV
        st.download_button(
            label="⬇ Download CSV Template",
            data=get_sample_csv_string(),
            file_name="eco_resilience_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ── Option B: CSV upload ───────────────────────────────────────────────
    elif "Upload" in data_source:
        uploaded = st.file_uploader(
            "Upload your ecological dataset (CSV)",
            type=["csv"],
            help="Must match the template column names. Download the template above.",
        )
        if uploaded is not None:
            try:
                raw_df = pd.read_csv(uploaded)
                st.success(f"✅ Loaded {len(raw_df)} rows from {uploaded.name}")
            except Exception as e:
                st.error(f"❌ Could not read CSV: {e}")
        else:
            st.info("⬆ Please upload a CSV file to proceed.")

    # ── Option C: Manual single-point entry ───────────────────────────────
    else:
        st.markdown("**Enter field measurements:**")
        with st.expander("🌧 Rainfall", expanded=True):
            rf_int  = st.slider("Intensity (mm/day)", 0.0, 200.0, 35.0, 1.0)
            rf_dur  = st.slider("Duration (hours)",    0.0,  72.0, 12.0, 0.5)
            rf_freq = st.slider("Frequency (events/mo)", 1, 30, 8)

        with st.expander("🌱 Vegetation & Soil"):
            ndvi_c  = st.slider("NDVI – Current",  0.0, 1.0, 0.55, 0.01)
            ndvi_b  = st.slider("NDVI – Baseline", 0.0, 1.0, 0.50, 0.01)
            sm      = st.slider("Soil Moisture",   0.0, 1.0, 0.40, 0.01)
            biodiv  = st.slider("Biodiversity Index (0–100)", 0, 100, 65)

        with st.expander("⛰ Terrain & Weather"):
            elev    = st.slider("Elevation (m)",  0, 5000, 400, 10)
            slope   = st.slider("Slope (°)",       0, 45,    8)
            temp    = st.slider("Temperature (°C)", -10, 50, 28)
            humid   = st.slider("Humidity (%)",    10, 100, 70)

        with st.expander("🗺 Land Use & Location"):
            land_use = st.selectbox(
                "Land-Use Class",
                ["Forest", "Agriculture", "Wetland", "Grassland", "Urban", "Barren"],
            )
            lat = st.number_input("Latitude",  -90.0,  90.0,  18.5, format="%.4f")
            lon = st.number_input("Longitude", -180.0, 180.0, 76.0, format="%.4f")

        raw_df = pd.DataFrame([{
            "latitude": lat, "longitude": lon,
            "rainfall_intensity_mm_day": rf_int,
            "rainfall_duration_hours": rf_dur,
            "rainfall_frequency_per_month": rf_freq,
            "soil_moisture": sm,
            "ndvi_current": ndvi_c, "ndvi_baseline": ndvi_b,
            "biodiversity_index": biodiv,
            "elevation_m": elev, "slope_degrees": slope,
            "temperature_celsius": temp, "humidity_percent": humid,
            "land_use": land_use,
        }])

    st.markdown("---")

    # ── ML settings ───────────────────────────────────────────────────────
    st.markdown("**⚙ ML Settings**")
    contamination = st.slider(
        "Anomaly Sensitivity",
        min_value=0.05, max_value=0.30, value=0.10, step=0.01,
        help="Isolation Forest contamination parameter – expected fraction of anomalies.",
    )

    # ── About ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.72rem;color:#7f8c8d;line-height:1.6">
      <b>Eco-Resilience AI v1.0</b><br>
      Indices: RRI · RunRI · ESI<br>
      Models: Isolation Forest · Random Forest<br>
      Built for environmental scientists & ecologists.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT AREA
# ══════════════════════════════════════════════════════════════════════════

# ── Hero banner ────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <h1>🌍 Eco-Resilience AI System</h1>
  <p>Multi-source ecological data analysis · Vulnerability mapping · AI-powered recovery prediction</p>
</div>
""", unsafe_allow_html=True)

# ── Guard: need data to proceed ───────────────────────────────────────────
if raw_df is None or len(raw_df) == 0:
    st.warning("👈 Please select or upload a dataset using the sidebar.")
    st.stop()

# ── Run the full pipeline ─────────────────────────────────────────────────
with st.spinner("⚙ Running ecological analysis pipeline…"):
    t0 = time.time()
    df, model_bundle = load_and_process(raw_df, contamination)
    elapsed = time.time() - t0


# ══════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════

tab_overview, tab_map, tab_ml, tab_data, tab_howitworks = st.tabs([
    "📊 Overview & Indices",
    "🗺 Interactive Map",
    "🤖 ML Insights",
    "📋 Data & Export",
    "📐 How It Works",
])


# ──────────────────────────────────────────────────────────────────────────
#  TAB 1 – OVERVIEW & INDICES
# ──────────────────────────────────────────────────────────────────────────
with tab_overview:

    # ── Top KPI row ────────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        mean_rri = df["rri"].mean()
        level    = "ok" if mean_rri >= 0.5 else ("warn" if mean_rri >= 0.3 else "alert")
        st.markdown(
            kpi_card("Avg. Rainfall Response Index", f"{mean_rri:.3f}",
                     classify_rri(mean_rri), level),
            unsafe_allow_html=True,
        )
    with col2:
        mean_ror = df["runoff_risk"].mean()
        level    = "alert" if mean_ror >= 0.6 else ("warn" if mean_ror >= 0.4 else "ok")
        st.markdown(
            kpi_card("Avg. Runoff Risk Index", f"{mean_ror:.3f}",
                     classify_runoff(mean_ror), level),
            unsafe_allow_html=True,
        )
    with col3:
        mean_rec = df["recovery_potential"].mean()
        level    = "ok" if mean_rec >= 0.55 else ("warn" if mean_rec >= 0.35 else "alert")
        st.markdown(
            kpi_card("Avg. Recovery Potential", f"{mean_rec:.3f}",
                     "Random Forest prediction", level),
            unsafe_allow_html=True,
        )
    with col4:
        n_anom = int(df["is_anomaly"].sum())
        pct    = 100 * n_anom / max(len(df), 1)
        level  = "alert" if pct > 20 else ("warn" if pct > 10 else "ok")
        st.markdown(
            kpi_card("Anomalous Zones", f"{n_anom}",
                     f"{pct:.1f}% of observations", level),
            unsafe_allow_html=True,
        )
    with col5:
        n_vuln = int((df["runoff_risk"] >= 0.65).sum())
        level  = "alert" if n_vuln > len(df) * 0.3 else "warn"
        st.markdown(
            kpi_card("High-Risk Zones", f"{n_vuln}",
                     "RunRI ≥ 0.65", level),
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="eco-divider">', unsafe_allow_html=True)

    # ── Gauge row ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📈 Index Gauges</div>', unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)
    with g1:
        st.plotly_chart(
            gauge_chart(df["rri"].mean(), "Avg. RRI (higher=better)", low_good=False),
            use_container_width=True,
        )
    with g2:
        st.plotly_chart(
            gauge_chart(df["runoff_risk"].mean(), "Avg. Runoff Risk (lower=better)", low_good=True),
            use_container_width=True,
        )
    with g3:
        st.plotly_chart(
            gauge_chart(df["recovery_potential"].mean(), "Avg. Recovery Potential", low_good=False),
            use_container_width=True,
        )

    st.markdown('<hr class="eco-divider">', unsafe_allow_html=True)

    # ── Distribution charts ────────────────────────────────────────────────
    st.markdown('<div class="section-title">📉 Index Distributions</div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)

    with d1:
        fig_hist = px.histogram(
            df, x="rri", nbins=25, color="rri_class",
            color_discrete_map={
                "High Resilience": "#27ae60",
                "Moderate Resilience": "#e67e22",
                "Low Resilience": "#e74c3c",
                "Critical": "#8e44ad",
            },
            title="Rainfall Response Index Distribution",
            labels={"rri": "RRI", "count": "Zones"},
        )
        fig_hist.update_layout(
            showlegend=True, height=280,
            margin=dict(t=40, b=30, l=30, r=10),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend_title_text="Class",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with d2:
        fig_box = px.box(
            df, x="land_use", y="runoff_risk",
            color="land_use",
            title="Runoff Risk by Land-Use Class",
            labels={"runoff_risk": "RunRI", "land_use": "Land Use"},
        )
        fig_box.update_layout(
            showlegend=False, height=280,
            margin=dict(t=40, b=30, l=30, r=10),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # ── Scatter: RRI vs Runoff Risk ────────────────────────────────────────
    st.markdown('<div class="section-title">🔍 RRI vs. Runoff Risk by Land Use</div>',
                unsafe_allow_html=True)

    fig_scatter = px.scatter(
        df, x="rri", y="runoff_risk",
        color="land_use", size="biodiversity_index",
        symbol="is_anomaly",
        symbol_map={True: "x", False: "circle"},
        hover_data=["rri_class", "runoff_class", "recovery_potential",
                    "ndvi_current", "soil_moisture"],
        title="Ecological Risk Landscape  (✗ = Anomaly, size = Biodiversity)",
        labels={"rri": "Rainfall Response Index (RRI)",
                "runoff_risk": "Runoff Risk Index"},
        opacity=0.82,
    )
    fig_scatter.add_shape(type="line", x0=0, y0=0, x1=1, y1=0,
                           line=dict(color="grey", dash="dot", width=1))
    fig_scatter.update_layout(
        height=380,
        margin=dict(t=50, b=30, l=30, r=10),
        plot_bgcolor="#f8f9fa", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Radar chart for average profile ───────────────────────────────────
    st.markdown('<div class="section-title">🕸 Average Ecological Profile by Land Use</div>',
                unsafe_allow_html=True)

    radar_cols = ["rri", "runoff_risk", "recovery_potential",
                  "ecosystem_sensitivity", "soil_moisture"]
    radar_labels = ["RRI", "Runoff Risk", "Recovery", "Sensitivity", "Soil Moisture"]

    profile_df = df.groupby("land_use")[radar_cols].mean().reset_index()

    fig_radar = go.Figure()
    colors_radar = px.colors.qualitative.Set2
    for i, row in profile_df.iterrows():
        vals = row[radar_cols].tolist()
        vals_closed = vals + [vals[0]]
        labels_closed = radar_labels + [radar_labels[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_closed, theta=labels_closed,
            fill="toself", opacity=0.55,
            name=row["land_use"],
            line=dict(color=colors_radar[i % len(colors_radar)], width=2),
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True, height=400,
        margin=dict(t=40, b=20, l=40, r=40),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Mathematical formulas ─────────────────────────────────────────────
    with st.expander("📐 Mathematical Foundations", expanded=False):
        st.markdown("""
        **Rainfall Response Index (RRI)**
        """)
        st.markdown("""
        <div class="formula-box">
        RRI = [0.35·NDVI_norm + 0.25·SM + 0.20·RF_eff + 0.20·BD_norm] × (1 − D_lu)<br><br>
        where:<br>
        &nbsp; NDVI_norm = (clip(NDVI_curr/NDVI_base, 0.5, 2.0) − 0.5) / 1.5<br>
        &nbsp; SM        = soil_moisture  ∈ [0, 1]<br>
        &nbsp; RF_eff    = exp(−0.5 × ((I − 25) / 20)²)  &nbsp;[Gaussian, optimum=25 mm/d]<br>
        &nbsp; BD_norm   = biodiversity / 100<br>
        &nbsp; D_lu      = land-use degradation penalty ∈ {0.05 … 0.70}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Runoff Risk Index (RunRI)**")
        st.markdown("""
        <div class="formula-box">
        RunRI = 0.20·S_norm + 0.25·SM + 0.25·C_lu + 0.15·tanh(I/50) + 0.15·(1−NDVI)<br><br>
        where:<br>
        &nbsp; S_norm = slope_degrees / 45<br>
        &nbsp; C_lu   = SCS Curve Number coefficient (Urban=0.85 … Wetland=0.10)<br>
        &nbsp; I      = rainfall intensity (mm/day)
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Ecosystem Sensitivity Index (ESI)**")
        st.markdown("""
        <div class="formula-box">
        ESI = 0.30·(1−RRI) + 0.25·RunRI + 0.20·CS + 0.15·E_sens + 0.10·(1−BD_norm)<br><br>
        where:<br>
        &nbsp; CS     = 0.6·|T−22|/30 + 0.4·(1−H/100)  &nbsp;[climate stress]<br>
        &nbsp; E_sens = log(1+elevation) / log(1+5000)   &nbsp;[elevation sensitivity]
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────
#  TAB 2 – INTERACTIVE MAP
# ──────────────────────────────────────────────────────────────────────────
with tab_map:
    st.markdown("""
    <div class="section-title">🗺 Ecological Vulnerability & Resilience Map</div>
    <p style="color:#7f8c8d;font-size:0.88rem;margin-bottom:12px">
      Each marker represents one observation point. Colour = RRI class.
      Marker size ∝ Runoff Risk. Toggle layers with the control panel (top-right).
    </p>
    """, unsafe_allow_html=True)

    # Build map (show spinner for large datasets)
    with st.spinner("Rendering map…"):
        eco_map = build_resilience_map(df)

    map_col, legend_col = st.columns([3, 1])

    with map_col:
        st_folium(eco_map, width="100%", height=580, returned_objects=[])

    with legend_col:
        st.markdown("**Layer Guide**")
        st.markdown("""
        | Layer | Description |
        |-------|-------------|
        | 📍 Eco Zones | Clustered observation points |
        | 🔥 Runoff Heatmap | Surface runoff intensity |
        | 🌿 Resilience Map | Ecosystem RRI density |
        | ⚠ Anomalies | ML-detected outliers |
        | 🔴 Vulnerable | High-risk shaded circles |
        """)

        st.markdown("---")
        st.markdown("**RRI Colour Scale**")
        st.markdown("""
        🟢 **≥ 0.70** — High resilience  
        🟠 **0.45–0.70** — Moderate  
        🔴 **0.25–0.45** — Low  
        🟣 **< 0.25** — Critical  
        """)

        st.markdown("---")
        n_critical = int((df["rri"] < 0.25).sum())
        n_extreme  = int((df["runoff_risk"] >= 0.75).sum())
        st.metric("Critical RRI zones",   n_critical)
        st.metric("Extreme runoff zones", n_extreme)

        st.markdown("---")
        st.caption(f"⏱ Pipeline completed in {elapsed:.2f}s")


# ──────────────────────────────────────────────────────────────────────────
#  TAB 3 – ML INSIGHTS
# ──────────────────────────────────────────────────────────────────────────
with tab_ml:
    ml_left, ml_right = st.columns([1, 1], gap="large")

    # ── Anomaly Detection Panel ───────────────────────────────────────────
    with ml_left:
        st.markdown('<div class="section-title">🔍 Isolation Forest Anomaly Detection</div>',
                    unsafe_allow_html=True)

        summary = anomaly_summary(df)
        c1, c2 = st.columns(2)
        c1.metric("Anomalies Detected",  summary["anomaly_count"])
        c2.metric("As % of Dataset",     f"{summary['anomaly_pct']:.1f}%")

        # Score distribution
        fig_anom = px.histogram(
            df, x="anomaly_score",
            color="is_anomaly",
            color_discrete_map={True: "#e74c3c", False: "#2ecc71"},
            nbins=30,
            barmode="overlay",
            opacity=0.75,
            title="Anomaly Score Distribution",
            labels={"anomaly_score": "Isolation Score (higher=more anomalous)",
                    "is_anomaly": "Is Anomaly"},
        )
        fig_anom.update_layout(
            height=260, margin=dict(t=40, b=30, l=30, r=10),
            plot_bgcolor="#f8f9fa", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_anom, use_container_width=True)

        # Top anomalies
        if len(summary["top_anomalies"]) > 0:
            st.markdown("**Top Anomalous Zones**")
            top = summary["top_anomalies"].copy()
            top["anomaly_score"] = top["anomaly_score"].round(4)
            top["rri"]           = top["rri"].round(4)
            st.dataframe(top, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("""
        **Algorithm:** Isolation Forest (Scikit-Learn)  
        **Contamination:** set in sidebar  
        **Features:** rainfall intensity, soil moisture, NDVI, biodiversity,
        temperature, humidity  
        **Interpretation:** Red points have multivariate signatures that
        cannot be explained by dominant patterns in the dataset — indicating
        potential data quality issues, localised ecological disturbances,
        or land-cover change.
        """)

    # ── Recovery Prediction Panel ─────────────────────────────────────────
    with ml_right:
        st.markdown('<div class="section-title">♻ Random Forest Recovery Predictor</div>',
                    unsafe_allow_html=True)

        metrics = model_bundle["metrics"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Train R²",  f"{metrics['train_r2']:.3f}")
        m2.metric("Test R²",   f"{metrics['test_r2']:.3f}")
        m3.metric("Test MAE",  f"{metrics['test_mae']:.4f}")

        # Recovery potential vs RRI scatter
        fig_rec = px.scatter(
            df, x="rri", y="recovery_potential",
            color="land_use", size="biodiversity_index",
            trendline="ols",
            title="Recovery Potential vs. RRI",
            labels={"rri": "RRI", "recovery_potential": "Recovery Potential"},
            opacity=0.80,
        )
        fig_rec.update_layout(
            height=260, margin=dict(t=40, b=30, l=30, r=10),
            plot_bgcolor="#f8f9fa", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_rec, use_container_width=True)

        # Feature importances
        st.markdown("**Top Feature Importances**")
        top_fi = model_bundle["importances"].head(10).reset_index()
        top_fi.columns = ["Feature", "Importance"]
        top_fi["Feature"] = top_fi["Feature"].str.replace("_", " ").str.title()

        fig_fi = px.bar(
            top_fi, x="Importance", y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Teal",
            title="",
        )
        fig_fi.update_layout(
            height=300, margin=dict(t=10, b=10, l=10, r=10),
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
            plot_bgcolor="#f8f9fa", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_fi, use_container_width=True)

        st.markdown("""
        **Algorithm:** Random Forest Regressor (300 trees, depth 10)  
        **Target:** Proxy recovery potential  
        (production: actual post-disturbance NDVI time-series)  
        **Top drivers:** NDVI, soil moisture, land-use class, biodiversity
        """)

    st.markdown('<hr class="eco-divider">', unsafe_allow_html=True)

    # ── Correlation matrix ─────────────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Feature Correlation Matrix</div>',
                unsafe_allow_html=True)
    corr_cols = [
        "rainfall_intensity_mm_day", "soil_moisture", "ndvi_current",
        "biodiversity_index", "slope_degrees", "temperature_celsius",
        "rri", "runoff_risk", "recovery_potential", "ecosystem_sensitivity",
    ]
    corr_labels = [
        "RF Intensity", "Soil Moisture", "NDVI", "Biodiversity",
        "Slope", "Temperature",
        "RRI", "Runoff Risk", "Recovery", "Eco. Sensitivity",
    ]
    corr_df   = df[corr_cols].corr()
    fig_corr  = px.imshow(
        corr_df.values,
        x=corr_labels, y=corr_labels,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        text_auto=".2f",
        title="Pearson Correlation Heatmap",
        aspect="auto",
    )
    fig_corr.update_layout(
        height=420, margin=dict(t=50, b=30, l=30, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_corr, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────
#  TAB 4 – DATA TABLE & EXPORT
# ──────────────────────────────────────────────────────────────────────────
with tab_data:
    st.markdown('<div class="section-title">📋 Processed Dataset</div>',
                unsafe_allow_html=True)

    # ── Filter controls ────────────────────────────────────────────────────
    f1, f2, f3 = st.columns(3)
    with f1:
        lu_filter = st.multiselect(
            "Filter by Land Use",
            options=sorted(df["land_use"].unique()),
            default=sorted(df["land_use"].unique()),
        )
    with f2:
        rri_range = st.slider("RRI Range", 0.0, 1.0, (0.0, 1.0), 0.01)
    with f3:
        show_anomaly_only = st.checkbox("Show anomalous zones only", value=False)

    filtered = df[
        df["land_use"].isin(lu_filter) &
        df["rri"].between(*rri_range)
    ]
    if show_anomaly_only:
        filtered = filtered[filtered["is_anomaly"]]

    # Display columns for table (friendly subset)
    display_cols = [
        "latitude", "longitude", "land_use",
        "rri", "rri_class", "runoff_risk", "runoff_class",
        "recovery_potential", "ecosystem_sensitivity",
        "is_anomaly", "anomaly_score",
        "ndvi_current", "soil_moisture", "rainfall_intensity_mm_day",
        "biodiversity_index", "elevation_m", "slope_degrees",
    ]

    st.dataframe(
        filtered[display_cols]
        .style.background_gradient(subset=["rri"], cmap="RdYlGn")
        .background_gradient(subset=["runoff_risk"], cmap="RdYlGn_r")
        .background_gradient(subset=["recovery_potential"], cmap="YlGn")
        .format({
            "latitude": "{:.4f}", "longitude": "{:.4f}",
            "rri": "{:.4f}", "runoff_risk": "{:.4f}",
            "recovery_potential": "{:.4f}", "ecosystem_sensitivity": "{:.4f}",
            "anomaly_score": "{:.4f}",
            "ndvi_current": "{:.3f}", "soil_moisture": "{:.3f}",
            "rainfall_intensity_mm_day": "{:.1f}",
            "biodiversity_index": "{:.1f}",
            "elevation_m": "{:.0f}", "slope_degrees": "{:.1f}",
        }),
        use_container_width=True,
        height=420,
    )

    st.caption(f"Showing {len(filtered)} of {len(df)} observation points after filters.")

    # ── Export ─────────────────────────────────────────────────────────────
    st.markdown("---")
    e1, e2 = st.columns(2)
    with e1:
        csv_out = filtered[display_cols].to_csv(index=False)
        st.download_button(
            label="⬇ Download Filtered Results (CSV)",
            data=csv_out,
            file_name="eco_resilience_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with e2:
        # Summary statistics
        summary_df = filtered[["rri", "runoff_risk", "recovery_potential",
                                "ecosystem_sensitivity"]].describe().round(4)
        st.download_button(
            label="⬇ Download Summary Statistics (CSV)",
            data=summary_df.to_csv(),
            file_name="eco_resilience_summary_stats.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown('<hr class="eco-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Summary Statistics</div>',
                unsafe_allow_html=True)
    st.dataframe(
        filtered[["rri", "runoff_risk", "recovery_potential",
                  "ecosystem_sensitivity", "anomaly_score"]]
        .describe().round(4),
        use_container_width=True,
    )
    # ── Tab 5: How It Works ───────────────────────────────────────────────────
with tab_howitworks:
    st.subheader("📐 How the Scores Are Calculated — In Plain English")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🌿 RRI — Rainfall Response Index")
        st.markdown("""
        **The question it answers:** Did the land respond well to the rain it received?

        **Simple analogy:** You water two plants equally. One grows lush and green — the other barely reacts. RRI measures which ecosystem is more like the first plant.

        | Score | Meaning |
        |-------|---------|
        | 0.70 – 1.0 | ✅ High Resilience |
        | 0.45 – 0.70 | 🟡 Moderate |
        | 0.25 – 0.45 | 🔴 Low |
        | 0 – 0.25 | 🚨 Critical |

        **What boosts the score:** Greener vegetation after rain, moist soil, moderate rainfall (~25mm/day is ideal), and high biodiversity.

        **What lowers the score:** Urban or barren land, extreme rainfall, dry soil.
        """)

    with col2:
        st.markdown("### 💧 RunRI — Runoff Risk Index")
        st.markdown("""
        **The question it answers:** Will the rain soak into the ground, or rush away as a flood?

        **Simple analogy:** Rain on a steep concrete road rushes off instantly (high risk). Rain on a flat meadow soaks in slowly (low risk).

        | Score | Meaning |
        |-------|---------|
        | 0 – 0.35 | ✅ Low Risk |
        | 0.35 – 0.55 | 🟡 Moderate |
        | 0.55 – 0.75 | 🔴 High Risk |
        | 0.75 – 1.0 | 🚨 Extreme Risk |

        **What increases risk:** Steep terrain, saturated soil, urban land, heavy rainfall, low vegetation.

        **What reduces risk:** Dense trees, flat land, wetlands and forests.
        """)

    with col3:
        st.markdown("### 🌡️ ESI — Ecosystem Sensitivity Index")
        st.markdown("""
        **The question it answers:** How easily could this ecosystem be pushed over the edge?

        **Simple analogy:** Some ecosystems are like a Jenga tower — one wrong move and they collapse. Others are like a dense shrub — you pull on it and it springs back. ESI measures how "Jenga-like" a place is.

        | Score | Meaning |
        |-------|---------|
        | 0 – 0.30 | ✅ Robust |
        | 0.30 – 0.50 | 🟡 Moderate |
        | 0.50 – 0.70 | 🔴 Sensitive |
        | 0.70 – 1.0 | 🚨 Very Fragile |

        **Built from:** Low RRI + High RunRI + Climate stress + High altitude + Low biodiversity.
        """)

    st.markdown("---")
    st.markdown("### 🤖 The Two AI Models")
    col4, col5 = st.columns(2)

    with col4:
        st.markdown("""
        **🔍 Anomaly Detection (Isolation Forest)**

        Like a detective — it looks at ALL locations and asks: *"Which one is behaving weirdly?"*
        It needs no labels — it figures out what's normal on its own. A location is flagged if it got normal rainfall but its vegetation or soil moisture don't match expectations.
        """)

    with col5:
        st.markdown("""
        **🔄 Recovery Predictor (Random Forest)**

        Like a doctor predicting how fast a patient heals. It looks at all features of a location and predicts a score from 0 to 1 — how quickly will this ecosystem bounce back after a flood, drought, or fire?

        **High recovery:** Forests, wetlands, high biodiversity, moist soil.
        **Low recovery:** Urban/barren land, steep slopes, degraded vegetation.
        """)
