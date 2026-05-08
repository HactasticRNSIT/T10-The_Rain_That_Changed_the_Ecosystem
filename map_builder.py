"""
utils/map_builder.py
====================
Builds a multi-layer interactive Folium map for the Eco-Resilience dashboard.

Layers produced
───────────────
1. Ecological Zone Markers (clustered) – colour-coded by RRI with rich popups.
2. Runoff Risk Heatmap               – red → orange gradient.
3. Resilience Heatmap                – red → green gradient.
4. Anomalous Zone Markers            – red exclamation icons.
5. Vulnerable Zone Polygons          – shaded circles around high-risk points.
"""

import folium
from folium.plugins import HeatMap, MarkerCluster, MiniMap, Fullscreen
import pandas as pd
import numpy as np


# ── Colour helpers ─────────────────────────────────────────────────────────

def _rri_color(rri: float) -> str:
    """Map RRI → hex colour (red=bad, amber=moderate, green=good)."""
    if   rri >= 0.70: return "#27ae60"  # emerald green
    elif rri >= 0.45: return "#e67e22"  # amber
    elif rri >= 0.25: return "#e74c3c"  # red
    else:             return "#8e44ad"  # deep purple = critical


def _runoff_color(risk: float) -> str:
    if   risk >= 0.75: return "darkred"
    elif risk >= 0.55: return "red"
    elif risk >= 0.35: return "orange"
    else:              return "green"


def _popup_html(row: pd.Series) -> str:
    """Render a styled HTML popup for a single observation point."""
    rri_class = row.get("rri_class", "—")
    risk_class = row.get("runoff_class", "—")
    anomaly_badge = (
        '<span style="color:#e74c3c;font-weight:bold;">⚠ ANOMALY</span>'
        if row.get("is_anomaly", False) else
        '<span style="color:#27ae60;">✔ Normal</span>'
    )
    return f"""
    <div style="font-family:'Segoe UI',Arial,sans-serif;width:260px;padding:4px">
      <div style="background:#1a2332;color:#ecf0f1;padding:8px 10px;border-radius:6px 6px 0 0;
                  font-size:13px;font-weight:700;letter-spacing:.5px">
        📍 Ecological Observation Point
      </div>
      <table style="width:100%;font-size:12px;border-collapse:collapse;margin-top:0">
        <tr style="background:#f0f4f8">
          <td style="padding:4px 8px"><b>Coordinates</b></td>
          <td style="padding:4px 8px">{row['latitude']:.4f}°N, {row['longitude']:.4f}°E</td>
        </tr>
        <tr>
          <td style="padding:4px 8px"><b>Land Use</b></td>
          <td style="padding:4px 8px">{row['land_use']}</td>
        </tr>
        <tr style="background:#f0f4f8">
          <td style="padding:4px 8px"><b>RRI</b></td>
          <td style="padding:4px 8px">{row['rri']:.3f} — <i>{rri_class}</i></td>
        </tr>
        <tr>
          <td style="padding:4px 8px"><b>Runoff Risk</b></td>
          <td style="padding:4px 8px">{row['runoff_risk']:.3f} — <i>{risk_class}</i></td>
        </tr>
        <tr style="background:#f0f4f8">
          <td style="padding:4px 8px"><b>Recovery Potential</b></td>
          <td style="padding:4px 8px">{row.get('recovery_potential', 0):.3f}</td>
        </tr>
        <tr>
          <td style="padding:4px 8px"><b>Eco. Sensitivity</b></td>
          <td style="padding:4px 8px">{row.get('ecosystem_sensitivity', 0):.3f}</td>
        </tr>
        <tr style="background:#f0f4f8">
          <td style="padding:4px 8px"><b>NDVI (current)</b></td>
          <td style="padding:4px 8px">{row['ndvi_current']:.3f}</td>
        </tr>
        <tr>
          <td style="padding:4px 8px"><b>Soil Moisture</b></td>
          <td style="padding:4px 8px">{row['soil_moisture']:.3f}</td>
        </tr>
        <tr style="background:#f0f4f8">
          <td style="padding:4px 8px"><b>Rainfall (mm/d)</b></td>
          <td style="padding:4px 8px">{row['rainfall_intensity_mm_day']:.1f}</td>
        </tr>
        <tr>
          <td style="padding:4px 8px"><b>Anomaly Status</b></td>
          <td style="padding:4px 8px">{anomaly_badge}</td>
        </tr>
      </table>
    </div>
    """


# ── Main builder ───────────────────────────────────────────────────────────

def build_resilience_map(df: pd.DataFrame) -> folium.Map:
    """
    Build and return a fully populated Folium map.

    Parameters
    ----------
    df : pd.DataFrame
        Output of the full pipeline (must include rri, runoff_risk,
        recovery_potential, ecosystem_sensitivity, is_anomaly, anomaly_score,
        rri_class, runoff_class columns).

    Returns
    -------
    folium.Map
    """
    center_lat = float(df["latitude"].mean())
    center_lon = float(df["longitude"].mean())

    # ── Base map ──────────────────────────────────────────────────────────
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles=None,                     # tiles added below for LayerControl
        prefer_canvas=True,
    )

    # ── Tile layers ───────────────────────────────────────────────────────
    folium.TileLayer(
        "CartoDB positron",
        name="🗺 Light (CartoDB)",
        control=True,
    ).add_to(m)
    folium.TileLayer(
        "CartoDB dark_matter",
        name="🌑 Dark Mode",
        control=True,
    ).add_to(m)
    folium.TileLayer(
        "OpenStreetMap",
        name="🛣 OpenStreetMap",
        control=True,
    ).add_to(m)

    # ── Layer 1: Clustered ecological zone markers ─────────────────────────
    cluster = MarkerCluster(name="📍 Ecological Zones", show=True)
    for _, row in df.iterrows():
        color = _rri_color(row["rri"])
        # Circle size encodes runoff risk
        radius = 7 + float(row["runoff_risk"]) * 12
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            weight=1.5,
            popup=folium.Popup(_popup_html(row), max_width=280),
            tooltip=(
                f"RRI: {row['rri']:.2f} | "
                f"Runoff: {row['runoff_risk']:.2f} | "
                f"{row['land_use']}"
            ),
        ).add_to(cluster)
    cluster.add_to(m)

    # ── Layer 2: Runoff Risk heatmap ───────────────────────────────────────
    heat_runoff = [
        [row["latitude"], row["longitude"], float(row["runoff_risk"])]
        for _, row in df.iterrows()
    ]
    HeatMap(
        heat_runoff,
        name="🔥 Runoff Risk Heatmap",
        min_opacity=0.3,
        radius=25,
        blur=18,
        gradient={0.2: "blue", 0.5: "lime", 0.8: "orange", 1.0: "red"},
        show=False,
    ).add_to(m)

    # ── Layer 3: Resilience (RRI) heatmap ─────────────────────────────────
    heat_rri = [
        [row["latitude"], row["longitude"], float(row["rri"])]
        for _, row in df.iterrows()
    ]
    HeatMap(
        heat_rri,
        name="🌿 Resilience Heatmap",
        min_opacity=0.3,
        radius=25,
        blur=18,
        gradient={0.0: "red", 0.35: "orange", 0.65: "yellow", 1.0: "#27ae60"},
        show=False,
    ).add_to(m)

    # ── Layer 4: Anomalous zone markers ───────────────────────────────────
    anomaly_group = folium.FeatureGroup(name="⚠ Anomalous Zones", show=True)
    for _, row in df[df["is_anomaly"]].iterrows():
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            icon=folium.Icon(color="red", icon="exclamation-sign", prefix="glyphicon"),
            popup=folium.Popup(
                f"<b>⚠ Anomalous Ecological Response</b><br>"
                f"Anomaly Score: {row['anomaly_score']:.3f}<br>"
                f"Land Use: {row['land_use']}<br>"
                f"RRI: {row['rri']:.3f}",
                max_width=200,
            ),
            tooltip="⚠ Anomaly Detected – Click for details",
        ).add_to(anomaly_group)
    anomaly_group.add_to(m)

    # ── Layer 5: Vulnerable zone shaded circles ───────────────────────────
    vuln_group = folium.FeatureGroup(name="🔴 Vulnerable Zones (RunRI > 0.65)", show=True)
    vulnerable = df[df["runoff_risk"] >= 0.65]
    for _, row in vulnerable.iterrows():
        folium.Circle(
            location=[row["latitude"], row["longitude"]],
            radius=20000 * float(row["runoff_risk"]),   # radius in metres
            color="#e74c3c",
            fill=True,
            fill_color="#e74c3c",
            fill_opacity=0.12,
            weight=1,
            tooltip=f"Vulnerable Zone | RunRI={row['runoff_risk']:.2f}",
        ).add_to(vuln_group)
    vuln_group.add_to(m)

    # ── Plugins ───────────────────────────────────────────────────────────
    MiniMap(toggle_display=True, tile_layer="CartoDB positron").add_to(m)
    Fullscreen(position="topright").add_to(m)

    # ── Legend HTML (injected as a custom control) ─────────────────────────
    legend_html = """
    <div style="
        position: fixed; bottom: 40px; left: 40px; z-index: 9999;
        background: rgba(26,35,50,0.92); border-radius: 10px;
        padding: 14px 18px; font-family:'Segoe UI',Arial,sans-serif;
        color: #ecf0f1; font-size: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.4);
        min-width: 180px;">
      <div style="font-size:13px;font-weight:700;margin-bottom:10px;
                  border-bottom:1px solid #34495e;padding-bottom:6px">
        🌍 RRI Resilience Legend
      </div>
      <div style="margin:4px 0">
        <span style="display:inline-block;width:14px;height:14px;border-radius:50%;
              background:#27ae60;margin-right:8px;vertical-align:middle"></span>
        High Resilience (RRI ≥ 0.70)
      </div>
      <div style="margin:4px 0">
        <span style="display:inline-block;width:14px;height:14px;border-radius:50%;
              background:#e67e22;margin-right:8px;vertical-align:middle"></span>
        Moderate (0.45 – 0.70)
      </div>
      <div style="margin:4px 0">
        <span style="display:inline-block;width:14px;height:14px;border-radius:50%;
              background:#e74c3c;margin-right:8px;vertical-align:middle"></span>
        Low Resilience (0.25 – 0.45)
      </div>
      <div style="margin:4px 0">
        <span style="display:inline-block;width:14px;height:14px;border-radius:50%;
              background:#8e44ad;margin-right:8px;vertical-align:middle"></span>
        Critical (&lt; 0.25)
      </div>
      <div style="margin-top:10px;border-top:1px solid #34495e;padding-top:8px;font-size:11px;color:#95a5a6">
        Circle size ∝ Runoff Risk
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # ── Layer control ─────────────────────────────────────────────────────
    folium.LayerControl(collapsed=False, position="topright").add_to(m)

    return m
