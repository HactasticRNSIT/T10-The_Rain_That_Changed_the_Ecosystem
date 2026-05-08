"""
utils/indices.py
================
Scientific computation of all ecological indices.

──────────────────────────────────────────────────────────────────────────────
INDEX 1 ─ Rainfall Response Index (RRI)
──────────────────────────────────────────────────────────────────────────────
Definition:
  RRI quantifies how efficiently an ecosystem converts rainfall into a
  positive vegetation response.  A high RRI indicates a healthy, intact
  ecosystem; a low RRI suggests degradation, water stress, or land-use change.

Formula (weighted composite, bounded [0, 1]):

  RRI = [0.35·NDVI_norm + 0.25·SM + 0.20·RF_eff + 0.20·BD_norm]
        × (1 − D_lu)

  where:
    NDVI_norm  = (R_ndvi − 0.5) / 1.5                   # normalised ratio
    R_ndvi     = clip(NDVI_current / NDVI_baseline, 0.5, 2.0)
    SM         = soil_moisture  ∈ [0, 1]
    RF_eff     = exp(−0.5 · ((I − μ_I) / σ_I)²)         # Gaussian bell, μ=25 mm/day
    BD_norm    = biodiversity_index / 100
    D_lu       = land-use degradation penalty  ∈ [0, 1]

Rationale:
  • NDVI ratio captures actual vs. expected vegetation greenness post-rain.
  • Soil moisture modulates infiltration capacity (Green-Ampt / Philip model).
  • Rainfall efficiency follows a Gaussian centred at 25 mm/day — too little
    causes water stress; too much causes surface runoff and erosion.
  • Biodiversity is a proxy for structural complexity and redundancy.
  • Degradation penalty encodes irreversible land-cover changes.

──────────────────────────────────────────────────────────────────────────────
INDEX 2 ─ Runoff Risk Index (RunRI)
──────────────────────────────────────────────────────────────────────────────
Definition:
  RunRI estimates the probability that rainfall will result in surface runoff
  rather than infiltration, using a modified Rational-Method / SCS-CN approach.

Formula:

  RunRI = 0.20·S_norm + 0.25·SM + 0.25·C_lu + 0.15·I_norm + 0.15·(1 − NDVI)

  where:
    S_norm  = slope_degrees / 45      # normalised slope
    SM      = soil_moisture           # saturation proxy
    C_lu    = SCS Curve Number coefficient by land use
    I_norm  = tanh(I / 50)            # non-linear intensity scaling
    NDVI    = vegetation cover proxy

──────────────────────────────────────────────────────────────────────────────
INDEX 3 ─ Ecosystem Sensitivity Index (ESI)
──────────────────────────────────────────────────────────────────────────────
  ESI = 0.30·(1−RRI) + 0.25·RunRI + 0.20·CS + 0.15·E_sens + 0.10·(1−BD_norm)

  where CS = climate stress composite; E_sens = elevation sensitivity.
"""

import numpy as np
import pandas as pd


# ── Look-up tables ─────────────────────────────────────────────────────────

# SCS Curve-Number inspired runoff coefficients
LAND_USE_RUNOFF: dict[str, float] = {
    "Forest":      0.15,
    "Grassland":   0.30,
    "Agriculture": 0.55,
    "Wetland":     0.10,
    "Urban":       0.85,
    "Barren":      0.70,
}

# Ecosystem degradation penalty (reduces RRI)
LAND_USE_DEGRADATION: dict[str, float] = {
    "Forest":      0.05,
    "Grassland":   0.15,
    "Agriculture": 0.35,
    "Wetland":     0.10,
    "Urban":       0.70,
    "Barren":      0.60,
}

# Ecological recovery speed (used by supervised model target engineering)
LAND_USE_RECOVERY: dict[str, float] = {
    "Forest":      0.90,
    "Wetland":     0.85,
    "Grassland":   0.70,
    "Agriculture": 0.50,
    "Barren":      0.30,
    "Urban":       0.20,
}


# ── Public API ─────────────────────────────────────────────────────────────

def calculate_rri(df: pd.DataFrame) -> pd.Series:
    """
    Compute the Rainfall Response Index for every row in *df*.

    Parameters
    ----------
    df : pd.DataFrame  –  must contain all standard pipeline columns.

    Returns
    -------
    pd.Series  –  RRI values ∈ [0, 1].
    """
    # 1. NDVI response ratio: current vs. long-term baseline
    r_ndvi      = (df["ndvi_current"] / df["ndvi_baseline"].clip(lower=0.01)).clip(0.5, 2.0)
    ndvi_norm   = (r_ndvi - 0.5) / 1.5          # rescale to [0, 1]

    # 2. Soil moisture (direct, already 0–1)
    sm          = df["soil_moisture"].clip(0.0, 1.0)

    # 3. Rainfall efficiency: Gaussian bell centred at 25 mm/day, σ = 20 mm/day
    #    Models the empirical "intermediate disturbance" optimum for vegetation.
    rf_eff      = np.exp(-0.5 * ((df["rainfall_intensity_mm_day"] - 25.0) / 20.0) ** 2)

    # 4. Biodiversity normalised
    bd_norm     = df["biodiversity_index"].clip(0, 100) / 100.0

    # 5. Degradation penalty from land use
    degradation = df["land_use"].map(LAND_USE_DEGRADATION).fillna(0.50)

    # 6. Weighted composite × resilience factor
    rri = (
        0.35 * ndvi_norm +
        0.25 * sm        +
        0.20 * rf_eff    +
        0.20 * bd_norm
    ) * (1.0 - degradation)

    return rri.clip(0.0, 1.0).rename("rri")


def calculate_runoff_risk(df: pd.DataFrame) -> pd.Series:
    """
    Compute the Runoff Risk Index for every row in *df*.

    Returns
    -------
    pd.Series  –  RunRI values ∈ [0, 1].
    """
    # 1. Slope contribution (steep terrain → faster runoff)
    slope_norm  = (df["slope_degrees"] / 45.0).clip(0.0, 1.0)

    # 2. Soil moisture (near-saturated soils can absorb no more water)
    sm          = df["soil_moisture"].clip(0.0, 1.0)

    # 3. Land-use SCS coefficient
    lu_coeff    = df["land_use"].map(LAND_USE_RUNOFF).fillna(0.50)

    # 4. Rainfall intensity (hyperbolic tangent for non-linear saturation)
    i_norm      = np.tanh(df["rainfall_intensity_mm_day"] / 50.0)

    # 5. Vegetation attenuation (canopy intercepts and delays runoff)
    veg         = df["ndvi_current"].clip(0.0, 1.0)

    runoff_risk = (
        0.20 * slope_norm    +
        0.25 * sm            +
        0.25 * lu_coeff      +
        0.15 * i_norm        +
        0.15 * (1.0 - veg)
    )

    return runoff_risk.clip(0.0, 1.0).rename("runoff_risk")


def calculate_ecosystem_sensitivity(df: pd.DataFrame) -> pd.Series:
    """
    Compute the Ecosystem Sensitivity Index for every row in *df*.

    This index must be called **after** RRI and RunRI have been attached
    as columns to *df* (which the pipeline does automatically).

    Returns
    -------
    pd.Series  –  ESI values ∈ [0, 1].
    """
    rri     = df["rri"]
    runoff  = df["runoff_risk"]

    # Climate stress: deviation from optimal 22 °C + low humidity penalty
    temp_stress     = (np.abs(df["temperature_celsius"] - 22.0) / 30.0).clip(0.0, 1.0)
    humidity_stress = 1.0 - df["humidity_percent"].clip(0.0, 100.0) / 100.0
    climate_stress  = (0.60 * temp_stress + 0.40 * humidity_stress).clip(0.0, 1.0)

    # Elevation sensitivity: high-altitude ecosystems are fragile (logarithmic)
    elev_sens = (np.log1p(df["elevation_m"]) / np.log1p(5000.0)).clip(0.0, 1.0)

    # Biodiversity deficit
    bd_deficit = 1.0 - df["biodiversity_index"].clip(0, 100) / 100.0

    esi = (
        0.30 * (1.0 - rri)   +
        0.25 * runoff        +
        0.20 * climate_stress +
        0.15 * elev_sens     +
        0.10 * bd_deficit
    ).clip(0.0, 1.0)

    return esi.rename("ecosystem_sensitivity")


def apply_all_indices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience wrapper: compute RRI, RunRI, and ESI and attach them as
    new columns.  The input DataFrame is not mutated.

    Returns
    -------
    pd.DataFrame  –  original columns + rri, runoff_risk, ecosystem_sensitivity.
    """
    out = df.copy()
    out["rri"]                  = calculate_rri(out).values
    out["runoff_risk"]          = calculate_runoff_risk(out).values
    out["ecosystem_sensitivity"] = calculate_ecosystem_sensitivity(out).values
    return out


def classify_rri(val: float) -> str:
    """Return a human-readable resilience class label."""
    if val >= 0.70:  return "High Resilience"
    if val >= 0.45:  return "Moderate Resilience"
    if val >= 0.25:  return "Low Resilience"
    return "Critical"


def classify_runoff(val: float) -> str:
    """Return a human-readable runoff risk label."""
    if val >= 0.75:  return "Extreme Risk"
    if val >= 0.55:  return "High Risk"
    if val >= 0.35:  return "Moderate Risk"
    return "Low Risk"
