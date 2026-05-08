"""
utils/data_generator.py
=======================
Synthetic ecological data generator for out-of-the-box testing.

All distributions are parameterised from real-world environmental studies:
  - Rainfall: log-normal (common for precipitation)
  - NDVI: beta distribution (bounded 0-1, right-skewed for vegetated terrain)
  - Soil moisture: beta distribution (bounded 0-1)
  - Biodiversity: Gaussian, clipped to [0, 100]
  - Elevation: log-normal (right-skewed, most land below 1000 m)
"""

import numpy as np
import pandas as pd


# ── Land-use class proportions (approximate global coverage mix) ───────────
LAND_USE_CLASSES = ["Forest", "Agriculture", "Urban", "Wetland", "Grassland", "Barren"]
LAND_USE_PROBS   = [0.25,     0.30,         0.15,    0.10,      0.15,         0.05]


def generate_dummy_data(n_samples: int = 60, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic ecological dataset.

    Parameters
    ----------
    n_samples : int
        Number of spatial observation points to generate.
    seed : int
        NumPy random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per observation point and all required
        input features for the Eco-Resilience pipeline.
    """
    rng = np.random.default_rng(seed)

    # ── Geographic coordinates ─────────────────────────────────────────────
    # Spread across the Indian subcontinent for a realistic demo
    lats = rng.uniform(8.0, 28.0, n_samples)    # 8°N – 28°N
    lons = rng.uniform(72.0, 88.0, n_samples)   # 72°E – 88°E

    # ── Rainfall variables ─────────────────────────────────────────────────
    # Intensity: log-normal with mean ≈ exp(3.5) ≈ 33 mm/day
    rainfall_intensity  = rng.lognormal(mean=3.5, sigma=0.8, size=n_samples)
    rainfall_duration   = rng.uniform(1.0, 72.0, n_samples)   # hours
    rainfall_frequency  = rng.uniform(1.0, 30.0, n_samples)   # events / month

    # ── Soil moisture [0, 1] ───────────────────────────────────────────────
    soil_moisture = rng.beta(2.0, 3.0, n_samples)  # skewed toward drier values

    # ── NDVI ──────────────────────────────────────────────────────────────
    # Current NDVI: beta, scaled to [0.1, 0.9]
    ndvi_current  = rng.beta(3.0, 2.0, n_samples) * 0.80 + 0.10
    # Baseline NDVI slightly below current (long-term average)
    ndvi_baseline = np.clip(ndvi_current - rng.normal(0.0, 0.05, n_samples), 0.05, 1.0)

    # ── Biodiversity index [0, 100] ────────────────────────────────────────
    biodiversity = np.clip(rng.normal(60.0, 20.0, n_samples), 0.0, 100.0)

    # ── Terrain ───────────────────────────────────────────────────────────
    elevation = np.clip(rng.lognormal(5.5, 1.0, n_samples), 0.0, 6000.0)  # metres
    slope     = np.clip(rng.exponential(5.0, n_samples), 0.0, 45.0)       # degrees

    # ── Weather ───────────────────────────────────────────────────────────
    temperature = rng.normal(28.0, 8.0, n_samples)    # °C
    humidity    = rng.uniform(30.0, 100.0, n_samples) # %

    # ── Land use (categorical) ────────────────────────────────────────────
    land_use = rng.choice(LAND_USE_CLASSES, size=n_samples, p=LAND_USE_PROBS)

    df = pd.DataFrame({
        "latitude":                    lats,
        "longitude":                   lons,
        "rainfall_intensity_mm_day":   rainfall_intensity,
        "rainfall_duration_hours":     rainfall_duration,
        "rainfall_frequency_per_month":rainfall_frequency,
        "soil_moisture":               soil_moisture,
        "ndvi_current":                ndvi_current,
        "ndvi_baseline":               ndvi_baseline,
        "biodiversity_index":          biodiversity,
        "elevation_m":                 elevation,
        "slope_degrees":               slope,
        "temperature_celsius":         temperature,
        "humidity_percent":            humidity,
        "land_use":                    land_use,
    })

    return df


def get_sample_csv_string() -> str:
    """Return a small sample CSV string for users to download as a template."""
    df = generate_dummy_data(n_samples=5, seed=0)
    return df.to_csv(index=False)
