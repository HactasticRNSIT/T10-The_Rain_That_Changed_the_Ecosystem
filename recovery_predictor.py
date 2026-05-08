"""
models/recovery_predictor.py
=============================
Supervised learning model for ecological recovery potential.

Scientific rationale
────────────────────
Ecological recovery potential (ERP) quantifies how quickly and completely
an ecosystem returns to its pre-disturbance state following a stressor
(drought, flood, fire).  It is analogous to the "engineering resilience"
concept of Holling (1973).

Model choice – Random Forest Regressor:
  • Handles mixed numerical/categorical features natively (after encoding).
  • Naturally captures non-linear feature interactions (e.g., NDVI × slope).
  • Robust to outliers and does not require feature scaling.
  • Provides out-of-bag error estimation and feature importance.

Target variable construction (for demo / out-of-the-box use):
  The synthetic target is a scientifically grounded proxy derived from:
    y = 0.30·LU_recovery + 0.25·NDVI + 0.20·SM + 0.15·BD + noise

  In production this would be replaced with actual time-series recovery
  measurements (e.g., months to return to baseline NDVI after disturbance).

Feature engineering applied:
  • Rainfall × NDVI interaction term (joint signal of water + vegetation).
  • Slope × Soil-moisture (terrain-hydrology coupling).
  • Label encoding of land-use category.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

from utils.indices import LAND_USE_RECOVERY


# ── Feature specification ──────────────────────────────────────────────────

BASE_FEATURES = [
    "rainfall_intensity_mm_day",
    "rainfall_duration_hours",
    "rainfall_frequency_per_month",
    "soil_moisture",
    "ndvi_current",
    "ndvi_baseline",
    "biodiversity_index",
    "elevation_m",
    "slope_degrees",
    "temperature_celsius",
    "humidity_percent",
]

ENGINEERED_FEATURES = [
    "rf_ndvi_interaction",   # rainfall × ndvi
    "slope_sm_coupling",     # slope × soil_moisture
    "ndvi_delta",            # ndvi_current − ndvi_baseline
]

CATEGORICAL_FEATURES = ["land_use_encoded"]


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to a copy of *df*."""
    out = df.copy()
    out["rf_ndvi_interaction"] = (
        out["rainfall_intensity_mm_day"] / 100.0 * out["ndvi_current"]
    )
    out["slope_sm_coupling"] = out["slope_degrees"] / 45.0 * out["soil_moisture"]
    out["ndvi_delta"]        = out["ndvi_current"] - out["ndvi_baseline"]
    return out


def _build_target(df: pd.DataFrame, noise_std: float = 0.03) -> pd.Series:
    """
    Construct a proxy target for recovery potential.

    In a production system this is replaced by observed recovery time-series.
    """
    rng          = np.random.default_rng(42)
    lu_recovery  = df["land_use"].map(LAND_USE_RECOVERY).fillna(0.50)
    bd_norm      = df["biodiversity_index"].clip(0, 100) / 100.0

    y = (
        0.30 * lu_recovery              +
        0.25 * df["ndvi_current"]       +
        0.20 * df["soil_moisture"]      +
        0.15 * bd_norm                  +
        0.10 * (1.0 - df["slope_degrees"] / 45.0)
    ) + rng.normal(0.0, noise_std, len(df))

    return y.clip(0.0, 1.0)


def train_recovery_model(
    df: pd.DataFrame,
    test_size: float = 0.20,
    n_estimators: int = 300,
    seed: int = 42,
) -> dict:
    """
    Train a Random Forest model to predict recovery potential.

    Parameters
    ----------
    df          : Raw input DataFrame (must include all standard columns).
    test_size   : Fraction reserved for evaluation.
    n_estimators: Number of trees.
    seed        : Random state.

    Returns
    -------
    dict with keys:
        model        – trained RandomForestRegressor
        label_enc    – fitted LabelEncoder for land_use
        feature_cols – list of feature column names used
        metrics      – dict { train_r2, test_r2, test_mae }
        importances  – pd.Series of feature importances
    """
    # ── Feature engineering ────────────────────────────────────────────────
    data = _engineer_features(df)

    # ── Encode land use ───────────────────────────────────────────────────
    le = LabelEncoder()
    data["land_use_encoded"] = le.fit_transform(data["land_use"])

    feature_cols = BASE_FEATURES + ENGINEERED_FEATURES + CATEGORICAL_FEATURES

    X = data[feature_cols].fillna(data[feature_cols].median())
    y = _build_target(data)

    # ── Train / test split ─────────────────────────────────────────────────
    if len(X) < 10:
        # Too few samples to split — train on everything, skip test metrics
        X_tr, y_tr = X, y
        X_te, y_te = X, y
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )

    # ── Model ─────────────────────────────────────────────────────────────
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=10,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X_tr, y_tr)

    # ── Metrics ───────────────────────────────────────────────────────────
    y_pred_tr = rf.predict(X_tr)
    y_pred_te = rf.predict(X_te)
    metrics = {
        "train_r2": float(r2_score(y_tr, y_pred_tr)),
        "test_r2":  float(r2_score(y_te, y_pred_te)),
        "test_mae": float(mean_absolute_error(y_te, y_pred_te)),
    }

    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(
        ascending=False
    )

    return {
        "model":        rf,
        "label_enc":    le,
        "feature_cols": feature_cols,
        "metrics":      metrics,
        "importances":  importances,
    }


def predict_recovery(
    result: dict,
    df: pd.DataFrame,
) -> np.ndarray:
    """
    Run inference on *df* using a previously trained model bundle.

    Parameters
    ----------
    result : dict returned by :func:`train_recovery_model`.
    df     : Raw input DataFrame.

    Returns
    -------
    np.ndarray of float ∈ [0, 1], one value per row.
    """
    model        = result["model"]
    le           = result["label_enc"]
    feature_cols = result["feature_cols"]

    data = _engineer_features(df)

    # Safe label encoding: unseen categories → first known class
    safe_lu = data["land_use"].where(
        data["land_use"].isin(le.classes_), other=le.classes_[0]
    )
    data["land_use_encoded"] = le.transform(safe_lu)

    X = data[feature_cols].fillna(data[feature_cols].median())
    return model.predict(X).clip(0.0, 1.0)
