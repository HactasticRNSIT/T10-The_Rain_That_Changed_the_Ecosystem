"""
models/anomaly_detection.py
============================
Unsupervised anomaly detection using Isolation Forest.

Scientific rationale
────────────────────
Isolation Forest (Liu et al., 2008) isolates observations by randomly
partitioning the feature space.  Anomalous points – those deviating from
expected rainfall-to-vegetation dynamics – require fewer splits to isolate
and therefore receive lower anomaly scores.

Feature set selected to capture multi-variate ecological surprise:
  • Rainfall intensity  → magnitude of the forcing event
  • Soil moisture       → antecedent condition
  • NDVI (current)      → vegetation response
  • Biodiversity index  → structural complexity
  • Temperature         → thermal modulation
  • Humidity            → atmospheric dryness / VPD proxy

An observation is flagged as anomalous if its Isolation Score falls in
the lowest *contamination* fraction of the dataset – meaning its
multivariate signature cannot be explained by the dominant ecological
patterns in the data.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# Features used for anomaly detection
ANOMALY_FEATURES = [
    "rainfall_intensity_mm_day",
    "soil_moisture",
    "ndvi_current",
    "biodiversity_index",
    "temperature_celsius",
    "humidity_percent",
]


def run_anomaly_detection(
    df: pd.DataFrame,
    contamination: float = 0.10,
    n_estimators: int = 300,
    seed: int = 42,
) -> tuple[pd.Series, pd.Series]:
    """
    Detect anomalous ecological zones using Isolation Forest.

    Parameters
    ----------
    df            : DataFrame with required feature columns.
    contamination : Expected proportion of outliers in the dataset (0–0.5).
    n_estimators  : Number of isolation trees.
    seed          : Random state for reproducibility.

    Returns
    -------
    is_anomaly    : pd.Series[bool]  – True where anomaly is detected.
    anomaly_score : pd.Series[float] – Normalised anomaly score ∈ [0, 1].
                    Higher score = more anomalous.
    """
    # ── 1. Build feature matrix ────────────────────────────────────────────
    X = df[ANOMALY_FEATURES].copy()
    # Impute any missing values with the column median
    X = X.fillna(X.median())

    # ── 2. Standardise (Isolation Forest is not scale-invariant in practice
    #        because of asymmetric value ranges across features) ────────────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── 3. Fit Isolation Forest ────────────────────────────────────────────
    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples="auto",
        random_state=seed,
        n_jobs=-1,
    )
    iso.fit(X_scaled)

    # ── 4. Predictions & scores ────────────────────────────────────────────
    # predict(): +1 = inlier, -1 = outlier
    labels        = iso.predict(X_scaled)
    raw_scores    = iso.score_samples(X_scaled)   # more negative = more anomalous

    # Normalise raw scores to [0, 1] where 1 = most anomalous
    min_s, max_s  = raw_scores.min(), raw_scores.max()
    norm_scores   = 1.0 - (raw_scores - min_s) / (max_s - min_s + 1e-9)

    is_anomaly    = pd.Series(labels == -1, index=df.index, name="is_anomaly")
    anomaly_score = pd.Series(norm_scores,  index=df.index, name="anomaly_score")

    return is_anomaly, anomaly_score


def anomaly_summary(df: pd.DataFrame) -> dict:
    """Return a summary dict for display in the dashboard."""
    n_total   = len(df)
    n_anomaly = int(df["is_anomaly"].sum())
    top = (
        df[df["is_anomaly"]]
        .nlargest(3, "anomaly_score")[
            ["latitude", "longitude", "land_use", "anomaly_score", "rri"]
        ]
        .reset_index(drop=True)
    )
    return {
        "total_points":    n_total,
        "anomaly_count":   n_anomaly,
        "anomaly_pct":     100.0 * n_anomaly / max(n_total, 1),
        "top_anomalies":   top,
    }
