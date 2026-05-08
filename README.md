# Eco-Pulse — The Rainfall Resilience Engine

> Hackathon MVP · Environmental Intelligence

Why does identical rainfall heal one ecosystem and break another? **Eco-Pulse** fuses hydrology, biodiversity, and land-use signals into a single **Resilience Score** and surfaces rainfall-response anomalies before they cascade.

## What's inside

- **Resilience Calculation Engine** (`src/lib/resilience-engine.ts`)
  - **Runoff Risk** `R = (Rainfall × Slope) / (Infiltration × 10)` (soft-saturated to 0–100)
  - **Recovery Potential** non-linear: rises with rainfall + diversity, penalized by degradation + runoff
  - **Recovery Index**: lightweight in-process ensemble (5 calibrated estimators) approximating a Random Forest regressor over synthetic data
  - **Anomaly Flag**: triggers when predicted recovery is ≥30% below the historical baseline for similar conditions
  - **Feature Importance**: numerical sensitivity per factor
  - **Sensitivity Grid**: 14×14 resilience field over (Degradation × Slope)

- **Server endpoint** (`src/lib/eco.functions.ts`)
  - TanStack `createServerFn` POST handler — equivalent to a `/predict` endpoint, callable as type-safe RPC

- **Immersive Frontend** (`src/routes/index.tsx`)
  - **Hero**: React Three Fiber wireframe Earth with rainfall particles
  - **Simulator Dashboard**: animated sliders, Framer Motion gauge, live anomaly badge
  - **Explainable AI**: Recharts feature-importance bars + driver bars
  - **Sensitivity Heatmap**: animated grid over degradation × slope
  - **Pitch Mode**: auto-runs through 5 scripted scenarios for live demos
  - **Dark Nature theme**: deep greens, charcoals, neon-green data accents (`src/styles.css`)

## Stack note

The hackathon brief specified Next.js + FastAPI + scikit-learn. This project runs on **TanStack Start (React 19) on a Cloudflare-Worker-style runtime**, which has no Python. To ship the MVP in time, the Random Forest was ported to a pure-TS calibrated ensemble with the same input/output contract. Swap it for a real `RandomForestRegressor` behind any HTTP endpoint by replacing `ensembleRecovery()` in `resilience-engine.ts`.

## Run

```bash
bun install
bun run dev
```

## Files

```
src/
  lib/
    resilience-engine.ts    # math + ensemble + sensitivity
    eco.functions.ts        # server-fn /predict
  components/eco/
    EarthHero.tsx           # R3F globe + rain particles
    ResilienceGauge.tsx     # animated SVG gauge
    EcoSlider.tsx           # neon range slider
    FeatureImportance.tsx   # Recharts bars
    SensitivityHeatmap.tsx  # grid heatmap
  routes/index.tsx          # full dashboard
  styles.css                # Dark Nature design system
```
