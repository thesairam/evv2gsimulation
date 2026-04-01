# ⚡ CLAUDE PROJECT SPEC — Energy Forecasting + LLM Analytics (15-Min Real Data)

---

## 🎯 Goal

Build a modular energy forecasting pipeline using **real data from free/open APIs** that predicts:

> **Net Power Output (kW) = Solar (estimated) + Wind (estimated) − Curtailment (derived)**

The system must:

* Use **real 15-minute interval data** from free APIs (NO synthetic/fake data)
* Fetch **historical weather** + **forecast weather** from **Open-Meteo** (free, no key)
* Estimate solar/wind generation from weather using simple physical models
* Use **XGBoost** for prediction
* Include a **live rolling dashboard** that updates every 15 minutes
* Show **past data, predictions, and actuals** side-by-side as they arrive
* Support **model self-improvement** — retrain on newly arrived actuals
* Include an **LLM layer for analysis, insights, and Q&A**
* Be **Dockerized** for easy launch

---

## 🧠 System Overview

```
Open-Meteo API (Historical + Forecast Weather)
        ↓
Solar/Wind Estimation (physical models from weather)
        ↓
Feature Engineering
        ↓
XGBoost Model (trains on historical, predicts next 15 min)
        ↓
Live Rolling Dashboard
   ├── Past actuals (scrolling window)
   ├── Predicted values (next 15 min)
   ├── Actual vs Predicted comparison (as actuals arrive)
   └── Running metrics (RMSE, MAE, MAPE, R²)
        ↓
🧠 LLM Analytics Layer (periodic analysis + Q&A)
        ↓
Insights + Recommendations
```

---

## 🧱 1. Data Sources — Real APIs Only

### Open-Meteo (FREE, no API key)

**Base URL:** `https://api.open-meteo.com/v1/forecast`

Fetch at **15-minute resolution** (`minutely_15`):

| Variable | Open-Meteo Parameter | Use |
|---|---|---|
| Temperature | `temperature_2m` | Feature |
| Wind Speed | `wind_speed_10m` | Wind power estimation |
| Wind Gusts | `wind_gusts_10m` | Optional feature |
| Cloud Cover | `cloud_cover` | Solar reduction factor |
| Solar Radiation | `shortwave_radiation` | Solar power estimation |
| Direct Radiation | `direct_normal_irradiance` | Better solar estimate |
| Diffuse Radiation | `diffuse_radiation` | Better solar estimate |
| Precipitation | `precipitation` | Feature |
| Relative Humidity | `relative_humidity_2m` | Feature |

**Historical weather:** Use Open-Meteo Archive API
`https://archive-api.open-meteo.com/v1/archive`

**Configuration:**

```python
LOCATION = {
    "latitude": 52.52,    # Berlin (default, configurable)
    "longitude": 13.41,
    "timezone": "Europe/Berlin"
}

# Plant capacity assumptions (configurable)
SOLAR_CAPACITY_KW = 500
WIND_CAPACITY_KW = 300
```

---

### Data Fetching Functions

```python
def fetch_historical_weather(days_back: int = 14) -> pd.DataFrame:
    """Fetch past N days of 15-min weather from Open-Meteo Archive API."""

def fetch_forecast_weather(hours_ahead: int = 24) -> pd.DataFrame:
    """Fetch upcoming forecast weather from Open-Meteo Forecast API."""

def fetch_latest_weather() -> pd.DataFrame:
    """Fetch latest available weather (for live updates)."""
```

All functions must return a DataFrame with:

```
datetime index (15-min, UTC or local timezone)
temperature_c
wind_speed_ms
wind_gusts_ms
cloud_cover_pct
shortwave_radiation_wm2
direct_radiation_wm2
diffuse_radiation_wm2
precipitation_mm
relative_humidity_pct
```

---

## ☀️ 2. Solar Power Estimation (from weather)

Estimate solar generation from radiation data:

```python
def estimate_solar_kw(radiation_wm2: float, capacity_kw: float = 500) -> float:
    """
    Simple model: solar_kw = capacity * (radiation / 1000) * efficiency
    - 1000 W/m² = standard test conditions (STC)
    - efficiency factor ~0.85 (inverter losses, temperature derating)
    """
    efficiency = 0.85
    return capacity_kw * (radiation_wm2 / 1000.0) * efficiency
```

Can also use direct + diffuse for better accuracy.

---

## 🌬️ 3. Wind Power Estimation (from weather)

Use wind speed + standard power curve:

```python
def estimate_wind_kw(wind_speed_ms: float, capacity_kw: float = 300) -> float:
    """
    Power curve:
    - Below cut-in (3 m/s): 0
    - 3-12 m/s: cubic growth
    - 12-25 m/s: rated capacity
    - Above cut-out (25 m/s): 0
    """
```

---

## 🚫 4. Curtailment & Net Generation

```python
# Assume a grid capacity limit
MAX_GRID_KW = 700
generation = solar_kw + wind_kw
curtailment = max(0, generation - MAX_GRID_KW)
net_generation_kw = generation - curtailment
```

---

## ⏱️ 5. Time Setup

* Frequency: `15min`
* 96 points per day
* Historical: fetch **14–30 days** for training
* Forecast: fetch **next 24h** for prediction display
* Live loop: every **15 minutes**, fetch new actual, compare to prediction

---

## 🧮 6. Feature Engineering

### Time Features

* Hour (float)
* Day of week (0-6)
* Is weekend (bool)

### Weather Features (direct from API)

* temperature_c
* wind_speed_ms
* cloud_cover_pct
* shortwave_radiation_wm2
* precipitation_mm
* relative_humidity_pct

### Lag Features (CRITICAL)

```
[1, 2, 4, 96] steps
```

Applied to:

* net_generation_kw (mandatory)
* solar_kw (estimated)
* wind_kw (estimated)

### Rolling Features

* 1-hour mean (4 steps)
* 6-hour mean (24 steps)

---

## 🎯 7. Target

```python
net_generation_kw = solar_kw + wind_kw - curtailment_kw
```

---

## 🤖 8. Model

```python
xgboost.XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
)
```

### Train/Test Split

* Time-based (NO shuffle)
* Last 20% = test set for initial evaluation
* In live mode: continuously retrain as new data arrives

### Online Learning / Retraining

Every N intervals (e.g., every hour = 4 intervals):

```python
def retrain_model(historical_df, model):
    """Retrain XGBoost on all available data including recent actuals."""
```

This allows the model to **improve over time** as more real data arrives.

---

## 📊 9. Metrics

Compute (updated in real-time during live mode):

* RMSE
* MAE
* MAPE
* R²

---

## 📈 10. Live Rolling Dashboard

### Main Visualization Window

A **continuously updating** plot (saves as HTML or PNG snapshots, or uses matplotlib animation):

```
┌─────────────────────────────────────────────────────────────┐
│                 ENERGY FORECAST DASHBOARD                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  [== Past 6h Actuals ==]  [Pred]  ← time axis →             │
│                                    │                         │
│  ───── Solar (actual)              │                         │
│  ───── Wind (actual)               │                         │
│  ━━━━━ Net Generation (actual)     │  ┈┈┈ Predicted next 15m │
│  - - - Demand estimate             │                         │
│                                    │                         │
│  When actual arrives for predicted  │                         │
│  interval → plot both + show error  │                         │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│  METRICS (running)                                           │
│  RMSE: 12.3 kW | MAE: 8.5 kW | MAPE: 6.2% | R²: 0.94     │
├─────────────────────────────────────────────────────────────┤
│  PREDICTION LOG                                              │
│  14:00 → Pred: 342 kW | Actual: 338 kW | Error: +4 kW      │
│  14:15 → Pred: 351 kW | Actual: 347 kW | Error: +4 kW      │
│  14:30 → Pred: 289 kW | Actual: (awaiting...)               │
└─────────────────────────────────────────────────────────────┘
```

### Required Plots

#### 1. Live Rolling Window (main dashboard)

* Past 6h of actual solar, wind, net generation
* Next 15-min prediction shown as a dot/bar ahead
* When actual arrives, overlay it and compute error
* Running metrics updated in real-time

#### 2. Prediction vs Actual Scatter

* All completed prediction/actual pairs
* 45-degree reference line
* Color-coded by time of day

#### 3. Error Over Time

* Running error (actual - predicted) over the session
* Shows if model is improving after retraining

#### 4. Feature Importance (XGBoost)

* Bar chart, updated after each retrain

#### 5. Weather Conditions Panel

* Current weather snapshot (temp, wind, cloud, radiation)
* Helpful for understanding prediction context

### Dashboard Output Options

* **Option A:** Save PNG snapshots to `output/` every 15 min
* **Option B:** HTML dashboard (using plotly or matplotlib inline)
* **Option C:** Terminal-based text dashboard (for headless Docker)

The default (Docker/headless) should use **Option A + Option C** together.

---

## 🧠 11. LLM Analytics Layer

### Purpose

Enhance the ML pipeline with an LLM that:

* Interprets results from real data
* Explains why predictions were good/bad in specific weather conditions
* Detects patterns (e.g., "model struggles during cloudy→sunny transitions")
* Suggests improvements
* Answers user questions

### LLM Input (Structured)

```json
{
  "metrics": {"rmse": 12.3, "mae": 8.5, "mape": 6.2, "r2": 0.94},
  "feature_importance": [
    {"feature": "gen_lag_1", "importance": 0.35},
    {"feature": "shortwave_radiation_wm2", "importance": 0.22}
  ],
  "recent_predictions": [
    {"time": "14:00", "predicted": 342, "actual": 338, "error": 4},
    {"time": "14:15", "predicted": 351, "actual": 347, "error": 4}
  ],
  "weather_context": {
    "current_temp": 18.5,
    "current_wind": 7.2,
    "current_cloud": 45,
    "current_radiation": 520
  },
  "metadata": {
    "resolution": "15min",
    "training_days": 14,
    "model": "XGBoost",
    "retrain_count": 3,
    "location": "Berlin (52.52, 13.41)"
  }
}
```

### Required Functions

```python
prepare_llm_input()
analyze_results()
ask_llm()
```

### LLM Responsibilities

1. **Performance Summary** — explain metrics in practical terms
2. **Diagnostics** — peak errors, bias, weather correlation issues
3. **Recommendations** — feature/model/data improvements
4. **Q&A Interface** — "Why was error high at 14:00?", "What weather pattern causes worst predictions?"

### LLM Integration

```python
from openai import OpenAI

client = OpenAI()

def analyze_results(llm_input):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

Falls back to **offline analysis** if no API key is provided.

---

## 🔁 12. Execution Flow

```
INITIAL PHASE:
  Fetch 14 days historical weather (Open-Meteo Archive)
        ↓
  Estimate solar/wind from weather
        ↓
  Create features (lags, rolling, time)
        ↓
  Train XGBoost model
        ↓
  Evaluate on test set
        ↓
  Show initial plots + metrics
        ↓
  Run LLM analysis on initial results

LIVE PHASE (every 15 minutes):
        ↓
  Fetch latest weather actual
        ↓
  Estimate solar/wind for that interval
        ↓
  Compare to previous prediction → log error
        ↓
  Create features for next interval
        ↓
  Predict next 15-min generation
        ↓
  Update dashboard (past + prediction + errors)
        ↓
  Every 1h (4 intervals): retrain model on all data
        ↓
  Update running metrics
        ↓
  Periodic LLM analysis (every retrain)
```

---

## 💾 13. Outputs

Save to `output/` directory:

* `energy_dataset.csv` — all historical + live data
* `xgb_energy_model.joblib` — latest trained model
* `predictions_log.csv` — all prediction/actual pairs
* `llm_analysis.txt` — latest LLM analysis
* `dashboard_*.png` — dashboard snapshots (every 15 min)

---

## 🧩 14. Code Structure

```
energy_forecast/
  __init__.py
  api_data.py          # Open-Meteo fetch functions
  estimation.py        # Solar/wind estimation from weather
  features.py          # Feature engineering
  model.py             # XGBoost train/predict/evaluate
  visualization.py     # Dashboard + static plots
  llm_analytics.py     # LLM integration
  config.py            # Location, capacity, API settings

main.py                # Pipeline entry point
Dockerfile
docker-compose.yml
requirements.txt
```

### Key Functions

```python
# api_data.py
fetch_historical_weather(days_back)
fetch_forecast_weather(hours_ahead)
fetch_latest_weather()

# estimation.py
estimate_solar_kw(radiation, capacity)
estimate_wind_kw(wind_speed, capacity)
compute_curtailment(solar, wind, grid_max)
compute_net_generation(solar, wind, curtailment)

# features.py
create_features(df)
get_feature_columns()

# model.py
train_model(df)
predict(model, df)
evaluate_model(y_true, y_pred)
retrain_model(df, existing_model)

# visualization.py
create_dashboard(history_df, predictions_log, metrics)
plot_prediction_scatter(predictions_log)
plot_error_over_time(predictions_log)
plot_feature_importance(model)
save_dashboard_snapshot(fig, path)

# llm_analytics.py
prepare_llm_input(metrics, features, predictions, weather)
analyze_results(llm_input, client)
ask_llm(llm_input, question, client)
```

---

## 🧠 15. Design Principles

* **Real data only** — no synthetic/generated data
* All data from **free, open APIs** (no API keys required for core functionality)
* Modular — easy to add new data sources or swap models
* Live-capable — designed for continuous 15-min update cycles
* Self-improving — model retrains periodically on new actuals
* Dockerized — single `docker compose up` to launch
* Graceful degradation — works offline for LLM (offline analysis mode)

---

## 🚀 16. Future Extensions

* Multi-step forecasting (t+1, t+4, t+96)
* ENTSO-E integration (real electricity prices, free API key)
* Battery/storage optimization layer
* Web dashboard (Flask/Streamlit)
* Alert system (prediction error exceeds threshold)
* Multi-location support

---

## ✅ Final Instruction for Claude

> Build the full pipeline as a clean Python project using the above specifications.
> Use ONLY real data from free/open APIs (Open-Meteo).
> No synthetic or generated data.
> Include a live rolling dashboard that shows past data, predictions, and actuals.
> Model should retrain periodically to improve.
> Dockerize so it is easy to launch with `docker compose up`.

Ensure:

* Modular structure
* Real API data (Open-Meteo, free, no key)
* Solar/wind estimated from weather data
* Clean feature engineering with lags + rolling
* Live dashboard with prediction vs actual comparison
* XGBoost model with periodic retraining
* Integrated LLM analytics + Q&A (with offline fallback)
* All outputs saved to `output/`

---
