"""
Feature engineering module for the energy forecasting pipeline.

Creates time features, lag features, and rolling features from real weather + estimation data.
"""

import pandas as pd


# Features the model will use (order matters for consistency)
FEATURE_COLUMNS = [
    # Time features
    "hour", "day_of_week", "is_weekend",
    # Weather features (from API)
    "temperature_c", "wind_speed_ms", "cloud_cover_pct",
    "shortwave_radiation_wm2", "precipitation_mm", "relative_humidity_pct",
    # Generation lags
    "gen_lag_1", "gen_lag_2", "gen_lag_4", "gen_lag_96",
    # Solar lags
    "solar_lag_1", "solar_lag_4", "solar_lag_96",
    # Wind lags
    "wind_lag_1", "wind_lag_4", "wind_lag_96",
    # Rolling means
    "gen_rolling_1h", "gen_rolling_6h",
]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all features for the XGBoost model.

    Expects a DataFrame with columns:
        net_generation_kw, solar_kw, wind_kw, plus weather columns.

    Returns:
        DataFrame with original columns plus engineered features, NaN rows dropped.
    """
    df = df.copy()

    # --- Time features ---
    df["hour"] = df.index.hour + df.index.minute / 60.0
    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    # --- Lag features for net generation ---
    for lag in [1, 2, 4, 96]:
        df[f"gen_lag_{lag}"] = df["net_generation_kw"].shift(lag)

    # --- Lag features for solar and wind ---
    for lag in [1, 4, 96]:
        df[f"solar_lag_{lag}"] = df["solar_kw"].shift(lag)
        df[f"wind_lag_{lag}"] = df["wind_kw"].shift(lag)

    # --- Rolling mean features for generation ---
    df["gen_rolling_1h"] = df["net_generation_kw"].shift(1).rolling(window=4).mean()
    df["gen_rolling_6h"] = df["net_generation_kw"].shift(1).rolling(window=24).mean()

    # Drop rows with NaN from lagging/rolling
    df = df.dropna(subset=FEATURE_COLUMNS)

    return df


def get_feature_columns() -> list:
    """Return the list of feature column names for the model."""
    return FEATURE_COLUMNS.copy()
