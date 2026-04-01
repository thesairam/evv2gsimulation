"""
API data module — fetches real weather data from Open-Meteo (free, no API key).

Provides 15-minute resolution weather data for:
  - Historical periods (archive API)
  - Forecast periods (forecast API)
  - Latest available weather (for live updates)
"""

import requests
import pandas as pd
from datetime import datetime, timedelta

from . import config


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename Open-Meteo columns to our internal names."""
    rename_map = {
        "temperature_2m": "temperature_c",
        "wind_speed_10m": "wind_speed_ms",
        "wind_gusts_10m": "wind_gusts_ms",
        "cloud_cover": "cloud_cover_pct",
        "shortwave_radiation": "shortwave_radiation_wm2",
        "direct_normal_irradiance": "direct_radiation_wm2",
        "diffuse_radiation": "diffuse_radiation_wm2",
        "precipitation": "precipitation_mm",
        "relative_humidity_2m": "relative_humidity_pct",
    }
    df = df.rename(columns=rename_map)
    return df


def fetch_historical_weather(days_back: int = None) -> pd.DataFrame:
    """
    Fetch past N days of 15-min weather using Open-Meteo Forecast API with past_days.

    The archive API only supports hourly data, so we use the forecast endpoint
    with the past_days parameter to get real 15-minute resolution data.

    Args:
        days_back: Number of days to fetch. Defaults to config.HISTORICAL_DAYS.

    Returns:
        DataFrame with datetime index and standardized weather columns.
    """
    if days_back is None:
        days_back = config.HISTORICAL_DAYS

    # Forecast API supports past_days up to 92 for minutely_15
    past_days = min(days_back, 92)

    params = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "minutely_15": ",".join(config.WEATHER_VARIABLES),
        "past_days": past_days,
        "forecast_days": 1,
        "timezone": config.TIMEZONE,
        "wind_speed_unit": "ms",
    }

    print(f"Fetching historical weather ({past_days} days back, 15-min resolution)...")
    resp = requests.get(config.OPEN_METEO_FORECAST_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    minutely = data["minutely_15"]
    df = pd.DataFrame(minutely)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    df = _standardize_columns(df)
    df.index.name = "datetime"

    # Keep only past data (exclude forecast portion)
    now = pd.Timestamp.now(tz=None)
    df = df[df.index <= now]

    print(f"  Fetched {len(df)} rows ({df.index[0]} → {df.index[-1]})")
    return df


def fetch_forecast_weather(hours_ahead: int = None) -> pd.DataFrame:
    """
    Fetch upcoming forecast weather from Open-Meteo Forecast API.

    Args:
        hours_ahead: Hours of forecast to fetch. Defaults to config.FORECAST_HOURS.

    Returns:
        DataFrame with datetime index and standardized weather columns.
    """
    if hours_ahead is None:
        hours_ahead = config.FORECAST_HOURS

    params = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "minutely_15": ",".join(config.WEATHER_VARIABLES),
        "forecast_hours": hours_ahead,
        "timezone": config.TIMEZONE,
        "wind_speed_unit": "ms",
    }

    print(f"Fetching forecast weather ({hours_ahead}h ahead)...")
    resp = requests.get(config.OPEN_METEO_FORECAST_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    minutely = data["minutely_15"]
    df = pd.DataFrame(minutely)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    df = _standardize_columns(df)
    df.index.name = "datetime"

    print(f"  Fetched {len(df)} forecast rows ({df.index[0]} → {df.index[-1]})")
    return df


def fetch_latest_weather() -> pd.DataFrame:
    """
    Fetch the most recent weather data available.
    Uses forecast API with a small window to get the latest observations.

    Returns:
        DataFrame with the most recent weather data points.
    """
    params = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "minutely_15": ",".join(config.WEATHER_VARIABLES),
        "past_minutely_15": 4,
        "forecast_minutely_15": 4,
        "timezone": config.TIMEZONE,
        "wind_speed_unit": "ms",
    }

    resp = requests.get(config.OPEN_METEO_FORECAST_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    minutely = data["minutely_15"]
    df = pd.DataFrame(minutely)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    df = _standardize_columns(df)
    df.index.name = "datetime"

    return df


def fetch_combined_data(historical_days: int = None, forecast_hours: int = None) -> tuple:
    """
    Fetch both historical and forecast data in one call.

    Returns:
        (historical_df, forecast_df) tuple of DataFrames.
    """
    historical_df = fetch_historical_weather(historical_days)
    forecast_df = fetch_forecast_weather(forecast_hours)
    return historical_df, forecast_df
