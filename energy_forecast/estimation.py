"""
Solar and wind power estimation from weather data.

Uses simple physical models:
  - Solar: radiation-based with efficiency factor
  - Wind: standard power curve with cut-in/rated/cut-out speeds
  - Curtailment: grid capacity constraint
"""

import numpy as np
import pandas as pd

from . import config


def estimate_solar_kw(radiation_wm2: np.ndarray,
                      capacity_kw: float = None) -> np.ndarray:
    """
    Estimate solar generation from shortwave radiation.

    Model: solar_kw = capacity * (radiation / 1000) * efficiency
    - 1000 W/m² = standard test conditions (STC)
    - efficiency ~0.85 accounts for inverter losses, temperature derating

    Args:
        radiation_wm2: Shortwave radiation in W/m².
        capacity_kw: Installed solar capacity. Defaults to config value.

    Returns:
        Estimated solar power output in kW.
    """
    if capacity_kw is None:
        capacity_kw = config.SOLAR_CAPACITY_KW

    radiation = np.asarray(radiation_wm2, dtype=float)
    solar_kw = capacity_kw * (radiation / 1000.0) * config.SOLAR_EFFICIENCY
    return np.maximum(solar_kw, 0)


def estimate_wind_kw(wind_speed_ms: np.ndarray,
                     capacity_kw: float = None) -> np.ndarray:
    """
    Estimate wind generation from wind speed using a standard power curve.

    Power curve:
    - Below cut-in (3 m/s): 0
    - Cut-in to rated (3-12 m/s): cubic growth
    - Rated to cut-out (12-25 m/s): rated capacity
    - Above cut-out (25 m/s): 0

    Args:
        wind_speed_ms: Wind speed in m/s.
        capacity_kw: Installed wind capacity. Defaults to config value.

    Returns:
        Estimated wind power output in kW.
    """
    if capacity_kw is None:
        capacity_kw = config.WIND_CAPACITY_KW

    ws = np.asarray(wind_speed_ms, dtype=float)
    cut_in = config.WIND_CUT_IN_MS
    rated = config.WIND_RATED_MS
    cut_out = config.WIND_CUT_OUT_MS

    wind_kw = np.zeros_like(ws)

    # Cubic region: cut-in to rated
    cubic_mask = (ws >= cut_in) & (ws < rated)
    wind_kw[cubic_mask] = capacity_kw * ((ws[cubic_mask] - cut_in) / (rated - cut_in)) ** 3

    # Rated region: rated to cut-out
    rated_mask = (ws >= rated) & (ws <= cut_out)
    wind_kw[rated_mask] = capacity_kw

    return wind_kw


def compute_curtailment(solar_kw: np.ndarray, wind_kw: np.ndarray,
                        max_grid_kw: float = None) -> np.ndarray:
    """
    Compute curtailment when total generation exceeds grid capacity.

    Args:
        solar_kw: Solar generation array.
        wind_kw: Wind generation array.
        max_grid_kw: Maximum grid capacity. Defaults to config value.

    Returns:
        Curtailed power in kW.
    """
    if max_grid_kw is None:
        max_grid_kw = config.MAX_GRID_KW

    generation = np.asarray(solar_kw) + np.asarray(wind_kw)
    curtailment = np.maximum(generation - max_grid_kw, 0)
    return curtailment


def compute_net_generation(solar_kw: np.ndarray, wind_kw: np.ndarray,
                           curtailment_kw: np.ndarray) -> np.ndarray:
    """Compute net generation = solar + wind - curtailment."""
    return np.asarray(solar_kw) + np.asarray(wind_kw) - np.asarray(curtailment_kw)


def add_estimations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add solar_kw, wind_kw, curtailment_kw, net_generation_kw columns
    to a weather DataFrame.

    Args:
        df: DataFrame with weather columns (shortwave_radiation_wm2, wind_speed_ms).

    Returns:
        DataFrame with added estimation columns.
    """
    df = df.copy()
    df["solar_kw"] = estimate_solar_kw(df["shortwave_radiation_wm2"].values)
    df["wind_kw"] = estimate_wind_kw(df["wind_speed_ms"].values)
    df["curtailment_kw"] = compute_curtailment(df["solar_kw"].values, df["wind_kw"].values)
    df["net_generation_kw"] = compute_net_generation(
        df["solar_kw"].values, df["wind_kw"].values, df["curtailment_kw"].values
    )
    return df
