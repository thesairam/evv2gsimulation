"""
Configuration for the energy forecasting pipeline.
All settings are centralized here.
"""

# Location (default: Berlin)
LATITUDE = 52.52
LONGITUDE = 13.41
TIMEZONE = "Europe/Berlin"

# Plant capacity assumptions
SOLAR_CAPACITY_KW = 500
WIND_CAPACITY_KW = 300

# Grid constraint
MAX_GRID_KW = 700

# Wind turbine power curve
WIND_CUT_IN_MS = 3.0
WIND_RATED_MS = 12.0
WIND_CUT_OUT_MS = 25.0

# Solar efficiency (inverter losses, temperature derating)
SOLAR_EFFICIENCY = 0.85

# Data settings
HISTORICAL_DAYS = 14
FORECAST_HOURS = 24

# Model retraining interval (number of 15-min steps; 4 = every hour)
RETRAIN_INTERVAL = 4

# Live loop interval (seconds); 900 = 15 minutes
LIVE_INTERVAL_SECONDS = 900

# Dashboard rolling window (hours of history to show)
DASHBOARD_WINDOW_HOURS = 6

# Open-Meteo API endpoints
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Weather variables to fetch (minutely_15 resolution)
WEATHER_VARIABLES = [
    "temperature_2m",
    "wind_speed_10m",
    "wind_gusts_10m",
    "cloud_cover",
    "shortwave_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
    "precipitation",
    "relative_humidity_2m",
]
