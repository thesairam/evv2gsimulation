"""Fetch ENTSO-E day-ahead prices for configured countries and plot them."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
import pandas as pd
from entsoe import EntsoePandasClient

# Use a non-interactive backend so the script works in headless environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  # isort: skip

CONFIG_NAME = "config.json"

COUNTRY_LABELS = {
    "DE_LU": "Germany/Luxembourg",
    "NL": "Netherlands",
    "FR": "France",
}


def load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def resolve_window(cfg: Dict) -> Tuple[pd.Timestamp, pd.Timestamp]:
    tz = cfg.get("timezone", "Europe/Brussels")
    now = pd.Timestamp.now(tz=tz)
    start_cfg = cfg.get("start_date")
    end_cfg = cfg.get("end_date")
    if start_cfg:
        start = pd.Timestamp(start_cfg, tz=tz)
    else:
        years = cfg.get("years_back", 2)
        start = now - pd.DateOffset(years=years)
    end = pd.Timestamp(end_cfg, tz=tz) if end_cfg else now
    return start, end


def fetch_series(
    client: EntsoePandasClient,
    country_code: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    _cfg: Dict,
) -> pd.Series:
    return client.query_day_ahead_prices(country_code=country_code, start=start, end=end)


def plot_series(df: pd.DataFrame, png_path: Path) -> None:
    rename_map = {code: COUNTRY_LABELS.get(code, code) for code in df.columns}
    ax = df.rename(columns=rename_map).plot(figsize=(12, 6))
    ax.set_title("ENTSO-E Day-Ahead Price (EUR/MWh)")
    ax.set_xlabel("Date")
    ax.set_ylabel("EUR/MWh")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path)


def main(config_path: Path | None = None) -> None:
    cfg_path = config_path or Path(__file__).with_name(CONFIG_NAME)
    cfg = load_config(cfg_path)

    client = EntsoePandasClient(api_key=cfg["api_token"])
    start, end = resolve_window(cfg)

    frames: Dict[str, pd.Series] = {}
    for country_code in cfg.get("countries", []):
        try:
            series = fetch_series(client, country_code, start, end, cfg)
            resample_rule = cfg.get("resample")
            if resample_rule:
                series = series.resample(resample_rule).mean()
            frames[country_code] = series
        except Exception as exc:  # pragma: no cover - simple logging
            print(f"Skipping {country_code}: {exc!r}")

    if not frames:
        raise SystemExit(f"No data fetched between {start} and {end}; check token, countries, or network.")

    df = pd.DataFrame(frames)
    cache_csv = cfg.get("cache_csv", "entsoe_day_ahead_prices.csv")
    plot_png = cfg.get("plot_png", "entsoe_day_ahead_prices.png")

    df.to_csv(cfg_path.parent / cache_csv, index_label="timestamp")
    plot_series(df, cfg_path.parent / plot_png)

    print(f"Wrote {cache_csv} and {plot_png} to {cfg_path.parent} from {start.date()} to {end.date()}.")


if __name__ == "__main__":
    main()
