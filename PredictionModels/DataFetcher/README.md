# DataFetcher

Fetch ENTSO-E day-ahead prices for multiple bidding zones and plot the result.

## Config (config.json)
- api_token: ENTSO-E API token.
- countries: list of bidding zone codes (e.g., "DE_LU", "NL", "FR").
- metric: label for the price series; currently informational only.
- market: day-ahead only (intraday removed for simplicity).
- start_date: ISO date/time (e.g., "2023-01-01") to override the rolling window start.
- end_date: ISO date/time to override the end; null uses "now" in timezone.
- years_back: integer years to look back when start_date is null (default 2).
- resample: pandas frequency alias to downsample (e.g., "D" daily, "W" weekly, "M" monthly); null keeps the original resolution.
- cache_csv: output filename for the fetched data.
- plot_png: output filename for the plot.
- timezone: tz name used for parsing start/end and determining "now" (e.g., "Europe/Brussels").

## Run
```
python3 -m pip install -r requirements.txt
python3 fetch_entsoe_data.py
```
Outputs: the CSV and PNG named in config.json, written beside the script.
