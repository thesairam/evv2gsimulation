#!/usr/bin/env python3
"""
Energy Forecasting Pipeline — Main Entry Point

Two phases:
  1. INITIAL: Fetch historical weather → estimate generation → train model → evaluate → plot
  2. LIVE: Every 15 min → fetch new weather → compare prediction → predict next → retrain hourly
"""

import argparse
import os
import time
import sys
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from energy_forecast import config
from energy_forecast.api_data import fetch_historical_weather, fetch_forecast_weather, fetch_latest_weather
from energy_forecast.estimation import add_estimations
from energy_forecast.features import create_features, get_feature_columns
from energy_forecast.model import (
    train_test_split_temporal, train_model, retrain_model,
    predict, evaluate_model, get_feature_importance, save_model,
)
from energy_forecast.visualization import (
    create_dashboard, plot_initial_evaluation, print_terminal_dashboard,
)
from energy_forecast.llm_analytics import (
    prepare_llm_input, analyze_results, save_analysis, get_openai_client,
)


def parse_args():
    p = argparse.ArgumentParser(description="Energy Forecasting Pipeline (Real Data)")
    p.add_argument("--days", type=int, default=config.HISTORICAL_DAYS,
                   help="Days of historical data to fetch")
    p.add_argument("--output-dir", type=str, default="output")
    p.add_argument("--no-live", action="store_true",
                   help="Run initial phase only, skip live loop")
    p.add_argument("--live-cycles", type=int, default=0,
                   help="Number of live cycles to run (0 = infinite)")
    p.add_argument("--fast-mode", action="store_true",
                   help="Use 60s intervals instead of 15min (for testing)")
    p.add_argument("--openai-key", type=str, default=None)
    return p.parse_args()


def initial_phase(args) -> tuple:
    """
    INITIAL PHASE:
      1. Fetch historical weather from Open-Meteo
      2. Estimate solar/wind from weather
      3. Create features
      4. Train XGBoost model
      5. Evaluate on test set
      6. Save initial plots + model
      7. Run LLM analysis

    Returns:
        (model, full_df, metrics, feature_importance, predictions_log)
    """
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # --- Step 1: Fetch historical weather ---
    print("\n" + "=" * 60)
    print("STEP 1: Fetching Historical Weather (Open-Meteo)")
    print("=" * 60)
    weather_df = fetch_historical_weather(days_back=args.days)

    # --- Step 2: Estimate generation ---
    print("\n" + "=" * 60)
    print("STEP 2: Estimating Solar/Wind Generation")
    print("=" * 60)
    df = add_estimations(weather_df)
    print(f"  Dataset: {len(df)} rows, {df.index[0]} → {df.index[-1]}")
    print(f"  Solar range: {df['solar_kw'].min():.1f} – {df['solar_kw'].max():.1f} kW")
    print(f"  Wind range:  {df['wind_kw'].min():.1f} – {df['wind_kw'].max():.1f} kW")
    print(f"  Net gen range: {df['net_generation_kw'].min():.1f} – {df['net_generation_kw'].max():.1f} kW")

    # Save raw dataset
    csv_path = os.path.join(output_dir, "energy_dataset.csv")
    df.to_csv(csv_path)
    print(f"  Dataset saved: {csv_path}")

    # --- Step 3: Feature engineering ---
    print("\n" + "=" * 60)
    print("STEP 3: Feature Engineering")
    print("=" * 60)
    df_feat = create_features(df)
    print(f"  Shape after features: {df_feat.shape} (dropped {len(df) - len(df_feat)} NaN rows)")

    # --- Step 4: Train/test split + training ---
    print("\n" + "=" * 60)
    print("STEP 4: Model Training")
    print("=" * 60)
    train_df, test_df = train_test_split_temporal(df_feat)
    print(f"  Train: {len(train_df)} | Test: {len(test_df)}")

    model = train_model(train_df)
    print("  XGBoost model trained.")

    model_path = os.path.join(output_dir, "xgb_energy_model.joblib")
    save_model(model, model_path)
    print(f"  Model saved: {model_path}")

    # --- Step 5: Evaluate ---
    print("\n" + "=" * 60)
    print("STEP 5: Evaluation")
    print("=" * 60)
    y_pred = predict(model, test_df)
    y_true = test_df["net_generation_kw"].values
    metrics = evaluate_model(y_true, y_pred)

    print(f"  RMSE: {metrics['rmse']} kW")
    print(f"  MAE:  {metrics['mae']} kW")
    print(f"  MAPE: {metrics['mape']}%")
    print(f"  R²:   {metrics['r2']}")

    fi = get_feature_importance(model)
    print("\n  Top 5 Features:")
    for _, row in fi.head(5).iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")

    # --- Step 6: Plots ---
    print("\n" + "=" * 60)
    print("STEP 6: Generating Evaluation Plots")
    print("=" * 60)
    plot_initial_evaluation(test_df, y_pred, fi, save_dir=output_dir)

    # --- Step 7: LLM analysis ---
    print("\n" + "=" * 60)
    print("STEP 7: LLM Analysis")
    print("=" * 60)
    predictions_log = pd.DataFrame({
        "datetime": test_df.index,
        "predicted": y_pred,
        "actual": y_true,
        "error": y_true - y_pred,
    })

    client = get_openai_client(args.openai_key)
    llm_input = prepare_llm_input(metrics, fi, predictions_log)
    analysis = analyze_results(llm_input, client=client)
    print(analysis)
    save_analysis(analysis, os.path.join(output_dir, "llm_analysis.txt"))

    return model, df_feat, metrics, fi, predictions_log


def live_phase(args, model, full_df, metrics, feature_importance, predictions_log):
    """
    LIVE PHASE:
      Every 15 minutes:
        1. Fetch latest weather
        2. Estimate generation (= "actual" for that interval)
        3. Compare to previous prediction
        4. Predict next 15-min interval
        5. Update dashboard
        6. Every hour: retrain model
    """
    output_dir = args.output_dir
    interval = 60 if args.fast_mode else config.LIVE_INTERVAL_SECONDS
    max_cycles = args.live_cycles if args.live_cycles > 0 else float("inf")
    retrain_interval = config.RETRAIN_INTERVAL

    print("\n" + "=" * 60)
    print("LIVE PHASE — Continuous Forecasting")
    print(f"  Interval: {interval}s | Retrain every {retrain_interval} cycles")
    if max_cycles != float("inf"):
        print(f"  Max cycles: {max_cycles}")
    print("=" * 60)

    cycle = 0
    retrain_count = 0
    pending_prediction = None
    live_log = pd.DataFrame(columns=["datetime", "predicted", "actual", "error"])
    client = get_openai_client(args.openai_key)

    while cycle < max_cycles:
        cycle += 1
        now = datetime.now()
        print(f"\n--- Live cycle {cycle} @ {now.strftime('%Y-%m-%d %H:%M:%S')} ---")

        try:
            # 1. Fetch latest weather
            latest_weather = fetch_latest_weather()
            latest_with_est = add_estimations(latest_weather)

            # 2. If we have a pending prediction, compare to actual
            if pending_prediction is not None and len(latest_with_est) > 0:
                # Use the closest available actual
                actual_val = latest_with_est["net_generation_kw"].iloc[0]
                pred_time = pending_prediction["datetime"]
                pred_val = pending_prediction["predicted_kw"]
                error = actual_val - pred_val

                # Update the pending row in live_log with actual values
                pending_mask = live_log["actual"].isna()
                if pending_mask.any():
                    idx = live_log[pending_mask].index[-1]
                    live_log.loc[idx, "actual"] = actual_val
                    live_log.loc[idx, "error"] = error
                else:
                    new_row = pd.DataFrame([{
                        "datetime": pred_time,
                        "predicted": pred_val,
                        "actual": actual_val,
                        "error": error,
                    }])
                    live_log = pd.concat([live_log, new_row], ignore_index=True)
                print(f"  ✓ Comparison: Pred={pred_val:.1f} | Actual={actual_val:.1f} | Error={error:+.1f} kW")

                # Append new data to full dataset for retraining
                new_data = latest_with_est.iloc[:1]
                full_df = pd.concat([full_df, create_features(
                    pd.concat([full_df.tail(96), add_estimations(latest_weather)])
                ).tail(1)]).drop_duplicates()

            # 3. Update running metrics
            completed = live_log.dropna(subset=["actual"])
            if len(completed) >= 2:
                metrics = evaluate_model(
                    completed["actual"].values.astype(float),
                    completed["predicted"].values.astype(float),
                )

            # 4. Retrain every N cycles
            if cycle % retrain_interval == 0 and len(full_df) > 100:
                print("  🔄 Retraining model...")
                model = retrain_model(full_df)
                retrain_count += 1
                feature_importance = get_feature_importance(model)
                save_model(model, os.path.join(output_dir, "xgb_energy_model.joblib"))
                print(f"  ✓ Retrained (count: {retrain_count})")

                # Periodic LLM analysis
                llm_input = prepare_llm_input(metrics, feature_importance, live_log,
                                              retrain_count=retrain_count)
                analysis = analyze_results(llm_input, client=client)
                save_analysis(analysis, os.path.join(output_dir, "llm_analysis.txt"))

            # 5. Make next prediction
            forecast_point = None
            if len(full_df) > 0:
                try:
                    last_row = full_df.tail(1)
                    feature_cols = get_feature_columns()
                    if all(c in last_row.columns for c in feature_cols):
                        next_pred = model.predict(last_row[feature_cols])[0]
                        next_time = last_row.index[-1] + pd.Timedelta(minutes=15)
                        pending_prediction = {
                            "datetime": next_time,
                            "predicted_kw": float(next_pred),
                        }
                        forecast_point = pending_prediction

                        # Log pending prediction
                        pending_row = pd.DataFrame([{
                            "datetime": next_time,
                            "predicted": float(next_pred),
                            "actual": np.nan,
                            "error": np.nan,
                        }])
                        live_log = pd.concat([live_log, pending_row], ignore_index=True)
                        print(f"  🔮 Next prediction: {next_time.strftime('%H:%M')} → {next_pred:.1f} kW")
                except Exception as e:
                    print(f"  ⚠ Prediction failed: {e}")

            # 6. Update dashboard
            window_hours = config.DASHBOARD_WINDOW_HOURS
            window_steps = window_hours * 4
            recent_history = full_df.tail(window_steps)

            # Terminal dashboard
            print_terminal_dashboard(metrics, live_log, forecast_point, retrain_count)

            # PNG dashboard
            snapshot_path = os.path.join(output_dir, f"dashboard_{cycle:04d}.png")
            create_dashboard(recent_history, live_log, metrics,
                             forecast_point=forecast_point, save_path=snapshot_path)

            # Also save a "latest" dashboard
            create_dashboard(recent_history, live_log, metrics,
                             forecast_point=forecast_point,
                             save_path=os.path.join(output_dir, "dashboard_latest.png"))

            # Save predictions log
            live_log.to_csv(os.path.join(output_dir, "predictions_log.csv"), index=False)

        except Exception as e:
            print(f"  ❌ Error in live cycle: {e}")

        # Wait for next interval
        if cycle < max_cycles:
            print(f"\n  ⏳ Next update in {interval}s...")
            time.sleep(interval)

    print("\n" + "=" * 60)
    print("LIVE PHASE COMPLETE")
    print(f"  Total cycles: {cycle} | Retrains: {retrain_count}")
    print("=" * 60)


def main():
    args = parse_args()

    # Run initial phase
    model, full_df, metrics, fi, predictions_log = initial_phase(args)

    print("\n" + "=" * 60)
    print("INITIAL PHASE COMPLETE")
    print("=" * 60)

    # Run live phase unless --no-live
    if not args.no_live:
        live_phase(args, model, full_df, metrics, fi, predictions_log)
    else:
        print("(Live phase skipped — use without --no-live to enable)")

    print(f"\nAll outputs saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
