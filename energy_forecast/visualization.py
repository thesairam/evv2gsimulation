"""
Visualization module — live rolling dashboard + static analysis plots.

Generates:
1. Live rolling dashboard (PNG snapshots + terminal text)
2. Prediction vs Actual scatter
3. Error over time
4. Feature importance
5. Initial evaluation plots
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


def create_dashboard(history_df: pd.DataFrame, predictions_log: pd.DataFrame,
                     metrics: dict, forecast_point: dict = None,
                     save_path: str = None):
    """
    Create the main live rolling dashboard with 4 panels.

    Args:
        history_df: Recent historical data (past ~6h) with solar_kw, wind_kw, net_generation_kw.
        predictions_log: DataFrame with columns: datetime, predicted, actual, error.
        metrics: Dict with rmse, mae, mape, r2.
        forecast_point: Optional dict with {'datetime': ..., 'predicted_kw': ...} for next prediction.
        save_path: Path to save the PNG snapshot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("ENERGY FORECAST — LIVE DASHBOARD", fontsize=16, fontweight="bold", y=0.98)

    # --- Panel 1: Rolling generation window ---
    ax1 = axes[0, 0]
    if len(history_df) > 0:
        ax1.plot(history_df.index, history_df["solar_kw"], label="Solar", color="#FFA500", linewidth=1.5)
        ax1.plot(history_df.index, history_df["wind_kw"], label="Wind", color="#4169E1", linewidth=1.5)
        ax1.plot(history_df.index, history_df["net_generation_kw"], label="Net Generation",
                 color="#2E8B57", linewidth=2.5)

    if forecast_point:
        ax1.scatter([forecast_point["datetime"]], [forecast_point["predicted_kw"]],
                    color="#FF4500", s=120, zorder=5, marker="D", label="Next Prediction")
        ax1.axvline(x=forecast_point["datetime"], color="#FF4500", linestyle=":", alpha=0.5)

    ax1.set_title("Generation (Rolling Window)", fontsize=12)
    ax1.set_ylabel("Power (kW)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    # --- Panel 2: Prediction vs Actual (scatter) ---
    ax2 = axes[0, 1]
    completed = predictions_log.dropna(subset=["actual"])
    if len(completed) > 0:
        hours = pd.to_datetime(completed["datetime"]).dt.hour
        scatter = ax2.scatter(completed["actual"], completed["predicted"],
                              c=hours, cmap="twilight", s=30, alpha=0.7, edgecolors="black", linewidth=0.3)
        plt.colorbar(scatter, ax=ax2, label="Hour of Day")

        all_vals = pd.concat([completed["actual"], completed["predicted"]])
        lims = [all_vals.min() * 0.9, all_vals.max() * 1.1]
        ax2.plot(lims, lims, "r--", linewidth=1, alpha=0.5, label="Perfect")
    ax2.set_title("Prediction vs Actual", fontsize=12)
    ax2.set_xlabel("Actual (kW)")
    ax2.set_ylabel("Predicted (kW)")
    if len(completed) > 0:
        ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Error over time ---
    ax3 = axes[1, 0]
    if len(completed) > 0:
        times = pd.to_datetime(completed["datetime"])
        errors = completed["error"].values
        colors = ["#2E8B57" if abs(e) < 20 else "#FFA500" if abs(e) < 50 else "#DC143C" for e in errors]
        ax3.bar(times, errors, color=colors, width=0.008, alpha=0.7)
        ax3.axhline(0, color="black", linewidth=0.8)

        # Running RMSE trend
        if len(completed) > 5:
            running_rmse = completed["error"].rolling(window=min(10, len(completed))).apply(
                lambda x: np.sqrt(np.mean(x**2)), raw=True
            )
            ax3.plot(times, running_rmse, color="#8B008B", linewidth=1.5,
                     linestyle="--", label="Running RMSE")
            ax3.legend(fontsize=8)

    ax3.set_title("Prediction Error Over Time", fontsize=12)
    ax3.set_ylabel("Error (kW)")
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    # --- Panel 4: Metrics + Prediction Log ---
    ax4 = axes[1, 1]
    ax4.axis("off")

    # Metrics box
    metric_text = (
        f"RUNNING METRICS\n"
        f"{'─' * 30}\n"
        f"RMSE:  {metrics.get('rmse', 'N/A'):>8} kW\n"
        f"MAE:   {metrics.get('mae', 'N/A'):>8} kW\n"
        f"MAPE:  {metrics.get('mape', 'N/A'):>8} %\n"
        f"R²:    {metrics.get('r2', 'N/A'):>8}\n"
        f"{'─' * 30}\n"
    )

    # Last 10 predictions log
    log_text = "PREDICTION LOG (recent)\n" + "─" * 40 + "\n"
    recent = predictions_log.tail(10)
    for _, row in recent.iterrows():
        t = row["datetime"]
        if isinstance(t, str):
            t_str = t[-5:]
        else:
            t_str = t.strftime("%H:%M")
        pred_str = f"{row['predicted']:.0f}"
        if pd.notna(row.get("actual")):
            act_str = f"{row['actual']:.0f}"
            err_str = f"{row['error']:+.0f}"
            log_text += f"{t_str} → Pred: {pred_str} | Act: {act_str} | Err: {err_str}\n"
        else:
            log_text += f"{t_str} → Pred: {pred_str} | Act: (awaiting...)\n"

    ax4.text(0.05, 0.95, metric_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0"))
    ax4.text(0.05, 0.45, log_text, transform=ax4.transAxes, fontsize=9,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f8f0"))

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  Dashboard saved: {save_path}")
    plt.close(fig)


def plot_initial_evaluation(test_df: pd.DataFrame, y_pred: np.ndarray,
                            feature_importance_df: pd.DataFrame,
                            save_dir: str = "output"):
    """
    Generate static evaluation plots after initial training.

    Creates:
    1. Prediction vs Actual time series (first 48h)
    2. Residual distribution
    3. Feature importance bar chart
    4. Generation profiles (sample day from test set)
    """
    os.makedirs(save_dir, exist_ok=True)

    # --- 1. Prediction vs Actual time series ---
    fig, ax = plt.subplots(figsize=(16, 6))
    steps = min(192, len(test_df))  # up to 48h
    t = test_df.index[:steps]
    ax.plot(t, test_df["net_generation_kw"].values[:steps], label="Actual", color="#2E8B57", linewidth=2)
    ax.plot(t, y_pred[:steps], label="Predicted", color="#FF4500", linewidth=2, linestyle="--")
    ax.set_title("Prediction vs Actual — Test Set", fontsize=14)
    ax.set_ylabel("Net Generation (kW)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "eval_pred_vs_actual.png"), dpi=150)
    plt.close(fig)

    # --- 2. Residual distribution ---
    residuals = test_df["net_generation_kw"].values - y_pred
    fig, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    axes2[0].hist(residuals, bins=50, color="#4169E1", edgecolor="black", alpha=0.7)
    axes2[0].axvline(0, color="red", linestyle="--")
    axes2[0].set_title("Residual Distribution")
    axes2[0].set_xlabel("Residual (kW)")
    axes2[1].scatter(range(len(residuals)), residuals, alpha=0.3, s=5, color="#4169E1")
    axes2[1].axhline(0, color="red", linestyle="--")
    axes2[1].set_title("Residuals Over Time")
    axes2[1].set_xlabel("Test Index")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "eval_residuals.png"), dpi=150)
    plt.close(fig)

    # --- 3. Feature importance ---
    fi = feature_importance_df.head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(fi["feature"][::-1], fi["importance"][::-1], color="#2E8B57", edgecolor="black")
    ax.set_title("Feature Importance (XGBoost)", fontsize=14)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "eval_feature_importance.png"), dpi=150)
    plt.close(fig)

    # --- 4. Generation profiles (sample day) ---
    dates = test_df.index.normalize().unique()
    if len(dates) > 0:
        day_data = test_df[test_df.index.normalize() == dates[0]]
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(day_data.index, day_data["solar_kw"], label="Solar", color="#FFA500", linewidth=2)
        ax.plot(day_data.index, day_data["wind_kw"], label="Wind", color="#4169E1", linewidth=2)
        ax.plot(day_data.index, day_data["net_generation_kw"], label="Net Generation",
                color="#2E8B57", linewidth=2.5)
        ax.set_title(f"Generation Profiles — {dates[0].strftime('%Y-%m-%d')}", fontsize=14)
        ax.set_ylabel("Power (kW)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "eval_generation_profiles.png"), dpi=150)
        plt.close(fig)

    print(f"  Evaluation plots saved to {save_dir}/")


def print_terminal_dashboard(metrics: dict, predictions_log: pd.DataFrame,
                             forecast_point: dict = None, retrain_count: int = 0):
    """
    Print a text-based dashboard to the terminal (for headless Docker).
    """
    print("\n" + "═" * 65)
    print("           ⚡ ENERGY FORECAST — LIVE DASHBOARD ⚡")
    print("═" * 65)

    print(f"\n  📊 RUNNING METRICS  (retrains: {retrain_count})")
    print(f"  ┌{'─' * 40}┐")
    print(f"  │  RMSE:  {metrics.get('rmse', 'N/A'):>8} kW                   │")
    print(f"  │  MAE:   {metrics.get('mae', 'N/A'):>8} kW                   │")
    print(f"  │  MAPE:  {metrics.get('mape', 'N/A'):>8} %                    │")
    print(f"  │  R²:    {metrics.get('r2', 'N/A'):>8}                       │")
    print(f"  └{'─' * 40}┘")

    if forecast_point:
        t = forecast_point["datetime"]
        t_str = t.strftime("%H:%M") if hasattr(t, "strftime") else str(t)
        print(f"\n  🔮 NEXT PREDICTION: {t_str} → {forecast_point['predicted_kw']:.1f} kW")

    print(f"\n  📋 PREDICTION LOG (last 10)")
    print(f"  {'─' * 55}")
    recent = predictions_log.tail(10)
    for _, row in recent.iterrows():
        t = row["datetime"]
        t_str = t.strftime("%H:%M") if hasattr(t, "strftime") else str(t)[-5:]
        pred_str = f"{row['predicted']:.0f}"
        if pd.notna(row.get("actual")):
            act_str = f"{row['actual']:.0f}"
            err_str = f"{row['error']:+.0f}"
            print(f"  {t_str}  Pred: {pred_str:>5} kW  |  Act: {act_str:>5} kW  |  Err: {err_str:>5} kW")
        else:
            print(f"  {t_str}  Pred: {pred_str:>5} kW  |  Act: (awaiting...)")

    print("═" * 65)
