"""
LLM Analytics Layer — interprets forecasting results using OpenAI or offline fallback.
"""

import json
import os
import numpy as np
import pandas as pd
from typing import Optional


def prepare_llm_input(metrics: dict, feature_importance_df: pd.DataFrame,
                      predictions_log: pd.DataFrame, weather_context: dict = None,
                      retrain_count: int = 0) -> dict:
    """Prepare structured input for LLM analysis."""
    top_features = feature_importance_df.head(10).to_dict("records")

    completed = predictions_log.dropna(subset=["actual"])
    recent_preds = []
    for _, row in completed.tail(20).iterrows():
        t = row["datetime"]
        t_str = t.strftime("%H:%M") if hasattr(t, "strftime") else str(t)
        recent_preds.append({
            "time": t_str,
            "predicted": round(float(row["predicted"]), 1),
            "actual": round(float(row["actual"]), 1),
            "error": round(float(row["error"]), 1),
        })

    # Diagnostics from completed predictions
    diagnostics = {}
    if len(completed) > 0:
        errors = completed["error"].values
        diagnostics = {
            "mean_error": round(float(np.mean(errors)), 2),
            "std_error": round(float(np.std(errors)), 2),
            "bias": "over-predicting" if np.mean(errors) < -1 else "under-predicting" if np.mean(errors) > 1 else "balanced",
            "count": len(completed),
        }

    return {
        "metrics": metrics,
        "feature_importance": top_features,
        "recent_predictions": recent_preds,
        "weather_context": weather_context or {},
        "diagnostics": diagnostics,
        "metadata": {
            "resolution": "15min",
            "model": "XGBoost",
            "retrain_count": retrain_count,
            "data_source": "Open-Meteo (real weather data)",
        },
    }


def analyze_results(llm_input: dict, client=None, model: str = "gpt-4o") -> str:
    """Run LLM analysis or fall back to offline."""
    prompt = f"""You are an energy analytics expert analyzing a real-time ML forecasting model.

The model predicts Net Power Output (kW) = Solar + Wind - Curtailment using XGBoost
on 15-minute resolution data from Open-Meteo weather API.

Results:

## Metrics
{json.dumps(llm_input['metrics'], indent=2)}

## Top Features
{json.dumps(llm_input['feature_importance'], indent=2)}

## Recent Predictions
{json.dumps(llm_input['recent_predictions'], indent=2)}

## Diagnostics
{json.dumps(llm_input['diagnostics'], indent=2)}

## Metadata
{json.dumps(llm_input['metadata'], indent=2)}

Provide:
1. **Performance Summary** — practical interpretation of metrics
2. **Diagnostics** — bias, peak vs night errors, patterns
3. **Recommendations** — improvements for features, data, model
Keep it concise and actionable."""

    if client is None:
        return _offline_analysis(llm_input)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content


def ask_llm(llm_input: dict, question: str, client=None, model: str = "gpt-4o") -> str:
    """Ask the LLM a question about the results."""
    prompt = f"""You are an energy analytics expert. Context:
Model: XGBoost, 15-min resolution, real weather data from Open-Meteo.
Metrics: {json.dumps(llm_input['metrics'])}
Top Features: {json.dumps(llm_input['feature_importance'][:5])}
Diagnostics: {json.dumps(llm_input['diagnostics'])}

User question: {question}

Answer concisely with specific numbers."""

    if client is None:
        return _offline_qa(llm_input, question)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content


def _offline_analysis(llm_input: dict) -> str:
    m = llm_input["metrics"]
    d = llm_input.get("diagnostics", {})
    fi = llm_input["feature_importance"]

    lines = [
        "=" * 60,
        "ENERGY FORECASTING — PERFORMANCE ANALYSIS (offline)",
        "=" * 60,
        "",
        "## Performance Summary",
        f"  RMSE: {m.get('rmse', 'N/A')} kW | MAE: {m.get('mae', 'N/A')} kW",
        f"  MAPE: {m.get('mape', 'N/A')}% | R²: {m.get('r2', 'N/A')}",
        "",
    ]

    r2 = m.get("r2", 0)
    if isinstance(r2, (int, float)):
        if r2 > 0.95:
            lines.append("  → Excellent fit. Suitable for operational use.")
        elif r2 > 0.85:
            lines.append("  → Good fit. Acceptable for forecasting.")
        elif r2 > 0.70:
            lines.append("  → Moderate fit. Consider improvements.")
        else:
            lines.append("  → Poor fit. Significant improvements needed.")

    if d:
        lines.extend([
            "",
            "## Diagnostics",
            f"  Bias: {d.get('bias', 'N/A')} (mean error: {d.get('mean_error', 'N/A')} kW)",
            f"  Error std: {d.get('std_error', 'N/A')} kW",
            f"  Completed predictions: {d.get('count', 0)}",
        ])

    lines.extend(["", "## Top Features"])
    for f in fi[:5]:
        lines.append(f"  {f['feature']}: {f['importance']:.4f}")

    lines.extend([
        "",
        "## Recommendations",
        "  - Monitor model drift as weather patterns change",
        "  - Consider adding more lag features (48h, 7-day)",
        "  - Evaluate if retraining frequency is optimal",
        "  - Watch for seasonal transitions affecting accuracy",
        "=" * 60,
    ])
    return "\n".join(lines)


def _offline_qa(llm_input: dict, question: str) -> str:
    m = llm_input["metrics"]
    fi = llm_input["feature_importance"]
    d = llm_input.get("diagnostics", {})

    return (
        f"[Offline mode] Metrics: RMSE={m.get('rmse')}kW, MAE={m.get('mae')}kW, "
        f"R²={m.get('r2')}. Bias: {d.get('bias', 'unknown')}. "
        f"Top feature: {fi[0]['feature'] if fi else 'N/A'}. "
        "Set OPENAI_API_KEY for detailed answers."
    )


def save_analysis(text: str, path: str):
    with open(path, "w") as f:
        f.write(text)


def get_openai_client(api_key: Optional[str] = None):
    try:
        from openai import OpenAI
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if key:
            return OpenAI(api_key=key)
        print("  No OpenAI API key found — using offline LLM mode.")
        return None
    except ImportError:
        print("  openai package not installed — using offline LLM mode.")
        return None
