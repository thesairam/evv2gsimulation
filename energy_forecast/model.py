"""
Model training, prediction, evaluation, and retraining.

Uses XGBoost for predicting net_generation_kw.
Supports periodic retraining for self-improvement in live mode.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

from .features import get_feature_columns


TARGET = "net_generation_kw"

MODEL_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.1,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}


def train_test_split_temporal(df: pd.DataFrame, test_fraction: float = 0.2):
    """Time-based train/test split (no shuffle)."""
    split_idx = int(len(df) * (1 - test_fraction))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def train_model(train_df: pd.DataFrame) -> xgb.XGBRegressor:
    """Train an XGBoost regressor on the training data."""
    feature_cols = get_feature_columns()
    X = train_df[feature_cols]
    y = train_df[TARGET]

    model = xgb.XGBRegressor(**MODEL_PARAMS)
    model.fit(X, y)
    return model


def retrain_model(full_df: pd.DataFrame) -> xgb.XGBRegressor:
    """Retrain on all available data (used during live phase)."""
    return train_model(full_df)


def predict(model: xgb.XGBRegressor, df: pd.DataFrame) -> np.ndarray:
    """Generate predictions."""
    feature_cols = get_feature_columns()
    return model.predict(df[feature_cols])


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute RMSE, MAE, MAPE, R²."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    mask = np.abs(y_true) > 1.0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = float("nan")

    return {
        "rmse": round(float(rmse), 2),
        "mae": round(float(mae), 2),
        "mape": round(float(mape), 2),
        "r2": round(float(r2), 4),
    }


def get_feature_importance(model: xgb.XGBRegressor) -> pd.DataFrame:
    """Get feature importance from the trained model."""
    feature_cols = get_feature_columns()
    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return fi


def save_model(model: xgb.XGBRegressor, path: str):
    """Save model to disk."""
    joblib.dump(model, path)


def load_model(path: str) -> xgb.XGBRegressor:
    """Load model from disk."""
    return joblib.load(path)
