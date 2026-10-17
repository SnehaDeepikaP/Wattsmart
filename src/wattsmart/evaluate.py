"""Model evaluation metrics, conformal interval calibration, and diagnostic plots."""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from wattsmart.config import FIGURES_DIR

logger = logging.getLogger(__name__)


def compute_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Symmetric Mean Absolute Percentage Error (SMAPE) in percentage."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    # Avoid zero division when both true and pred are zero
    mask = denom != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100.0)


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Percentage Error (MAPE) in percentage with zero protection."""
    # Filter out tiny values < 1e-4 to avoid massive explosion
    mask = np.abs(y_true) > 1e-4
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def evaluate_predictions(
    y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]
) -> Dict[str, float]:
    """Calculate RMSE, MAE, R2, MAPE, and SMAPE for regression predictions.

    Args:
        y_true: True target values.
        y_pred: Predicted values.

    Returns:
        Dictionary of formatted evaluation metrics.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(yt, yp))
    mse = float(mean_squared_error(yt, yp))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(yt, yp))
    mape = compute_mape(yt, yp)
    smape = compute_smape(yt, yp)

    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
        "mape": round(mape, 2),
        "smape": round(smape, 2),
    }


def calibrate_conformal_interval(
    y_val: np.ndarray,
    y_val_pred: np.ndarray,
    confidence_level: float = 0.90,
) -> float:
    """Calibrate split conformal prediction interval radius.

    Computes empirical (1 - alpha) quantile of absolute residuals on calibration data.
    Provides mathematically sound finite-sample coverage guarantees.

    Args:
        y_val: Calibration true values.
        y_val_pred: Calibration point predictions.
        confidence_level: Target coverage probability (e.g. 0.90 for 90% coverage).

    Returns:
        conformal_radius: Margin of error added/subtracted around point prediction.
    """
    residuals = np.abs(np.asarray(y_val) - np.asarray(y_val_pred))
    n = len(residuals)
    # Standard conformal quantile index: ceil((n + 1) * confidence_level) / n
    quantile_level = min(1.0, np.ceil((n + 1) * confidence_level) / n)
    conformal_radius = float(np.quantile(residuals, quantile_level))
    logger.info(
        "Calibrated conformal prediction interval: radius=%.2f Wh at %.1f%% confidence",
        conformal_radius,
        confidence_level * 100,
    )
    return round(conformal_radius, 2)


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Trained Model",
    save_path: Optional[Path] = None,
) -> Path:
    """Generate and save an Actual vs Predicted scatter plot."""
    path = save_path or (FIGURES_DIR / "actual_vs_predicted.png")
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.35, edgecolors="none", color="#1f77b4", s=30)
    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot(
        [min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect 1:1 Fit"
    )

    ax.set_title(
        f"Actual vs. Predicted Appliance Energy - {model_name}",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Actual Consumption (Wh)", fontsize=11)
    ax.set_ylabel("Predicted Consumption (Wh)", fontsize=11)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    logger.info("Saved actual vs predicted plot to %s", path)
    return path


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Trained Model",
    save_path: Optional[Path] = None,
) -> Path:
    """Generate and save dual-panel residual diagnostics plot."""
    path = save_path or (FIGURES_DIR / "residuals.png")
    path.parent.mkdir(parents=True, exist_ok=True)

    residuals = np.asarray(y_true) - np.asarray(y_pred)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter of residuals vs predicted
    ax1.scatter(y_pred, residuals, alpha=0.35, color="#2ca02c", edgecolors="none", s=30)
    ax1.axhline(0, color="red", linestyle="--", lw=2)
    ax1.set_title(
        f"Residuals vs. Fitted - {model_name}", fontsize=12, fontweight="bold"
    )
    ax1.set_xlabel("Fitted Values (Wh)", fontsize=11)
    ax1.set_ylabel("Residuals (Actual - Predicted, Wh)", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Histogram/KDE of residuals
    sns.histplot(residuals, kde=True, ax=ax2, color="#2ca02c", bins=30)
    ax2.axvline(0, color="red", linestyle="--", lw=2)
    ax2.set_title("Residual Distribution", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Residual (Wh)", fontsize=11)
    ax2.set_ylabel("Frequency", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    logger.info("Saved residuals plot to %s", path)
    return path
