"""End-to-end model training, cross-validation, Optuna tuning, and evaluation pipeline.

Reproducible execution via: python -m wattsmart.train
Outputs:
- models/model.joblib
- models/metrics.json
- models/feature_list.json
- models/shap_summary.png
- docs/figures/actual_vs_predicted.png
- docs/figures/residuals.png
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from wattsmart.config import (
    CONFORMAL_CONFIDENCE_LEVEL,
    FEATURE_LIST_PATH,
    FIGURES_DIR,
    METRICS_PATH,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    SHAP_SUMMARY_PATH,
    TARGET_COL,
    TRAIN_TEST_SPLIT_RATIO,
    TS_CV_SPLITS,
)
from wattsmart.data import (
    clean_data,
    get_chronological_split,
    get_dataset_medians,
    load_raw_data,
    resample_data,
    save_processed_data,
    validate_schema,
)
from wattsmart.evaluate import (
    calibrate_conformal_interval,
    evaluate_predictions,
    plot_actual_vs_predicted,
    plot_residuals,
)
from wattsmart.features import engineer_features, get_feature_and_target_matrices

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("wattsmart.train")

# Quiet Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_baselines(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> Dict[str, Dict[str, float]]:
    """Compute performance for Mean Baseline and Persistence (Lag-1) Baseline."""
    y_train = train_df[target_col].values
    y_test = test_df[target_col].values

    # Mean Baseline: Predict overall historical mean
    mean_val = float(np.mean(y_train))
    mean_preds = np.full(len(y_test), mean_val)
    mean_metrics = evaluate_predictions(y_test, mean_preds)

    # Persistence Baseline: Predict immediate previous period value
    # For test[0], the immediate predecessor is train[-1]
    full_series = pd.concat([train_df[target_col], test_df[target_col]])
    persistence_series = full_series.shift(1).iloc[len(train_df) :]
    persistence_preds = persistence_series.values
    persistence_metrics = evaluate_predictions(y_test, persistence_preds)

    logger.info(
        "Mean Baseline: MAE=%.2f, RMSE=%.2f, R2=%.4f, SMAPE=%.2f%%",
        mean_metrics["mae"],
        mean_metrics["rmse"],
        mean_metrics["r2"],
        mean_metrics["smape"],
    )
    logger.info(
        "Persistence Baseline: MAE=%.2f, RMSE=%.2f, R2=%.4f, SMAPE=%.2f%%",
        persistence_metrics["mae"],
        persistence_metrics["rmse"],
        persistence_metrics["r2"],
        persistence_metrics["smape"],
    )

    return {
        "Mean Baseline": mean_metrics,
        "Persistence Baseline": persistence_metrics,
    }


def evaluate_candidate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, BaseEstimator]]:
    """Train and evaluate candidate models on chronological test split."""
    candidates = {
        "Ridge": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=10.0, random_state=RANDOM_STATE)),
            ]
        ),
        "RandomForest": Pipeline(
            [
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=100,
                        max_depth=12,
                        min_samples_split=4,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "LightGBM": Pipeline(
            [
                (
                    "regressor",
                    LGBMRegressor(
                        n_estimators=120,
                        max_depth=5,
                        learning_rate=0.05,
                        num_leaves=31,
                        random_state=RANDOM_STATE,
                        verbose=-1,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            [
                (
                    "regressor",
                    XGBRegressor(
                        n_estimators=120,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    results: Dict[str, Dict[str, float]] = {}
    fitted_models: Dict[str, BaseEstimator] = {}

    for name, pipeline in candidates.items():
        logger.info("Training candidate model: %s...", name)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        metrics = evaluate_predictions(y_test, preds)
        results[name] = metrics
        fitted_models[name] = pipeline
        logger.info(
            "%s: MAE=%.2f, RMSE=%.2f, R2=%.4f, SMAPE=%.2f%%",
            name,
            metrics["mae"],
            metrics["rmse"],
            metrics["r2"],
            metrics["smape"],
        )

    return results, fitted_models


def tune_best_model_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 20,
) -> Dict[str, Any]:
    """Tune XGBoost / LightGBM hyperparameters using TimeSeriesSplit cross-validation."""
    logger.info(
        "Running Optuna hyperparameter optimization with TimeSeriesSplit (k=%d)...",
        TS_CV_SPLITS,
    )
    tscv = TimeSeriesSplit(n_splits=TS_CV_SPLITS)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 200, step=20),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.1),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0, step=0.1
            ),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }

        cv_maes = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model = XGBRegressor(**params)
            model.fit(X_tr, y_tr)
            val_preds = model.predict(X_val)
            cv_maes.append(float(np.mean(np.abs(y_val - val_preds))))

        return float(np.mean(cv_maes))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, timeout=120)
    logger.info("Optuna optimization completed. Best CV MAE: %.2f", study.best_value)
    logger.info("Best parameters: %s", study.best_params)
    return study.best_params


def generate_shap_artifacts(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[Dict[str, float], Path]:
    """Compute SHAP values using TreeExplainer and save summary plot."""
    import shap

    logger.info("Generating SHAP summary plot...")
    # Extract underlying regressor from pipeline
    regressor = model.named_steps.get("regressor", model)

    # Use a representative sample of test set for fast and clear explanation
    sample_size = min(250, len(X_test))
    X_sample = X_test.iloc[:sample_size]

    try:
        explainer = shap.TreeExplainer(regressor)
        shap_values = explainer.shap_values(X_sample)

        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=feature_cols,
            show=False,
            max_display=15,
        )
        plt.title(
            "SHAP Feature Importance Summary (Wattsmart 2.0)",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()
        SHAP_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(SHAP_SUMMARY_PATH, dpi=300, bbox_inches="tight")
        # Also copy to docs/figures for documentation
        docs_shap = FIGURES_DIR / "shap_summary.png"
        fig.savefig(docs_shap, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Compute mean absolute SHAP value per feature
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        feature_importance_dict = {
            feature_cols[i]: round(float(mean_abs_shap[i]), 4)
            for i in np.argsort(mean_abs_shap)[::-1]
        }
    except Exception as exc:
        logger.warning(
            "SHAP TreeExplainer encountered %s; falling back to feature importances.",
            exc,
        )
        importances = getattr(
            regressor, "feature_importances_", np.zeros(len(feature_cols))
        )
        feature_importance_dict = {
            feature_cols[i]: round(float(importances[i]), 4)
            for i in np.argsort(importances)[::-1]
        }

    return feature_importance_dict, SHAP_SUMMARY_PATH


def train_pipeline() -> Dict[str, Any]:
    """Execute reproducible end-to-end training pipeline."""
    np.random.seed(RANDOM_STATE)

    # 1. Load and validate data
    logger.info("Step 1: Loading and validating raw dataset...")
    df_raw = load_raw_data()
    val_info = validate_schema(df_raw)
    logger.info(
        "Validation passed: %d rows, %d cols.", val_info["rows"], val_info["columns"]
    )

    # 2. Clean and resample
    logger.info("Step 2: Cleaning and resampling to %s...", "10min")
    df_clean = clean_data(df_raw)
    df_resampled = resample_data(df_clean, freq="10min")
    save_processed_data(df_resampled)

    # 3. Feature engineering
    logger.info("Step 3: Engineering leak-free features...")
    df_features = engineer_features(df_resampled, target_col=TARGET_COL)

    # 4. Chronological split
    logger.info("Step 4: Chronological 80/20 train/test split...")
    train_df, test_df = get_chronological_split(
        df_features, split_ratio=TRAIN_TEST_SPLIT_RATIO
    )
    X_train, y_train, feature_cols = get_feature_and_target_matrices(
        train_df, TARGET_COL
    )
    X_test, y_test, _ = get_feature_and_target_matrices(test_df, TARGET_COL)

    # Compute training set medians for graceful inference defaults
    dataset_medians = get_dataset_medians(train_df)

    # 5. Baseline models
    logger.info("Step 5: Evaluating baseline models...")
    comparison_table = run_baselines(train_df, test_df, target_col=TARGET_COL)

    # 6. Candidate model evaluation
    logger.info("Step 6: Evaluating candidate models...")
    candidate_metrics, candidate_models = evaluate_candidate_models(
        X_train, y_train, X_test, y_test
    )
    comparison_table.update(candidate_metrics)

    # 7. Hyperparameter tuning on best model (XGBoost)
    logger.info("Step 7: Tuning best model with Optuna...")
    best_params = tune_best_model_optuna(X_train, y_train, n_trials=20)

    # 8. Train final production pipeline
    logger.info("Step 8: Assembling final production pipeline...")
    final_pipeline = Pipeline(
        [
            (
                "regressor",
                XGBRegressor(
                    **best_params,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    final_pipeline.fit(X_train, y_train)

    # Evaluate final tuned model
    final_test_preds = final_pipeline.predict(X_test)
    final_metrics = evaluate_predictions(y_test, final_test_preds)
    comparison_table["Tuned_XGBoost (Final)"] = final_metrics
    logger.info(
        "Final Tuned XGBoost: MAE=%.2f, RMSE=%.2f, R2=%.4f, SMAPE=%.2f%%",
        final_metrics["mae"],
        final_metrics["rmse"],
        final_metrics["r2"],
        final_metrics["smape"],
    )

    # 9. Conformal Prediction Interval Calibration
    logger.info(
        "Step 9: Calibrating conformal prediction interval (%.0f%%)...",
        CONFORMAL_CONFIDENCE_LEVEL * 100,
    )
    conformal_radius = calibrate_conformal_interval(
        y_test.values,
        final_test_preds,
        confidence_level=CONFORMAL_CONFIDENCE_LEVEL,
    )

    # 10. Generate diagnostic plots
    logger.info("Step 10: Generating diagnostic plots...")
    plot_actual_vs_predicted(
        y_test.values, final_test_preds, model_name="Tuned XGBoost"
    )
    plot_residuals(y_test.values, final_test_preds, model_name="Tuned XGBoost")

    # 11. SHAP analysis
    logger.info("Step 11: Computing SHAP feature importances...")
    feature_importance_dict, _ = generate_shap_artifacts(
        final_pipeline, X_train, X_test, feature_cols
    )

    # 12. Save model and metadata artifacts
    logger.info("Step 12: Saving production artifacts...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, MODEL_PATH)
    logger.info("Saved model pipeline to %s", MODEL_PATH)

    # Metadata record
    metadata = {
        "model_type": "XGBRegressor",
        "pipeline_steps": ["XGBRegressor"],
        "hyperparameters": {
            k: float(v) if isinstance(v, (int, float, np.floating)) else v
            for k, v in best_params.items()
        },
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "dataset_info": {
            "start_date": val_info["start_date"],
            "end_date": val_info["end_date"],
            "raw_rows": val_info["rows"],
            "resampled_rows": len(df_resampled),
            "resampling_frequency": "10min",
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "features_count": len(feature_cols),
        },
        "conformal_prediction": {
            "confidence_level": CONFORMAL_CONFIDENCE_LEVEL,
            "residual_radius_wh": conformal_radius,
            "method": "split_conformal_quantile",
        },
        "metrics": final_metrics,
        "model_comparison": comparison_table,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metrics to %s", METRICS_PATH)

    # Save feature list and medians for fast, deterministic inference
    features_metadata = {
        "features": feature_cols,
        "feature_importances": feature_importance_dict,
        "dataset_medians": dataset_medians,
    }
    with open(FEATURE_LIST_PATH, "w", encoding="utf-8") as f:
        json.dump(features_metadata, f, indent=2)
    logger.info("Saved feature list to %s", FEATURE_LIST_PATH)

    logger.info("Training pipeline completed successfully.")
    return metadata


if __name__ == "__main__":
    train_pipeline()
