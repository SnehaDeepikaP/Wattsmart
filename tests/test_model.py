"""Unit and performance tests for trained ML pipeline, baselines, and conformal intervals."""

import json

import joblib

from wattsmart.config import METRICS_PATH, MODEL_PATH
from wattsmart.predict import Predictor


def test_model_artifact_exists_and_loads():
    """Verify saved model pipeline and metrics file exist and load cleanly."""
    assert MODEL_PATH.exists(), f"Model file not found at {MODEL_PATH}"
    assert METRICS_PATH.exists(), f"Metrics file not found at {METRICS_PATH}"

    pipeline = joblib.load(MODEL_PATH)
    assert pipeline is not None


def test_model_beats_mean_baseline_mae():
    """MINIMUM-PERFORMANCE TEST: Assert final model MAE strictly beats the Mean Baseline MAE.

    Mean baseline MAE on chronological test set is ~24.16 Wh.
    The tuned model must achieve a strictly lower MAE.
    """
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    model_mae = data["metrics"]["mae"]
    mean_baseline_mae = data["model_comparison"]["Mean Baseline"]["mae"]
    persistence_mae = data["model_comparison"]["Persistence Baseline"]["mae"]

    assert (
        model_mae < mean_baseline_mae
    ), f"Model MAE ({model_mae} Wh) failed to beat Mean Baseline MAE ({mean_baseline_mae} Wh)!"
    assert (
        model_mae < persistence_mae
    ), f"Model MAE ({model_mae} Wh) failed to beat Persistence Baseline MAE ({persistence_mae} Wh)!"
    # Require at least 40% improvement over mean baseline
    mae_reduction = (mean_baseline_mae - model_mae) / mean_baseline_mae
    assert (
        mae_reduction > 0.40
    ), f"MAE improvement ({mae_reduction:.1%}) below 40% threshold."


def test_conformal_prediction_bounds():
    """Verify predictor generates valid confidence intervals [lower <= prediction <= upper]."""
    predictor = Predictor.get_instance()
    res = predictor.predict(
        {
            "indoor_temp": 21.0,
            "indoor_humidity": 45.0,
            "outdoor_temp": 5.0,
            "outdoor_humidity": 80.0,
            "hour": 18,
        }
    )

    pred = res["prediction_wh"]
    interval = res["interval"]

    assert (
        interval["lower_wh"] <= pred
    ), "Interval lower bound exceeds point prediction!"
    assert (
        interval["upper_wh"] >= pred
    ), "Interval upper bound is less than point prediction!"
    assert interval["lower_wh"] >= 0.0, "Interval lower bound is negative!"
    assert interval["confidence_level"] == 0.90
    assert interval["radius_wh"] > 0.0
