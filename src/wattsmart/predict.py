"""Inference and forecasting engine for Wattsmart.

Provides point predictions, conformal prediction intervals, autoregressive multi-step forecasts,
and what-if scenario simulations.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from wattsmart.config import (
    FEATURE_LIST_PATH,
    METRICS_PATH,
    MODEL_PATH,
    PROCESSED_DATA_PATH,
    TARGET_COL,
)
from wattsmart.features import prepare_inference_features

logger = logging.getLogger(__name__)


class Predictor:
    """Singleton predictor that caches trained model, metadata, and feature definitions."""

    _instance: Optional["Predictor"] = None

    def __init__(
        self,
        model_path: Optional[Path] = None,
        feature_list_path: Optional[Path] = None,
        metrics_path: Optional[Path] = None,
    ):
        self.model_path = model_path or MODEL_PATH
        self.feature_list_path = feature_list_path or FEATURE_LIST_PATH
        self.metrics_path = metrics_path or METRICS_PATH

        self.model = None
        self.feature_list: List[str] = []
        self.dataset_medians: Dict[str, float] = {}
        self.feature_importances: Dict[str, float] = {}
        self.conformal_radius: float = 18.0
        self.confidence_level: float = 0.90
        self.metrics_data: Dict[str, Any] = {}

        self.load_artifacts()

    @classmethod
    def get_instance(cls) -> "Predictor":
        """Get or initialize singleton Predictor instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_artifacts(self) -> None:
        """Load trained pipeline and feature/metrics metadata from disk."""
        if not self.model_path.exists():
            logger.warning(
                "Model file not found at %s. Predictor initialized without model.",
                self.model_path,
            )
            return

        logger.info("Loading model from %s", self.model_path)
        self.model = joblib.load(self.model_path)

        if self.feature_list_path.exists():
            with open(self.feature_list_path, "r", encoding="utf-8") as f:
                fdata = json.load(f)
                self.feature_list = fdata.get("features", [])
                self.dataset_medians = fdata.get("dataset_medians", {})
                self.feature_importances = fdata.get("feature_importances", {})

        if self.metrics_path.exists():
            with open(self.metrics_path, "r", encoding="utf-8") as f:
                self.metrics_data = json.load(f)
                conformal_info = self.metrics_data.get("conformal_prediction", {})
                self.conformal_radius = float(
                    conformal_info.get("residual_radius_wh", 18.0)
                )
                self.confidence_level = float(
                    conformal_info.get("confidence_level", 0.90)
                )

    def predict(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Generate point prediction with calibrated conformal prediction interval.

        Args:
            input_dict: Dictionary with sensor readings and environmental conditions.

        Returns:
            Dictionary containing prediction, confidence interval, and metadata.
        """
        if self.model is None:
            self.load_artifacts()
            if self.model is None:
                raise RuntimeError(
                    "Model is not loaded. Please run model training first."
                )

        # Prepare 1-row feature vector matching trained schema
        X_df, defaults_applied = prepare_inference_features(
            input_dict, self.feature_list, self.dataset_medians
        )

        pred_val = float(self.model.predict(X_df)[0])
        # Physical lower bound clipping (appliances cannot consume negative power)
        pred_val = max(10.0, pred_val)
        lower_bound = max(0.0, round(pred_val - self.conformal_radius, 2))
        upper_bound = round(pred_val + self.conformal_radius, 2)

        return {
            "prediction_wh": round(pred_val, 2),
            "unit": "Wh",
            "interval": {
                "lower_wh": lower_bound,
                "upper_wh": upper_bound,
                "confidence_level": self.confidence_level,
                "radius_wh": self.conformal_radius,
                "method": "split_conformal_quantile",
            },
            "inputs_used": {k: v for k, v in input_dict.items() if v is not None},
            "defaults_applied": defaults_applied,
        }

    def forecast(self, hours: int = 12) -> Dict[str, Any]:
        """Generate forward multi-hour forecast starting from latest observed historical data.

        Args:
            hours: Number of hours to forecast (1 to 24).

        Returns:
            Dictionary containing time-series predictions and interval bands.
        """
        if self.model is None:
            self.load_artifacts()

        hours = max(1, min(24, int(hours)))
        # Resampling is 10-minute intervals: 6 steps per hour
        total_steps = hours * 6

        # Load recent history from processed data
        if not PROCESSED_DATA_PATH.exists():
            raise FileNotFoundError(
                f"Processed dataset not found at: {PROCESSED_DATA_PATH}"
            )

        df_processed = pd.read_csv(
            PROCESSED_DATA_PATH, index_col="date", parse_dates=True
        )
        # Use latest 24 steps (4 hours) as initial seed buffer
        seed_history = df_processed.iloc[-48:].copy()

        forecast_timestamps = []
        point_forecasts = []
        lower_forecasts = []
        upper_forecasts = []

        last_ts = seed_history.index[-1]
        simulated_history = seed_history.copy()

        for step in range(1, total_steps + 1):
            next_ts = last_ts + pd.Timedelta(minutes=10 * step)
            hour_val = next_ts.hour
            dow_val = next_ts.dayofweek

            # Build feature dictionary using recent rolling simulated history
            recent_appliances = simulated_history[TARGET_COL].values
            lag1 = (
                float(recent_appliances[-1]) if len(recent_appliances) >= 1 else 160.0
            )
            lag2 = float(recent_appliances[-2]) if len(recent_appliances) >= 2 else lag1
            lag6 = float(recent_appliances[-6]) if len(recent_appliances) >= 6 else lag1
            lag12 = (
                float(recent_appliances[-12]) if len(recent_appliances) >= 12 else lag1
            )

            roll_mean_6 = (
                float(np.mean(recent_appliances[-6:]))
                if len(recent_appliances) >= 6
                else lag1
            )
            roll_std_6 = (
                float(np.std(recent_appliances[-6:]))
                if len(recent_appliances) >= 6
                else 5.0
            )
            roll_mean_12 = (
                float(np.mean(recent_appliances[-12:]))
                if len(recent_appliances) >= 12
                else lag1
            )

            step_input = {
                "indoor_temp": self.dataset_medians.get("T1", 21.0),
                "indoor_humidity": self.dataset_medians.get("RH_1", 40.0),
                "outdoor_temp": self.dataset_medians.get("T_out", 5.0),
                "outdoor_humidity": self.dataset_medians.get("RH_out", 80.0),
                "windspeed": self.dataset_medians.get("Windspeed", 4.0),
                "pressure": self.dataset_medians.get("Press_mm_hg", 755.0),
                "hour": hour_val,
                "day_of_week": dow_val,
                "appliances_lag_1": lag1,
                "appliances_lag_2": lag2,
                "appliances_lag_6": lag6,
                "appliances_lag_12": lag12,
                "appliances_roll_mean_6": roll_mean_6,
                "appliances_roll_std_6": roll_std_6,
                "appliances_roll_mean_12": roll_mean_12,
            }

            res = self.predict(step_input)
            pred_wh = res["prediction_wh"]
            point_forecasts.append(pred_wh)
            lower_forecasts.append(res["interval"]["lower_wh"])
            upper_forecasts.append(res["interval"]["upper_wh"])
            forecast_timestamps.append(next_ts.strftime("%Y-%m-%d %H:%M"))

            # Append predicted value into simulated history for subsequent autoregressive lags
            new_row = simulated_history.iloc[-1:].copy()
            new_row.index = [next_ts]
            new_row[TARGET_COL] = pred_wh
            simulated_history = pd.concat([simulated_history, new_row])

        return {
            "forecast_hours": hours,
            "steps": total_steps,
            "interval_minutes": 10,
            "start_time": forecast_timestamps[0],
            "end_time": forecast_timestamps[-1],
            "unit": "Wh",
            "series": [
                {
                    "timestamp": ts,
                    "prediction_wh": p,
                    "lower_wh": low,
                    "upper_wh": high,
                }
                for ts, p, low, high in zip(
                    forecast_timestamps,
                    point_forecasts,
                    lower_forecasts,
                    upper_forecasts,
                )
            ],
        }

    def whatif(
        self, base_scenario: Dict[str, Any], modifications: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate energy impact of hypothetical changes against a baseline scenario.

        Args:
            base_scenario: Baseline parameters.
            modifications: Changes applied to the base scenario.

        Returns:
            Dictionary comparing base vs modified predictions and delta.
        """
        base_pred = self.predict(base_scenario)
        base_wh = base_pred["prediction_wh"]

        modified_scenario = dict(base_scenario)
        modified_scenario.update(modifications)
        mod_pred = self.predict(modified_scenario)
        mod_wh = mod_pred["prediction_wh"]

        delta_wh = round(mod_wh - base_wh, 2)
        pct_change = round((delta_wh / base_wh) * 100.0, 2) if base_wh > 0 else 0.0

        if delta_wh < 0:
            direction = "reduction"
            impact_text = (
                f"Conserves {abs(delta_wh)} Wh ({abs(pct_change)}% energy savings)"
            )
        elif delta_wh > 0:
            direction = "increase"
            impact_text = f"Consumes +{delta_wh} Wh (+{pct_change}% additional energy)"
        else:
            direction = "neutral"
            impact_text = "No predicted change in energy consumption"

        return {
            "base_prediction_wh": base_wh,
            "modified_prediction_wh": mod_wh,
            "delta_wh": delta_wh,
            "percent_change": pct_change,
            "direction": direction,
            "impact_summary": impact_text,
            "unit": "Wh",
            "modifications_applied": modifications,
        }
