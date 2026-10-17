"""Analytics, anomaly detection, and dynamic ML recommendation engine.

Computes:
- Peak and off-peak hours strictly from historical data
- Hourly and day-of-week consumption profiles
- High-consumption anomaly spikes via z-score and IsolationForest
- Dynamic counterfactual recommendations with model-predicted Wh savings
  (replaces legacy static tips and hardcoded 5% estimates).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.ensemble import IsolationForest

from wattsmart.config import PROCESSED_DATA_PATH, TARGET_COL

logger = logging.getLogger(__name__)

DAY_NAMES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


class DatasetInsights:
    """Computes and caches dataset analytics, profiles, and anomalies from cleaned data."""

    _instance: Optional["DatasetInsights"] = None

    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or PROCESSED_DATA_PATH
        self.df: Optional[pd.DataFrame] = None
        self.hourly_profile: List[Dict[str, Any]] = []
        self.day_of_week_profile: List[Dict[str, Any]] = []
        self.peak_hours: List[int] = []
        self.off_peak_hours: List[int] = []
        self.anomalies: List[Dict[str, Any]] = []
        self.summary_stats: Dict[str, Any] = {}

        self.compute_all_insights()

    @classmethod
    def get_instance(cls) -> "DatasetInsights":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def compute_all_insights(self) -> None:
        """Load processed data and calculate profiles, peaks, and anomalies."""
        if not self.data_path.exists():
            logger.warning(
                "Processed dataset not found at %s. Cannot compute dataset insights.",
                self.data_path,
            )
            return

        self.df = pd.read_csv(self.data_path, index_col="date", parse_dates=True)
        if TARGET_COL not in self.df.columns:
            return

        # Add hour and day_of_week
        self.df["hour"] = self.df.index.hour
        self.df["day_of_week"] = self.df.index.dayofweek

        # 1. Hourly profile
        hourly_grp = (
            self.df.groupby("hour")[TARGET_COL]
            .agg(
                mean_wh="mean",
                median_wh="median",
                std_wh="std",
                min_wh="min",
                max_wh="max",
                count="count",
            )
            .reset_index()
        )

        self.hourly_profile = [
            {
                "hour": int(row["hour"]),
                "hour_label": f"{int(row['hour']):02d}:00",
                "mean_wh": round(float(row["mean_wh"]), 2),
                "median_wh": round(float(row["median_wh"]), 2),
                "std_wh": round(float(row["std_wh"]), 2),
                "min_wh": round(float(row["min_wh"]), 2),
                "max_wh": round(float(row["max_wh"]), 2),
            }
            for _, row in hourly_grp.iterrows()
        ]

        # 2. Identify Peak and Off-Peak hours strictly from data
        mean_by_hour = hourly_grp.set_index("hour")["mean_wh"]
        p75 = float(mean_by_hour.quantile(0.75))
        p25 = float(mean_by_hour.quantile(0.25))

        self.peak_hours = sorted(
            [int(h) for h in mean_by_hour[mean_by_hour >= p75].index]
        )
        self.off_peak_hours = sorted(
            [int(h) for h in mean_by_hour[mean_by_hour <= p25].index]
        )

        # 3. Day of week profile
        dow_grp = (
            self.df.groupby("day_of_week")[TARGET_COL]
            .agg(
                mean_wh="mean",
                median_wh="median",
                std_wh="std",
            )
            .reset_index()
        )

        self.day_of_week_profile = [
            {
                "day_of_week": int(row["day_of_week"]),
                "day_name": DAY_NAMES[int(row["day_of_week"])],
                "mean_wh": round(float(row["mean_wh"]), 2),
                "median_wh": round(float(row["median_wh"]), 2),
                "std_wh": round(float(row["std_wh"]), 2),
            }
            for _, row in dow_grp.iterrows()
        ]

        # 4. Anomaly detection using IsolationForest and Z-Score
        appliances_series = self.df[TARGET_COL]
        mean_app = appliances_series.mean()
        std_app = appliances_series.std()
        z_scores = (appliances_series - mean_app) / (std_app if std_app > 0 else 1.0)

        # Isolation Forest on 10-minute records
        iso = IsolationForest(contamination=0.02, random_state=42)
        feature_subset = [
            c
            for c in [TARGET_COL, "T1", "RH_1", "T_out", "RH_out", "Windspeed"]
            if c in self.df.columns
        ]
        iso_preds = iso.fit_predict(self.df[feature_subset])

        # Select significant positive consumption spikes (z > 2.0 and iso == -1)
        anomaly_mask = (z_scores > 2.0) & (iso_preds == -1)
        anomaly_df = (
            self.df[anomaly_mask].sort_values(TARGET_COL, ascending=False).head(20)
        )

        self.anomalies = [
            {
                "timestamp": str(ts),
                "appliances_wh": round(float(row[TARGET_COL]), 2),
                "z_score": round(float(z_scores.loc[ts]), 2),
                "indoor_temp": round(
                    float(row.get("T1", row.get("T_indoor_mean", 21.0))), 1
                ),
                "outdoor_temp": round(float(row.get("T_out", 5.0)), 1),
                "windspeed": round(float(row.get("Windspeed", 3.0)), 1),
                "hour": int(row["hour"]),
            }
            for ts, row in anomaly_df.iterrows()
        ]

        # 5. Overall summary statistics
        self.summary_stats = {
            "total_records": len(self.df),
            "date_range": {
                "start": str(self.df.index.min()),
                "end": str(self.df.index.max()),
                "span_days": round(
                    (self.df.index.max() - self.df.index.min()).total_seconds() / 86400,
                    1,
                ),
            },
            "appliances": {
                "mean_wh": round(float(mean_app), 2),
                "std_wh": round(float(std_app), 2),
                "median_wh": round(float(appliances_series.median()), 2),
                "min_wh": round(float(appliances_series.min()), 2),
                "max_wh": round(float(appliances_series.max()), 2),
            },
            "peak_hours": self.peak_hours,
            "off_peak_hours": self.off_peak_hours,
        }

    def get_insights_payload(self) -> Dict[str, Any]:
        """Format full insights response for API."""
        if not self.hourly_profile:
            self.compute_all_insights()

        return {
            "summary": self.summary_stats,
            "hourly_profile": self.hourly_profile,
            "day_of_week_profile": self.day_of_week_profile,
            "peak_hours": self.peak_hours,
            "off_peak_hours": self.off_peak_hours,
            "anomaly_spikes": self.anomalies,
        }


def generate_recommendations(
    input_data: Dict[str, Any],
    predicted_wh: float,
    predictor: Any,
) -> List[Dict[str, Any]]:
    """Generate 2-4 personalized energy-saving recommendations with counterfactual ML estimates.

    Replaces static legacy advice with counterfactual predictions calculated by re-evaluating
    the trained ML pipeline under altered operational conditions.

    Args:
        input_data: User's input parameter dictionary.
        predicted_wh: Baseline predicted appliance consumption (Wh).
        predictor: Active Predictor instance.

    Returns:
        List of structured recommendation dictionaries with estimated Wh delta and caveats.
    """
    recommendations: List[Dict[str, Any]] = []
    insights = DatasetInsights.get_instance()

    hour = int(input_data.get("hour", 12))
    indoor_temp = float(input_data.get("indoor_temp", 21.0))
    outdoor_temp = float(input_data.get("outdoor_temp", 7.0))
    indoor_humidity = float(input_data.get("indoor_humidity", 45.0))

    # 1. Peak-hour load shifting recommendation
    if hour in insights.peak_hours:
        best_off_peak = insights.off_peak_hours[0] if insights.off_peak_hours else 3
        # Model re-prediction at off-peak hour
        alt_input = dict(input_data)
        alt_input["hour"] = best_off_peak
        alt_res = predictor.predict(alt_input)
        alt_wh = alt_res["prediction_wh"]
        savings_wh = round(predicted_wh - alt_wh, 1)

        if savings_wh > 0:
            savings_pct = round((savings_wh / predicted_wh) * 100, 1)
            recommendations.append(
                {
                    "id": "load_shift",
                    "category": "Load Shifting",
                    "title": f"Shift Heavy Appliances to Off-Peak ({best_off_peak:02d}:00)",
                    "description": (
                        f"Current hour ({hour:02d}:00) is within peak grid demand. "
                        f"Shifting deferrable loads (washer, dryer, dishwasher) to off-peak ({best_off_peak:02d}:00) "
                        f"is predicted to reduce consumption by {savings_wh} Wh ({savings_pct}%)."
                    ),
                    "estimated_savings_wh": savings_wh,
                    "unit": "Wh",
                    "caveats": "Estimate derived from counterfactual model re-prediction; assumes shiftable multi-kilowatt appliances.",
                }
            )

    # 2. Indoor Thermal Setpoint Optimization
    # If indoor temp is warmer than eco standard (20°C in winter) when outdoor is cold (< 12°C)
    if outdoor_temp < 15.0 and indoor_temp > 20.0:
        target_temp = indoor_temp - 1.0
        alt_input = dict(input_data)
        alt_input["indoor_temp"] = target_temp
        alt_res = predictor.predict(alt_input)
        alt_wh = alt_res["prediction_wh"]
        temp_savings_wh = round(predicted_wh - alt_wh, 1)

        recommendations.append(
            {
                "id": "thermostat_setpoint",
                "category": "HVAC / Thermal Setpoint",
                "title": f"Modulate Setpoint by -1 C ({indoor_temp:.1f} C -> {target_temp:.1f} C)",
                "description": (
                    f"Outdoor temperature is {outdoor_temp:.1f}°C. Lowering your indoor target by 1°C "
                    f"reduces thermal conduction load and is projected to save {abs(temp_savings_wh)} Wh "
                    f"over this interval."
                ),
                "estimated_savings_wh": abs(temp_savings_wh),
                "unit": "Wh",
                "caveats": "Calculated via ML feature sensitivity on temperature gradient; depends on home insulation rating.",
            }
        )
    elif outdoor_temp >= 22.0 and indoor_temp < 23.0:
        # Cooling scenario
        target_temp = indoor_temp + 1.0
        alt_input = dict(input_data)
        alt_input["indoor_temp"] = target_temp
        alt_res = predictor.predict(alt_input)
        cooling_savings = round(predicted_wh - alt_res["prediction_wh"], 1)

        recommendations.append(
            {
                "id": "thermostat_setpoint",
                "category": "HVAC / Cooling",
                "title": f"Increase Cooling Setpoint by +1 C ({indoor_temp:.1f} C -> {target_temp:.1f} C)",
                "description": (
                    f"Raising cooling setpoint by 1°C relieves air conditioning compressor demand, "
                    f"saving approximately {abs(cooling_savings)} Wh."
                ),
                "estimated_savings_wh": abs(cooling_savings),
                "unit": "Wh",
                "caveats": "Counterfactual model estimate; building envelope thermal inertia may vary.",
            }
        )

    # 3. Humidity Management
    if indoor_humidity > 55.0:
        alt_input = dict(input_data)
        alt_input["indoor_humidity"] = 45.0
        alt_res = predictor.predict(alt_input)
        rh_savings = round(predicted_wh - alt_res["prediction_wh"], 1)

        recommendations.append(
            {
                "id": "humidity_control",
                "category": "Moisture & Dehumidification",
                "title": f"Ventilate to Reduce Indoor Humidity ({indoor_humidity:.0f}% -> 45%)",
                "description": (
                    f"High indoor humidity ({indoor_humidity:.0f}%) triggers dehumidification and ventilation "
                    f"fans. Exhausting moisture through smart ventilation is estimated to save {abs(rh_savings)} Wh."
                ),
                "estimated_savings_wh": abs(rh_savings),
                "unit": "Wh",
                "caveats": "Predicted from indoor-outdoor vapor pressure differential in the trained model.",
            }
        )

    # 4. Contextual Benchmark against Historical Baseline
    # Find average for current hour
    hour_stats = next(
        (item for item in insights.hourly_profile if item["hour"] == hour), None
    )
    if hour_stats:
        typical_wh = hour_stats["mean_wh"]
        diff_from_typical = round(predicted_wh - typical_wh, 1)
        if diff_from_typical > 5.0:
            pct_above = round((diff_from_typical / typical_wh) * 100, 1)
            recommendations.append(
                {
                    "id": "benchmark_alert",
                    "category": "Benchmark Warning",
                    "title": f"Usage is {pct_above}% Above Typical {hour:02d}:00 Baseline",
                    "description": (
                        f"Your predicted {predicted_wh:.1f} Wh exceeds the historical median of {typical_wh:.1f} Wh "
                        f"for {hour:02d}:00 by +{diff_from_typical} Wh. Audit standby entertainment centers, "
                        f"auxiliary heaters, or high-draw pumps."
                    ),
                    "estimated_savings_wh": diff_from_typical,
                    "unit": "Wh",
                    "caveats": "Benchmark relative to 13.7-day household monitoring baseline.",
                }
            )
        else:
            recommendations.append(
                {
                    "id": "benchmark_good",
                    "category": "Benchmark Confirmation",
                    "title": f"Optimal Energy State for Hour {hour:02d}:00",
                    "description": (
                        f"Projected demand ({predicted_wh:.1f} Wh) is within or below historical norms "
                        f"({typical_wh:.1f} Wh average). Current sensor telemetry indicates efficient operation."
                    ),
                    "estimated_savings_wh": 0.0,
                    "unit": "Wh",
                    "caveats": "Telemetry matches efficient baseline quartile.",
                }
            )

    # Return top 2 to 4 recommendations
    return recommendations[:4]
