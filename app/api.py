"""API routes and request validation for Wattsmart."""

import logging
from typing import Optional

from flask import Blueprint, jsonify, request
from pydantic import BaseModel, Field, ValidationError

from wattsmart.insights import DatasetInsights, generate_recommendations
from wattsmart.predict import Predictor

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__, url_prefix="/api")


# --- Pydantic Validation Schemas ---


class PredictInputSchema(BaseModel):
    """Schema for single-step appliance energy prediction."""

    indoor_temp: Optional[float] = Field(
        None, ge=-10.0, le=45.0, description="Indoor temperature in °C"
    )
    indoor_humidity: Optional[float] = Field(
        None, ge=5.0, le=95.0, description="Indoor relative humidity in %"
    )
    outdoor_temp: Optional[float] = Field(
        None, ge=-30.0, le=50.0, description="Outdoor temperature in °C"
    )
    outdoor_humidity: Optional[float] = Field(
        None, ge=5.0, le=100.0, description="Outdoor relative humidity in %"
    )
    windspeed: Optional[float] = Field(
        None, ge=0.0, le=50.0, description="Windspeed in m/s"
    )
    pressure: Optional[float] = Field(
        None, ge=650.0, le=850.0, description="Atmospheric pressure in mm Hg"
    )
    hour: Optional[int] = Field(None, ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: Optional[int] = Field(
        None, ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)"
    )

    # Legacy field aliases for backwards compatibility
    T_in: Optional[float] = Field(None, ge=-10.0, le=45.0)
    RH_in: Optional[float] = Field(None, ge=5.0, le=95.0)
    T_out: Optional[float] = Field(None, ge=-30.0, le=50.0)
    RH_out: Optional[float] = Field(None, ge=5.0, le=100.0)
    Windspeed: Optional[float] = Field(None, ge=0.0, le=50.0)
    Press_mm_hg: Optional[float] = Field(None, ge=650.0, le=850.0)


class ForecastInputSchema(BaseModel):
    """Schema for multi-hour autoregressive forecast."""

    hours: int = Field(
        default=12, ge=1, le=24, description="Forecast horizon in hours (1 to 24)"
    )


class WhatIfInputSchema(BaseModel):
    """Schema for counterfactual what-if scenario comparison."""

    base_scenario: PredictInputSchema
    modifications: PredictInputSchema


# --- API Routes ---


@api_bp.route("/predict", methods=["POST"])
def predict_endpoint():
    """Predict household appliance energy consumption with conformal prediction intervals.

    Returns:
        JSON with predicted Wh, confidence interval bounds, applied defaults, and dynamic tips.
    """
    raw_json = request.get_json(silent=True)
    if raw_json is None:
        return (
            jsonify(
                {
                    "error": "Bad Request",
                    "message": "Request body must be a valid JSON object.",
                }
            ),
            400,
        )

    try:
        validated = PredictInputSchema.model_validate(raw_json)
    except ValidationError as exc:
        errors = [
            {"field": ".".join(str(loc) for loc in err["loc"]), "message": err["msg"]}
            for err in exc.errors()
        ]
        return (
            jsonify(
                {
                    "error": "Validation Error",
                    "message": "Invalid input parameter ranges.",
                    "details": errors,
                }
            ),
            400,
        )

    predictor = Predictor.get_instance()
    # Normalize input dict
    clean_dict = validated.model_dump(exclude_none=True)
    if "T_in" in clean_dict and "indoor_temp" not in clean_dict:
        clean_dict["indoor_temp"] = clean_dict["T_in"]
    if "RH_in" in clean_dict and "indoor_humidity" not in clean_dict:
        clean_dict["indoor_humidity"] = clean_dict["RH_in"]
    if "T_out" in clean_dict and "outdoor_temp" not in clean_dict:
        clean_dict["outdoor_temp"] = clean_dict["T_out"]
    if "RH_out" in clean_dict and "outdoor_humidity" not in clean_dict:
        clean_dict["outdoor_humidity"] = clean_dict["RH_out"]
    if "Windspeed" in clean_dict and "windspeed" not in clean_dict:
        clean_dict["windspeed"] = clean_dict["Windspeed"]
    if "Press_mm_hg" in clean_dict and "pressure" not in clean_dict:
        clean_dict["pressure"] = clean_dict["Press_mm_hg"]

    pred_res = predictor.predict(clean_dict)
    recs = generate_recommendations(clean_dict, pred_res["prediction_wh"], predictor)

    return (
        jsonify(
            {
                "status": "success",
                "prediction_wh": pred_res["prediction_wh"],
                "unit": "Wh",
                "confidence_interval": pred_res["interval"],
                "inputs_used": pred_res["inputs_used"],
                "defaults_applied": pred_res["defaults_applied"],
                "recommendations": recs,
            }
        ),
        200,
    )


@api_bp.route("/forecast", methods=["POST"])
def forecast_endpoint():
    """Generate forward N-hour autoregressive forecast with confidence interval bands."""
    raw_json = request.get_json(silent=True) or {}

    try:
        validated = ForecastInputSchema.model_validate(raw_json)
    except ValidationError as exc:
        errors = [
            {"field": ".".join(str(loc) for loc in err["loc"]), "message": err["msg"]}
            for err in exc.errors()
        ]
        return (
            jsonify(
                {
                    "error": "Validation Error",
                    "message": "Invalid forecast horizon. Must be between 1 and 24 hours.",
                    "details": errors,
                }
            ),
            400,
        )

    predictor = Predictor.get_instance()
    forecast_data = predictor.forecast(hours=validated.hours)
    return (
        jsonify(
            {
                "status": "success",
                "forecast": forecast_data,
            }
        ),
        200,
    )


@api_bp.route("/whatif", methods=["POST"])
def whatif_endpoint():
    """Simulate counterfactual energy delta between base scenario and modified conditions."""
    raw_json = request.get_json(silent=True)
    if raw_json is None:
        return (
            jsonify(
                {
                    "error": "Bad Request",
                    "message": "Request body must be a valid JSON object.",
                }
            ),
            400,
        )

    try:
        validated = WhatIfInputSchema.model_validate(raw_json)
    except ValidationError as exc:
        errors = [
            {"field": ".".join(str(loc) for loc in err["loc"]), "message": err["msg"]}
            for err in exc.errors()
        ]
        return (
            jsonify(
                {
                    "error": "Validation Error",
                    "message": "Invalid what-if scenario parameters.",
                    "details": errors,
                }
            ),
            400,
        )

    predictor = Predictor.get_instance()
    base_dict = validated.base_scenario.model_dump(exclude_none=True)
    mod_dict = validated.modifications.model_dump(exclude_none=True)

    result = predictor.whatif(base_dict, mod_dict)
    return (
        jsonify(
            {
                "status": "success",
                "whatif": result,
            }
        ),
        200,
    )


@api_bp.route("/insights", methods=["GET"])
def insights_endpoint():
    """Return data-derived peak hours, off-peak hours, hourly profiles, and anomaly spikes."""
    insights = DatasetInsights.get_instance()
    payload = insights.get_insights_payload()
    return (
        jsonify(
            {
                "status": "success",
                "insights": payload,
            }
        ),
        200,
    )


@api_bp.route("/model-info", methods=["GET"])
def model_info_endpoint():
    """Return model metadata, benchmark comparison against baselines, and feature importances."""
    predictor = Predictor.get_instance()
    return (
        jsonify(
            {
                "status": "success",
                "model_type": predictor.metrics_data.get("model_type", "XGBRegressor"),
                "metrics": predictor.metrics_data.get("metrics", {}),
                "model_comparison": predictor.metrics_data.get("model_comparison", {}),
                "conformal_prediction": predictor.metrics_data.get(
                    "conformal_prediction", {}
                ),
                "dataset_info": predictor.metrics_data.get("dataset_info", {}),
                "training_timestamp_utc": predictor.metrics_data.get(
                    "training_timestamp_utc"
                ),
                "top_features": sorted(
                    [
                        {"feature": k, "importance": v}
                        for k, v in predictor.feature_importances.items()
                    ],
                    key=lambda x: x["importance"],
                    reverse=True,
                )[:15],
            }
        ),
        200,
    )
