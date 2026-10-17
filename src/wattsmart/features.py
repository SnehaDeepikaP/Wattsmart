"""Feature engineering module for Wattsmart.

Implements leak-free feature engineering including:
- Temporal features (hour, day_of_week, weekend, sin/cos cyclical encodings)
- Environmental deltas (indoor-outdoor temperature & humidity differentials)
- Past-only lag features (e.g. lag 1, 2, 6, 12)
- Past-only rolling window statistics (rolling mean/std shifted by 1)
- Inference feature preparation with robust fallback defaults.
"""

import logging
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from wattsmart.config import (
    INDOOR_RH_COLS,
    INDOOR_TEMP_COLS,
    OUTDOOR_WEATHER_COLS,
    TARGET_COL,
)

logger = logging.getLogger(__name__)

# Core lag steps for 10-minute intervals
LAG_STEPS: List[int] = [1, 2, 6, 12]
ROLLING_WINDOWS: List[int] = [6, 12]


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal and cyclical sinusoidal features from DatetimeIndex."""
    data = df.copy()
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError(
            "DataFrame index must be a DatetimeIndex for temporal feature extraction."
        )

    data["hour"] = data.index.hour
    data["day_of_week"] = data.index.dayofweek
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
    # Cyclical hour encodings for smooth 23h -> 0h continuity
    data["sin_hour"] = np.sin(2 * np.pi * data["hour"] / 24.0)
    data["cos_hour"] = np.cos(2 * np.pi * data["hour"] / 24.0)
    return data


def add_environmental_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute indoor aggregation and indoor-outdoor differentials."""
    data = df.copy()

    # Calculate indoor mean temperature
    available_temp_cols = [c for c in INDOOR_TEMP_COLS if c in data.columns]
    if available_temp_cols:
        data["T_indoor_mean"] = data[available_temp_cols].mean(axis=1)
    elif "T_in" in data.columns:
        data["T_indoor_mean"] = data["T_in"]
    else:
        data["T_indoor_mean"] = 21.0

    # Calculate indoor mean relative humidity
    available_rh_cols = [c for c in INDOOR_RH_COLS if c in data.columns]
    if available_rh_cols:
        data["RH_indoor_mean"] = data[available_rh_cols].mean(axis=1)
    elif "RH_in" in data.columns:
        data["RH_indoor_mean"] = data["RH_in"]
    else:
        data["RH_indoor_mean"] = 40.0

    # Indoor - outdoor differentials (drives thermal and vapor conduction)
    if "T_out" in data.columns:
        data["temp_diff"] = data["T_indoor_mean"] - data["T_out"]
    else:
        data["temp_diff"] = 0.0

    if "RH_out" in data.columns:
        data["rh_diff"] = data["RH_indoor_mean"] - data["RH_out"]
    else:
        data["rh_diff"] = 0.0

    return data


def add_lag_and_rolling_features(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """Create leak-free lag and rolling statistics on the target variable.

    Strict Leakage Prevention:
    - Lag features use shift(k) where k >= 1 (only observations from past time steps).
    - Rolling features MUST use shift(1) prior to .rolling() so that the current time step's
      target value is NEVER included in rolling computations.

    Args:
        df: Cleaned DataFrame with DatetimeIndex sorted chronologically.
        target_col: Target column name.

    Returns:
        DataFrame with lag and rolling columns appended.
    """
    data = df.copy()
    if target_col not in data.columns:
        logger.warning(
            "Target column '%s' not present; skipping target lags.", target_col
        )
        return data

    target_series = data[target_col]

    # Explicit lag features
    for lag in LAG_STEPS:
        data[f"appliances_lag_{lag}"] = target_series.shift(lag)

    # Rolling window statistics (strictly shifted by 1 to prevent target leakage)
    shifted_target = target_series.shift(1)
    for window in ROLLING_WINDOWS:
        data[f"appliances_roll_mean_{window}"] = shifted_target.rolling(
            window=window, min_periods=1
        ).mean()
        data[f"appliances_roll_std_{window}"] = (
            shifted_target.rolling(window=window, min_periods=1).std().fillna(0.0)
        )

    return data


def engineer_features(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    drop_na_rows: bool = True,
) -> pd.DataFrame:
    """Full feature engineering pipeline for tabular time-series modeling.

    Args:
        df: Cleaned DataFrame.
        target_col: Target column name.
        drop_na_rows: Whether to drop initial rows that have NaN from lag shifts.

    Returns:
        Enriched DataFrame with all features.
    """
    data = add_temporal_features(df)
    data = add_environmental_features(data)
    data = add_lag_and_rolling_features(data, target_col=target_col)

    if drop_na_rows:
        data = data.dropna()

    return data


def get_feature_and_target_matrices(
    df_features: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Separate feature matrix X and target y, excluding target from X.

    Args:
        df_features: Feature-engineered DataFrame.
        target_col: Name of target column.

    Returns:
        (X, y, feature_column_names)
    """
    if target_col in df_features.columns:
        y = df_features[target_col].copy()
        X = df_features.drop(columns=[target_col]).copy()
    else:
        y = None
        X = df_features.copy()

    feature_cols = list(X.columns)
    return X, y, feature_cols


def prepare_inference_features(
    input_data: Dict[str, Union[float, int]],
    feature_list: List[str],
    dataset_medians: Dict[str, float],
) -> Tuple[pd.DataFrame, List[str]]:
    """Construct a single-row feature vector matching the trained model schema.

    Fills optional missing features using dataset medians and computes all
    derived temporal, cyclical, and environmental differential features.

    Args:
        input_data: User-supplied dictionary of parameters.
        feature_list: Exact ordered list of columns used during model training.
        dataset_medians: Training set column medians for imputation.

    Returns:
        (X_inference, list_of_defaults_applied)
    """
    defaults_applied: List[str] = []
    row: Dict[str, float] = {}

    # Extract primary user inputs with graceful fallback to dataset medians
    indoor_temp = input_data.get("indoor_temp", input_data.get("T_in"))
    if indoor_temp is None:
        indoor_temp = dataset_medians.get("T1", 21.0)
        defaults_applied.append("indoor_temp")

    indoor_humidity = input_data.get("indoor_humidity", input_data.get("RH_in"))
    if indoor_humidity is None:
        indoor_humidity = dataset_medians.get("RH_1", 40.0)
        defaults_applied.append("indoor_humidity")

    outdoor_temp = input_data.get("outdoor_temp", input_data.get("T_out"))
    if outdoor_temp is None:
        outdoor_temp = dataset_medians.get("T_out", 7.0)
        defaults_applied.append("outdoor_temp")

    outdoor_humidity = input_data.get("outdoor_humidity", input_data.get("RH_out"))
    if outdoor_humidity is None:
        outdoor_humidity = dataset_medians.get("RH_out", 80.0)
        defaults_applied.append("outdoor_humidity")

    windspeed = input_data.get("windspeed", input_data.get("Windspeed"))
    if windspeed is None:
        windspeed = dataset_medians.get("Windspeed", 4.0)
        defaults_applied.append("windspeed")

    pressure = input_data.get("pressure", input_data.get("Press_mm_hg"))
    if pressure is None:
        pressure = dataset_medians.get("Press_mm_hg", 755.0)
        defaults_applied.append("pressure")

    hour = input_data.get("hour")
    if hour is None:
        hour = 12.0
        defaults_applied.append("hour")

    day_of_week = input_data.get("day_of_week")
    if day_of_week is None:
        day_of_week = 2.0  # Wednesday
        defaults_applied.append("day_of_week")

    is_weekend = 1.0 if float(day_of_week) >= 5 else 0.0

    # Cyclical hour features
    sin_hour = float(np.sin(2 * np.pi * float(hour) / 24.0))
    cos_hour = float(np.cos(2 * np.pi * float(hour) / 24.0))

    # Differentials
    temp_diff = float(indoor_temp) - float(outdoor_temp)
    rh_diff = float(indoor_humidity) - float(outdoor_humidity)

    # Base sensor mappings
    base_sensors = {
        "T_indoor_mean": float(indoor_temp),
        "RH_indoor_mean": float(indoor_humidity),
        "T_out": float(outdoor_temp),
        "RH_out": float(outdoor_humidity),
        "Windspeed": float(windspeed),
        "Press_mm_hg": float(pressure),
        "hour": float(hour),
        "day_of_week": float(day_of_week),
        "is_weekend": is_weekend,
        "sin_hour": sin_hour,
        "cos_hour": cos_hour,
        "temp_diff": temp_diff,
        "rh_diff": rh_diff,
    }

    # Populate each required model feature in exact column order
    for col in feature_list:
        if col in base_sensors:
            row[col] = base_sensors[col]
        elif col in input_data and input_data[col] is not None:
            row[col] = float(input_data[col])
        elif col in INDOOR_TEMP_COLS:
            # Approximate specific room temp from indoor_temp plus historical room offset
            room_offset = dataset_medians.get(col, 20.0) - dataset_medians.get(
                "T1", 20.0
            )
            row[col] = float(indoor_temp) + room_offset
            defaults_applied.append(col)
        elif col in INDOOR_RH_COLS:
            # Approximate specific room RH from indoor_humidity plus room offset
            room_rh_offset = dataset_medians.get(col, 40.0) - dataset_medians.get(
                "RH_1", 40.0
            )
            row[col] = float(indoor_humidity) + room_rh_offset
            defaults_applied.append(col)
        elif col in OUTDOOR_WEATHER_COLS:
            row[col] = float(dataset_medians.get(col, 0.0))
            defaults_applied.append(col)
        elif col.startswith("appliances_"):
            # Lags / rolling target features: use hourly or overall dataset median
            default_val = dataset_medians.get(
                col, dataset_medians.get("Appliances", 160.0)
            )
            row[col] = float(default_val)
            defaults_applied.append(col)
        else:
            row[col] = float(dataset_medians.get(col, 0.0))
            defaults_applied.append(col)

    df_inf = pd.DataFrame([row], columns=feature_list)
    return df_inf, sorted(list(set(defaults_applied)))
