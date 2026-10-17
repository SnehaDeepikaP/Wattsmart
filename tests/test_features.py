"""Unit tests for feature engineering, temporal encodings, and strict target leakage prevention."""

import numpy as np
import pytest

from wattsmart.features import (
    add_environmental_features,
    add_lag_and_rolling_features,
    add_temporal_features,
)


def test_feature_leakage_check(sample_timeseries_df):
    """CRITICAL TEST: Assert that lag and rolling features at time t use only data strictly < t.

    1. Lag_k at step i must strictly equal target at step i-k.
    2. Rolling mean/std at step i must be calculated strictly from periods prior to step i.
    3. Mutating target at step i must NEVER affect lag or rolling features at step i.
    """
    df = sample_timeseries_df.copy()
    features_df = add_lag_and_rolling_features(df, target_col="Appliances")

    # Drop NaNs to inspect populated window
    features_clean = features_df.dropna()

    # 1. Check exact equivalence: lag_1[t_i] == target[t_{i-1}]
    for i in range(12, 25):
        current_time = features_clean.index[i]
        prev_time = features_clean.index[i - 1]
        lag6_time = features_clean.index[i - 6]

        expected_lag1 = df.loc[prev_time, "Appliances"]
        actual_lag1 = features_clean.loc[current_time, "appliances_lag_1"]
        assert actual_lag1 == pytest.approx(
            expected_lag1
        ), f"Lag 1 at {current_time} leaked or mismatched!"

        expected_lag6 = df.loc[lag6_time, "Appliances"]
        actual_lag6 = features_clean.loc[current_time, "appliances_lag_6"]
        assert actual_lag6 == pytest.approx(
            expected_lag6
        ), f"Lag 6 at {current_time} leaked or mismatched!"

        # 2. Check rolling mean includes exactly [t_{i-6} : t_{i-1}] and excludes t_i
        past_6_values = df.loc[:prev_time, "Appliances"].iloc[-6:].values
        expected_roll_mean_6 = float(np.mean(past_6_values))
        actual_roll_mean_6 = features_clean.loc[current_time, "appliances_roll_mean_6"]
        assert actual_roll_mean_6 == pytest.approx(
            expected_roll_mean_6
        ), "Rolling mean 6 leaked current time step!"

    # 3. Mutation test: modifying Appliances at time t must NOT alter features at time t
    test_idx = features_clean.index[20]
    original_lag1 = features_clean.loc[test_idx, "appliances_lag_1"]
    original_roll_mean = features_clean.loc[test_idx, "appliances_roll_mean_6"]

    # Mutate target at test_idx
    mutated_df = df.copy()
    mutated_df.loc[test_idx, "Appliances"] = 999999.0
    mutated_features = add_lag_and_rolling_features(mutated_df, target_col="Appliances")

    assert mutated_features.loc[test_idx, "appliances_lag_1"] == pytest.approx(
        original_lag1
    ), "Target mutation at time t corrupted lag_1 at time t! Leakage detected."
    assert mutated_features.loc[test_idx, "appliances_roll_mean_6"] == pytest.approx(
        original_roll_mean
    ), "Target mutation at time t corrupted rolling mean at time t! Leakage detected."


def test_cyclical_temporal_encodings(sample_timeseries_df):
    """Verify hour cyclical transformation satisfies sin^2(x) + cos^2(x) == 1."""
    df = add_temporal_features(sample_timeseries_df)
    assert "sin_hour" in df.columns
    assert "cos_hour" in df.columns

    sin_sq_plus_cos_sq = df["sin_hour"] ** 2 + df["cos_hour"] ** 2
    assert np.allclose(sin_sq_plus_cos_sq, 1.0, atol=1e-6)

    # Verify is_weekend logic
    assert set(df["is_weekend"].unique()).issubset({0, 1})


def test_environmental_differentials(sample_timeseries_df):
    """Verify temp_diff and rh_diff calculate indoor - outdoor gradients."""
    df = add_environmental_features(sample_timeseries_df)
    assert "temp_diff" in df.columns
    assert "rh_diff" in df.columns

    expected_temp_diff = df["T_indoor_mean"] - df["T_out"]
    assert np.allclose(df["temp_diff"], expected_temp_diff)
