"""Unit tests for data loading, schema validation, column cleaning, and chronological splitting."""

import pandas as pd
import pytest

from wattsmart.config import RAW_DATA_PATH, TARGET_COL
from wattsmart.data import (
    DataValidationError,
    clean_data,
    get_chronological_split,
    load_raw_data,
    resample_data,
    validate_schema,
)


def test_raw_data_loading():
    """Verify raw data loads into a DatetimeIndex DataFrame with expected target column."""
    df = load_raw_data(RAW_DATA_PATH)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert TARGET_COL in df.columns
    assert len(df) == 19735
    assert df.index.is_monotonic_increasing


def test_schema_validation():
    """Verify validation passes on clean raw data and captures expected summary fields."""
    df = load_raw_data(RAW_DATA_PATH)
    summary = validate_schema(df)
    assert summary["is_valid"] is True
    assert summary["duplicate_timestamps"] == 0
    assert summary["total_nulls"] == 0
    assert summary["target_mean"] > 0


def test_schema_validation_rejects_negative_target():
    """Verify schema validation raises DataValidationError when negative consumption is present."""
    df = load_raw_data(RAW_DATA_PATH).iloc[:50].copy()
    df.iloc[5, df.columns.get_loc(TARGET_COL)] = -25.0
    with pytest.raises(DataValidationError, match="negative"):
        validate_schema(df)


def test_cleaning_drops_redundant_and_noise_columns():
    """Verify clean_data drops exact duplicates T2, RH_2 and random noise rv1, rv2."""
    df = load_raw_data(RAW_DATA_PATH).iloc[:50].copy()
    cleaned = clean_data(df)
    for col in ["T2", "RH_2", "rv1", "rv2"]:
        assert col not in cleaned.columns


def test_resampling_aggregation():
    """Verify resampling aggregates 1-minute data into 10-minute intervals without nulls."""
    df = (
        load_raw_data(RAW_DATA_PATH).iloc[:60].copy()
    )  # 60 1-min observations = 6 10-min bins
    resampled = resample_data(df, freq="10min")
    assert len(resampled) == 6
    assert not resampled.isnull().any().any()


def test_chronological_split_prevents_leakage():
    """Verify chronological split strictly segregates past train data from future test data."""
    df = load_raw_data(RAW_DATA_PATH).iloc[:100].copy()
    train_df, test_df = get_chronological_split(df, split_ratio=0.8)

    assert len(train_df) == 80
    assert len(test_df) == 20
    assert train_df.index.max() < test_df.index.min()
    # Ensure no overlapping timestamps between sets
    assert len(set(train_df.index).intersection(set(test_df.index))) == 0
