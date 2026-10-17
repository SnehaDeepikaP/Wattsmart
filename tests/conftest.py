"""Pytest configuration and shared fixtures for Wattsmart test suite."""

import numpy as np
import pandas as pd
import pytest

from app.main import create_app


@pytest.fixture(scope="session")
def app():
    """Create test Flask application instance."""
    app_instance = create_app({"TESTING": True, "DEBUG": False})
    return app_instance


@pytest.fixture(scope="session")
def client(app):
    """Create Flask test client."""
    return app.test_client()


@pytest.fixture(scope="session")
def sample_timeseries_df():
    """Create a controlled synthetic time-series DataFrame for unit testing."""
    dates = pd.date_range("2026-01-01 00:00:00", periods=100, freq="10min")
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "Appliances": np.random.uniform(50.0, 300.0, size=len(dates)),
            "T1": np.random.uniform(18.0, 24.0, size=len(dates)),
            "RH_1": np.random.uniform(30.0, 60.0, size=len(dates)),
            "T_out": np.random.uniform(0.0, 10.0, size=len(dates)),
            "RH_out": np.random.uniform(60.0, 95.0, size=len(dates)),
            "Windspeed": np.random.uniform(1.0, 10.0, size=len(dates)),
            "Press_mm_hg": np.random.uniform(740.0, 770.0, size=len(dates)),
            "Tdewpoint": np.random.uniform(-2.0, 5.0, size=len(dates)),
            "Visibility": np.random.uniform(20.0, 45.0, size=len(dates)),
        },
        index=dates,
    )
    return df
