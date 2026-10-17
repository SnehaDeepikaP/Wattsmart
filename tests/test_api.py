"""Functional API integration tests for all Flask routes and endpoints."""


def test_health_check_endpoint(client):
    """Verify /health returns 200 and healthy status."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["version"] == "2.0.0"


def test_predict_endpoint_success(client):
    """Verify POST /api/predict returns valid prediction, interval, and recommendations."""
    payload = {
        "indoor_temp": 21.5,
        "indoor_humidity": 45.0,
        "outdoor_temp": 6.0,
        "outdoor_humidity": 80.0,
        "windspeed": 3.5,
        "pressure": 756.0,
        "hour": 18,
        "day_of_week": 2,
    }
    resp = client.post("/api/predict", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "success"
    assert data["unit"] == "Wh"
    assert "prediction_wh" in data
    assert "confidence_interval" in data
    assert len(data["recommendations"]) >= 1

    # Check that recommendation estimates are numbers in Wh
    for rec in data["recommendations"]:
        assert "estimated_savings_wh" in rec
        assert rec["unit"] == "Wh"
        assert "caveats" in rec


def test_predict_validation_errors(client):
    """Verify POST /api/predict rejects invalid parameter ranges with HTTP 400."""
    # Out of range hour (hour must be 0-23)
    resp = client.post("/api/predict", json={"hour": 99})
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["error"] == "Validation Error"

    # Out of range temperature (indoor_temp > 45°C)
    resp = client.post("/api/predict", json={"indoor_temp": 95.0})
    assert resp.status_code == 400

    # Non-JSON payload
    resp = client.post(
        "/api/predict", data="invalid non json", content_type="text/plain"
    )
    assert resp.status_code == 400


def test_forecast_endpoint(client):
    """Verify POST /api/forecast returns multi-step forward projection."""
    resp = client.post("/api/forecast", json={"hours": 6})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "success"
    series = data["forecast"]["series"]
    assert len(series) == 36  # 6 hours * 6 10-min steps = 36 steps
    for item in series:
        assert "prediction_wh" in item
        assert "lower_wh" in item
        assert "upper_wh" in item
        assert item["lower_wh"] <= item["upper_wh"]


def test_whatif_endpoint(client):
    """Verify POST /api/whatif computes counterfactual delta between base and modified states."""
    payload = {
        "base_scenario": {"indoor_temp": 22.0, "hour": 18},
        "modifications": {"indoor_temp": 20.0, "hour": 22},
    }
    resp = client.post("/api/whatif", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    whatif = data["whatif"]
    assert "base_prediction_wh" in whatif
    assert "modified_prediction_wh" in whatif
    assert "delta_wh" in whatif
    assert "percent_change" in whatif
    assert "impact_summary" in whatif


def test_insights_endpoint(client):
    """Verify GET /api/insights returns peak hours, off-peak hours, profiles, and anomalies."""
    resp = client.get("/api/insights")
    assert resp.status_code == 200
    data = resp.get_json()
    insights = data["insights"]
    assert len(insights["peak_hours"]) > 0
    assert len(insights["off_peak_hours"]) > 0
    assert len(insights["hourly_profile"]) == 24
    assert len(insights["day_of_week_profile"]) == 7
    assert len(insights["anomaly_spikes"]) > 0


def test_model_info_endpoint(client):
    """Verify GET /api/model-info returns real metrics, comparison table, and features."""
    resp = client.get("/api/model-info")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "metrics" in data
    assert "model_comparison" in data
    assert "top_features" in data
    assert len(data["top_features"]) > 0


def test_html_pages(client):
    """Verify all frontend HTML routes render HTTP 200."""
    routes = ["/", "/forecast", "/whatif", "/insights", "/model"]
    for route in routes:
        resp = client.get(route)
        assert resp.status_code == 200, f"Route {route} failed with {resp.status_code}"
        assert b"Wattsmart" in resp.data
