import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client

@pytest.fixture
def valid_borrower_payload():
    return {
        "status_checking": "A11",
        "duration_months": 24.0,
        "credit_history": "A32",
        "purpose": "A40",
        "credit_amount": 4500.0,
        "savings_account": "A61",
        "employment_since": "A73",
        "installment_rate_pct": 4.0,
        "personal_status_sex": "A93",
        "other_debtors": "A101",
        "residence_since_years": 4.0,
        "property": "A121",
        "age_years": 35.0,
        "other_installment_plans": "A143",
        "housing": "A152",
        "existing_credits": 1.0,
        "job": "A173",
        "people_liable": 1.0,
        "telephone": "A191",
        "foreign_worker": "A201"
    }

def test_read_root(client):
    response = client.get("/")
    assert response.status_code == 200
    json_data = response.json()
    assert "name" in json_data
    assert json_data["name"] == "CreditRiskML Platform API"
    assert "documentation" in json_data
    assert "model_version" in json_data

def test_get_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "healthy"
    assert json_data["database_connected"] is True
    assert json_data["model_loaded"] is True
    assert "timestamp" in json_data

def test_get_model_metadata(client):
    response = client.get("/model")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert data["status"] == "Champion"
    assert "decision_threshold" in data
    assert "validation_metrics" in data
    assert "test_metrics" in data

def test_predict_single_borrower(client, valid_borrower_payload):
    response = client.post("/predict", json=valid_borrower_payload)
    assert response.status_code == 200
    data = response.json()
    assert "request_id" in data
    assert data["prediction"] in (0, 1)
    assert 0.0 <= data["default_probability"] <= 1.0
    assert data["risk_level"] in ("Low", "Medium", "High")
    assert "decision_threshold" in data
    assert "top_risk_factors" in data
    assert isinstance(data["top_risk_factors"], list)

def test_predict_batch_borrowers(client, valid_borrower_payload):
    payload = {
        "applications": [
            valid_borrower_payload,
            {**valid_borrower_payload, "credit_amount": 1200.0, "duration_months": 12.0}
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2
    for pred in data["predictions"]:
        assert pred["prediction"] in (0, 1)
        assert 0.0 <= pred["default_probability"] <= 1.0
        assert pred["risk_level"] in ("Low", "Medium", "High")

def test_predict_invalid_input(client, valid_borrower_payload):
    invalid_payload = {**valid_borrower_payload, "duration_months": -5.0}
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422

def test_metrics_endpoint(client, valid_borrower_payload):
    # Perform a request to generate metrics
    client.post("/predict", json=valid_borrower_payload)
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert data["total_requests"] >= 1
    assert "latency_p50_ms" in data
    assert "latency_avg_ms" in data

def test_monitoring_drift_endpoint(client):
    response = client.get("/monitoring/drift")
    assert response.status_code == 200
    data = response.json()
    assert "total_features" in data
    assert "drifted_features_count" in data
    assert "dataset_drift_detected" in data
    assert isinstance(data["features"], list)

def test_monitoring_performance_endpoint(client):
    response = client.get("/monitoring/performance")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "samples_count" in data

def test_dashboard_endpoints(client):
    # Test GET /dashboard returns HTML
    resp_dash = client.get("/dashboard")
    assert resp_dash.status_code == 200
    assert "text/html" in resp_dash.headers.get("content-type", "")
    assert "CreditRiskML" in resp_dash.text

    # Test GET / with text/html header returns dashboard
    resp_html = client.get("/", headers={"accept": "text/html,application/xhtml+xml"})
    assert resp_html.status_code == 200
    assert "text/html" in resp_html.headers.get("content-type", "")
