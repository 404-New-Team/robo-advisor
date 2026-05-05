from unittest.mock import AsyncMock, patch
from tests.conftest import MOCK_OPTIMIZE_RESULT


VALID_REQUEST = {
    "tickers": ["005930", "069500", "SPY", "QQQ", "GLD"],
    "risk_level": "moderate",
    "start_date": "2021-01-01",
    "end_date": "2026-01-01",
}


def test_optimize_success(client):
    with patch("app.routers.optimize.call_optimize", new_callable=AsyncMock, return_value=MOCK_OPTIMIZE_RESULT):
        response = client.post("/optimize", json=VALID_REQUEST)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "weights" in data
    assert "metrics" in data


def test_optimize_weights_sum_to_one(client):
    with patch("app.routers.optimize.call_optimize", new_callable=AsyncMock, return_value=MOCK_OPTIMIZE_RESULT):
        response = client.post("/optimize", json=VALID_REQUEST)
    weights = response.json()["weights"]
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_optimize_invalid_risk_level(client):
    bad_request = {**VALID_REQUEST, "risk_level": "extreme"}
    response = client.post("/optimize", json=bad_request)
    assert response.status_code == 422


def test_optimize_missing_tickers(client):
    bad_request = {**VALID_REQUEST, "tickers": []}
    response = client.post("/optimize", json=bad_request)
    assert response.status_code == 422


def test_optimize_risk_tags_in_response(client):
    with patch("app.routers.optimize.call_optimize", new_callable=AsyncMock, return_value=MOCK_OPTIMIZE_RESULT):
        response = client.post("/optimize", json=VALID_REQUEST)
    data = response.json()
    assert "risk_tags" in data
    assert isinstance(data["risk_tags"], list)
    if data["risk_tags"]:
        tag = data["risk_tags"][0]
        assert "asset" in tag
        assert "severity" in tag
