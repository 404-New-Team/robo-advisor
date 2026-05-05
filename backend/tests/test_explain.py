from unittest.mock import AsyncMock, patch
from tests.conftest import MOCK_SHAP_RESULT


VALID_REQUEST = {
    "tickers": ["005930", "069500", "SPY", "QQQ", "GLD"],
    "target_asset": "005930",
    "date": "2026-05-03",
}


def test_explain_success(client):
    with patch("app.routers.explain.call_shap", new_callable=AsyncMock, return_value=MOCK_SHAP_RESULT):
        response = client.post("/explain", json=VALID_REQUEST)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "shap_values" in data
    assert "explanation" in data


def test_explain_target_not_in_tickers(client):
    bad_request = {**VALID_REQUEST, "target_asset": "AAPL"}
    response = client.post("/explain", json=bad_request)
    assert response.status_code == 404


def test_explain_shap_values_structure(client):
    with patch("app.routers.explain.call_shap", new_callable=AsyncMock, return_value=MOCK_SHAP_RESULT):
        response = client.post("/explain", json=VALID_REQUEST)
    shap_values = response.json()["shap_values"]
    assert isinstance(shap_values, dict)
    for v in shap_values.values():
        assert isinstance(v, float)


def test_explain_plot_urls_returned(client):
    with patch("app.routers.explain.call_shap", new_callable=AsyncMock, return_value=MOCK_SHAP_RESULT):
        response = client.post("/explain", json=VALID_REQUEST)
    data = response.json()
    assert "force_plot_url" in data
    assert "summary_plot_url" in data
