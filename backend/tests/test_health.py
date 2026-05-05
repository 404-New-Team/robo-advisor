from unittest.mock import AsyncMock, patch


def test_health_returns_ok(client):
    mock_models = {"rl_engine": True, "rag_agent": True, "shap_explainer": True}
    with patch("app.routers.health.check_ai_health", new_callable=AsyncMock, return_value=mock_models):
        response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "timestamp" in data
    assert "version" in data


def test_health_models_structure(client):
    mock_models = {"rl_engine": True, "rag_agent": False, "shap_explainer": True}
    with patch("app.routers.health.check_ai_health", new_callable=AsyncMock, return_value=mock_models):
        response = client.get("/health")
    data = response.json()
    assert "models" in data
    assert isinstance(data["models"]["rl_engine"], bool)
    assert isinstance(data["models"]["rag_agent"], bool)
    assert isinstance(data["models"]["shap_explainer"], bool)


def test_health_ai_down_still_returns_200(client):
    with patch("app.routers.health.check_ai_health", new_callable=AsyncMock, return_value={}):
        response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["models"]["rl_engine"] is False
