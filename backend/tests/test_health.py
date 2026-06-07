from unittest.mock import AsyncMock, patch

from app.services import ai_client


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


async def test_check_ai_health_uses_health_endpoint(monkeypatch):
    class FakeResponse:
        status_code = 200

        def json(self):
            return {"rl_engine": True, "rag_agent": True, "shap_explainer": False}

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, path):
            assert path == "/health"
            return FakeResponse()

    endpoints = []

    def fake_get_client(endpoint):
        endpoints.append(endpoint)
        return FakeClient()

    monkeypatch.setattr(ai_client, "_get_client", fake_get_client)

    result = await ai_client.check_ai_health()

    assert endpoints == ["/health"]
    assert result == {"rl_engine": True, "rag_agent": True, "shap_explainer": False}

