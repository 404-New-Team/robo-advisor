from unittest.mock import AsyncMock, patch
from tests.conftest import MOCK_RESEARCH_RESULT


VALID_REQUEST = {
    "tickers": ["005930"],
    "max_results": 5,
}


def test_research_success(client):
    with patch("app.routers.research.call_research", new_callable=AsyncMock, return_value=MOCK_RESEARCH_RESULT):
        response = client.post("/research", json=VALID_REQUEST)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "summary" in data
    assert "sources" in data


def test_research_empty_tickers_rejected(client):
    response = client.post("/research", json={"tickers": []})
    assert response.status_code == 422


def test_research_risk_events_structure(client):
    with patch("app.routers.research.call_research", new_callable=AsyncMock, return_value=MOCK_RESEARCH_RESULT):
        response = client.post("/research", json=VALID_REQUEST)
    risk_events = response.json()["risk_events"]
    assert isinstance(risk_events, list)
    if risk_events:
        event = risk_events[0]
        assert "type" in event
        assert "severity" in event
        assert event["severity"] in ("low", "moderate", "high")


def test_research_reasoning_trace_present(client):
    with patch("app.routers.research.call_research", new_callable=AsyncMock, return_value=MOCK_RESEARCH_RESULT):
        response = client.post("/research", json=VALID_REQUEST)
    data = response.json()
    assert "reasoning_trace" in data
    assert isinstance(data["reasoning_trace"], list)


def test_research_multiple_tickers(client):
    request = {**VALID_REQUEST, "tickers": ["005930", "000660"]}
    result = {**MOCK_RESEARCH_RESULT, "tickers": ["005930", "000660"]}
    with patch("app.routers.research.call_research", new_callable=AsyncMock, return_value=result):
        response = client.post("/research", json=request)
    assert response.status_code == 200
    assert response.json()["tickers"] == ["005930", "000660"]


def test_research_forwards_portfolio_context(client):
    request = {
        **VALID_REQUEST,
        "portfolio_context": {
            "risk_level": "moderate",
            "investment_amount": 30000000,
            "selected_tickers": ["005930", "000660"],
            "excluded_tickers": [],
            "active_tickers": ["005930", "000660"],
            "weights": {"005930": 0.6, "000660": 0.4},
        },
    }
    with patch("app.routers.research.call_research", new_callable=AsyncMock, return_value=MOCK_RESEARCH_RESULT) as mocked:
        response = client.post("/research", json=request)
    assert response.status_code == 200
    assert mocked.await_args.args[0]["portfolio_context"]["active_tickers"] == ["005930", "000660"]
