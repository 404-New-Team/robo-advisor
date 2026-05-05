from unittest.mock import AsyncMock, patch
from tests.conftest import MOCK_RESEARCH_RESULT


VALID_REQUEST = {
    "query": "삼성전자 최근 리스크 분석해줘",
    "ticker": "005930",
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


def test_research_empty_query_rejected(client):
    bad_request = {**VALID_REQUEST, "query": ""}
    response = client.post("/research", json=bad_request)
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
        assert event["severity"] in ("LOW", "MEDIUM", "HIGH")


def test_research_reasoning_trace_present(client):
    with patch("app.routers.research.call_research", new_callable=AsyncMock, return_value=MOCK_RESEARCH_RESULT):
        response = client.post("/research", json=VALID_REQUEST)
    data = response.json()
    assert "reasoning_trace" in data
    assert isinstance(data["reasoning_trace"], list)


def test_research_without_ticker(client):
    request_no_ticker = {"query": "글로벌 시장 리스크 분석해줘"}
    result = {**MOCK_RESEARCH_RESULT, "ticker": None}
    with patch("app.routers.research.call_research", new_callable=AsyncMock, return_value=result):
        response = client.post("/research", json=request_no_ticker)
    assert response.status_code == 200
