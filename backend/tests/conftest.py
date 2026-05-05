import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.database import get_db


class FakeDB:
    def add(self, obj): pass
    def commit(self): pass
    def close(self): pass


def override_get_db():
    yield FakeDB()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


MOCK_OPTIMIZE_RESULT = {
    "status": "success",
    "weights": {"005930": 0.20, "069500": 0.15, "SPY": 0.30, "QQQ": 0.20, "GLD": 0.15},
    "metrics": {
        "expected_return": 0.112,
        "sharpe_ratio": 1.43,
        "max_drawdown": -0.087,
        "volatility": 0.134,
    },
    "risk_tags": [
        {"asset": "005930", "type": "규제", "severity": "high", "source": "삼성전자 미국 수출 규제 강화 우려"}
    ],
}

MOCK_SHAP_RESULT = {
    "status": "success",
    "target_asset": "005930",
    "final_weight": 0.20,
    "shap_values": {
        "momentum_7d": 0.032,
        "volatility_30d": -0.028,
        "news_risk_score": -0.041,
        "rsi": 0.015,
        "market_cap_weight": 0.019,
    },
    "explanation": "삼성전자의 비중이 20%로 결정된 주요 원인은 뉴스 리스크 점수(-0.041)입니다.",
    "force_plot_url": "/static/shap/force_005930_20260503.png",
    "summary_plot_url": "/static/shap/summary_20260503.png",
}

MOCK_RESEARCH_RESULT = {
    "status": "success",
    "ticker": "005930",
    "summary": "삼성전자는 미국의 반도체 수출 규제 강화로 단기 리스크가 높은 상황입니다.",
    "risk_events": [
        {
            "type": "규제",
            "description": "미국 반도체 수출 규제 강화",
            "severity": "HIGH",
            "detected_at": "2026-05-02",
        }
    ],
    "sources": [
        {
            "title": "삼성전자, 미국 수출 규제 직격탄",
            "url": "https://n.news.naver.com/article/test",
            "published_at": "2026-05-02",
            "relevance_score": 0.92,
        }
    ],
    "reasoning_trace": [
        "1. '삼성전자 리스크' 키워드로 초기 검색",
        "2. 규제 관련 뉴스 5건 확보 → 분석 진행",
    ],
    "self_correction_count": 1,
}

MOCK_BACKTEST_RESULT = {
    "status": "success",
    "strategy": "drl",
    "metrics": {
        "total_return": 0.534,
        "sharpe_ratio": 1.67,
        "sortino_ratio": 2.14,
        "calmar_ratio": 1.89,
        "max_drawdown": -0.143,
        "volatility": 0.118,
        "win_rate": 0.612,
    },
    "benchmark_comparison": {
        "kospi_return": 0.221,
        "sp500_return": 0.387,
        "strategy_alpha": 0.147,
    },
    "walk_forward_results": [
        {"period": "2021-01~2022-01", "return": 0.112, "sharpe": 1.43},
        {"period": "2022-01~2023-01", "return": -0.043, "sharpe": 0.61},
    ],
}
