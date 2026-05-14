from __future__ import annotations

import os
from typing import Any

from mock_data import get_backtest_response, get_explain_response, get_optimize_response, get_research_response

try:
    import requests
except ImportError:
    requests = None


API_BASE_URL = os.getenv("ROBBY_API_BASE_URL", "http://localhost:8000")
USE_MOCK = os.getenv("ROBBY_USE_MOCK", "true").lower() != "false"


def _request(method: str, path: str, **kwargs: Any) -> dict:
    if requests is None:
        raise RuntimeError("requests 패키지가 설치되어 있지 않습니다.")
    response = requests.request(method, f"{API_BASE_URL}{path}", timeout=5, **kwargs)
    response.raise_for_status()
    return response.json()


def health() -> dict:
    if USE_MOCK:
        return {"status": "ok", "model_loaded": True, "mode": "mock"}
    return _request("GET", "/health")


def optimize_portfolio(risk_level: str, tickers: list[str], excluded: list[str] | None = None) -> dict:
    if USE_MOCK:
        return get_optimize_response(risk_level=risk_level, tickers=tickers, excluded=excluded)
    payload = {"risk_level": risk_level, "tickers": tickers}
    return _request("POST", "/optimize", json=payload)


def research(query: str, ticker: str | None = None, max_results: int = 5) -> dict:
    if USE_MOCK:
        return get_research_response(ticker=ticker, query=query, max_results=max_results)
    payload = {"query": query, "ticker": ticker, "max_results": max_results}
    return _request("POST", "/research", json=payload)


def explain(tickers: list[str], target_asset: str) -> dict:
    if USE_MOCK:
        return get_explain_response(target_asset=target_asset)
    payload = {"tickers": tickers, "target_asset": target_asset}
    return _request("POST", "/explain", json=payload)


def backtest(tickers: list[str], strategy: str) -> dict:
    if USE_MOCK:
        return get_backtest_response(strategy=strategy)
    params = {"tickers": ",".join(tickers), "strategy": strategy}
    return _request("GET", "/backtest", params=params)
