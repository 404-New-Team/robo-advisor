from __future__ import annotations

import os
from typing import Any

try:
    import requests
except ImportError:
    requests = None


API_BASE_URL = (os.getenv("API_BASE_URL") or os.getenv("ROBBY_API_BASE_URL", "http://localhost:8000")).rstrip("/")
REQUEST_TIMEOUT = float(os.getenv("ROBBY_API_TIMEOUT", "8"))
USE_MOCK = os.getenv("ROBBY_USE_MOCK", "false").lower() == "true"


class ApiClientError(RuntimeError):
    pass


def _error_message(response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text or response.reason

    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, dict):
        message = detail.get("message") or response.reason
        extra = detail.get("detail")
        return f"{message}: {extra}" if extra else message
    if isinstance(detail, str):
        return detail
    if isinstance(body, dict):
        return body.get("message") or response.reason
    return response.reason


def _request(method: str, path: str, **kwargs: Any) -> dict:
    if requests is None:
        raise ApiClientError("requests 패키지가 설치되어 있지 않습니다.")
    try:
        response = requests.request(method, f"{API_BASE_URL}{path}", timeout=REQUEST_TIMEOUT, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as error:
        response = error.response
        message = _error_message(response) if response is not None else str(error)
        status = response.status_code if response is not None else "HTTP"
        raise ApiClientError(f"{method} {path} 실패 ({status}): {message}") from error
    except requests.exceptions.JSONDecodeError as error:
        raise ApiClientError(f"{method} {path} 응답이 JSON 형식이 아닙니다.") from error
    except requests.exceptions.RequestException as error:
        raise ApiClientError(f"{method} {path} 연결 실패: {error}") from error


def health() -> dict:
    if USE_MOCK:
        return {"status": "ok", "model_loaded": True, "mode": "mock"}
    data = _request("GET", "/health")
    models = data.get("models", {})
    return {
        **data,
        "mode": "backend",
        "model_loaded": all(models.values()) if models else False,
    }


def optimize_portfolio(risk_level: str, tickers: list[str], excluded: list[str] | None = None) -> dict:
    if USE_MOCK:
        from mock_data import get_optimize_response

        return get_optimize_response(risk_level=risk_level, tickers=tickers, excluded=excluded)
    excluded_set = set(excluded or [])
    filtered_tickers = [ticker for ticker in tickers if ticker not in excluded_set] or tickers
    payload = {"risk_level": risk_level, "tickers": filtered_tickers}
    return _request("POST", "/optimize", json=payload)


def research(query: str, ticker: str | None = None, max_results: int = 5) -> dict:
    if USE_MOCK:
        from mock_data import get_research_response

        return get_research_response(ticker=ticker, query=query, max_results=max_results)
    payload = {"query": query, "ticker": ticker, "max_results": max_results}
    return _request("POST", "/research", json=payload)


def explain(tickers: list[str], target_asset: str) -> dict:
    if USE_MOCK:
        from mock_data import get_explain_response

        return get_explain_response(target_asset=target_asset)
    payload = {"tickers": tickers, "target_asset": target_asset}
    return _request("POST", "/explain", json=payload)


def backtest(tickers: list[str], strategy: str) -> dict:
    if USE_MOCK:
        from mock_data import get_backtest_response

        return get_backtest_response(strategy=strategy)
    params = {"tickers": ",".join(tickers), "strategy": strategy}
    return _request("GET", "/backtest", params=params)


def strategy_backtests(tickers: list[str], strategies: list[str]) -> list[dict]:
    return [backtest(tickers, strategy) for strategy in strategies]
