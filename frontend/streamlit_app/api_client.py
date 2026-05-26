from __future__ import annotations

import os
from typing import Any

try:
    import requests
except ImportError:
    requests = None


API_BASE_URL = (os.getenv("API_BASE_URL") or os.getenv("ROBBY_API_BASE_URL", "http://localhost:8000")).rstrip("/")
REQUEST_TIMEOUT = float(os.getenv("ROBBY_API_TIMEOUT", "30"))
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


def _request(method: str, path: str, token: str | None = None, **kwargs: Any) -> dict:
    if requests is None:
        raise ApiClientError("requests 패키지가 설치되어 있지 않습니다.")
    headers = kwargs.pop("headers", {}) or {}
    if token:
        headers = {**headers, "Authorization": f"Bearer {token}"}
    if headers:
        kwargs["headers"] = headers
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


def register(email: str, username: str, password: str) -> dict:
    if USE_MOCK:
        return {"id": 1, "email": email, "username": username}
    payload = {"email": email, "username": username, "password": password}
    return _request("POST", "/auth/register", json=payload)


def login(email: str, password: str) -> dict:
    if USE_MOCK:
        return {"access_token": "mock-token", "token_type": "bearer"}
    payload = {"email": email, "password": password}
    return _request("POST", "/auth/login", json=payload)


def me(token: str) -> dict:
    if USE_MOCK:
        return {"id": 1, "email": "demo@example.com", "username": "demo"}
    return _request("GET", "/auth/me", token=token)


def get_user_tickers(token: str) -> dict:
    if USE_MOCK:
        from reference_data import get_default_tickers

        return {"tickers": get_default_tickers()}
    return _request("GET", "/users/tickers", token=token)


def add_user_ticker(token: str, ticker: str) -> dict:
    if USE_MOCK:
        return {"tickers": [ticker]}
    return _request("POST", "/users/tickers", token=token, json={"ticker": ticker})


def delete_user_ticker(token: str, ticker: str) -> dict:
    if USE_MOCK:
        return {"tickers": []}
    return _request("DELETE", f"/users/tickers/{ticker}", token=token)


def optimize_portfolio(risk_level: str, tickers: list[str], excluded: list[str] | None = None, token: str | None = None) -> dict:
    if USE_MOCK:
        from mock_data import get_optimize_response

        return get_optimize_response(risk_level=risk_level, tickers=tickers, excluded=excluded)
    excluded_set = set(excluded or [])
    filtered_tickers = [ticker for ticker in tickers if ticker not in excluded_set] or tickers
    payload = {"risk_level": risk_level, "tickers": filtered_tickers}
    return _request("POST", "/optimize", token=token, json=payload)


def research(
    tickers: list[str],
    max_results: int = 5,
    token: str | None = None,
    portfolio_context: dict[str, Any] | None = None,
) -> dict:
    research_tickers = [ticker for ticker in tickers if ticker]
    if USE_MOCK:
        from mock_data import get_research_response

        return get_research_response(
            tickers=research_tickers,
            max_results=max_results,
            portfolio_context=portfolio_context,
        )
    payload = {"tickers": research_tickers, "max_results": max_results}
    if portfolio_context is not None:
        payload["portfolio_context"] = portfolio_context
    return _request("POST", "/research", token=token, json=payload)


def explain(tickers: list[str], target_asset: str, token: str | None = None) -> dict:
    if USE_MOCK:
        from mock_data import get_explain_response

        return get_explain_response(target_asset=target_asset)
    payload = {"tickers": tickers, "target_asset": target_asset}
    return _request("POST", "/explain", token=token, json=payload)


def backtest(tickers: list[str], strategy: str, token: str | None = None) -> dict:
    if USE_MOCK:
        from mock_data import get_backtest_response

        return get_backtest_response(strategy=strategy)
    params = {"tickers": tickers, "strategy": strategy}
    return _request("GET", "/backtest", token=token, params=params)


def strategy_backtests(tickers: list[str], strategies: list[str], token: str | None = None) -> list[dict]:
    return [backtest(tickers, strategy, token=token) for strategy in strategies]


def allocation(weights: dict, total_amount: float, token: str | None = None) -> dict:
    if USE_MOCK:
        items = []
        for ticker, weight in weights.items():
            target = total_amount * weight
            price = 50000
            shares = int(target // price)
            items.append({
                "ticker": ticker,
                "weight": weight,
                "current_price": price,
                "target_amount": round(target, 2),
                "integer_shares": shares,
                "fractional_shares": round(target / price, 4),
                "actual_amount": float(shares * price),
                "leftover": round(target - shares * price, 2),
            })
        total_invested = sum(i["actual_amount"] for i in items)
        return {
            "status": "success",
            "total_amount": total_amount,
            "total_invested": total_invested,
            "total_leftover": round(total_amount - total_invested, 2),
            "items": items,
        }
    payload = {"weights": weights, "total_amount": total_amount}
    return _request("POST", "/allocation", token=token, json=payload)
