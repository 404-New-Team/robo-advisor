import httpx
from fastapi import HTTPException, status
from app.config import settings


class AIServiceError(Exception):
    def __init__(self, status_code: int, message: str, detail: str | None = None):
        self.status_code = status_code
        self.message = message
        self.detail = detail
        super().__init__(message)


def _get_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=settings.ai_service_url,
        timeout=settings.ai_timeout,
    )


async def call_optimize(payload: dict) -> dict:
    async with _get_client() as client:
        try:
            response = await client.post("/ai/optimize", json=payload)
        except httpx.TimeoutException:
            raise AIServiceError(504, "AI 서비스 응답 시간 초과", f"{settings.ai_timeout}초 초과")
        except httpx.RequestError as e:
            raise AIServiceError(503, "AI 서비스 연결 실패", str(e))

    if response.status_code != 200:
        raise AIServiceError(response.status_code, "AI optimize 오류", response.text)
    return response.json()


async def call_shap(payload: dict) -> dict:
    async with _get_client() as client:
        try:
            response = await client.post("/ai/shap", json=payload)
        except httpx.TimeoutException:
            raise AIServiceError(504, "AI 서비스 응답 시간 초과", f"{settings.ai_timeout}초 초과")
        except httpx.RequestError as e:
            raise AIServiceError(503, "AI 서비스 연결 실패", str(e))

    if response.status_code != 200:
        raise AIServiceError(response.status_code, "AI SHAP 오류", response.text)
    return response.json()


async def call_research(payload: dict) -> dict:
    async with _get_client() as client:
        try:
            response = await client.post("/ai/research", json=payload)
        except httpx.TimeoutException:
            raise AIServiceError(504, "AI 서비스 응답 시간 초과", f"{settings.ai_timeout}초 초과")
        except httpx.RequestError as e:
            raise AIServiceError(503, "AI 서비스 연결 실패", str(e))

    if response.status_code != 200:
        raise AIServiceError(response.status_code, "AI research 오류", response.text)
    return response.json()


async def call_backtest(payload: dict) -> dict:
    async with _get_client() as client:
        try:
            response = await client.post("/ai/backtest", json=payload)
        except httpx.TimeoutException:
            raise AIServiceError(504, "AI 서비스 응답 시간 초과", f"{settings.ai_timeout}초 초과")
        except httpx.RequestError as e:
            raise AIServiceError(503, "AI 서비스 연결 실패", str(e))

    if response.status_code != 200:
        raise AIServiceError(response.status_code, "AI backtest 오류", response.text)
    return response.json()


async def check_ai_health() -> dict:
    async with _get_client() as client:
        try:
            response = await client.get("/health")
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
    return {"rl_engine": False, "rag_agent": False, "shap_explainer": False}
