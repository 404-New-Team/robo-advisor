from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
import os

from app.config import settings
from app.routers import health, optimize, explain, research, backtest, auth, users, allocation


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from app.database import init_db
        init_db()
    except Exception as e:
        print(f"[WARN] DB init skipped: {e}")
    os.makedirs("static/shap", exist_ok=True)
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "에이전틱 RAG 및 강화학습 기반 통합 자율 로보어드바이저 백엔드 API.\n\n"
        "포트폴리오 최적화(DRL/MVO/Equal-Weight), SHAP 의사결정 해석, "
        "RAG 투자 리서치, Walk-Forward 백테스트를 제공합니다."
    ),
    lifespan=lifespan,
)

_static_dir = "static"
os.makedirs(_static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=_static_dir), name="static")

app.include_router(health.router)
app.include_router(optimize.router)
app.include_router(explain.router)
app.include_router(research.router)
app.include_router(backtest.router)
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(allocation.router)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    detail = exc.detail
    if isinstance(detail, dict):
        message = detail.get("message", "요청 처리 중 오류가 발생했습니다.")
        detail_str = detail.get("detail")
    else:
        message = str(detail)
        detail_str = None
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "code": exc.status_code, "message": message, "detail": detail_str},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    first = errors[0] if errors else {}
    loc = " → ".join(str(l) for l in first.get("loc", []) if l != "body")
    message = f"{loc}: {first.get('msg', '입력값 유효성 오류')}" if loc else first.get("msg", "입력값 유효성 오류")
    return JSONResponse(
        status_code=422,
        content={"status": "error", "code": 422, "message": message, "detail": None},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"status": "error", "code": 500, "message": "서버 내부 오류가 발생했습니다.", "detail": str(exc)},
    )