from datetime import datetime, timezone
from fastapi import APIRouter
from app.services.ai_client import check_ai_health
from app.config import settings

router = APIRouter(tags=["Health"])


@router.get("/health", summary="서버 상태 및 모델 로드 여부 확인")
async def health_check():
    models = await check_ai_health()
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models": {
            "rl_engine": models.get("rl_engine", False),
            "rag_agent": models.get("rag_agent", False),
            "shap_explainer": models.get("shap_explainer", False),
        },
        "version": settings.app_version,
    }
