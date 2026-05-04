from ..envs.risk_state import RISK_TAG_NAMES, RiskTag

__all__ = ["RISK_TAG_NAMES", "RiskTag", "RISK_DETECTION_TOOL"]

# Claude tool_use 스키마 — 구조화된 출력으로 파싱 실패 방지
RISK_DETECTION_TOOL = {
    "name": "report_risk_events",
    "description": "뉴스 텍스트에서 탐지된 금융 리스크 이벤트를 구조화된 형식으로 보고합니다.",
    "input_schema": {
        "type": "object",
        "properties": {
            "risk_tags": {
                "type": "array",
                "description": "탐지된 리스크 태그 목록 (탐지되지 않은 리스크는 level=0)",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "enum": RISK_TAG_NAMES,
                        },
                        "level": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "리스크 강도 (0=없음, 1=극도로 높음)",
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "탐지 신뢰도",
                        },
                        "source": {
                            "type": "string",
                            "description": "리스크 근거 한 줄 요약",
                        },
                    },
                    "required": ["name", "level", "confidence", "source"],
                },
            }
        },
        "required": ["risk_tags"],
    },
}
