"""
Claude API를 활용한 뉴스 → 리스크 태그 변환 에이전트.

tool_use로 구조화된 출력을 강제하여 파싱 실패를 방지한다.
"""

import anthropic
from ..envs.risk_state import RiskTag, RISK_TAG_NAMES
from .risk_tags import RISK_DETECTION_TOOL

SYSTEM_PROMPT = """당신은 금융 시장 리스크 분석 전문가입니다.
뉴스 텍스트를 읽고 다음 5가지 리스크 유형을 평가하세요:

- regulatory_risk: 규제 변경, 정부 정책, 법률 리스크
- earnings_shock: 실적 쇼크, 어닝 서프라이즈/미스, 가이던스 하향
- geopolitical_risk: 지정학적 리스크, 전쟁, 무역 분쟁, 제재
- market_stress: 시장 변동성 급등, 유동성 위기, 신용 스프레드 확대
- liquidity_risk: 유동성 부족, 거래 중단, 뱅크런

탐지되지 않은 리스크 유형은 level=0.0으로 설정하세요.
level은 뉴스 내용의 심각도, confidence는 뉴스 신뢰도를 반영합니다."""


class RiskDetector:
    def __init__(self, model: str = "claude-opus-4-7", max_tokens: int = 1024):
        self.client = anthropic.Anthropic()  # ANTHROPIC_API_KEY 환경변수 자동 사용
        self.model = model
        self.max_tokens = max_tokens

    def detect(self, news_texts: list, context: list = None) -> list:
        """
        뉴스 텍스트 목록 분석 → RiskTag 목록 반환.
        context: NewsStore.search()로 가져온 유사 과거 기사 목록 (선택).
        빈 입력이면 즉시 빈 리스트 반환.
        """
        if not news_texts:
            return []

        combined = "\n\n---\n\n".join(
            f"[뉴스 {i+1}]\n{text}" for i, text in enumerate(news_texts)
        )

        context_section = ""
        if context:
            past = "\n".join(
                f"- {c['metadata'].get('title', c['text'][:80])}"
                for c in context[:3]
            )
            context_section = f"\n\n[유사 과거 사례 참고]\n{past}"

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            tools=[RISK_DETECTION_TOOL],
            tool_choice={"type": "tool", "name": "report_risk_events"},
            messages=[{"role": "user", "content": f"다음 뉴스를 분석하세요:\n\n{combined}{context_section}"}],
        )

        return self._parse_response(response)

    def _parse_response(self, response) -> list:
        for block in response.content:
            if block.type == "tool_use" and block.name == "report_risk_events":
                return [
                    RiskTag(
                        name=t["name"],
                        level=float(t["level"]),
                        confidence=float(t["confidence"]),
                        source=t.get("source", ""),
                    )
                    for t in block.input.get("risk_tags", [])
                    if t["name"] in RISK_TAG_NAMES
                ]
        return []
