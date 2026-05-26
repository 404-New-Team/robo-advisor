from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional, TypedDict

from ..envs.risk_state import RiskTag

if TYPE_CHECKING:
    from .news_store import NewsStore

try:
    from langgraph.graph import END, StateGraph
except Exception:
    END = "__end__"
    StateGraph = None


RISK_KEYWORDS = {
    "regulatory_risk": [
        "regulation",
        "regulatory",
        "law",
        "policy",
        "probe",
        "lawsuit",
        "antitrust",
        "sec",
        "rate",
    ],
    "earnings_shock": [
        "earnings",
        "revenue",
        "profit",
        "loss",
        "guidance",
        "forecast",
        "miss",
        "beat",
    ],
    "geopolitical_risk": [
        "war",
        "conflict",
        "sanction",
        "tariff",
        "trade",
        "geopolitical",
        "export control",
    ],
    "market_stress": [
        "volatility",
        "selloff",
        "correction",
        "crash",
        "credit spread",
        "stress",
        "recession",
    ],
    "liquidity_risk": [
        "liquidity",
        "funding",
        "debt",
        "bankruptcy",
        "default",
        "cash flow",
    ],
}


class ResearchState(TypedDict, total=False):
    query: str
    original_query: str
    ticker: Optional[str]
    plan: list[str]
    search_queries: list[str]
    retrieved: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    relevance_score: float
    attempts: int
    needs_rewrite: bool
    answer: str
    risk_tags: list[dict[str, Any]]
    trace: list[str]
    # self-correction
    answer_verified: bool
    correction_count: int
    correction_reasons: list[str]


@dataclass
class Citation:
    id: str
    title: str
    source: str
    published: str = ""
    url: str = ""
    snippet: str = ""
    relevance_score: float = 0.0


@dataclass
class ResearchReport:
    query: str
    answer: str
    citations: list[Citation]
    risk_tags: list[RiskTag]
    reasoning_trace: list[str]
    retrieved_count: int
    self_corrected: bool
    correction_count: int
    correction_reasons: list[str]
    relevance_score: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["risk_tags"] = [asdict(tag) for tag in self.risk_tags]
        return data


@dataclass
class AgenticRAGConfig:
    n_results: int = 5
    min_documents: int = 2
    min_relevance_score: float = 0.08
    max_rewrites: int = 2
    max_corrections: int = 2
    min_answer_quality: float = 0.5
    llm_model: str = "claude-3-5-sonnet-latest"
    max_tokens: int = 1200


class AgenticRAGResearchAgent:
    """
    LangGraph based research workflow:
    plan -> retrieve -> grade -> rewrite if needed -> analyze.

    The public run_research() method returns plain dictionaries so a later
    FastAPI layer can wrap it without importing Streamlit or model internals.
    """

    def __init__(
        self,
        news_store: Optional["NewsStore"] = None,
        config: Optional[AgenticRAGConfig] = None,
        llm_generate: Optional[Callable[[str, list[dict[str, Any]], list[dict[str, Any]]], str]] = None,
    ):
        if news_store is None:
            from .news_store import NewsStore

            news_store = NewsStore()
        self.news_store = news_store
        self.config = config or AgenticRAGConfig()
        self.llm_generate = llm_generate
        self.graph = self._build_graph()

    def run_research(self, query: str, ticker: Optional[str] = None) -> dict[str, Any]:
        report = self.run(query=query, ticker=ticker)
        return report.to_dict()

    def run(self, query: str, ticker: Optional[str] = None) -> ResearchReport:
        if not query or not query.strip():
            raise ValueError("query must not be empty")

        state: ResearchState = {
            "query": query.strip(),
            "original_query": query.strip(),
            "ticker": ticker or self._extract_ticker(query),
            "attempts": 0,
            "trace": [],
        }

        if self.graph is not None:
            final_state = self.graph.invoke(state)
        else:
            final_state = self._run_without_langgraph(state)

        correction_count = int(final_state.get("correction_count", 0))
        citations = [Citation(**item) for item in final_state.get("citations", [])]
        risk_tags = [RiskTag(**item) for item in final_state.get("risk_tags", [])]
        return ResearchReport(
            query=final_state["original_query"],
            answer=final_state.get("answer", ""),
            citations=citations,
            risk_tags=risk_tags,
            reasoning_trace=final_state.get("trace", []),
            retrieved_count=len(final_state.get("retrieved", [])),
            self_corrected=correction_count > 0,
            correction_count=correction_count,
            correction_reasons=final_state.get("correction_reasons", []),
            relevance_score=round(float(final_state.get("relevance_score", 0.0)), 4),
        )

    # ------------------------------------------------------------------

    def _build_graph(self):
        if StateGraph is None:
            return None

        workflow = StateGraph(ResearchState)
        workflow.add_node("plan", self._plan)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("grade", self._grade)
        workflow.add_node("rewrite", self._rewrite)
        workflow.add_node("analyze", self._analyze)
        workflow.add_node("verify", self._verify)
        workflow.add_node("correct", self._correct)

        workflow.set_entry_point("plan")
        workflow.add_edge("plan", "retrieve")
        workflow.add_edge("retrieve", "grade")
        workflow.add_conditional_edges(
            "grade",
            self._route_after_grade,
            {"rewrite": "rewrite", "analyze": "analyze"},
        )
        workflow.add_edge("rewrite", "retrieve")
        workflow.add_edge("analyze", "verify")
        workflow.add_conditional_edges(
            "verify",
            self._route_after_verify,
            {"correct": "correct", END: END},
        )
        workflow.add_edge("correct", "verify")
        return workflow.compile()

    def _run_without_langgraph(self, state: ResearchState) -> ResearchState:
        state = self._plan(state)
        while True:
            state = self._retrieve(state)
            state = self._grade(state)
            if self._route_after_grade(state) == "analyze":
                state = self._analyze(state)
                break
            state = self._rewrite(state)

        while True:
            state = self._verify(state)
            if self._route_after_verify(state) == END:
                return state
            state = self._correct(state)

    def _plan(self, state: ResearchState) -> ResearchState:
        ticker = state.get("ticker")
        query = state["query"]
        plan = [
            "Identify the investment question and likely ticker/company focus.",
            "Retrieve recent financial news and disclosures from the vector store.",
            "Grade retrieval quality; rewrite and retry if evidence is weak.",
            "Synthesize an investment research view with source citations.",
            "Convert detected events into risk tags for the RL pipeline.",
        ]
        search_queries = [query]
        if ticker:
            search_queries.extend(
                [
                    f"{ticker} earnings guidance revenue risk",
                    f"{ticker} regulation lawsuit tariff market risk",
                ]
            )
        else:
            search_queries.append(f"{query} earnings regulation market risk")

        state["plan"] = plan
        state["search_queries"] = search_queries
        state.setdefault("trace", []).append(
            f"plan: created {len(search_queries)} retrieval queries for ticker={ticker or 'N/A'}"
        )
        return state

    def _retrieve(self, state: ResearchState) -> ResearchState:
        retrieved: list[dict[str, Any]] = []
        seen: set[str] = set()
        for search_query in state.get("search_queries", [state["query"]]):
            for item in self.news_store.search(search_query, n=self.config.n_results):
                metadata = item.get("metadata", {})
                key = metadata.get("url") or metadata.get("title") or item.get("text", "")[:120]
                if key in seen:
                    continue
                seen.add(key)
                scored = dict(item)
                scored["query"] = search_query
                scored["relevance_score"] = self._score_document(state["original_query"], item)
                retrieved.append(scored)

        retrieved.sort(key=lambda item: item.get("relevance_score", 0.0), reverse=True)
        state["retrieved"] = retrieved[: self.config.n_results * 2]
        state.setdefault("trace", []).append(
            f"retrieve: found {len(state['retrieved'])} unique documents"
        )
        return state

    def _grade(self, state: ResearchState) -> ResearchState:
        docs = state.get("retrieved", [])
        scores = [float(doc.get("relevance_score", 0.0)) for doc in docs]
        relevance_score = sum(scores[: self.config.n_results]) / max(min(len(scores), self.config.n_results), 1)
        needs_rewrite = (
            len(docs) < self.config.min_documents
            or relevance_score < self.config.min_relevance_score
        )
        state["relevance_score"] = relevance_score
        state["needs_rewrite"] = needs_rewrite
        state.setdefault("trace", []).append(
            f"grade: relevance={relevance_score:.3f}, needs_rewrite={needs_rewrite}"
        )
        return state

    def _rewrite(self, state: ResearchState) -> ResearchState:
        attempts = int(state.get("attempts", 0)) + 1
        ticker = state.get("ticker")
        base = state["original_query"]
        if ticker:
            rewritten = [
                f"{ticker} latest earnings shock analyst outlook",
                f"{ticker} regulatory geopolitical liquidity market stress",
                f"{ticker} financial news investment risk event",
            ]
        else:
            rewritten = [
                f"{base} financial news source citation",
                f"{base} earnings regulation liquidity volatility",
                f"{base} investment risk event market impact",
            ]

        state["attempts"] = attempts
        state["search_queries"] = rewritten
        state.setdefault("trace", []).append(
            f"rewrite: attempt {attempts}, expanded query set to {len(rewritten)} queries"
        )
        return state

    def _analyze(self, state: ResearchState) -> ResearchState:
        docs = state.get("retrieved", [])[: self.config.n_results]
        citations = [self._to_citation(doc, idx) for idx, doc in enumerate(docs, start=1)]
        risk_tags = self._infer_risk_tags(docs)

        if self.llm_generate is not None:
            answer = self.llm_generate(state["original_query"], docs, citations)
        elif os.environ.get("ANTHROPIC_API_KEY"):
            answer = self._generate_with_claude(state["original_query"], docs, citations, risk_tags)
        else:
            answer = self._generate_extractive_report(state["original_query"], citations, risk_tags)

        state["citations"] = citations
        state["risk_tags"] = [asdict(tag) for tag in risk_tags]
        state["answer"] = answer
        state.setdefault("trace", []).append(
            f"analyze: produced report with {len(citations)} citations and {len(risk_tags)} risk tags"
        )
        return state

    def _route_after_grade(self, state: ResearchState) -> str:
        if state.get("needs_rewrite") and int(state.get("attempts", 0)) < self.config.max_rewrites:
            return "rewrite"
        return "analyze"

    def _verify(self, state: ResearchState) -> ResearchState:
        """생성된 답변의 품질을 검증한다."""
        answer = state.get("answer", "")
        citations = state.get("citations", [])
        risk_tags = state.get("risk_tags", [])
        reasons: list[str] = []

        # 1. 답변 길이 부족
        if len(answer.strip()) < 80:
            reasons.append("answer_too_short")


        # 3. 리스크 태그가 탐지됐는데 답변에 언급 없음
        detected_risks = [t["name"] for t in risk_tags if t.get("level", 0) > 0.3]
        if detected_risks:
            risk_keywords = {"regulatory_risk": "규제", "earnings_shock": "실적", "geopolitical_risk": "지정학", "market_stress": "시장", "liquidity_risk": "유동성"}
            if not any(risk_keywords.get(r, r) in answer for r in detected_risks):
                reasons.append("risk_tags_not_reflected")

        # 4. Claude API로 품질 점수 평가 (ANTHROPIC_API_KEY 있을 때)
        if not reasons and os.environ.get("ANTHROPIC_API_KEY"):
            quality_score = self._score_answer_with_llm(answer, citations)
            if quality_score < self.config.min_answer_quality:
                reasons.append(f"low_quality_score:{quality_score:.2f}")
        elif not reasons:
            # API 없을 때: 핵심 투자 용어 포함 여부로 단순 검증
            investment_terms = ["리스크", "투자", "수익", "손실", "시장", "포트폴리오", "risk", "investment", "market"]
            if not any(term in answer.lower() for term in investment_terms):
                reasons.append("missing_investment_context")

        state["answer_verified"] = len(reasons) == 0
        state["correction_reasons"] = state.get("correction_reasons", []) + reasons
        state.setdefault("trace", []).append(
            f"verify: {'pass' if state['answer_verified'] else f'fail({reasons})'}"
        )
        return state

    def _correct(self, state: ResearchState) -> ResearchState:
        """검증 실패 원인을 바탕으로 답변을 재생성한다."""
        correction_count = int(state.get("correction_count", 0)) + 1
        reasons = state.get("correction_reasons", [])
        citations = state.get("citations", [])
        risk_tags_raw = state.get("risk_tags", [])
        risk_tags = [RiskTag(**t) for t in risk_tags_raw]

        correction_instruction = self._build_correction_instruction(reasons)

        if os.environ.get("ANTHROPIC_API_KEY"):
            docs = state.get("retrieved", [])[: self.config.n_results]
            corrected_answer = self._generate_corrected_with_claude(
                state["original_query"], docs, citations, risk_tags, correction_instruction
            )
        else:
            corrected_answer = self._generate_corrected_extractive(
                state["original_query"], citations, risk_tags, correction_instruction
            )

        state["answer"] = corrected_answer
        state["correction_count"] = correction_count
        state["answer_verified"] = False
        state.setdefault("trace", []).append(
            f"correct: attempt {correction_count}, reasons={reasons}"
        )
        return state

    def _route_after_verify(self, state: ResearchState) -> str:
        if not state.get("answer_verified") and int(state.get("correction_count", 0)) < self.config.max_corrections:
            return "correct"
        return END

    def _score_answer_with_llm(self, answer: str, citations: list[dict[str, Any]]) -> float:
        """Claude로 답변 품질을 0~1 점수로 평가한다."""
        try:
            import anthropic
            sources_summary = "; ".join(c.get("title", "") for c in citations[:3])
            prompt = (
                "Rate this investment research answer quality from 0.0 to 1.0. "
                "Criteria: factual grounding, citation usage, actionable insight, clarity. "
                "Reply with only a float number.\n\n"
                f"Sources: {sources_summary}\n\nAnswer: {answer[:800]}"
            )
            client = anthropic.Anthropic()
            response = client.messages.create(
                model=self.config.llm_model,
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
            return max(0.0, min(1.0, float(text.strip())))
        except Exception:
            return 1.0  # API 오류 시 검증 통과

    def _build_correction_instruction(self, reasons: list[str]) -> str:
        parts = []
        if "answer_too_short" in reasons:
            parts.append("답변을 더 상세하게 작성하세요 (최소 3문단).")
        if "risk_tags_not_reflected" in reasons:
            parts.append("탐지된 리스크 이벤트(규제, 실적, 지정학, 유동성 등)를 답변에 명시하세요.")
        if any("low_quality_score" in r for r in reasons):
            parts.append("투자 판단에 실질적으로 도움이 되는 구체적인 인사이트를 포함하세요.")
        if "missing_investment_context" in reasons:
            parts.append("투자 리스크, 시장 영향, 포트폴리오 관점의 내용을 반드시 포함하세요.")
        return " ".join(parts) if parts else "답변의 완성도와 명확성을 높이세요."

    def _generate_corrected_with_claude(
        self,
        query: str,
        docs: list[dict[str, Any]],
        citations: list[dict[str, Any]],
        risk_tags: list[RiskTag],
        instruction: str,
    ) -> str:
        try:
            import anthropic
        except Exception:
            return self._generate_corrected_extractive(query, citations, risk_tags, instruction)

        context = "\n\n".join(
            f"[{idx}] {doc.get('metadata', {}).get('title', 'Untitled')}\n{doc.get('text', '')[:1000]}"
            for idx, doc in enumerate(docs, start=1)
        )
        risk_line = ", ".join(
            f"{self._RISK_NAME_KO.get(t.name, t.name)}(수준={t.level:.2f})"
            for t in risk_tags if t.level > 0.2
        ) or "없음"
        prompt = (
            "아래 정보를 바탕으로 수정된 한국어 투자 의견을 작성하세요. "
            "섹션 제목 없이 의견 내용만 출력하세요.\n"
            f"수정 요구사항: {instruction}\n\n"
            "작성 기준:\n"
            "- 탐지된 리스크가 포트폴리오에 미치는 영향을 쉬운 한국어 문장으로 3문장 이상 기술\n"
            "- 뉴스 제목·출처명·[1] 같은 인용 마커 사용 금지\n"
            "- 리스크 유형별 의미와 투자자가 취해야 할 행동을 구체적으로 안내\n\n"
            f"탐지된 리스크 태그: {risk_line}\n\n"
            f"질문: {query}\n\n출처:\n{context}"
        )
        client = anthropic.Anthropic()
        try:
            response = client.messages.create(
                model=self.config.llm_model,
                max_tokens=self.config.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
        except Exception:
            return self._generate_corrected_extractive(query, citations, risk_tags, instruction)

    def _generate_corrected_extractive(
        self,
        query: str,
        citations: list[dict[str, Any]],
        risk_tags: list[RiskTag],
        instruction: str,
    ) -> str:
        _, opinion_text = self._build_summary_and_opinion(citations, risk_tags)
        return opinion_text

    def _to_citation(self, item: dict[str, Any], idx: int) -> dict[str, Any]:
        metadata = item.get("metadata", {})
        text = item.get("text", "")
        return {
            "id": str(metadata.get("id") or idx),
            "title": str(metadata.get("title") or "Untitled source"),
            "source": str(metadata.get("source") or "unknown"),
            "published": str(metadata.get("published") or ""),
            "url": str(metadata.get("url") or ""),
            "snippet": self._clean_snippet(text),
            "relevance_score": round(float(item.get("relevance_score", 0.0)), 4),
        }

    def _score_document(self, query: str, item: dict[str, Any]) -> float:
        text = f"{item.get('text', '')} {item.get('metadata', {}).get('title', '')}".lower()
        query_tokens = self._tokens(query)
        if not query_tokens:
            return float(item.get("score", 0.0) or 0.0)

        overlap = sum(1 for token in query_tokens if token in text) / len(query_tokens)
        vector_score = float(item.get("score", 0.0) or 0.0)
        return max(overlap, vector_score)

    def _infer_risk_tags(self, docs: list[dict[str, Any]]) -> list[RiskTag]:
        corpus = "\n".join(
            f"{doc.get('metadata', {}).get('title', '')}\n{doc.get('text', '')}" for doc in docs
        ).lower()
        tags: list[RiskTag] = []
        for name, keywords in RISK_KEYWORDS.items():
            hits = [keyword for keyword in keywords if keyword in corpus]
            if not hits:
                continue
            level = min(1.0, 0.25 + 0.15 * len(hits))
            confidence = min(1.0, 0.45 + 0.1 * len(hits))
            tags.append(
                RiskTag(
                    name=name,
                    level=level,
                    confidence=confidence,
                    source=", ".join(hits[:4]),
                )
            )
        return tags

    def _generate_with_claude(
        self,
        query: str,
        docs: list[dict[str, Any]],
        citations: list[dict[str, Any]],
        risk_tags: list[RiskTag],
    ) -> str:
        try:
            import anthropic
        except Exception:
            return self._generate_extractive_report(query, [Citation(**c) for c in citations], risk_tags)

        context = "\n\n".join(
            f"[{idx}] {doc.get('metadata', {}).get('title', 'Untitled')}\n{doc.get('text', '')[:1200]}"
            for idx, doc in enumerate(docs, start=1)
        )
        risk_line = ", ".join(
            f"{self._RISK_NAME_KO.get(t.name, t.name)}(수준={t.level:.2f})"
            for t in risk_tags if t.level > 0.2
        ) or "없음"
        prompt = (
            "아래 정보를 바탕으로 한국어 투자 의견을 작성하세요. "
            "섹션 제목 없이 의견 내용만 출력하세요.\n\n"
            "작성 기준:\n"
            "- 탐지된 리스크가 포트폴리오에 미치는 영향을 쉬운 한국어 문장으로 3문장 이상 기술\n"
            "- 뉴스 제목·출처명·[1] 같은 인용 마커 사용 금지\n"
            "- 리스크 유형별 의미와 투자자가 취해야 할 행동을 구체적으로 안내\n\n"
            f"탐지된 리스크 태그: {risk_line}\n"
            f"질문: {query}\n\n출처:\n{context}"
        )
        client = anthropic.Anthropic()
        try:
            response = client.messages.create(
                model=self.config.llm_model,
                max_tokens=self.config.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
        except Exception:
            return self._generate_extractive_report(query, [Citation(**c) for c in citations], risk_tags)

    _RISK_NAME_KO: dict[str, str] = {
        "regulatory_risk": "규제",
        "earnings_shock": "실적",
        "geopolitical_risk": "지정학",
        "market_stress": "시장 변동성",
        "liquidity_risk": "유동성",
    }

    _RISK_DETAIL_KO: dict[str, str] = {
        "regulatory_risk": "정책·법률 변화가 사업 수익성에 직접 영향을 줄 수 있습니다",
        "earnings_shock": "실적·매출 변동으로 단기 주가 변동성이 확대될 수 있습니다",
        "geopolitical_risk": "무역 제재·분쟁 등 지정학적 불확실성이 공급망과 수출 실적에 영향을 줄 수 있습니다",
        "market_stress": "전반적인 시장 변동성 확대 국면으로 포트폴리오 베타 노출을 점검해야 합니다",
        "liquidity_risk": "자금 조달 비용 상승 및 부채 구조 변화가 현금 흐름에 부담을 줄 수 있습니다",
    }

    def _build_summary_and_opinion(
        self,
        citations: list,
        risk_tags: list[RiskTag],
    ) -> tuple[str, str]:
        """(요약, 투자 의견) 문자열 쌍 생성 — 리스크 태그 기반 자연어, 인용 마커·출처 제목 없음."""
        active = [t for t in risk_tags if t.level > 0.2]

        if active:
            top = max(active, key=lambda t: t.level)
            top_ko = self._RISK_NAME_KO.get(top.name, top.name)
            detail = self._RISK_DETAIL_KO.get(top.name, "포트폴리오 영향을 면밀히 점검해야 합니다")
            other_tags = sorted(
                [t for t in active if t.name != top.name],
                key=lambda t: t.level,
                reverse=True,
            )
            other_str = (
                f" 이와 함께 {', '.join(self._RISK_NAME_KO.get(t.name, t.name) for t in other_tags)} "
                "리스크도 동시에 모니터링이 필요합니다."
                if other_tags else ""
            )

            level_desc = "높음" if top.level >= 0.7 else "중간" if top.level >= 0.4 else "낮음"

            summary = (
                f"{top_ko} 리스크가 탐지되었습니다(수준: {level_desc}). "
                "리스크 완화 시그널이 확인되기 전까지 해당 자산의 비중 축소를 권고합니다."
            )
            opinion = (
                f"{detail}. "
                f"현재 {top_ko} 리스크 수준은 {top.level:.2f}로 {level_desc}으로 평가되며, "
                "이는 단기적으로 포트폴리오 변동성을 확대시킬 수 있는 요인입니다. "
                "실적 개선·규제 명확화·시장 안정 등의 완화 시그널이 나타나기 전까지 "
                f"해당 자산의 비중을 현 수준 이하로 유지할 것을 권고합니다.{other_str} "
                "향후 관련 동향을 지속 점검하며 비중을 점진적으로 재조정하는 전략을 권고합니다."
            )
        else:
            summary = (
                "현재 포트폴리오에서 뚜렷한 고위험 신호가 감지되지 않았습니다. "
                "현 비중을 유지하는 기조를 권고합니다."
            )
            opinion = (
                "분석 결과 즉각적인 대응이 필요한 고위험 이벤트는 관측되지 않았습니다. "
                "다만 시장 환경은 언제든 변화할 수 있으므로, "
                "실적·규제·유동성 관련 동향을 지속 모니터링하며 현 비중을 유지하는 기조를 권고합니다."
            )
        return summary, opinion

    def _generate_extractive_report(
        self,
        _query: str,
        citations: list[Citation] | list[dict[str, Any]],
        risk_tags: list[RiskTag],
    ) -> str:
        normalized = [c if isinstance(c, Citation) else Citation(**c) for c in citations]
        if not normalized:
            return "검색된 근거 문서가 부족합니다. 뉴스/공시 데이터를 먼저 적재한 뒤 다시 검색해야 합니다."

        _, opinion_text = self._build_summary_and_opinion(normalized, risk_tags)
        return opinion_text

    @staticmethod
    def _extract_ticker(query: str) -> Optional[str]:
        match = re.search(r"\b[A-Z]{1,5}\b", query)
        return match.group(0) if match else None

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {token for token in re.findall(r"[a-zA-Z0-9가-힣]{2,}", text.lower())}

    @staticmethod
    def _clean_snippet(text: str, max_length: int = 240) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if len(cleaned) <= max_length:
            return cleaned
        return cleaned[: max_length - 3].rstrip() + "..."
