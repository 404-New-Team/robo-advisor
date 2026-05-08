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

        citations = [Citation(**item) for item in final_state.get("citations", [])]
        risk_tags = [RiskTag(**item) for item in final_state.get("risk_tags", [])]
        return ResearchReport(
            query=final_state["original_query"],
            answer=final_state.get("answer", ""),
            citations=citations,
            risk_tags=risk_tags,
            reasoning_trace=final_state.get("trace", []),
            retrieved_count=len(final_state.get("retrieved", [])),
            self_corrected=final_state.get("attempts", 0) > 0,
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

        workflow.set_entry_point("plan")
        workflow.add_edge("plan", "retrieve")
        workflow.add_edge("retrieve", "grade")
        workflow.add_conditional_edges(
            "grade",
            self._route_after_grade,
            {"rewrite": "rewrite", "analyze": "analyze"},
        )
        workflow.add_edge("rewrite", "retrieve")
        workflow.add_edge("analyze", END)
        return workflow.compile()

    def _run_without_langgraph(self, state: ResearchState) -> ResearchState:
        state = self._plan(state)
        while True:
            state = self._retrieve(state)
            state = self._grade(state)
            if self._route_after_grade(state) == "analyze":
                return self._analyze(state)
            state = self._rewrite(state)

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
        prompt = (
            "Write a concise Korean investment research report. "
            "Use bracket citations like [1], [2] tied to the supplied sources. "
            "Include: summary, key evidence, risk events, and investment view. "
            "Do not invent facts beyond the sources.\n\n"
            f"Question: {query}\n\nSources:\n{context}"
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

    def _generate_extractive_report(
        self,
        query: str,
        citations: list[Citation] | list[dict[str, Any]],
        risk_tags: list[RiskTag],
    ) -> str:
        normalized = [c if isinstance(c, Citation) else Citation(**c) for c in citations]
        if not normalized:
            return (
                f"질문: {query}\n\n"
                "검색된 근거 문서가 부족합니다. 뉴스/공시 데이터를 먼저 적재한 뒤 다시 검색해야 합니다."
            )

        evidence_lines = [
            f"[{idx}] {citation.title} - {citation.snippet}"
            for idx, citation in enumerate(normalized, start=1)
        ]
        risk_line = ", ".join(f"{tag.name}({tag.level:.2f})" for tag in risk_tags) or "뚜렷한 위험 태그 없음"
        return (
            f"질문: {query}\n\n"
            "요약: 검색된 금융 뉴스 근거를 기준으로 투자 리스크와 핵심 이벤트를 정리했습니다.\n\n"
            "주요 근거:\n"
            + "\n".join(evidence_lines)
            + "\n\n"
            f"탐지된 리스크 태그: {risk_line}\n\n"
            "투자 의견: 위 근거만으로는 확정적 매수/매도 판단보다 리스크 모니터링이 우선입니다. "
            "실적, 규제, 유동성 관련 후속 뉴스가 포트폴리오 비중 조정의 핵심 입력값입니다."
        )

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
