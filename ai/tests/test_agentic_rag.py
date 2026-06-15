from src.research.agentic_rag import AgenticRAGConfig, AgenticRAGResearchAgent
from src.research.news_store import NewsStore


class DeterministicEmbeddingFunction:
    def name(self):
        return "default"

    def __call__(self, input):
        return self._embed(input)

    def embed_query(self, input):
        return self._embed(input)

    def embed_documents(self, input):
        return self._embed(input)

    def _embed(self, input):
        vectors = []
        for text in input:
            length = float(len(text) % 17)
            checksum = float(sum(ord(ch) for ch in text) % 31)
            vectors.append([length, checksum, 1.0])
        return vectors


def test_agentic_rag_uses_local_chromadb_citations(tmp_path):
    store = NewsStore(
        persist_dir=str(tmp_path / "chroma"),
        collection_name="agentic_rag_news_test",
        embedding_function=DeterministicEmbeddingFunction(),
    )
    store.add(
        [
            {
                "title": "AAPL earnings beat with regulatory pressure",
                "text": "AAPL reported an earnings beat but faces regulatory pressure and market volatility.",
                "published": "2026-05-01",
                "source": "Example Finance",
                "url": "https://example.com/aapl-earnings",
            },
            {
                "title": "AAPL liquidity risk remains contained",
                "text": "Analysts said AAPL liquidity risk is contained despite wider market stress.",
                "published": "2026-05-02",
                "source": "Example Markets",
                "url": "https://example.com/aapl-liquidity",
            },
        ]
    )

    agent = AgenticRAGResearchAgent(
        news_store=store,
        config=AgenticRAGConfig(n_results=3),
        llm_generate=lambda query, docs, citations: "Mock report with citations [1].",
    )
    result = agent.run_research("AAPL investment risk")

    assert result["answer"] == "Mock report with citations [1]."
    assert result["citations"]
    assert result["citations"][0]["url"].startswith("https://example.com/")
    assert result["risk_tags"]
    assert result["reasoning_trace"]


class CapturingNewsStore:
    def __init__(self):
        self.queries = []

    def search(self, query, n=5):
        self.queries.append(query)
        ticker = "005930" if "005930" in query else "000660"
        return [
            {
                "text": f"{query} earnings regulation market risk portfolio weight evidence",
                "metadata": {
                    "id": query,
                    "title": f"{ticker} portfolio risk news",
                    "source": "Example Finance",
                    "published": "2026-05-01",
                    "url": f"https://example.com/{ticker}",
                },
                "score": 0.9,
            }
        ]


def test_agentic_rag_expands_portfolio_context_queries(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    store = CapturingNewsStore()
    agent = AgenticRAGResearchAgent(news_store=store, config=AgenticRAGConfig(n_results=3))
    context = {
        "risk_level": "moderate",
        "investment_amount": 30000000,
        "selected_tickers": ["005930", "000660"],
        "excluded_tickers": [],
        "active_tickers": ["005930", "000660"],
        "weights": {"005930": 0.6, "000660": 0.4},
        "ticker_names": {"005930": "삼성전자", "000660": "SK하이닉스"},
    }

    result = agent.run_research("현재 포트폴리오 리스크 요약", portfolio_context=context)

    assert any("005930" in query for query in store.queries)
    assert any("000660" in query for query in store.queries)
    assert "포트폴리오 구성/비중 기준" in result["answer"]
    assert "삼성전자 005930 60.0%" in result["answer"]
