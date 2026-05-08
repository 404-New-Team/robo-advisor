from ai.research.agentic_rag import AgenticRAGConfig, AgenticRAGResearchAgent
from ai.research.news_store import NewsStore


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
