from src.research.documents import normalize_article
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


class FakeCollection:
    def __init__(self):
        self.ids = []
        self.documents = []
        self.metadatas = []

    def get(self, include=None):
        return {"ids": self.ids}

    def add(self, ids, documents, metadatas):
        self.ids.extend(ids)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)

    def count(self):
        return len(self.ids)

    def query(self, query_texts, n_results, include=None):
        return {
            "documents": [self.documents[:n_results]],
            "metadatas": [self.metadatas[:n_results]],
            "distances": [[0.25 for _ in self.documents[:n_results]]],
        }


def test_normalize_article_preserves_citation_metadata():
    doc = normalize_article(
        {
            "title": " <b>AAPL earnings beat</b> ",
            "summary": "<p>Revenue rose despite market volatility.</p>",
            "published": "2026-05-01",
            "source": "Example Finance",
            "url": "https://example.com/aapl",
        },
        provider="example",
    )

    assert doc.id
    assert doc.title == "AAPL earnings beat"
    assert doc.summary == "Revenue rose despite market volatility."
    assert doc.url == "https://example.com/aapl"
    assert doc.provider == "example"


def test_news_store_add_and_search_returns_citation_ready_items():
    store = NewsStore.__new__(NewsStore)
    store.collection = FakeCollection()

    added = store.add(
        [
            {
                "title": "AAPL earnings beat",
                "text": "AAPL earnings beat while regulatory pressure remains.",
                "published": "2026-05-01",
                "source": "Example Finance",
                "url": "https://example.com/aapl",
            }
        ]
    )
    results = store.search("AAPL regulatory risk", n=1)

    assert added == 1
    assert results[0]["metadata"]["title"] == "AAPL earnings beat"
    assert results[0]["metadata"]["url"] == "https://example.com/aapl"
    assert results[0]["metadata"]["document_type"] == "news"
    assert results[0]["score"] > 0


def test_news_store_with_local_chromadb(tmp_path):
    store = NewsStore(
        persist_dir=str(tmp_path / "chroma"),
        collection_name="financial_news_test",
        embedding_function=DeterministicEmbeddingFunction(),
    )

    added = store.add(
        [
            {
                "title": "AAPL regulatory pressure",
                "text": "AAPL faces regulatory pressure after earnings guidance.",
                "published": "2026-05-02",
                "source": "Example Finance",
                "url": "https://example.com/aapl-regulatory",
            },
            {
                "title": "MSFT market volatility",
                "text": "MSFT remains exposed to market volatility and liquidity risk.",
                "published": "2026-05-03",
                "source": "Example Finance",
                "url": "https://example.com/msft-market",
            },
        ]
    )
    results = store.search("AAPL regulation earnings", n=2)
    citations = store.citations("AAPL regulation earnings", n=2)

    assert added == 2
    assert store.count() == 2
    assert results
    assert results[0]["metadata"]["url"]
    assert results[0]["metadata"]["document_type"] == "news"
    assert citations[0]["score"] > 0
