from ai.research.documents import normalize_article
from ai.research.news_store import NewsStore


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
