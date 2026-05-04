"""
ChromaDB 기반 뉴스 임베딩 저장소.

뉴스를 임베딩해서 저장하고, 유사 과거 사례를 검색한다.
RiskDetector가 컨텍스트로 활용한다.
"""

import os
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

PERSIST_DIR = str(Path(__file__).parent.parent / ".cache" / "chromadb")


class NewsStore:
    def __init__(self, persist_dir: str = PERSIST_DIR):
        host = os.environ.get("CHROMA_HOST")
        port = int(os.environ.get("CHROMA_PORT", 8000))

        if host:
            # 팀 공유 서버 모드: CHROMA_HOST 환경변수 설정 시 사용
            self.client = chromadb.HttpClient(host=host, port=port)
        else:
            # 로컬 모드 (기본값)
            self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="financial_news",
            embedding_function=DefaultEmbeddingFunction(),
        )

    def add(self, articles: list[dict]) -> int:
        """기사 임베딩 후 저장. 이미 존재하는 ID는 건너뜀."""
        existing = set(self.collection.get(include=[])["ids"])
        new = [a for a in articles if a["id"] not in existing]
        if not new:
            return 0
        self.collection.add(
            ids=[a["id"] for a in new],
            documents=[a["text"] for a in new],
            metadatas=[
                {
                    "title":     a["title"],
                    "source":    a["source"],
                    "published": a["published"],
                }
                for a in new
            ],
        )
        return len(new)

    def search(self, query: str, n: int = 5) -> list[dict]:
        """쿼리와 의미적으로 유사한 과거 기사 반환."""
        total = self.collection.count()
        if total == 0:
            return []
        results = self.collection.query(
            query_texts=[query],
            n_results=min(n, total),
        )
        return [
            {"text": doc, "metadata": meta}
            for doc, meta in zip(
                results["documents"][0],
                results["metadatas"][0],
            )
        ]

    def search_by_risk(self, n_per_type: int = 2) -> list[str]:
        """
        리스크 유형별 키워드로 검색해 관련 기사를 우선 선별.
        단순히 앞에서 N건 자르는 대신 실제 리스크 관련 기사를 추출한다.
        """
        RISK_QUERIES = {
            "regulatory_risk":  "regulation policy government law central bank rate",
            "earnings_shock":   "earnings revenue profit loss guidance forecast miss beat",
            "geopolitical_risk": "war conflict sanctions trade dispute tariff geopolitical",
            "market_stress":    "volatility crash correction selloff credit spread market stress",
            "liquidity_risk":   "liquidity funding debt bankruptcy bank run capital",
        }
        seen, selected = set(), []
        for query in RISK_QUERIES.values():
            for item in self.search(query, n=n_per_type):
                title = item["metadata"].get("title", "")
                if title not in seen:
                    seen.add(title)
                    selected.append(item["text"])
        return selected

    def count(self) -> int:
        return self.collection.count()
