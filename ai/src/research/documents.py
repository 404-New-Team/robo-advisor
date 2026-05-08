from __future__ import annotations

import hashlib
import html
import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class NewsDocument:
    id: str
    title: str
    text: str
    summary: str
    published: str
    source: str
    url: str
    provider: str = ""
    document_type: str = "news"

    def to_chroma_document(self) -> str:
        return f"{self.title}\n{self.summary or self.text}".strip()

    def to_metadata(self) -> dict[str, str]:
        data = asdict(self)
        data.pop("text", None)
        return {key: _as_metadata_value(value) for key, value in data.items()}

    def to_article_dict(self) -> dict[str, str]:
        return asdict(self)


def normalize_article(article: dict[str, Any], provider: str = "") -> NewsDocument:
    title = clean_text(str(article.get("title") or "Untitled source"))
    raw_text = str(article.get("text") or article.get("body") or article.get("summary") or title)
    text = clean_text(raw_text)
    summary = clean_text(str(article.get("summary") or text))[:300]
    url = clean_text(str(article.get("url") or article.get("link") or ""))
    source = clean_text(str(article.get("source") or provider or "unknown"))
    published = clean_text(str(article.get("published") or article.get("published_at") or article.get("date") or ""))
    doc_id = clean_text(str(article.get("id") or "")) or stable_article_id(
        url=url,
        title=title,
        published=published,
        source=source,
    )

    return NewsDocument(
        id=doc_id,
        title=title,
        text=text,
        summary=summary,
        published=published,
        source=source,
        url=url,
        provider=provider,
    )


def normalize_articles(articles: list[dict[str, Any]], provider: str = "") -> list[NewsDocument]:
    normalized = []
    seen = set()
    for article in articles:
        doc = normalize_article(article, provider=provider)
        if doc.id in seen:
            continue
        seen.add(doc.id)
        normalized.append(doc)
    return normalized


def search_result_to_citation_item(
    document: str,
    metadata: dict[str, Any] | None,
    score: float,
    distance: float | None = None,
) -> dict[str, Any]:
    metadata = metadata or {}
    return {
        "text": document,
        "metadata": {
            "id": str(metadata.get("id") or stable_article_id("", document[:120], "", "")),
            "title": str(metadata.get("title") or "Untitled source"),
            "source": str(metadata.get("source") or "unknown"),
            "published": str(metadata.get("published") or ""),
            "url": str(metadata.get("url") or ""),
            "summary": str(metadata.get("summary") or clean_text(document)[:300]),
            "provider": str(metadata.get("provider") or ""),
            "document_type": str(metadata.get("document_type") or "news"),
        },
        "score": float(score),
        "distance": distance,
    }


def clean_text(value: str) -> str:
    unescaped = html.unescape(value)
    no_tags = re.sub(r"<[^>]+>", " ", unescaped)
    return re.sub(r"\s+", " ", no_tags).strip()


def stable_article_id(url: str, title: str, published: str, source: str) -> str:
    key = "|".join([url, title, published, source])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def _as_metadata_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)
