"""CLI for ingesting RSS financial news into ChromaDB."""

from __future__ import annotations

import argparse
import os

from .news_fetcher import fetch_all
from .news_store import NewsStore


def ingest_news(max_per_feed: int = 20) -> dict[str, int]:
    """Fetch RSS articles and add new documents to the configured NewsStore."""
    store = NewsStore()
    before = store.count()
    articles = fetch_all(max_per_feed=max_per_feed)
    added = store.add(articles)
    after = store.count()
    return {
        "before": before,
        "fetched": len(articles),
        "added": added,
        "after": after,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest RSS financial news into ChromaDB.")
    parser.add_argument("--max-per-feed", type=int, default=int(os.getenv("NEWS_MAX_PER_FEED", "20")))
    args = parser.parse_args()

    result = ingest_news(max_per_feed=args.max_per_feed)
    print(
        "news ingest: "
        f"before={result['before']} "
        f"fetched={result['fetched']} "
        f"added={result['added']} "
        f"after={result['after']}"
    )


if __name__ == "__main__":
    main()
