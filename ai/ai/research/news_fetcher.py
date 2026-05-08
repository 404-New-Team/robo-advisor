"""
RSS 피드 기반 금융 뉴스 수집 모듈.
"""

import feedparser

RSS_FEEDS = {
    "yahoo_finance":  "https://finance.yahoo.com/news/rssindex",
    "marketwatch":    "https://www.marketwatch.com/rss/topstories",
    "reuters_biz":    "https://feeds.reuters.com/reuters/businessNews",
}


def fetch_feed(url: str, max_articles: int = 20) -> list[dict]:
    feed = feedparser.parse(url)
    articles = []
    for entry in feed.entries[:max_articles]:
        title = entry.get("title", "")
        body  = entry.get("summary", "")
        if not title and not body:
            continue
        articles.append({
            "id":        entry.get("id") or entry.get("link", ""),
            "title":     title,
            "text":      f"{title}\n{body}".strip(),
            "summary":   body[:300],
            "published": entry.get("published", ""),
            "source":    feed.feed.get("title", url),
            "url":       entry.get("link", ""),
        })
    return articles


def fetch_all(max_per_feed: int = 10) -> list[dict]:
    all_articles = []
    for name, url in RSS_FEEDS.items():
        try:
            articles = fetch_feed(url, max_per_feed)
            all_articles.extend(articles)
            print(f"  [{name}] {len(articles)}건 수집")
        except Exception as e:
            print(f"  [{name}] 수집 실패: {e}")
    return all_articles
