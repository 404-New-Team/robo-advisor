from ai.data.market_data import fetch_prices
from ai.envs.portfolio_env import PortfolioEnv
from ai.research.news_fetcher import fetch_all
from ai.research.news_store import NewsStore
from ai.research.risk_detector import RiskDetector

def main():
    # 1. 뉴스 수집
    print("=== 1. 뉴스 수집 ===")
    articles = fetch_all(max_per_feed=10)
    print(f"  총 {len(articles)}건 수집\n")

    # 2. ChromaDB 저장
    print("=== 2. ChromaDB 저장 ===")
    store = NewsStore()
    added = store.add(articles)
    print(f"  신규 저장: {added}건 | 누적: {store.count()}건\n")

    # 3. 리스크 관련 기사 선별 + 탐지
    print("=== 3. 리스크 탐지 ===")
    news_texts = store.search_by_risk(n_per_type=2)  # 리스크 유형별 키워드로 선별
    context    = store.search(query=" ".join(news_texts)[:500], n=3)
    print(f"  선별된 기사: {len(news_texts)}건\n")

    detector = RiskDetector()
    tags = detector.detect(news_texts, context=context)

    active_tags = [t for t in tags if t.level > 0.0]
    if active_tags:
        for tag in active_tags:
            print(f"  {tag.name:<25} level={tag.level:.2f}  confidence={tag.confidence:.2f}")
            print(f"    근거: {tag.source}")
    else:
        print("  탐지된 리스크 없음")

    # 4. RL 환경에 태그 주입
    print("\n=== 4. RL 환경 주입 ===")
    prices = fetch_prices(["AAPL", "MSFT", "GOOGL", "AMZN", "META"], "2022-01-01", "2024-12-31")
    env = PortfolioEnv(prices)
    obs, _ = env.reset()        # reset 먼저
    env.inject_risk_tags(tags)  # 그 다음 주입
    obs = env._get_obs()        # 주입된 상태로 관측값 갱신

    risk_obs = obs[25:30]
    print(f"  관측 공간 shape: {obs.shape}")
    print(f"  리스크 태그 관측값: {dict(zip([t.name for t in tags], risk_obs.tolist()))}")


if __name__ == "__main__":
    main()
