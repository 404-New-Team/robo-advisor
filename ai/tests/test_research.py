from pathlib import Path
import sys


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.data.market_data import fetch_prices
from src.envs.portfolio_env import PortfolioEnv
from src.research.news_fetcher import fetch_all
from src.research.news_store import NewsStore
from src.research.risk_detector import RiskDetector


def main():
    print("=== 1. 뉴스 수집 ===")
    articles = fetch_all(max_per_feed=10)
    print(f"  총 {len(articles)}건 수집\n")

    print("=== 2. ChromaDB 저장 ===")
    store = NewsStore()
    added = store.add(articles)
    print(f"  신규 저장: {added}건 | 누적: {store.count()}건\n")

    print("=== 3. 리스크 탐지 ===")
    news_texts = store.search_by_risk(n_per_type=2)
    context = store.search(query=" ".join(news_texts)[:500], n=3)
    print(f"  선별된 기사: {len(news_texts)}건\n")

    detector = RiskDetector()
    tags = detector.detect(news_texts, context=context)

    active_tags = [tag for tag in tags if tag.level > 0.0]
    if active_tags:
        for tag in active_tags:
            print(f"  {tag.name:<25} level={tag.level:.2f}  confidence={tag.confidence:.2f}")
            print(f"    근거: {tag.source}")
    else:
        print("  탐지된 리스크 없음")

    print("\n=== 4. RL 환경 주입 ===")
    prices = fetch_prices(["AAPL", "MSFT", "GOOGL", "AMZN", "META"], "2022-01-01", "2024-12-31")
    env = PortfolioEnv(prices)
    env.reset()
    env.inject_risk_tags(tags)
    obs = env._get_obs()

    risk_obs = obs[25:30]
    print(f"  관측 공간 shape: {obs.shape}")
    print(f"  리스크 태그 관측값: {dict(zip([tag.name for tag in tags], risk_obs.tolist()))}")


if __name__ == "__main__":
    main()
