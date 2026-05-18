from pathlib import Path
import sys


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.data.market_data import fetch_prices
from src.envs.portfolio_env import PortfolioEnv


def run_env_smoke() -> None:
    prices = fetch_prices(
        ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
        "2022-01-01",
        "2024-12-31",
    )
    env = PortfolioEnv(prices)

    obs, _ = env.reset()
    print(f"observation shape: {obs.shape}")
    assert obs.shape == env.observation_space.shape

    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"step {i + 1} | reward: {reward:.4f} | "
            f"value: {info['portfolio_value']:.4f} | drawdown: {info['drawdown']:.4f}"
        )
        assert obs.shape == env.observation_space.shape
        assert "portfolio_value" in info
        assert not terminated
        if truncated:
            break


def test_portfolio_env_smoke():
    run_env_smoke()


if __name__ == "__main__":
    run_env_smoke()
