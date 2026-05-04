from ai.data.market_data import fetch_prices
from ai.envs.portfolio_env import PortfolioEnv

prices = fetch_prices(["AAPL", "MSFT", "GOOGL", "AMZN", "META"], "2022-01-01", "2024-12-31")
env = PortfolioEnv(prices)

obs, _ = env.reset()
print(f"관측 공간 shape: {obs.shape}")  # (35,) 이어야 함

for i in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"step {i+1} | reward: {reward:.4f} | value: {info['portfolio_value']:.4f} | drawdown: {info['drawdown']:.4f}")
