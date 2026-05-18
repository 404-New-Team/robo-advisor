from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.ppo_agent import PPOAgent
from src.data.market_data import fetch_prices
from src.envs.portfolio_env import PortfolioEnv
from src.envs.risk_state import RiskState

AI_DIR = Path(__file__).resolve().parent
CONFIG_PATH = AI_DIR / "src" / "config" / "settings.yaml"
CHECKPOINT_DIR = AI_DIR / "checkpoints"
BEST_PATH = CHECKPOINT_DIR / "portfolio_ppo_best"
SCORE_PATH = CHECKPOINT_DIR / "best_score.txt"


class TrainingLogger(BaseCallback):
    """롤아웃마다 평균 에피소드 보상을 기록해 학습 곡선을 그린다."""

    def __init__(self):
        super().__init__()
        self.timesteps: list[int]   = []
        self.mean_rewards: list[float] = []

    def _on_rollout_end(self) -> bool:
        val = self.model.logger.name_to_value.get("rollout/ep_rew_mean")
        if val is not None:
            self.timesteps.append(self.num_timesteps)
            self.mean_rewards.append(float(val))
        return True

    def _on_step(self) -> bool:
        return True


def make_env_fn(prices, cfg):
    env_cfg = cfg["environment"]
    def _init():
        return PortfolioEnv(
            prices=prices,
            risk_state=RiskState(),
            window_size=env_cfg["window_size"],
            transaction_cost=env_cfg["transaction_cost"],
            slippage=env_cfg.get("slippage", 0.0005),
            max_drawdown_threshold=env_cfg.get("max_drawdown_threshold", 0.15),
            risk_penalty_lambda=cfg["reward"]["risk_penalty_lambda"],
            drawdown_penalty_mu=cfg["reward"]["drawdown_penalty_mu"],
        )
    return _init


def evaluate(agent, env, n_episodes=3):
    rewards, final_values = [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done, total_reward = False, 0.0
        while not done:
            action = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        rewards.append(total_reward)
        final_values.append(info["portfolio_value"])
    return float(np.mean(rewards)), float(np.mean(final_values))


def buy_and_hold_value(prices) -> float:
    """균등 비중 Buy & Hold의 최종 포트폴리오 가치."""
    returns = prices.iloc[-1] / prices.iloc[0] - 1
    return float(1.0 + returns.mean())


def plot_learning_curve(logger: TrainingLogger, save_path: Path) -> None:
    if not logger.timesteps:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(logger.timesteps, logger.mean_rewards)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("PPO Learning Curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"  학습 곡선 저장: {save_path}")


def load_best_score() -> float:
    if SCORE_PATH.exists():
        try:
            return float(SCORE_PATH.read_text().strip())
        except ValueError:
            return float("-inf")
    return float("-inf")


def save_best_score(score: float) -> None:
    SCORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCORE_PATH.write_text(str(score))


def main():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env_cfg   = cfg["environment"]
    train_cfg = cfg["training"]

    print("데이터 로드 중...")
    train_prices = fetch_prices(env_cfg["tickers"], start="2019-01-01", end="2025-06-30")
    eval_prices  = fetch_prices(env_cfg["tickers"], start="2025-07-01", end="2026-05-01")
    print(f"  학습: {len(train_prices)}행  |  평가: {len(eval_prices)}행")

    n_envs    = train_cfg.get("n_envs", 1)
    env_fn    = make_env_fn(train_prices, cfg)
    train_env = DummyVecEnv([env_fn] * n_envs) if n_envs > 1 else env_fn()
    eval_env  = make_env_fn(eval_prices, cfg)()

    print(f"\nPPO 학습 시작")
    print(f"  timesteps={train_cfg['total_timesteps']:,}  |  n_envs={n_envs}  |  lr={train_cfg['learning_rate']}")
    logger = TrainingLogger()
    agent = PPOAgent(
        env=train_env,
        learning_rate=train_cfg["learning_rate"],
        batch_size=train_cfg["batch_size"],
    )
    agent.train(
        total_timesteps=train_cfg["total_timesteps"],
        checkpoint_dir=str(CHECKPOINT_DIR),
        callbacks=[logger],
    )

    print("\n학습 곡선 저장 중...")
    plot_learning_curve(logger, CHECKPOINT_DIR / "learning_curve.png")

    print("\n평가 중 (2025-07 ~ 2026-05 데이터)...")
    mean_reward, mean_value = evaluate(agent, eval_env)
    bnh_value = buy_and_hold_value(eval_prices)

    print(f"  PPO   포트폴리오 가치: {mean_value:.4f}  ({(mean_value - 1) * 100:+.2f}%)")
    print(f"  B&H   포트폴리오 가치: {bnh_value:.4f}  ({(bnh_value - 1) * 100:+.2f}%)")
    print(f"  초과 수익:             {(mean_value - bnh_value) * 100:+.2f}%p")

    best_score = load_best_score()
    best_model_exists = BEST_PATH.with_suffix(".zip").exists()
    if mean_reward > best_score or not best_model_exists:
        agent.save(str(BEST_PATH))
        save_best_score(mean_reward)
        print(f"\n최고 성능 갱신 ({best_score:.4f} → {mean_reward:.4f})")
        print(f"모델 저장 완료: {BEST_PATH}.zip")
    else:
        print(f"\n성능 미달 (현재={mean_reward:.4f} < 최고={best_score:.4f}), 저장 생략")


if __name__ == "__main__":
    main()
