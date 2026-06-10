"""
최종 PPO 모델 훈련 스크립트.

실험에서 검증된 최적 하이퍼파라미터로 전체 데이터셋을 학습하고 체크포인트를 저장한다.
Walk-Forward 평가가 아닌 단일 모델 훈련용이므로, 전체 기간 데이터를 훈련에 사용한다.

저장 경로:
  checkpoints/production/seed_{seed}/final_model.zip  — 각 시드별 모델
  checkpoints/portfolio_ppo_best.zip                  — API 로딩용 (seed=3 복사)

사용법:
  cd ai
  python experiments/train_final_model.py
  python experiments/train_final_model.py --timesteps 300000
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.market_data import fetch_prices
from src.envs.portfolio_env import PortfolioEnv
from src.envs.risk_state import RiskState
from src.agents.ppo_agent import PPOAgent
from stable_baselines3.common.vec_env import DummyVecEnv

CONFIG_PATH = Path(__file__).parent.parent / "src" / "config" / "settings.yaml"
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"
PRODUCTION_DIR = CHECKPOINT_DIR / "production"

OPTIMAL_SEEDS = [3, 41, 25, 73, 16]
PRIMARY_SEED = 3  # API portfolio_ppo_best.zip에 복사할 시드


def main() -> None:
    parser = argparse.ArgumentParser(description="최종 PPO 모델 훈련 (production)")
    parser.add_argument("--start",          type=str,   default="2017-01-01")
    parser.add_argument("--end",            type=str,   default="2025-12-31")
    parser.add_argument("--timesteps",      type=int,   default=225_000)
    parser.add_argument("--learning_rate",  type=float, default=1e-4)
    parser.add_argument("--lambda_val",     type=float, default=0.5)
    parser.add_argument("--window_size",    type=int,   default=30)
    parser.add_argument("--n_envs",         type=int,   default=4)
    parser.add_argument("--seeds",          type=int,   nargs="+", default=OPTIMAL_SEEDS)
    args = parser.parse_args()

    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    env_cfg = cfg["environment"]

    print("시장 데이터 로딩 중...")
    prices = fetch_prices(
        tickers=env_cfg["tickers"],
        start=args.start,
        end=args.end,
    )
    print(f"데이터 shape: {prices.shape}  기간: {prices.index[0].date()} ~ {prices.index[-1].date()}")

    print(f"\n=== 최종 모델 훈련 설정 ===")
    print(f"  learning_rate = {args.learning_rate}")
    print(f"  lambda        = {args.lambda_val}")
    print(f"  window_size   = {args.window_size}")
    print(f"  timesteps     = {args.timesteps:,}")
    print(f"  n_envs        = {args.n_envs}")
    print(f"  seeds         = {args.seeds}")
    print("=" * 30)

    PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)

    def make_env():
        return PortfolioEnv(
            prices=prices,
            risk_state=RiskState(),
            window_size=args.window_size,
            transaction_cost=env_cfg["transaction_cost"],
            slippage=env_cfg.get("slippage", 0.0005),
            max_drawdown_threshold=env_cfg.get("max_drawdown_threshold", 0.25),
            risk_penalty_lambda=args.lambda_val,
        )

    for i, seed in enumerate(args.seeds):
        print(f"\n[{i+1}/{len(args.seeds)}] seed={seed} 훈련 중...")

        if args.n_envs > 1:
            env = DummyVecEnv([make_env] * args.n_envs)
        else:
            env = make_env()

        seed_dir = PRODUCTION_DIR / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        agent = PPOAgent(
            env=env,
            learning_rate=args.learning_rate,
            batch_size=256,
            seed=seed,
            verbose=1,
        )
        agent.train(
            total_timesteps=args.timesteps,
            checkpoint_dir=str(seed_dir),
        )
        save_path = seed_dir / "final_model"
        agent.save(str(save_path))
        print(f"  저장 완료: {save_path}.zip")

        if args.n_envs > 1:
            env.close()

    # API 로딩용: PRIMARY_SEED 모델을 portfolio_ppo_best.zip으로 복사
    primary_src = PRODUCTION_DIR / f"seed_{PRIMARY_SEED}" / "final_model.zip"
    best_dst = CHECKPOINT_DIR / "portfolio_ppo_best.zip"
    if primary_src.exists():
        shutil.copy2(primary_src, best_dst)
        print(f"\nAPI 모델 복사 완료: {primary_src} → {best_dst}")
    else:
        print(f"\n[경고] {primary_src} 가 없어 portfolio_ppo_best.zip 생성 건너뜀")

    # 훈련 메타데이터 저장
    meta = {
        "seeds": args.seeds,
        "primary_seed": PRIMARY_SEED,
        "learning_rate": args.learning_rate,
        "risk_penalty_lambda": args.lambda_val,
        "window_size": args.window_size,
        "timesteps": args.timesteps,
        "data_start": args.start,
        "data_end": args.end,
    }
    meta_path = PRODUCTION_DIR / "config.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n훈련 완료. 모델 저장 경로: {PRODUCTION_DIR}")
    print(f"API 체크포인트:          {best_dst}")


if __name__ == "__main__":
    main()
