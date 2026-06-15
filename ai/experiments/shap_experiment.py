"""
SHAP 해석 실험 스크립트.

학습된 PPO 에이전트의 의사결정을 SHAP으로 시각화한다.
  - Summary Plot  : 전체 관측값에 걸친 특성 중요도 분포
  - Force Plot    : 단일 스텝의 특성별 push/pull 기여도
  - Waterfall Plot: 단일 스텝의 특성 기여도 누적 bar

사용법:
  cd ai
  python experiments/shap_experiment.py
  python experiments/shap_experiment.py --checkpoint checkpoints/portfolio_ppo_best
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.market_data import fetch_prices
from src.envs.portfolio_env import PortfolioEnv
from src.envs.risk_state import RiskState
from src.agents.ppo_agent import PPOAgent
from src.xai.shap_explainer import SHAPExplainer

AI_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = AI_DIR / "src" / "config" / "settings.yaml"
DEFAULT_CHECKPOINT = AI_DIR / "checkpoints" / "portfolio_ppo_best"
DEMO_CHECKPOINT_DIR = AI_DIR / "checkpoints" / "shap_demo"
SAVE_DIR = AI_DIR / "experiments" / "results" / "shap"


def resolve_checkpoint(path: str | None) -> Path:
    checkpoint = Path(path).expanduser() if path else DEFAULT_CHECKPOINT
    if not checkpoint.is_absolute():
        cwd_candidate = Path.cwd() / checkpoint
        ai_candidate = AI_DIR / checkpoint
        checkpoint = cwd_candidate if cwd_candidate.with_suffix(".zip").exists() else ai_candidate
    return checkpoint


def build_feature_names(tickers: list, tag_names: list) -> list:
    suffixes = ["ret1d", "ret5d", "ret20d", "vol20d", "mom20d",
                "rsi14", "macd", "macd_signal", "bb_upper", "bb_lower", "bb_position"]
    market = [f"{t}_{s}" for t in tickers for s in suffixes]
    weights = [f"w_{t}" for t in tickers]
    return market + tag_names + weights


def collect_observations(agent: PPOAgent, env: PortfolioEnv, n: int) -> np.ndarray:
    """환경을 rollout해 n개 관측값을 수집한다."""
    obs_list = []
    obs, _ = env.reset()
    done = False
    while len(obs_list) < n:
        obs_list.append(obs.copy())
        action = agent.predict(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
            done = False
    return np.array(obs_list[:n])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="학습된 모델 경로 (없으면 단기 학습)")
    parser.add_argument("--demo_train", action="store_true",
                        help="Run a short demo training if no checkpoint is available.")
    parser.add_argument("--n_background", type=int, default=50,
                        help="KernelExplainer 배경 샘플 수 (클수록 정확하나 느림)")
    parser.add_argument("--n_summary", type=int, default=30,
                        help="Summary plot 대상 관측값 수")
    parser.add_argument("--output_idx", type=int, default=0,
                        help="Force/Waterfall 설명 대상 자산 인덱스")
    args = parser.parse_args()

    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env_cfg = cfg["environment"]
    tickers = env_cfg["tickers"]

    print("시장 데이터 로딩 중...")
    prices = fetch_prices(tickers, start="2023-01-01", end="2024-12-31")
    print(f"  shape={prices.shape}  기간={prices.index[0].date()} ~ {prices.index[-1].date()}")

    env = PortfolioEnv(
        prices=prices,
        risk_state=RiskState(),
        window_size=env_cfg["window_size"],
        transaction_cost=env_cfg["transaction_cost"],
        slippage=env_cfg.get("slippage", 0.0005),
        max_drawdown_threshold=env_cfg.get("max_drawdown_threshold", 0.15),
    )

    # ── 모델 로드 or 단기 학습 ─────────────────────────────────
    args.checkpoint = str(resolve_checkpoint(args.checkpoint))
    if Path(args.checkpoint).with_suffix(".zip").exists():
        print(f"\n체크포인트 로드: {args.checkpoint}")
        try:
            agent = PPOAgent.load(args.checkpoint, env=env)
        except ValueError as e:
            if not args.demo_train:
                raise
            print(f"  [경고] 체크포인트 로드 실패: {e}")
            print("  관측 공간이 변경됐습니다 (구 모델 호환 불가).")
            print("  시연용 단기 학습으로 대체합니다 (10,000 스텝)...")
            agent = PPOAgent(env=env, learning_rate=cfg["training"]["learning_rate"])
            agent.train(total_timesteps=10_000, checkpoint_dir=str(DEMO_CHECKPOINT_DIR))
    else:
        if not args.demo_train:
            raise FileNotFoundError(
                f"Checkpoint not found: {Path(args.checkpoint).with_suffix('.zip')}. "
                "Run `python ai/train.py` first, pass --checkpoint, or use --demo_train "
                "only for a short demonstration model."
            )
        print("\n체크포인트 없음 — 시연용 단기 학습 (10,000 스텝)")
        agent = PPOAgent(env=env, learning_rate=cfg["training"]["learning_rate"])
        agent.train(total_timesteps=10_000, checkpoint_dir=str(DEMO_CHECKPOINT_DIR))

    # ── 관측값 수집 ────────────────────────────────────────────
    print(f"\n관측값 수집 중 (background={args.n_background}, summary={args.n_summary})...")
    all_obs = collect_observations(agent, env, args.n_background + args.n_summary)
    background_obs = all_obs[:args.n_background]
    summary_obs = all_obs[args.n_background:]
    explain_obs = all_obs[0]  # Force / Waterfall 대상 (1개)

    # ── SHAPExplainer 초기화 ───────────────────────────────────
    feature_names = build_feature_names(tickers, cfg["risk"]["tag_names"])
    explainer = SHAPExplainer(agent, feature_names)

    print(f"\nKernelExplainer 초기화 중 (background {args.n_background}개)...")
    print("  ※ KernelExplainer는 느립니다. 잠시 기다려주세요.")
    explainer.fit(background_obs, n_background=args.n_background)

    # ── 상위 특성 출력 ─────────────────────────────────────────
    print("\n[Top-5 결정 요인]")
    top5 = explainer.top_k_features(explain_obs, k=5)
    for rank, (name, val) in enumerate(top5, 1):
        print(f"  {rank}. {name:<30s}  |SHAP|={val:.4f}")

    # ── 플롯 생성 ──────────────────────────────────────────────
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    ticker_name = tickers[args.output_idx]
    print(f"\n플롯 생성 중 (asset_idx={args.output_idx} → {ticker_name})...")

    explainer.plot_summary(
        summary_obs,
        max_display=20,
        save_path=str(SAVE_DIR / "summary_plot.png"),
    )
    explainer.plot_force(
        explain_obs,
        output_idx=args.output_idx,
        save_path=str(SAVE_DIR / "force_plot.png"),
    )
    explainer.plot_waterfall(
        explain_obs,
        output_idx=args.output_idx,
        max_display=15,
        save_path=str(SAVE_DIR / "waterfall_plot.png"),
    )

    print(f"\n완료. 플롯 저장 위치: {SAVE_DIR}")


if __name__ == "__main__":
    main()
