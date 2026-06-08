"""
Walk-Forward 백테스트 실행 스크립트.

DRL(PPO) 단독 성능을 Walk-Forward 방식으로 검증한다.
폴드별 CAGR / Sharpe / MDD 등 12개 지표를 출력하며,
n_seeds 앙상블로 훈련 분산을 줄일 수 있다.

파일명 규칙:
  walk_forward_tm{TM}_tt{TT}_ts{TS}_s{S}[_lr{LR}][_l{L}][_w{W}].json
  - _lr{LR}: learning_rate가 기본값(3e-4)이 아닌 경우만 포함
  - _l{L}  : risk_penalty_lambda가 기본값(0.1)이 아닌 경우만 포함
  - _w{W}  : window_size가 settings.yaml 기본값과 다른 경우만 포함

사용법:
  cd ai
  python experiments/walk_forward_experiment.py
  python experiments/walk_forward_experiment.py --drl_timesteps 300000 --n_seeds 5
  python experiments/walk_forward_experiment.py --learning_rate 1e-4
  python experiments/walk_forward_experiment.py --risk_penalty_lambda 2.0
  python experiments/walk_forward_experiment.py --window_size 30
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.market_data import fetch_prices
from src.backtest.walk_forward import WalkForwardBacktest, WalkForwardConfig

CONFIG_PATH = Path(__file__).parent.parent / "src" / "config" / "settings.yaml"
RESULT_DIR = Path(__file__).parent / "results"

_DEFAULT_LR = 3e-4
_DEFAULT_LAMBDA = 0.1


def _numpy_default(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _lr_tag(lr: float) -> str:
    """1e-4 → 'lr1e-4', 3e-4 → 'lr3e-4'  (exponent 앞 0 제거)"""
    s = f"{lr:.0e}"
    s = re.sub(r"e([+-])0*(\d+)", lambda m: f"e{m.group(1)}{m.group(2)}", s)
    return f"lr{s}"


def _lambda_tag(lam: float) -> str:
    return "l" + str(lam).replace(".", "p")


def main():
    parser = argparse.ArgumentParser(description="DRL Walk-Forward 단독 성능 검증")
    parser.add_argument("--start",               type=str,   default="2019-01-01")
    parser.add_argument("--end",                 type=str,   default="2024-12-31")
    parser.add_argument("--train_months",        type=int,   default=24,       help="훈련 기간(월)")
    parser.add_argument("--test_months",         type=int,   default=6,        help="테스트 기간(월)")
    parser.add_argument("--step_months",         type=int,   default=6,        help="슬라이딩 스텝(월)")
    parser.add_argument("--drl_timesteps",       type=int,   default=150_000,  help="폴드당 학습 스텝")
    parser.add_argument("--n_seeds",             type=int,   default=1,        help="앙상블 시드 수")
    parser.add_argument("--n_envs",              type=int,   default=4,        help="병렬 환경 수 (VecEnv)")
    parser.add_argument("--learning_rate",       type=float, default=_DEFAULT_LR,     help="PPO learning rate (기본값 3e-4)")
    parser.add_argument("--risk_penalty_lambda", type=float, default=_DEFAULT_LAMBDA, help="리스크 패널티 람다 (기본값 0.1)")
    parser.add_argument("--window_size",         type=int,   default=None,     help="관측 윈도우 크기 N (미지정 시 settings.yaml 값 사용)")
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

    default_window = env_cfg["window_size"]
    window_size = args.window_size if args.window_size is not None else default_window

    wf_cfg = WalkForwardConfig(
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=args.step_months,
        train_timesteps=args.drl_timesteps,
        n_seeds=args.n_seeds,
        n_envs=args.n_envs,
        learning_rate=args.learning_rate,
        window_size=window_size,
        transaction_cost=env_cfg["transaction_cost"],
        slippage=env_cfg.get("slippage", 0.0005),
        max_drawdown_threshold=env_cfg.get("max_drawdown_threshold", 0.25),
        risk_penalty_lambda=args.risk_penalty_lambda,
    )

    backtest = WalkForwardBacktest(prices=prices, config=wf_cfg)
    result = backtest.run(verbose=True)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "config": {
            "train_months":          wf_cfg.train_months,
            "test_months":           wf_cfg.test_months,
            "step_months":           wf_cfg.step_months,
            "train_timesteps":       wf_cfg.train_timesteps,
            "n_seeds":               wf_cfg.n_seeds,
            "n_envs":                wf_cfg.n_envs,
            "learning_rate":         wf_cfg.learning_rate,
            "window_size":           wf_cfg.window_size,
            "max_drawdown_threshold": wf_cfg.max_drawdown_threshold,
            "risk_penalty_lambda":   wf_cfg.risk_penalty_lambda,
        },
        "summary": {
            "n_folds":          len(result.folds),
            "mean_cagr":        result.mean_cagr,
            "std_cagr":         result.std_cagr,
            "mean_sharpe":      result.mean_sharpe,
            "std_sharpe":       result.std_sharpe,
            "mean_max_drawdown": result.mean_max_drawdown,
            "std_max_drawdown":  result.std_max_drawdown,
        },
        "folds": [
            {
                "fold_idx":    fm.fold_idx,
                "test_start":  fm.test_start,
                "test_end":    fm.test_end,
                "n_train_bars": fm.n_train_bars,
                "n_test_bars": fm.n_test_bars,
                **fm.metrics.as_dict(),
            }
            for fm in result.folds
        ],
    }

    # 파일명: 기본값과 다를 때만 태그 추가
    base = f"walk_forward_tm{args.train_months}_tt{args.test_months}_ts{args.drl_timesteps}_s{args.n_seeds}"
    if args.learning_rate != _DEFAULT_LR:
        base += f"_{_lr_tag(args.learning_rate)}"
    if args.risk_penalty_lambda != _DEFAULT_LAMBDA:
        base += f"_{_lambda_tag(args.risk_penalty_lambda)}"
    if window_size != default_window:
        base += f"_w{window_size}"
    fname = base + ".json"

    save_path = RESULT_DIR / fname
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=_numpy_default)

    print(f"\n결과 저장 완료: {save_path}")


if __name__ == "__main__":
    main()
