"""
보상 함수 변형 3가지 비교 실험 스크립트.

실행:
  cd ai
  python experiments/reward_experiment.py

결과:
  - 콘솔에 ANOVA 테이블 출력
  - experiments/results/anova_result.json 저장
  - experiments/results/reward_distribution.png 저장
"""

import sys
from pathlib import Path

# ai/ 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from src.data.market_data import fetch_prices
from src.envs.portfolio_env import RewardVariant
from src.research.anova_analysis import collect_episode_rewards, run_anova, report

RESULTS_DIR = Path(__file__).parent / "results"
VARIANTS = [v.value for v in RewardVariant]
N_EPISODES = 30
ALPHA = 0.05


def plot_reward_distributions(
    rewards_by_variant: dict[str, list[float]],
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 박스플롯
    ax = axes[0]
    ax.boxplot(
        [rewards_by_variant[v] for v in VARIANTS],
        labels=VARIANTS,
        patch_artist=True,
        boxprops=dict(facecolor="#AED6F1", color="#1A5276"),
        medianprops=dict(color="#C0392B", linewidth=2),
    )
    ax.set_title("Episode Reward Distribution by Reward Variant")
    ax.set_ylabel("Total Episode Reward")
    ax.set_xlabel("Reward Variant")
    ax.grid(True, alpha=0.3)

    # 평균 + 95% CI 막대그래프
    ax2 = axes[1]
    means = [np.mean(rewards_by_variant[v]) for v in VARIANTS]
    stds  = [np.std(rewards_by_variant[v], ddof=1) for v in VARIANTS]
    ns    = [len(rewards_by_variant[v]) for v in VARIANTS]
    cis   = [1.96 * s / np.sqrt(n) for s, n in zip(stds, ns)]

    colors = ["#85C1E9", "#82E0AA", "#F1948A"]
    bars = ax2.bar(VARIANTS, means, yerr=cis, capsize=6, color=colors, edgecolor="gray", alpha=0.85)
    ax2.set_title("Mean ± 95% CI by Reward Variant")
    ax2.set_ylabel("Mean Episode Reward")
    ax2.set_xlabel("Reward Variant")
    ax2.grid(True, alpha=0.3, axis="y")
    for bar, mean in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                 f"{mean:.4f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"분포 시각화 저장: {save_path}")


def main():
    print("데이터 로드 중...")
    prices = fetch_prices(
        tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
        start="2022-01-01",
        end="2023-12-31",
    )
    print(f"  {len(prices)}행 로드 완료\n")

    print(f"각 변형({', '.join(VARIANTS)}) × {N_EPISODES} 에피소드 수행 중...")
    rewards_by_variant = collect_episode_rewards(
        prices=prices,
        variant_names=VARIANTS,
        n_episodes=N_EPISODES,
    )
    for name, rewards in rewards_by_variant.items():
        print(f"  {name}: mean={np.mean(rewards):.4f}, std={np.std(rewards, ddof=1):.4f}")

    print("\nANOVA 검증 중...")
    result = run_anova(rewards_by_variant, alpha=ALPHA)
    report(result, save_path=str(RESULTS_DIR / "anova_result.json"))

    print("\n분포 시각화 중...")
    plot_reward_distributions(rewards_by_variant, RESULTS_DIR / "reward_distribution.png")

    if result.significant:
        best = max(result.group_means, key=lambda k: result.group_means[k])
        print(f"\n결론: 변형 간 통계적으로 유의미한 차이 있음 (p={result.p_value:.4f})")
        print(f"      가장 높은 평균 보상: {best} ({result.group_means[best]:.4f})")
    else:
        print(f"\n결론: 변형 간 통계적으로 유의미한 차이 없음 (p={result.p_value:.4f}, α={ALPHA})")


if __name__ == "__main__":
    main()
