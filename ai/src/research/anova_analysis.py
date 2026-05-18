"""
보상 함수 변형 3가지에 대한 ANOVA 통계 검증 모듈.

사용 흐름:
  1. collect_episode_rewards() — 각 variant로 n_episodes 수행, 에피소드 총 보상 수집
  2. run_anova()              — one-way ANOVA + Tukey HSD post-hoc
  3. report()                — 콘솔 출력 및 JSON 저장
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class ANOVAResult:
    f_statistic: float
    p_value: float
    significant: bool              # p < alpha
    alpha: float
    group_means: dict[str, float]
    group_stds: dict[str, float]
    group_ns: dict[str, int]
    tukey_results: list[dict]      # pairwise post-hoc


def collect_episode_rewards(
    prices,
    variant_names: list[str],
    n_episodes: int = 30,
    window_size: int = 20,
    transaction_cost: float = 0.001,
) -> dict[str, list[float]]:
    """각 RewardVariant로 n_episodes 무작위 에피소드를 수행, 총 보상 목록 반환."""
    from ..envs.portfolio_env import PortfolioEnv, RewardVariant

    results: dict[str, list[float]] = {}
    rng = np.random.default_rng(42)

    for name in variant_names:
        variant = RewardVariant(name)
        env = PortfolioEnv(
            prices=prices,
            window_size=window_size,
            transaction_cost=transaction_cost,
            reward_variant=variant,
        )
        episode_rewards: list[float] = []
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=int(rng.integers(0, 10_000)))
            total_reward = 0.0
            done = False
            while not done:
                # 무작위 행동 (정책 없이 보상 분포 특성만 비교)
                action = rng.uniform(-1, 1, size=env.action_space.shape).astype(np.float32)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
            episode_rewards.append(total_reward)
        results[name] = episode_rewards

    return results


def run_anova(
    rewards_by_variant: dict[str, list[float]],
    alpha: float = 0.05,
) -> ANOVAResult:
    """one-way ANOVA + Tukey HSD post-hoc 검증."""
    from scipy import stats

    groups = list(rewards_by_variant.keys())
    data = [np.array(rewards_by_variant[g]) for g in groups]

    f_stat, p_val = stats.f_oneway(*data)

    group_means = {g: float(np.mean(d)) for g, d in zip(groups, data)}
    group_stds  = {g: float(np.std(d, ddof=1)) for g, d in zip(groups, data)}
    group_ns    = {g: len(d) for g, d in zip(groups, data)}

    tukey_results = _tukey_hsd(groups, data, alpha)

    return ANOVAResult(
        f_statistic=round(float(f_stat), 4),
        p_value=round(float(p_val), 6),
        significant=bool(p_val < alpha),
        alpha=alpha,
        group_means=group_means,
        group_stds=group_stds,
        group_ns=group_ns,
        tukey_results=tukey_results,
    )


def _tukey_hsd(
    groups: list[str],
    data: list[np.ndarray],
    alpha: float,
) -> list[dict]:
    """Tukey HSD pairwise 비교 (statsmodels 없이 직접 구현)."""
    from scipy import stats
    from itertools import combinations

    n_total = sum(len(d) for d in data)
    k = len(groups)

    # MSE: within-group mean square error
    grand_mean = np.mean(np.concatenate(data))
    ss_within = sum(np.sum((d - np.mean(d)) ** 2) for d in data)
    df_within = n_total - k
    mse = ss_within / df_within if df_within > 0 else 1e-8

    results = []
    for (i, g1), (j, g2) in combinations(enumerate(groups), 2):
        n1, n2 = len(data[i]), len(data[j])
        mean_diff = float(np.mean(data[i]) - np.mean(data[j]))
        se = float(np.sqrt(mse * (1 / n1 + 1 / n2) / 2))
        q_stat = abs(mean_diff) / (se + 1e-8)

        # studentized range 분포 근사 (t-분포 기반 보수적 근사)
        t_stat = q_stat / np.sqrt(2)
        p_approx = float(2 * stats.t.sf(t_stat, df=df_within))

        results.append({
            "group1": g1,
            "group2": g2,
            "mean_diff": round(mean_diff, 4),
            "q_statistic": round(q_stat, 4),
            "p_value_approx": round(p_approx, 6),
            "significant": p_approx < alpha,
        })

    return results


def report(result: ANOVAResult, save_path: Optional[str] = None) -> None:
    """결과를 콘솔에 출력하고 save_path가 있으면 JSON으로 저장."""
    print("\n" + "=" * 60)
    print("REWARD VARIANT ANOVA RESULTS")
    print("=" * 60)
    print(f"F-statistic : {result.f_statistic:.4f}")
    print(f"p-value     : {result.p_value:.6f}  ({'significant ✓' if result.significant else 'not significant ✗'} at α={result.alpha})")
    print()
    print(f"{'Variant':<15} {'Mean':>10} {'Std':>10} {'N':>5}")
    print("-" * 45)
    for g in result.group_means:
        print(
            f"{g:<15} "
            f"{result.group_means[g]:>10.4f} "
            f"{result.group_stds[g]:>10.4f} "
            f"{result.group_ns[g]:>5d}"
        )
    print()
    print("Tukey HSD Pairwise Comparisons:")
    print(f"  {'Group1':<12} {'Group2':<12} {'Diff':>8} {'q-stat':>8} {'p-approx':>10} {'Sig':>5}")
    print("  " + "-" * 58)
    for t in result.tukey_results:
        sig = "✓" if t["significant"] else " "
        print(
            f"  {t['group1']:<12} {t['group2']:<12} "
            f"{t['mean_diff']:>8.4f} {t['q_statistic']:>8.4f} "
            f"{t['p_value_approx']:>10.6f} {sig:>5}"
        )
    print("=" * 60)

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, ensure_ascii=False, indent=2)
        print(f"결과 저장: {path}")
