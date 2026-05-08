"""
Safe-Guard 모니터링: RL 에이전트의 행동을 실시간으로 검증하고 위험 제약을 강제.

규제 관점의 XAI 요건 충족을 위해 모든 개입 기록을 유지한다.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class SafeGuardConfig:
    max_drawdown: float = 0.20        # 낙폭 20% 초과 시 균등 분산 강제
    max_position: float = 0.40        # 단일 종목 비중 40% 한도
    max_aggregate_risk: float = 0.75  # 집계 리스크 0.75 초과 시 포지션 축소


class SafeGuardMonitor:
    def __init__(self, config: SafeGuardConfig = None):
        self.config = config or SafeGuardConfig()
        self.violations: list = []

    def validate(self, weights: np.ndarray, info: dict) -> np.ndarray:
        """
        포트폴리오 비중을 검증하고 제약 위반 시 조정하여 반환.
        weights: softmax 적용된 포트폴리오 비중 (합=1)
        info: PortfolioEnv._get_info() 반환값
        """
        adjusted = weights.copy()

        # 1. 최대 낙폭 초과 → 균등 분산으로 강제
        if info["drawdown"] > self.config.max_drawdown:
            self._record("max_drawdown", info["drawdown"])
            return np.ones_like(adjusted) / len(adjusted)

        # 2. 단일 포지션 한도
        if np.max(adjusted) > self.config.max_position:
            self._record("max_position", float(np.max(adjusted)))
            adjusted = np.clip(adjusted, 0.0, self.config.max_position)
            adjusted /= adjusted.sum()

        # 3. 집계 리스크 과다 → 비중 비례 축소
        aggregate_risk = float(np.mean(info["risk_state"]))
        if aggregate_risk > self.config.max_aggregate_risk:
            self._record("aggregate_risk", aggregate_risk)
            scale = max(1.0 - (aggregate_risk - self.config.max_aggregate_risk), 0.1)
            adjusted = adjusted * scale + (1 - scale) / len(adjusted)

        return adjusted

    def weights_to_action(self, weights: np.ndarray) -> np.ndarray:
        """검증된 비중 → env logit action 변환. softmax(log(w)) = w 성질 활용."""
        return np.log(weights + 1e-8).astype(np.float32)

    def _record(self, rule: str, value: float) -> None:
        self.violations.append({"rule": rule, "value": round(value, 4)})

    def summary(self) -> dict:
        from collections import Counter
        return dict(Counter(v["rule"] for v in self.violations))
