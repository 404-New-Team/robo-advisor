from dataclasses import dataclass
from typing import Dict
import numpy as np


RISK_TAG_NAMES = [
    "regulatory_risk",
    "earnings_shock",
    "geopolitical_risk",
    "market_stress",
    "liquidity_risk",
]


@dataclass
class RiskTag:
    name: str
    level: float = 0.0       # 0.0 ~ 1.0
    confidence: float = 0.0  # 0.0 ~ 1.0
    source: str = ""

    def __post_init__(self):
        self.level = float(np.clip(self.level, 0.0, 1.0))
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))


class RiskState:
    """
    리서치 에이전트로부터 업데이트되는 리스크 상태.
    각 태그는 매 스텝 decay되고, 새 신호가 들어오면 confidence-weighted max로 갱신.
    """

    def __init__(self, tag_names: list = None, decay_rate: float = 0.95):
        self.tag_names = tag_names or RISK_TAG_NAMES
        self.decay_rate = decay_rate
        self._levels: Dict[str, float] = {name: 0.0 for name in self.tag_names}

    def update(self, tags: list) -> None:
        for tag in tags:
            if tag.name in self._levels:
                new_signal = tag.level * tag.confidence
                self._levels[tag.name] = max(self._levels[tag.name], new_signal)

    def step_decay(self) -> None:
        """매 환경 스텝마다 호출 — 리스크 신호를 점진적으로 감쇠."""
        for name in self._levels:
            self._levels[name] *= self.decay_rate

    def to_array(self) -> np.ndarray:
        return np.array([self._levels[name] for name in self.tag_names], dtype=np.float32)

    def reset(self) -> None:
        self._levels = {name: 0.0 for name in self.tag_names}

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v:.3f}" for k, v in self._levels.items())
        return f"RiskState({items})"
