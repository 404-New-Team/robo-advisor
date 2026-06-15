"""
SHAP 기반 RL 의사결정 해석기.

규제 요건(XAI 의무화) 대응: 포트폴리오 조정 결정의 근거를 특성 중요도로 설명.
KernelExplainer를 사용하므로 모델 구조에 무관하게 적용 가능.

Plot 종류:
  summary_plot  : 여러 관측값에 걸친 특성 중요도 분포 (bar / beeswarm)
  force_plot    : 단일 관측값의 특성별 기여도 (push/pull 형태)
  waterfall_plot: 특성별 SHAP 기여도 누적 (bar 형태, 읽기 쉬움)

SHAP 버전 호환성:
  KernelExplainer.shap_values()는 버전에 따라
  - list of (n_samples, n_features)  — 구버전
  - ndarray (n_outputs, n_samples, n_features) — 신버전
  두 형식을 모두 처리한다.
"""

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import shap


class SHAPExplainer:
    def __init__(self, agent, feature_names: list):
        self.agent = agent
        self.feature_names = feature_names
        self._explainer: Optional[shap.KernelExplainer] = None

    # ------------------------------------------------------------------
    # 초기화
    # ------------------------------------------------------------------

    def fit(self, background_obs: np.ndarray, n_background: int = 100) -> None:
        """학습 완료 후 1회 실행 — 배경 데이터로 KernelExplainer 초기화."""
        background = shap.sample(background_obs, min(n_background, len(background_obs)))

        def predict_fn(obs_batch: np.ndarray) -> np.ndarray:
            return np.array([self.agent.predict(obs, deterministic=True) for obs in obs_batch])

        self._explainer = shap.KernelExplainer(predict_fn, background)

    # ------------------------------------------------------------------
    # 수치 해석
    # ------------------------------------------------------------------

    def explain(self, obs: np.ndarray) -> dict:
        """단일 관측값의 SHAP 값과 특성 중요도 반환."""
        self._check_fitted()
        shap_values = self._explainer.shap_values(obs.reshape(1, -1), silent=True)
        sv_arr = self._to_array(shap_values)  # → (n_outputs, n_samples, n_features)

        # 출력별 절댓값 평균 후 첫 번째 샘플 → (n_features,)
        importance = np.mean(np.abs(sv_arr[:, 0, :]), axis=0).flatten()

        return {
            "feature_names": self.feature_names,
            "shap_importance": importance,
            "base_value": self._explainer.expected_value,
        }

    def top_k_features(self, obs: np.ndarray, k: int = 5) -> list:
        """상위 k개 결정 요인 반환 — 의사결정 근거 보고용."""
        result = self.explain(obs)
        importance = result["shap_importance"]          # 항상 1D
        top_idx = importance.argsort()[::-1][:k]
        return [(self.feature_names[int(i)], round(float(importance[i]), 6)) for i in top_idx]

    # ------------------------------------------------------------------
    # Summary Plot
    # ------------------------------------------------------------------

    def plot_summary(
        self,
        observations: np.ndarray,
        max_display: int = 20,
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> None:
        """
        Summary plot: 여러 관측값에 걸친 특성 중요도 시각화 (horizontal bar).

        shap.summary_plot의 shape 호환성 문제를 피하기 위해
        matplotlib으로 직접 렌더링한다.

        Args:
            observations : 설명할 관측값 배열 (n_obs, n_features)
            max_display  : 표시할 최대 특성 수 (중요도 상위 n개)
            save_path    : 저장 경로 (None이면 저장 안 함)
            show         : plt.show() 호출 여부
        """
        self._check_fitted()
        print(f"[SHAP] Summary plot 계산 중 ({len(observations)}개 샘플)...")
        shap_values = self._explainer.shap_values(observations, silent=True)
        sv_arr = self._to_array(shap_values)  # (n_outputs, n_obs, n_features)

        # 모든 출력·샘플에 걸쳐 절댓값 평균 → (n_features,)
        mean_abs = np.mean(np.abs(sv_arr), axis=(0, 1))

        n_show = min(max_display, len(self.feature_names))
        top_idx = np.argsort(mean_abs)[-n_show:]          # 중요도 낮은→높은 순 (barh용)
        top_names = [self.feature_names[int(i)] for i in top_idx]
        top_vals  = mean_abs[top_idx]

        fig, ax = plt.subplots(figsize=(9, max(4, n_show * 0.38)))
        bars = ax.barh(top_names, top_vals, color="#4c72b0", edgecolor="white", height=0.7)
        ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
        ax.set_title("SHAP Feature Importance (Mean |SHAP| across samples & outputs)",
                     fontsize=12, pad=10)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        self._save_and_show(save_path, show, "Summary plot")

    # ------------------------------------------------------------------
    # Force Plot
    # ------------------------------------------------------------------

    def plot_force(
        self,
        obs: np.ndarray,
        output_idx: int = 0,
        max_display: int = 15,
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> None:
        """
        Force plot: 단일 관측값의 특성별 SHAP 기여도 (diverging bar, matplotlib).

        빨간 막대 → 예측값을 올리는 특성 / 파란 막대 → 내리는 특성.

        Args:
            obs         : 설명할 단일 관측값 (n_features,)
            output_idx  : multi-output 시 설명할 자산(행동) 인덱스
            max_display : 표시할 최대 특성 수 (|SHAP| 상위 n개)
            save_path   : 저장 경로
            show        : plt.show() 호출 여부
        """
        self._check_fitted()
        shap_values = self._explainer.shap_values(obs.reshape(1, -1), silent=True)
        sv, base = self._extract_output(shap_values, output_idx, single=True)
        sv = sv.flatten()

        n_show = min(max_display, len(self.feature_names))
        top_idx = np.argsort(np.abs(sv))[-n_show:]
        top_sv   = sv[top_idx]
        top_names = [self.feature_names[int(i)] for i in top_idx]

        # 값 오름차순 정렬 (음수 → 양수)
        order = np.argsort(top_sv)
        top_sv    = top_sv[order]
        top_names = [top_names[i] for i in order]
        colors = ["#e74c3c" if v > 0 else "#3498db" for v in top_sv]

        fig, ax = plt.subplots(figsize=(9, max(4, n_show * 0.42)))
        ax.barh(top_names, top_sv, color=colors, edgecolor="white", height=0.72)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP Value  (red = pushes up, blue = pushes down)", fontsize=10)
        ax.set_title(
            f"SHAP Force Plot  (asset_idx={output_idx}, base={base:.4f})",
            fontsize=12, pad=10,
        )
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        self._save_and_show(save_path, show, "Force plot")

    # ------------------------------------------------------------------
    # Waterfall Plot
    # ------------------------------------------------------------------

    def plot_waterfall(
        self,
        obs: np.ndarray,
        output_idx: int = 0,
        max_display: int = 15,
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> None:
        """
        Waterfall plot: base value에서 출발해 SHAP 기여도를 누적하는 bridge chart.

        force_plot보다 읽기 쉽고 논문/보고서에 적합.

        Args:
            obs         : 설명할 단일 관측값 (n_features,)
            output_idx  : multi-output 시 설명할 자산(행동) 인덱스
            max_display : 표시할 최대 특성 수 (나머지는 "others"로 합산)
            save_path   : 저장 경로
            show        : plt.show() 호출 여부
        """
        self._check_fitted()
        shap_values = self._explainer.shap_values(obs.reshape(1, -1), silent=True)
        sv, base = self._extract_output(shap_values, output_idx, single=True)
        sv = sv.flatten()

        # 상위 max_display-1개 + others
        n_feat = len(sv)
        n_show = min(max_display - 1, n_feat)
        top_idx = np.argsort(np.abs(sv))[-n_show:][::-1]  # 중요도 내림차순
        others_val = float(np.sum(sv) - np.sum(sv[top_idx]))

        names  = [self.feature_names[int(i)] for i in top_idx]
        values = sv[top_idx].tolist()
        if abs(others_val) > 1e-10:
            names.append(f"others ({n_feat - n_show} feats)")
            values.append(others_val)

        # Bridge (waterfall) 계산
        running = base
        lefts, widths, colors_list = [], [], []
        for v in values:
            lefts.append(min(running, running + v))
            widths.append(abs(v))
            colors_list.append("#e74c3c" if v > 0 else "#3498db")
            running += v
        final = running

        fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.46 + 1.5)))
        bars = ax.barh(names, widths, left=lefts, color=colors_list,
                       edgecolor="white", height=0.72)

        # base / final 점선
        ax.axvline(base,  color="gray",  linewidth=1, linestyle="--", label=f"base={base:.4f}")
        ax.axvline(final, color="black", linewidth=1.2, linestyle="-", label=f"output={final:.4f}")

        ax.set_xlabel("Model Output Value", fontsize=10)
        ax.set_title(
            f"SHAP Waterfall Plot  (asset_idx={output_idx})",
            fontsize=12, pad=10,
        )
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="x", alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        self._save_and_show(save_path, show, "Waterfall plot")

    # ------------------------------------------------------------------
    # 편의 메서드: 3종 플롯 한 번에 생성
    # ------------------------------------------------------------------

    def plot_all(
        self,
        obs: np.ndarray,
        background_obs: np.ndarray,
        save_dir: str = "outputs/shap",
        output_idx: int = 0,
        show: bool = False,
    ) -> None:
        """Summary / Force / Waterfall plot을 한 번에 생성·저장한다."""
        d = Path(save_dir)
        self.plot_summary(background_obs, save_path=str(d / "summary_plot.png"), show=show)
        self.plot_force(obs, output_idx=output_idx, save_path=str(d / "force_plot.png"), show=show)
        self.plot_waterfall(obs, output_idx=output_idx, save_path=str(d / "waterfall_plot.png"), show=show)

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _to_array(self, shap_values) -> np.ndarray:
        """
        shap_values를 항상 (n_outputs, n_samples, n_features) 3D ndarray로 정규화.

        KernelExplainer는 버전에 따라 두 형식 중 하나를 반환한다:
          - list of (n_samples, n_features) arrays → stack → (n_outputs, n_samples, n_features)
          - ndarray (n_samples, n_features)        → unsqueeze → (1, n_samples, n_features)
          - ndarray (n_outputs, n_samples, n_features) → 그대로
        np.array(list) 대신 np.stack을 써서 object-array 생성을 방지한다.
        """
        if isinstance(shap_values, list):
            arr = np.stack([np.asarray(sv, dtype=float) for sv in shap_values], axis=0)
        else:
            arr = np.asarray(shap_values, dtype=float)

        n_features = len(self.feature_names)
        if arr.ndim == 2:
            arr = arr[np.newaxis]        # (n_samples, n_features) → (1, n_samples, n_features)
        elif arr.ndim == 3 and arr.shape[1] == n_features:
            arr = np.moveaxis(arr, -1, 0)  # (samples, features, outputs) -> (outputs, samples, features)
        elif arr.ndim != 3:
            arr = arr.reshape(1, 1, -1)
        return arr

    def _extract_output(
        self, shap_values, output_idx: int, single: bool
    ) -> Tuple[np.ndarray, float]:
        """지정 output_idx의 SHAP 값(sv)과 base value를 추출한다."""
        arr = self._to_array(shap_values)  # (n_outputs, n_samples, n_features)
        idx = min(output_idx, arr.shape[0] - 1)

        sv = arr[idx, 0] if single else arr[idx]  # single → (n_features,)

        ev = self._explainer.expected_value
        if hasattr(ev, "__len__") and len(ev) > idx:
            base = float(ev[idx])
        elif hasattr(ev, "__len__"):
            base = float(np.mean(ev))
        else:
            base = float(ev)

        return sv, base

    def _check_fitted(self) -> None:
        if self._explainer is None:
            raise RuntimeError("fit()을 먼저 호출하세요.")

    @staticmethod
    def _save_and_show(save_path: Optional[str], show: bool, label: str) -> None:
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[SHAP] {label} 저장: {save_path}")
        if show:
            plt.show()
        plt.close("all")
