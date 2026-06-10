from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from api_client import anova
from ui import configure_page, load_api_data, render_sidebar

configure_page("ANOVA 검증")

state = render_sidebar()

st.title("ANOVA 성과 검증")
st.caption(
    "검증 1 · 2 (One-way ANOVA) 및 검증 3 (Two-way ANOVA) 결과를 표시합니다.  \n"
    "**실험 기간**: 2017-01-01 ~ 2025-12-31  |  **폴드 구성**: train=24mo / test=6mo / step=6mo  |  "
    "**최적 하이퍼파라미터**: lr=1e-4, λ=0.5, window=30, n_seeds=5"
)

# ─── 실행 패널 ────────────────────────────────────────────────────────────────
with st.expander("실행 설정", expanded=False):
    col_a, col_b = st.columns(2)
    alpha = col_a.number_input("유의 수준 α", min_value=0.01, max_value=0.20, value=0.05, step=0.01)
    n_ep = col_b.number_input("보상 변형 에피소드 수", min_value=5, max_value=50, value=20, step=5)

run_btn = st.button("ANOVA 검증 실행", type="primary", use_container_width=True)

if run_btn:
    result = load_api_data(
        "ANOVA 검증",
        anova,
        tickers=state["active_tickers"],
        alpha=float(alpha),
        n_episodes_reward=int(n_ep),
        token=state["access_token"],
        start_date="2017-01-01",
        end_date="2025-12-31",
    )
    st.session_state["anova_result"] = result
    st.toast("ANOVA 검증이 완료됐습니다.")

result = st.session_state.get("anova_result")
if result is None:
    st.info("'ANOVA 검증 실행' 버튼을 눌러 분석을 시작하세요.")
    st.stop()


# ─── 유틸 ─────────────────────────────────────────────────────────────────────
def _fmt_p(p: float | None) -> str:
    if p is None or p != p:
        return "N/A"
    return f"{p:.6f}"


def _render_oneways(data: dict, title: str, group_label: str) -> None:
    """One-way ANOVA 결과 렌더링."""
    st.subheader(title)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("F-통계량", f"{data['f_statistic']:.4f}")
    c2.metric("p-value", _fmt_p(data.get("p_value")))
    c3.metric("η² 효과 크기", f"{data['eta_squared']:.4f}")
    c4.metric("판정", "유의 ✅" if data["significant"] else "비유의 ❌")

    st.caption(f"효과 크기 해석: **{data['eta_squared_interp']}**")

    # 그룹 통계 표
    groups = list(data["group_means"].keys())
    group_df = pd.DataFrame({
        group_label: groups,
        "평균": [round(data["group_means"][g], 4) for g in groups],
        "표준편차": [round(data["group_stds"][g], 4) for g in groups],
        "N": [data["group_ns"][g] for g in groups],
    })
    st.dataframe(group_df, use_container_width=True, hide_index=True)

    # 그룹 평균 막대 차트
    fig = px.bar(
        group_df,
        x=group_label,
        y="평균",
        error_y="표준편차",
        color=group_label,
        title=f"{group_label}별 평균 (±1 SD)",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Tukey HSD
    tukey = data.get("tukey_results", [])
    if tukey and data["significant"]:
        st.markdown("**Tukey HSD 사후 검정**")
        tukey_df = pd.DataFrame([
            {
                "그룹 1": t["group1"],
                "그룹 2": t["group2"],
                "평균 차이": round(t["mean_diff"], 4),
                "q-통계량": round(t["q_statistic"], 4),
                "p-근사값": _fmt_p(t.get("p_value_approx")),
                "유의": "✅" if t["significant"] else "❌",
            }
            for t in tukey
        ])
        st.dataframe(tukey_df, use_container_width=True, hide_index=True)
    elif tukey:
        st.caption("p ≥ α이므로 사후 검정(Tukey HSD) 생략.")


# ─── 검증 1 ────────────────────────────────────────────────────────────────────
st.divider()
with st.container():
    st.info(
        "**검증 1**: 보상 함수 변형(R1_LOGRET · R2_SHARPE · R3_FULL) 간 에피소드 보상 분포에 "
        "통계적으로 유의미한 차이가 있는가?  \n"
        "→ 유의하다면 보상 함수 설계가 에이전트 학습에 실질적 영향을 줌을 의미합니다."
    )
    _render_oneways(
        result["verification1_reward"],
        "검증 1 — 보상 함수 변형별 성과 비교 (One-way ANOVA)",
        "보상 함수 변형",
    )

# ─── 검증 2 ────────────────────────────────────────────────────────────────────
st.divider()
with st.container():
    st.info(
        "**검증 2**: Walk-Forward 폴드 CAGR 기준으로 DRL · MVO · 동일가중 세 전략 간 "
        "성과에 통계적으로 유의미한 차이가 있는가?  \n"
        "→ 유의하다면 DRL이 단순 벤치마크 대비 의미 있는 성과 우위를 가짐을 의미합니다."
    )
    _render_oneways(
        result["verification2_strategy"],
        "검증 2 — DRL vs MVO vs 동일가중 전략 비교 (One-way ANOVA)",
        "전략",
    )

# ─── 검증 3 ────────────────────────────────────────────────────────────────────
st.divider()
v3 = result.get("verification3_regime", {})

st.subheader("검증 3 — 시장 국면별 전략 성과 비교 (Two-way ANOVA)")
st.info(
    "**검증 3**: 전략(DRL/MVO/EW)과 시장 국면(Bull/Sideways/Bear)의 주효과 및 상호작용이 "
    "폴드 CAGR에 유의미한 영향을 주는가?  \n"
    "→ 상호작용 효과가 유의하다면 DRL이 특정 시장 국면에서 차별적 강점을 가짐을 의미합니다."
)

if "error" in v3:
    st.warning(f"검증 3 계산 실패: {v3['error']}")
else:
    # 주효과/상호작용 요약 테이블
    effects = {
        "전략 (Strategy)": v3["strategy_effect"],
        "시장 국면 (Regime)": v3["regime_effect"],
        "상호작용 (Interaction)": v3["interaction_effect"],
    }
    eff_rows = []
    for name, eff in effects.items():
        eff_rows.append({
            "요인": name,
            "F-통계량": round(eff["f_statistic"], 4),
            "p-value": _fmt_p(eff.get("p_value")),
            "Partial η²": round(eff["eta_sq_partial"], 4),
            "유의": "✅" if eff["significant"] else "❌",
        })
    st.dataframe(pd.DataFrame(eff_rows), use_container_width=True, hide_index=True)

    col_l, col_r = st.columns(2)

    with col_l:
        # 전략별 평균 CAGR
        st.markdown("**전략별 평균 CAGR**")
        strat_df = pd.DataFrame([
            {"전략": k, "평균 CAGR": round(v, 4)}
            for k, v in v3.get("strategy_means", {}).items()
        ])
        if not strat_df.empty:
            fig_s = px.bar(strat_df, x="전략", y="평균 CAGR", color="전략",
                           color_discrete_sequence=px.colors.qualitative.Set2)
            fig_s.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_s, use_container_width=True)

    with col_r:
        # 국면별 평균 CAGR
        st.markdown("**시장 국면별 평균 CAGR**")
        regime_df = pd.DataFrame([
            {"국면": k, "평균 CAGR": round(v, 4), "폴드 수": v3.get("regime_counts", {}).get(k, 0)}
            for k, v in v3.get("regime_means", {}).items()
        ])
        if not regime_df.empty:
            color_map = {"Bull": "#2ecc71", "Sideways": "#f39c12", "Bear": "#e74c3c"}
            fig_r = px.bar(
                regime_df, x="국면", y="평균 CAGR", color="국면",
                color_discrete_map=color_map,
                text="폴드 수",
            )
            fig_r.update_traces(texttemplate="n=%{text}", textposition="outside")
            fig_r.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_r, use_container_width=True)

    # 셀 평균 히트맵 (전략 × 국면)
    cell_means = v3.get("cell_means", {})
    cell_ns = v3.get("cell_ns", {})
    if cell_means:
        st.markdown("**셀 평균 CAGR (전략 × 시장 국면)**")
        try:
            import ast
            strategies_seen = sorted({ast.literal_eval(k)[0] for k in cell_means})
            regimes_seen = sorted({ast.literal_eval(k)[1] for k in cell_means})
            z = []
            text = []
            for s in strategies_seen:
                row_z = []
                row_text = []
                for r in regimes_seen:
                    key = str((s, r))
                    val = cell_means.get(key)
                    n = cell_ns.get(key, 0)
                    row_z.append(val if val is not None else float("nan"))
                    row_text.append(f"{val:.3f}<br>(n={n})" if val is not None else "N/A")
                z.append(row_z)
                text.append(row_text)

            fig_heat = go.Figure(data=go.Heatmap(
                z=z,
                x=regimes_seen,
                y=strategies_seen,
                text=text,
                texttemplate="%{text}",
                colorscale="RdYlGn",
                colorbar_title="평균 CAGR",
            ))
            fig_heat.update_layout(
                xaxis_title="시장 국면",
                yaxis_title="전략",
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        except Exception:
            st.dataframe(pd.DataFrame(cell_means, index=[0]), use_container_width=True)

    # ANOVA 원본 테이블
    anova_table = v3.get("anova_table", [])
    if anova_table:
        with st.expander("ANOVA 원본 테이블", expanded=False):
            table_df = pd.DataFrame([
                {
                    "요인": row.get("source", ""),
                    "SS": round(row.get("SS", 0), 6),
                    "df": row.get("df", ""),
                    "MS": round(row.get("MS", 0), 6) if row.get("MS") is not None else "",
                    "F": round(row.get("F", 0), 4) if row.get("F") is not None else "",
                    "p": round(row.get("p", 1), 6) if row.get("p") is not None else "",
                }
                for row in anova_table
            ])
            st.dataframe(table_df, use_container_width=True, hide_index=True)
