from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from api_client import anova
from ui import configure_page, load_api_data, render_sidebar

configure_page("투자 방식 비교")

state = render_sidebar()

st.title("투자 방식 비교")
st.caption(
    "여러 투자 방식의 과거 성과 차이가 우연인지, 의미 있는 차이인지 확인합니다.  \n"
    "**비교 기간**: 2017-01-01 ~ 2025-12-31"
)

# ─── 실행 패널 ────────────────────────────────────────────────────────────────
with st.expander("고급 설정", expanded=False):
    col_a, col_b = st.columns(2)
    alpha = col_a.number_input("차이를 인정할 기준", min_value=0.01, max_value=0.20, value=0.05, step=0.01)
    n_ep = col_b.number_input("반복 실험 횟수", min_value=5, max_value=50, value=20, step=5)

run_btn = st.button("투자 방식 비교 실행", type="primary", use_container_width=True)

if run_btn:
    result = load_api_data(
        "투자 방식 비교",
        anova,
        tickers=state["active_tickers"],
        alpha=float(alpha),
        n_episodes_reward=int(n_ep),
        token=state["access_token"],
        start_date="2017-01-01",
        end_date="2025-12-31",
    )
    st.session_state["anova_result"] = result
    st.toast("투자 방식 비교가 완료됐습니다.")

result = st.session_state.get("anova_result")
if result is None:
    st.info("'투자 방식 비교 실행' 버튼을 눌러 분석을 시작하세요.")
    st.stop()


# ─── 유틸 ─────────────────────────────────────────────────────────────────────
def _fmt_p(p: float | None) -> str:
    if p is None or p != p:
        return "N/A"
    return f"{p:.6f}"


def _friendly_label(value: str) -> str:
    labels = {
        "DRL": "AI 추천",
        "drl": "AI 추천",
        "MVO": "수익/위험 균형",
        "mvo": "수익/위험 균형",
        "EqualWeight": "같은 비율",
        "equal_weight": "같은 비율",
        "EW": "같은 비율",
        "Bull": "상승장",
        "Sideways": "횡보장",
        "Bear": "하락장",
        "R1_LOGRET": "수익 중심",
        "R2_SHARPE": "위험 대비 수익 중심",
        "R3_FULL": "수익과 위험 함께 고려",
    }
    return labels.get(str(value), str(value))


def _render_oneways(data: dict, title: str, group_label: str) -> None:
    st.subheader(title)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("차이 크기", f"{data['f_statistic']:.4f}")
    c2.metric("우연일 가능성", _fmt_p(data.get("p_value")))
    c3.metric("영향 정도", f"{data['eta_squared']:.4f}")
    c4.metric("판정", "차이 있음 ✅" if data["significant"] else "뚜렷한 차이 없음")

    st.caption(f"영향 정도 해석: **{data['eta_squared_interp']}**")

    # 그룹 통계 표
    groups = list(data["group_means"].keys())
    group_df = pd.DataFrame({
        group_label: [_friendly_label(g) for g in groups],
        "평균": [round(data["group_means"][g], 4) for g in groups],
        "흔들림": [round(data["group_stds"][g], 4) for g in groups],
        "비교 횟수": [data["group_ns"][g] for g in groups],
    })
    st.dataframe(group_df, use_container_width=True, hide_index=True)

    # 그룹 평균 막대 차트
    fig = px.bar(
        group_df,
        x=group_label,
        y="평균",
        error_y="흔들림",
        color=group_label,
        title=f"{group_label}별 평균",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Tukey HSD
    tukey = data.get("tukey_results", [])
    if tukey and data["significant"]:
        st.markdown("**어떤 그룹끼리 다른지 확인**")
        tukey_df = pd.DataFrame([
            {
                "그룹 1": _friendly_label(t["group1"]),
                "그룹 2": _friendly_label(t["group2"]),
                "평균 차이": round(t["mean_diff"], 4),
                "차이 크기": round(t["q_statistic"], 4),
                "우연일 가능성": _fmt_p(t.get("p_value_approx")),
                "차이 있음": "✅" if t["significant"] else "❌",
            }
            for t in tukey
        ])
        st.dataframe(tukey_df, use_container_width=True, hide_index=True)
    elif tukey:
        st.caption("뚜렷한 차이가 없어 추가 비교를 생략했습니다.")


# ─── 검증 1 ────────────────────────────────────────────────────────────────────
st.divider()
with st.container():
    st.info(
        "**비교 1**: AI가 학습할 때 쓰는 점수 계산 방식에 따라 결과가 달라지는지 확인합니다."
    )
    _render_oneways(
        result["verification1_reward"],
        "비교 1 — AI 학습 점수 방식별 결과",
        "점수 방식",
    )

# ─── 검증 2 ────────────────────────────────────────────────────────────────────
st.divider()
with st.container():
    st.info(
        "**비교 2**: AI 추천, 수익/위험 균형, 같은 비율 투자 방식의 과거 성과가 실제로 다른지 확인합니다."
    )
    _render_oneways(
        result["verification2_strategy"],
        "비교 2 — 투자 방식별 과거 성과",
        "투자 방식",
    )

# ─── 검증 3 ────────────────────────────────────────────────────────────────────
st.divider()
v3 = result.get("verification3_regime", {})

st.subheader("비교 3 — 시장 상황별 투자 방식 비교")
st.info(
    "**비교 3**: 상승장, 횡보장, 하락장에서 어떤 투자 방식이 더 잘 버티는지 확인합니다."
)

if "error" in v3:
    st.warning(f"검증 3 계산 실패: {v3['error']}")
else:
    # 주효과/상호작용 요약 테이블
    effects = {
        "투자 방식": v3["strategy_effect"],
        "시장 상황": v3["regime_effect"],
        "투자 방식 × 시장 상황": v3["interaction_effect"],
    }
    eff_rows = []
    for name, eff in effects.items():
        eff_rows.append({
            "요인": name,
            "차이 크기": round(eff["f_statistic"], 4),
            "우연일 가능성": _fmt_p(eff.get("p_value")),
            "영향 정도": round(eff["eta_sq_partial"], 4),
            "차이 있음": "✅" if eff["significant"] else "❌",
        })
    st.dataframe(pd.DataFrame(eff_rows), use_container_width=True, hide_index=True)

    col_l, col_r = st.columns(2)

    with col_l:
        # 전략별 평균 CAGR
        st.markdown("**투자 방식별 평균 수익**")
        strat_df = pd.DataFrame([
            {"투자 방식": _friendly_label(k), "평균 수익": round(v, 4)}
            for k, v in v3.get("strategy_means", {}).items()
        ])
        if not strat_df.empty:
            fig_s = px.bar(strat_df, x="투자 방식", y="평균 수익", color="투자 방식",
                           color_discrete_sequence=px.colors.qualitative.Set2)
            fig_s.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_s, use_container_width=True)

    with col_r:
        # 국면별 평균 CAGR
        st.markdown("**시장 상황별 평균 수익**")
        regime_df = pd.DataFrame([
            {"시장 상황": _friendly_label(k), "평균 수익": round(v, 4), "비교 횟수": v3.get("regime_counts", {}).get(k, 0)}
            for k, v in v3.get("regime_means", {}).items()
        ])
        if not regime_df.empty:
            color_map = {"상승장": "#2ecc71", "횡보장": "#f39c12", "하락장": "#e74c3c"}
            fig_r = px.bar(
                regime_df, x="시장 상황", y="평균 수익", color="시장 상황",
                color_discrete_map=color_map,
                text="비교 횟수",
            )
            fig_r.update_traces(texttemplate="n=%{text}", textposition="outside")
            fig_r.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_r, use_container_width=True)

    # 셀 평균 히트맵 (전략 × 국면)
    cell_means = v3.get("cell_means", {})
    cell_ns = v3.get("cell_ns", {})
    if cell_means:
        st.markdown("**투자 방식과 시장 상황을 함께 본 평균 수익**")
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
                x=[_friendly_label(r) for r in regimes_seen],
                y=[_friendly_label(s) for s in strategies_seen],
                text=text,
                texttemplate="%{text}",
                colorscale="RdYlGn",
                colorbar_title="평균 수익",
            ))
            fig_heat.update_layout(
                xaxis_title="시장 상황",
                yaxis_title="투자 방식",
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        except Exception:
            st.dataframe(pd.DataFrame(cell_means, index=[0]), use_container_width=True)

    # ANOVA 원본 테이블
    anova_table = v3.get("anova_table", [])
    if anova_table:
        with st.expander("통계 계산 원본 보기", expanded=False):
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
