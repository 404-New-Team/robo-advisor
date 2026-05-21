from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from api_client import backtest, strategy_backtests
from reference_data import STRATEGY_LABELS
from ui import COLOR_SEQUENCE, configure_page, load_api_data, percent_dataframe, render_sidebar, strategy_comparison_from_results


configure_page("ANOVA 검증")

state = render_sidebar()

st.title("ANOVA 성과 검증")

strategy = st.selectbox("전략", list(STRATEGY_LABELS.keys()), format_func=lambda value: STRATEGY_LABELS[value])
result = load_api_data("백테스트", backtest, state["active_tickers"], strategy)
metrics = result["metrics"]

cols = st.columns(4)
cols[0].metric("누적수익률", f"{metrics['total_return'] * 100:.1f}%")
cols[1].metric("Sharpe", f"{metrics['sharpe_ratio']:.2f}")
cols[2].metric("MDD", f"{metrics['max_drawdown'] * 100:.1f}%", delta_color="inverse")
cols[3].metric("승률", f"{metrics['win_rate'] * 100:.1f}%")

comparison_results = load_api_data("전략 비교", strategy_backtests, state["active_tickers"], list(STRATEGY_LABELS.keys()))
comparison_df = strategy_comparison_from_results(comparison_results)

left, right = st.columns([1, 1])
with left:
    st.subheader("전략 비교")
    comparison_fig = px.bar(
        comparison_df,
        x="전략",
        y=["누적수익률", "Sharpe", "승률"],
        barmode="group",
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    comparison_fig.update_layout(xaxis_title="", yaxis_title="", legend_title_text="", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(comparison_fig, use_container_width=True, key="anova_strategy_comparison_chart")
with right:
    st.subheader("Walk-Forward")
    walk_df = pd.DataFrame(result["walk_forward_results"], columns=["period", "return", "sharpe"])
    walk_fig = px.bar(walk_df, x="period", y="return", color="sharpe", color_continuous_scale="Blues")
    walk_fig.update_layout(xaxis_title="기간", yaxis_title="수익률", coloraxis_colorbar_title="Sharpe", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(walk_fig, use_container_width=True, key="anova_walk_forward_chart")

st.subheader("통계 검정")
st.info("현재 백엔드 공개 API는 /backtest까지 제공하므로 ANOVA p-value와 효과 크기는 API 추가 후 연결합니다.")

st.subheader("전략 비교 원본")
st.dataframe(percent_dataframe(comparison_df, ["누적수익률", "MDD", "승률"]), use_container_width=True, hide_index=True)
