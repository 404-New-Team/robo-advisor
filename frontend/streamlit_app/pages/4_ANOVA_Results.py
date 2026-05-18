from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from api_client import backtest
from mock_data import STRATEGY_LABELS, get_anova_result, get_regime_table, get_strategy_comparison
from ui import COLOR_SEQUENCE, configure_page, percent_dataframe, render_sidebar


configure_page("ANOVA 검증")

state = render_sidebar()

st.title("ANOVA 성과 검증")

strategy = st.selectbox("전략", list(STRATEGY_LABELS.keys()), format_func=lambda value: STRATEGY_LABELS[value])
result = backtest(state["selected_tickers"], strategy)
metrics = result["metrics"]

cols = st.columns(4)
cols[0].metric("누적수익률", f"{metrics['total_return'] * 100:.1f}%")
cols[1].metric("Sharpe", f"{metrics['sharpe_ratio']:.2f}")
cols[2].metric("MDD", f"{metrics['max_drawdown'] * 100:.1f}%", delta_color="inverse")
cols[3].metric("승률", f"{metrics['win_rate'] * 100:.1f}%")

comparison_df = get_strategy_comparison()
anova = get_anova_result()

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
    st.plotly_chart(comparison_fig, use_container_width=True)
with right:
    st.subheader("Walk-Forward")
    walk_df = pd.DataFrame(result["walk_forward_results"])
    walk_fig = px.bar(walk_df, x="period", y="return", color="sharpe", color_continuous_scale="Blues")
    walk_fig.update_layout(xaxis_title="기간", yaxis_title="수익률", coloraxis_colorbar_title="Sharpe", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(walk_fig, use_container_width=True)

st.subheader("통계 검정")
anova_cols = st.columns(3)
anova_cols[0].metric("p-value", f"{anova['p_value']:.3f}")
anova_cols[1].metric("효과 크기 η²", f"{anova['eta_squared']:.2f}")
anova_cols[2].metric("F-statistic", f"{anova['f_statistic']:.2f}")
st.success(anova["conclusion"])

st.subheader("시장 국면별 결과")
regime_df = get_regime_table()
st.dataframe(percent_dataframe(regime_df, ["DRL", "MVO", "동일가중"]), use_container_width=True, hide_index=True)
