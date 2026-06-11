from pathlib import Path
import sys

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from api_client import backtest, optimize_portfolio
from reference_data import get_order_preview, get_weight_table
from ui import allocation_chart, configure_page, format_money, load_api_data, performance_chart, render_metric_row, render_sidebar, strategy_comparison_from_results, walk_forward_performance_frame


configure_page("포트폴리오")

state = render_sidebar()
result = load_api_data(
    "포트폴리오 최적화",
    optimize_portfolio,
    risk_level=state["risk_level"],
    tickers=state["selected_tickers"],
    excluded=state["excluded_tickers"],
    token=state["access_token"],
)
_BT_START, _BT_END = "2017-01-01", "2025-12-31"
backtest_result = load_api_data(
    "과거 성과", backtest, state["active_tickers"], "drl",
    token=state["access_token"], start_date=_BT_START, end_date=_BT_END,
)
backtest_results_all = [backtest_result]
weights = result["weights"]
weight_df = get_weight_table(weights)

st.title("포트폴리오 구성")
st.caption(f"{state['risk_label']} · 투자 가능 금액 {format_money(state['investment_amount'])}")

render_metric_row(result["metrics"])

left, right = st.columns([1, 1.15])
with left:
    st.subheader("추천 투자 비율")
    st.plotly_chart(allocation_chart(weight_df), use_container_width=True, key="portfolio_allocation_chart")
with right:
    st.subheader("투자 비율 직접 조정")
    editor_df = weight_df[["티커", "종목", "섹터", "비중"]].copy()
    editor_df["비중"] = (editor_df["비중"] * 100).round(1)
    edited_df = st.data_editor(
        editor_df,
        use_container_width=True,
        hide_index=True,
        disabled=["티커", "종목", "섹터"],
        column_config={
            "비중": st.column_config.NumberColumn("비중(%)", min_value=0.0, max_value=60.0, step=0.5, format="%.1f")
        },
    )
    total_weight = edited_df["비중"].sum()

adjusted_weights = {
    row["티커"]: row["비중"] / max(total_weight, 1)
    for _, row in edited_df.iterrows()
}
order_df = get_order_preview(adjusted_weights, state["investment_amount"])
order_display_df = order_df.copy()
order_display_df["목표 비중"] = (order_display_df["목표 비중"] * 100).round(1)
order_display_df["매수 금액"] = order_display_df["매수 금액"].map(format_money)

st.subheader("얼마씩 살지 미리보기")
st.dataframe(
    order_display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "목표 비중": st.column_config.ProgressColumn("목표 비중", min_value=0, max_value=100, format="%.1f%%"),
        "매수 금액": st.column_config.TextColumn("매수 금액"),
    },
)

st.subheader("과거 기간별 수익 흐름")
st.caption(f"기간: {_BT_START} ~ {_BT_END}")
strategy_cache_key = f"portfolio_strategy_backtests:{','.join(state['active_tickers'])}:{_BT_START}:{_BT_END}"
if st.button("다른 방식과 비교하기", use_container_width=True, key="portfolio_strategy_backtest_run"):
    strategy_results = [backtest_result]
    for _strategy in ("mvo", "equal_weight"):
        try:
            strategy_results.append(backtest(
                state["active_tickers"], _strategy,
                token=state["access_token"], start_date=_BT_START, end_date=_BT_END,
            ))
        except Exception as _e:
            st.warning(f"{_strategy} 과거 성과 계산 실패: {_e}")
    st.session_state[strategy_cache_key] = strategy_results
backtest_results_all = st.session_state.get(strategy_cache_key, backtest_results_all)
st.plotly_chart(
    performance_chart(walk_forward_performance_frame(backtest_results_all)),
    use_container_width=True,
    key="portfolio_walk_forward_chart",
)

cmp_df = strategy_comparison_from_results(backtest_results_all)
if not cmp_df.empty:
    st.markdown("**투자 방식별 성과 요약**")
    st.dataframe(
        cmp_df.style.format({
            "총 수익": "{:.1%}",
            "위험 대비 수익": "{:.3f}",
            "최대 하락폭": "{:.1%}",
            "수익 난 기간 비율": "{:.1%}",
        }),
        use_container_width=True,
        hide_index=True,
    )

comparison_df = pd.DataFrame(
    [
        {"항목": "최초 추천", "예상 수익": result["metrics"]["expected_return"], "최대 하락폭": result["metrics"]["max_drawdown"]},
        {
            "항목": "직접 조정",
            "예상 수익": result["metrics"]["expected_return"] - 0.006 + abs(total_weight - 100) * -0.0002,
            "최대 하락폭": result["metrics"]["max_drawdown"] - 0.004,
        },
    ]
)
st.dataframe(
    comparison_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "예상 수익": st.column_config.NumberColumn("예상 수익", format="%.3f"),
        "최대 하락폭": st.column_config.NumberColumn("최대 하락폭", format="%.3f"),
    },
)
