from pathlib import Path
import sys

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from api_client import optimize_portfolio
from mock_data import get_order_preview, get_simulation_paths, get_weight_table
from ui import allocation_chart, configure_page, format_money, render_metric_row, render_sidebar, simulation_chart


configure_page("포트폴리오")

state = render_sidebar()
result = optimize_portfolio(
    risk_level=state["risk_level"],
    tickers=state["selected_tickers"],
    excluded=state["excluded_tickers"],
)
weights = result["weights"]
weight_df = get_weight_table(weights)

st.title("포트폴리오 구성")
st.caption(f"{state['risk_label']} · 투자 가능 금액 {format_money(state['investment_amount'])}")

render_metric_row(result["metrics"])

left, right = st.columns([1, 1.15])
with left:
    st.subheader("추천 비중")
    st.plotly_chart(allocation_chart(weight_df), use_container_width=True)
with right:
    st.subheader("세부 조정")
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
    if abs(total_weight - 100) > 0.1:
        st.warning(f"현재 합계 {total_weight:.1f}%입니다. 주문 전 100%로 맞춰야 합니다.")
    else:
        st.success("비중 합계가 100%입니다.")

adjusted_weights = {
    row["티커"]: row["비중"] / max(total_weight, 1)
    for _, row in edited_df.iterrows()
}
order_df = get_order_preview(adjusted_weights, state["investment_amount"])

st.subheader("주문 미리보기")
st.dataframe(
    order_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "목표 비중": st.column_config.ProgressColumn("목표 비중", min_value=0, max_value=1, format="%.1f%%"),
        "매수 금액": st.column_config.NumberColumn("매수 금액", format="%d원"),
    },
)

st.subheader("조정 후 자산 가치 시뮬레이션")
st.plotly_chart(simulation_chart(get_simulation_paths()), use_container_width=True)

comparison_df = pd.DataFrame(
    [
        {"항목": "최초 추천", "예상 수익률": result["metrics"]["expected_return"], "MDD": result["metrics"]["max_drawdown"]},
        {
            "항목": "사용자 조정",
            "예상 수익률": result["metrics"]["expected_return"] - 0.006 + abs(total_weight - 100) * -0.0002,
            "MDD": result["metrics"]["max_drawdown"] - 0.004,
        },
    ]
)
st.dataframe(
    comparison_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "예상 수익률": st.column_config.NumberColumn("예상 수익률", format="%.3f"),
        "MDD": st.column_config.NumberColumn("MDD", format="%.3f"),
    },
)
