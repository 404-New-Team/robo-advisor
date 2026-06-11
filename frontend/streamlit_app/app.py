import streamlit as st

from api_client import allocation, backtest, health, optimize_portfolio, research
from reference_data import get_asset_name, get_weight_table
from ui import allocation_chart, configure_page, load_api_data, performance_chart, render_allocation_table, render_metric_row, render_sidebar, walk_forward_performance_frame


configure_page("통합 대시보드")

state = render_sidebar()
health_state = load_api_data("상태 확인", health)
optimize_result = load_api_data(
    "포트폴리오 최적화",
    optimize_portfolio,
    risk_level=state["risk_level"],
    tickers=state["selected_tickers"],
    excluded=state["excluded_tickers"],
    token=state["access_token"],
)
st.session_state["latest_portfolio_weights"] = optimize_result["weights"]
portfolio_context = {
    "risk_level": state["risk_level"],
    "investment_amount": state["investment_amount"],
    "selected_tickers": state["selected_tickers"],
    "excluded_tickers": state["excluded_tickers"],
    "active_tickers": state["active_tickers"],
    "weights": optimize_result["weights"],
    "ticker_names": {ticker: get_asset_name(ticker) for ticker in state["active_tickers"]},
}
backtest_result = load_api_data("과거 성과", backtest, state["active_tickers"], "drl", token=state["access_token"])
backtest_results_all = [backtest_result]
allocation_result = load_api_data(
    "주문 수량 계산",
    allocation,
    weights=optimize_result["weights"],
    total_amount=state["investment_amount"],
    token=state["access_token"],
)
weight_df = get_weight_table(optimize_result["weights"])

st.title("Robby 통합 관제 대시보드")
st.caption(f"모드: {health_state['mode']} · 모델 로드: {'완료' if health_state['model_loaded'] else '대기'}")

render_metric_row(optimize_result["metrics"])

left, right = st.columns([1.05, 1])
with left:
    st.subheader("추천 포트폴리오")
    st.plotly_chart(allocation_chart(weight_df), use_container_width=True, key="dashboard_allocation_chart")
with right:
    st.subheader("과거 기간별 수익 흐름")
    st.plotly_chart(
        performance_chart(walk_forward_performance_frame(backtest_results_all)),
        use_container_width=True,
        key="dashboard_walk_forward_chart",
    )

tab_summary, tab_research, tab_simulation = st.tabs(["내 투자 구성", "뉴스 분석 근거", "예상 흐름"])

with tab_summary:
    display_weight_df = weight_df.copy()
    display_weight_df["비중"] = (display_weight_df["비중"] * 100).round(1)
    st.dataframe(
        display_weight_df,
        use_container_width=True,
        hide_index=True,
        column_config={"비중": st.column_config.ProgressColumn("비중", min_value=0, max_value=100, format="%.1f%%")},
    )
    metric_cols = st.columns(4)
    metrics = backtest_result["metrics"]
    metric_cols[0].metric("과거 기준 총 수익", f"{metrics['total_return'] * 100:.1f}%")
    metric_cols[1].metric("하락 위험 대비 수익", f"{metrics['sortino_ratio']:.2f}")
    metric_cols[2].metric("하락폭 대비 회복력", f"{metrics['calmar_ratio']:.2f}")
    metric_cols[3].metric("수익 난 기간 비율", f"{metrics['win_rate'] * 100:.1f}%")

    st.divider()
    st.subheader("얼마씩 살지 계산")
    render_allocation_table(allocation_result)

with tab_research:
    research_key = f"dashboard_research_result:{','.join(state['active_tickers'])}:{state['risk_level']}"
    if st.button("뉴스 분석 실행", type="primary", use_container_width=True, key="dashboard_research_run"):
        st.session_state[research_key] = load_api_data(
            "뉴스 분석",
            research,
            tickers=state["active_tickers"],
            max_results=3,
            token=state["access_token"],
            portfolio_context=portfolio_context,
        )
    research_result = st.session_state.get(research_key)
    if research_result is None:
        st.info("뉴스 분석 실행 버튼을 눌러 현재 투자 구성 기준 분석을 시작하세요.")
    else:
        st.write(research_result["summary"])
        for index, step in enumerate(research_result["reasoning_trace"], start=1):
            st.text(f"{index}. {step}")

with tab_simulation:
    st.info("아직 미래 예측 기능은 없어 과거 기간별 수익 흐름을 표시합니다.")
    st.plotly_chart(
        performance_chart(walk_forward_performance_frame(backtest_results_all)),
        use_container_width=True,
        key="dashboard_simulation_walk_forward_chart",
    )
