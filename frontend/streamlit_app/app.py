import streamlit as st

from api_client import backtest, health, optimize_portfolio, research
from reference_data import get_weight_table
from ui import allocation_chart, configure_page, load_api_data, performance_chart, render_metric_row, render_sidebar, walk_forward_performance_frame


configure_page("통합 대시보드")

state = render_sidebar()
health_state = load_api_data("상태 확인", health)
optimize_result = load_api_data(
    "포트폴리오 최적화",
    optimize_portfolio,
    risk_level=state["risk_level"],
    tickers=state["selected_tickers"],
    excluded=state["excluded_tickers"],
)
research_result = load_api_data("리서치", research, "현재 포트폴리오 리스크 요약", max_results=3)
backtest_result = load_api_data("백테스트", backtest, state["active_tickers"], "drl")
weight_df = get_weight_table(optimize_result["weights"])

st.title("Robby 통합 관제 대시보드")
st.caption(f"모드: {health_state['mode']} · 모델 로드: {'완료' if health_state['model_loaded'] else '대기'}")

render_metric_row(optimize_result["metrics"])

left, right = st.columns([1.05, 1])
with left:
    st.subheader("추천 포트폴리오")
    st.plotly_chart(allocation_chart(weight_df), use_container_width=True, key="dashboard_allocation_chart")
with right:
    st.subheader("Walk-Forward 성과")
    st.plotly_chart(
        performance_chart(walk_forward_performance_frame([backtest_result])),
        use_container_width=True,
        key="dashboard_walk_forward_chart",
    )

tab_summary, tab_research, tab_simulation = st.tabs(["포트폴리오", "리서치 근거", "시뮬레이션"])

with tab_summary:
    st.dataframe(
        weight_df,
        use_container_width=True,
        hide_index=True,
        column_config={"비중": st.column_config.ProgressColumn("비중", min_value=0, max_value=1, format="%.1f%%")},
    )
    metric_cols = st.columns(4)
    metrics = backtest_result["metrics"]
    metric_cols[0].metric("백테스트 누적수익률", f"{metrics['total_return'] * 100:.1f}%")
    metric_cols[1].metric("Sortino", f"{metrics['sortino_ratio']:.2f}")
    metric_cols[2].metric("Calmar", f"{metrics['calmar_ratio']:.2f}")
    metric_cols[3].metric("승률", f"{metrics['win_rate'] * 100:.1f}%")

with tab_research:
    st.write(research_result["summary"])
    for index, step in enumerate(research_result["reasoning_trace"], start=1):
        st.text(f"{index}. {step}")

with tab_simulation:
    st.info("미래 경로 시뮬레이션 API가 아직 없어 백테스트 Walk-Forward 결과를 표시합니다.")
    st.plotly_chart(
        performance_chart(walk_forward_performance_frame([backtest_result])),
        use_container_width=True,
        key="dashboard_simulation_walk_forward_chart",
    )
