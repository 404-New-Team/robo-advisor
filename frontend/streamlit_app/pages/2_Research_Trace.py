from pathlib import Path
import sys

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from api_client import research
from reference_data import get_universe
from ui import configure_page, load_api_data, render_sidebar


configure_page("리서치 추론")

state = render_sidebar()
universe = get_universe()

st.title("Reasoning Trace")


def _ticker_label(value: str | None) -> str:
    if value is None:
        return "전체 포트폴리오"
    row = universe.loc[universe["ticker"] == value, "name"]
    name = row.iloc[0] if not row.empty else value
    return f"{name} ({value})"


def _latest_weights(active_tickers: list[str]) -> dict[str, float]:
    weights = st.session_state.get("latest_portfolio_weights")
    if not isinstance(weights, dict):
        return {}
    active = set(active_tickers)
    return {
        str(ticker): float(weight)
        for ticker, weight in weights.items()
        if ticker in active and isinstance(weight, int | float)
    }


def _portfolio_context() -> dict:
    active_tickers = state["active_tickers"]
    return {
        "risk_level": state["risk_level"],
        "investment_amount": state["investment_amount"],
        "selected_tickers": state["selected_tickers"],
        "excluded_tickers": state["excluded_tickers"],
        "active_tickers": active_tickers,
        "weights": _latest_weights(active_tickers),
        "ticker_names": {
            ticker: _ticker_label(ticker).rsplit(" (", 1)[0]
            for ticker in active_tickers
        },
    }


cols = st.columns([1, 1, 2])
ticker = cols[0].selectbox(
    "대상 종목",
    [None] + state["active_tickers"],
    format_func=_ticker_label,
)
max_results = cols[1].slider("출처 수", min_value=3, max_value=10, value=5)
submitted = cols[2].button("리서치 실행", type="primary", use_container_width=True)

if submitted:
    research_tickers = state["active_tickers"] if ticker is None else [ticker]
    result = load_api_data(
        "리서치",
        research,
        tickers=research_tickers,
        max_results=max_results,
        token=state["access_token"],
        portfolio_context=_portfolio_context(),
    )
    st.session_state["research_trace_result"] = result
    st.toast("백엔드 리서치 결과를 갱신했습니다.")

result = st.session_state.get("research_trace_result")
if result is None:
    st.info("리서치 실행 버튼을 눌러 현재 포트폴리오 기준 분석을 시작하세요.")
    st.stop()

st.subheader("요약")
st.write(result["summary"])

left, right = st.columns([1, 1])
with left:
    st.subheader("리스크 이벤트")
    risk_df = pd.DataFrame(result["risk_events"], columns=["type", "description", "severity", "detected_at"])
    st.dataframe(risk_df, use_container_width=True, hide_index=True)
with right:
    st.subheader("자체 검증")
    st.metric("Self-Correction", f"{result['self_correction_count']}회")
    trace_log = "\n".join(f"{index:02d}  {step}" for index, step in enumerate(result["reasoning_trace"], start=1))
    with st.expander("검증 로그", expanded=False):
        st.code(trace_log or "검증 로그가 없습니다.", language="text")

st.subheader("출처")
source_df = pd.DataFrame(result["sources"], columns=["title", "url", "published_at", "relevance_score"])
source_df["relevance_score"] = source_df["relevance_score"] * 100
st.dataframe(
    source_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "url": st.column_config.LinkColumn("URL"),
        "relevance_score": st.column_config.ProgressColumn("관련도", min_value=0, max_value=100, format="%.0f%%"),
    },
)
