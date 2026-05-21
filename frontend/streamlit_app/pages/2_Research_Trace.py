from pathlib import Path
import sys

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from api_client import research
from reference_data import get_default_tickers, get_universe
from ui import configure_page, load_api_data, render_sidebar


configure_page("리서치 추론")

state = render_sidebar()
universe = get_universe()

st.title("Reasoning Trace")

query = st.text_area("투자 질문", value="현재 포트폴리오의 주요 리스크와 비중 조정 근거를 요약해줘.", height=90)
cols = st.columns([1, 1, 2])
ticker = cols[0].selectbox(
    "대상 종목",
    [None] + get_default_tickers(),
    format_func=lambda value: "전체 포트폴리오" if value is None else f"{universe.loc[universe['ticker'] == value, 'name'].iloc[0]} ({value})",
)
max_results = cols[1].slider("출처 수", min_value=3, max_value=10, value=5)
submitted = cols[2].button("리서치 실행", type="primary", use_container_width=True)

result = load_api_data("리서치", research, query, ticker=ticker, max_results=max_results)

if submitted:
    st.toast("백엔드 리서치 결과를 갱신했습니다.")

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
    for step in result["reasoning_trace"]:
        st.info(step)

st.subheader("출처")
source_df = pd.DataFrame(result["sources"], columns=["title", "url", "published_at", "relevance_score"])
st.dataframe(
    source_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "url": st.column_config.LinkColumn("URL"),
        "relevance_score": st.column_config.ProgressColumn("관련도", min_value=0, max_value=1, format="%.0f%%"),
    },
)
