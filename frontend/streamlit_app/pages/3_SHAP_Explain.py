from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from api_client import explain
from reference_data import get_default_tickers, get_universe
from ui import COLOR_SEQUENCE, configure_page, format_percent, load_api_data, render_sidebar, shap_summary_from_results


configure_page("SHAP 해석")

state = render_sidebar()
universe = get_universe()

st.title("SHAP 의사결정 해석")

target = st.selectbox(
    "해석 대상",
    state["active_tickers"] or get_default_tickers(),
    format_func=lambda ticker: f"{universe.loc[universe['ticker'] == ticker, 'name'].iloc[0]} ({ticker})",
)
result = load_api_data("SHAP 해석", explain, state["active_tickers"], target)
shap_df = pd.DataFrame(
    [{"피처": key, "기여도": value, "방향": "확대" if value >= 0 else "축소"} for key, value in result["shap_values"].items()],
    columns=["피처", "기여도", "방향"],
).sort_values("기여도")

cols = st.columns(3)
cols[0].metric("최종 비중", format_percent(result["final_weight"]))
cols[1].metric("양의 기여 합", format_percent(shap_df.loc[shap_df["기여도"] > 0, "기여도"].sum()))
cols[2].metric("음의 기여 합", format_percent(shap_df.loc[shap_df["기여도"] < 0, "기여도"].sum()), delta_color="inverse")

st.write(result["explanation"])

left, right = st.columns([1.05, 1])
with left:
    st.subheader("피처별 기여도")
    fig = px.bar(
        shap_df,
        x="기여도",
        y="피처",
        color="방향",
        orientation="h",
        color_discrete_map={"확대": "#2563eb", "축소": "#dc2626"},
    )
    fig.update_layout(xaxis_title="SHAP value", yaxis_title="", legend_title_text="", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True, key="shap_feature_chart")
with right:
    st.subheader("전체 종목 중요도")
    summary_results = [
        load_api_data("SHAP 요약", explain, state["active_tickers"], ticker)
        for ticker in (state["active_tickers"] or get_default_tickers())[:6]
    ]
    summary_df = shap_summary_from_results(summary_results)
    summary_fig = px.scatter(
        summary_df,
        x="기여도",
        y="피처",
        color="종목",
        size=summary_df["기여도"].abs(),
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    summary_fig.update_layout(xaxis_title="기여도", yaxis_title="", legend_title_text="", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(summary_fig, use_container_width=True, key="shap_summary_chart")

st.subheader("원본 값")
st.dataframe(shap_df, use_container_width=True, hide_index=True)
