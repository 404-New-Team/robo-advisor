from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from api_client import add_user_ticker, delete_user_ticker, get_user_tickers, login, me, register
from reference_data import PROFILE_LABELS, STRATEGY_LABELS, get_asset_label, get_default_tickers, get_universe


COLOR_SEQUENCE = ["#2563eb", "#16a34a", "#f97316", "#dc2626", "#7c3aed", "#0891b2", "#ca8a04", "#4b5563"]


def configure_page(title: str) -> None:
    st.set_page_config(page_title=f"Robby | {title}", page_icon="R", layout="wide")
    css = "\n".join(
        [
            "<style>",
            ".block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1320px; }",
            "[data-testid='stMetric'] { border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px 16px; background: #ffffff; }",
            "[data-testid='stSidebar'] { background: #f8fafc; }",
            "[data-testid='stSidebar'] * { color: #111827; }",
            "[data-testid='stSidebarNav'] a { color: #111827; }",
            "[data-testid='stSidebarNav'] a:hover { background: #e0ecff; }",
            "[data-testid='stSidebarNav'] a[aria-current='page'] { background: #dbeafe; color: #1d4ed8; font-weight: 700; }",
            "h1, h2, h3 { letter-spacing: 0; }",
            "@media (prefers-color-scheme: dark) {",
            "  [data-testid='stMetric'] { border-color: #334155; background: #111827; }",
            "  [data-testid='stSidebar'] { background: #0f172a; }",
            "  [data-testid='stSidebar'] * { color: #e5e7eb; }",
            "  [data-testid='stSidebarNav'] a { color: #e5e7eb; }",
            "  [data-testid='stSidebarNav'] a:hover { background: #1e293b; }",
            "  [data-testid='stSidebarNav'] a[aria-current='page'] { background: #1d4ed8; color: #ffffff; font-weight: 700; }",
            "}",
            "</style>",
        ]
    )
    st.markdown(css, unsafe_allow_html=True)


def _clear_auth_state() -> None:
    for key in ["access_token", "auth_user", "saved_tickers", "saved_tickers_loaded_for"]:
        st.session_state.pop(key, None)


def _load_current_user(token: str) -> dict | None:
    if st.session_state.get("auth_user"):
        return st.session_state["auth_user"]
    try:
        user = me(token)
    except Exception as error:
        _clear_auth_state()
        st.sidebar.error(f"로그인 세션 확인 실패: {error}")
        return None
    st.session_state["auth_user"] = user
    return user


def _load_saved_tickers(token: str | None) -> list[str]:
    if not token:
        return []
    if st.session_state.get("saved_tickers_loaded_for") == token:
        return st.session_state.get("saved_tickers", [])
    try:
        tickers = get_user_tickers(token).get("tickers", [])
    except Exception as error:
        st.sidebar.warning(f"저장 종목 조회 실패: {error}")
        tickers = []
    st.session_state["saved_tickers"] = tickers
    st.session_state["saved_tickers_loaded_for"] = token
    return tickers


def _sync_saved_tickers(token: str, selected: list[str]) -> None:
    current = set(_load_saved_tickers(token))
    desired = set(selected)
    for ticker in desired - current:
        add_user_ticker(token, ticker)
    for ticker in current - desired:
        delete_user_ticker(token, ticker)
    st.session_state["saved_tickers"] = selected
    st.session_state["saved_tickers_loaded_for"] = token


def render_auth_panel() -> tuple[str | None, list[str]]:
    st.sidebar.subheader("계정")
    token = st.session_state.get("access_token")
    if token:
        user = _load_current_user(token)
        if user:
            st.sidebar.caption(f"{user.get('username', '')} · {user.get('email', '')}")
            if st.sidebar.button("로그아웃", use_container_width=True, key="auth_logout"):
                _clear_auth_state()
                st.rerun()
            return token, _load_saved_tickers(token)
        return None, []

    email = st.sidebar.text_input("이메일", key="auth_login_email")
    password = st.sidebar.text_input("비밀번호", type="password", key="auth_login_password")
    if st.sidebar.button("로그인", use_container_width=True, key="auth_login_submit"):
        try:
            token_data = login(email, password)
            token = token_data["access_token"]
            st.session_state["access_token"] = token
            st.session_state["auth_user"] = me(token)
            st.sidebar.success("로그인했습니다.")
            st.rerun()
        except Exception as error:
            st.sidebar.error(f"로그인 실패: {error}")

    with st.sidebar.expander("회원가입"):
        reg_email = st.text_input("회원 이메일", key="auth_register_email")
        reg_username = st.text_input("이름", key="auth_register_username")
        reg_password = st.text_input("회원 비밀번호", type="password", key="auth_register_password")
        if st.button("가입 후 로그인", use_container_width=True, key="auth_register_submit"):
            try:
                register(reg_email, reg_username, reg_password)
                token_data = login(reg_email, reg_password)
                token = token_data["access_token"]
                st.session_state["access_token"] = token
                st.session_state["auth_user"] = me(token)
                st.success("회원가입했습니다.")
                st.rerun()
            except Exception as error:
                st.error(f"회원가입 실패: {error}")

    return None, []


def render_sidebar() -> dict:
    universe = get_universe()
    st.sidebar.title("Robby")
    token, saved_tickers = render_auth_panel()
    default_tickers = [ticker for ticker in saved_tickers if ticker in set(universe["ticker"])] or get_default_tickers()
    risk_label = st.sidebar.radio("투자 성향", list(PROFILE_LABELS.values()), index=1)
    risk_level = next(key for key, value in PROFILE_LABELS.items() if value == risk_label)
    investment_amount = st.sidebar.number_input("투자 가능 금액", min_value=1_000_000, max_value=500_000_000, value=30_000_000, step=1_000_000)
    selected = st.sidebar.multiselect(
        "투자 대상",
        universe["ticker"].tolist(),
        default=default_tickers,
        format_func=get_asset_label,
    )
    if token and st.sidebar.button("투자 대상 저장", use_container_width=True, key="save_user_tickers"):
        try:
            _sync_saved_tickers(token, selected)
            st.sidebar.success("투자 대상을 저장했습니다.")
        except Exception as error:
            st.sidebar.error(f"투자 대상 저장 실패: {error}")
    excluded = st.sidebar.multiselect(
        "제외 종목",
        selected,
        default=[],
        format_func=get_asset_label,
    )
    if not selected:
        st.sidebar.warning("투자 대상이 비어 있어 기본 유니버스를 사용합니다.")
        selected = get_default_tickers()
        excluded = []
    if selected and len(excluded) == len(selected):
        st.sidebar.warning("전체 종목을 제외할 수 없어 마지막 제외 항목을 해제합니다.")
        excluded = excluded[:-1]
    horizon = st.sidebar.selectbox("시뮬레이션 기간", ["6개월", "12개월", "24개월"], index=1)
    active = [ticker for ticker in selected if ticker not in set(excluded)] or selected
    return {
        "risk_level": risk_level,
        "risk_label": risk_label,
        "investment_amount": int(investment_amount),
        "selected_tickers": selected,
        "excluded_tickers": excluded,
        "active_tickers": active,
        "horizon": horizon,
        "access_token": token,
    }


def load_api_data(label: str, func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as error:
        st.error(f"{label} API 호출 실패: {error}")
        st.stop()


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def format_money(value: int | float) -> str:
    return f"{int(value):,}원"


def render_metric_row(metrics: dict) -> None:
    cols = st.columns(4)
    cols[0].metric("예상 수익률", format_percent(metrics["expected_return"]), "+2.8%p")
    cols[1].metric("Sharpe", f"{metrics['sharpe_ratio']:.2f}", "+0.31")
    cols[2].metric("MDD", format_percent(metrics["max_drawdown"]), "-3.4%p", delta_color="inverse")
    cols[3].metric("변동성", format_percent(metrics["volatility"]), "-1.1%p", delta_color="inverse")


def allocation_chart(weight_df: pd.DataFrame):
    chart_df = weight_df.copy()
    chart_df["표시명"] = chart_df["종목"] + " " + chart_df["티커"]
    fig = px.pie(
        chart_df,
        values="비중",
        names="표시명",
        hole=0.45,
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig.update_traces(textposition="inside", texttemplate="%{label}<br>%{percent:.1%}")
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), legend_title_text="")
    return fig


def performance_chart(df: pd.DataFrame):
    fig = px.line(
        df,
        x="날짜",
        y=[column for column in df.columns if column != "날짜"],
        markers=True,
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig.update_layout(
        yaxis_title="기준가",
        xaxis_title="",
        legend_title_text="",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig


def walk_forward_performance_frame(results: list[dict]) -> pd.DataFrame:
    rows_by_period = {}
    for result in results:
        strategy = result.get("strategy", "strategy")
        label = STRATEGY_LABELS.get(strategy, strategy)
        cumulative = 100.0
        items = result.get("walk_forward_results", []) or [{"period": "N/A", "return": 0.0}]
        for item in items:
            cumulative *= 1 + float(item.get("return", 0.0))
            period = item.get("period", "N/A")
            rows_by_period.setdefault(period, {"날짜": period})
            rows_by_period[period][label] = cumulative
    return pd.DataFrame(rows_by_period.values())


def strategy_comparison_from_results(results: list[dict]) -> pd.DataFrame:
    rows = []
    for result in results:
        strategy = result.get("strategy", "")
        metrics = result.get("metrics", {})
        rows.append(
            {
                "전략": STRATEGY_LABELS.get(strategy, strategy),
                "누적수익률": metrics.get("total_return", 0.0),
                "Sharpe": metrics.get("sharpe_ratio", 0.0),
                "MDD": metrics.get("max_drawdown", 0.0),
                "승률": metrics.get("win_rate", 0.0),
            }
        )
    return pd.DataFrame(rows)


def shap_summary_from_results(results: list[dict]) -> pd.DataFrame:
    rows = []
    for result in results:
        asset = result.get("target_asset", "")
        for feature, value in result.get("shap_values", {}).items():
            rows.append({"종목": asset, "피처": feature, "기여도": value})
    return pd.DataFrame(rows, columns=["종목", "피처", "기여도"])


def simulation_chart(df: pd.DataFrame):
    fig = px.line(
        df,
        x="날짜",
        y="자산가치",
        color="경로",
        markers=True,
        color_discrete_sequence=["#16a34a", "#2563eb", "#dc2626"],
    )
    fig.update_layout(
        yaxis_title="자산가치 지수",
        xaxis_title="",
        legend_title_text="",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig


def render_allocation_table(allocation_result: dict) -> None:
    items = allocation_result.get("items", [])
    if not items:
        st.warning("주문 수량 데이터가 없습니다.")
        return

    summary_cols = st.columns(3)
    summary_cols[0].metric("총 투자 예정금", format_money(allocation_result["total_amount"]))
    summary_cols[1].metric("실제 투자금", format_money(allocation_result["total_invested"]))
    summary_cols[2].metric("잔여금", format_money(allocation_result["total_leftover"]))

    rows = []
    for item in items:
        rows.append({
            "티커": item["ticker"],
            "비중": item["weight"],
            "현재가(원)": item["current_price"],
            "목표금액(원)": int(item["target_amount"]),
            "정수매수(주)": item["integer_shares"],
            "소수점(참고)": item["fractional_shares"],
            "실제투자금(원)": int(item["actual_amount"]),
            "잔여금(원)": int(item["leftover"]),
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "비중": st.column_config.ProgressColumn("비중", min_value=0, max_value=1, format="%.1f%%"),
            "현재가(원)": st.column_config.NumberColumn("현재가(원)", format="%d"),
            "목표금액(원)": st.column_config.NumberColumn("목표금액(원)", format="%d"),
            "정수매수(주)": st.column_config.NumberColumn("정수매수(주)", format="%d"),
            "소수점(참고)": st.column_config.NumberColumn("소수점(참고)", format="%.4f"),
            "실제투자금(원)": st.column_config.NumberColumn("실제투자금(원)", format="%d"),
            "잔여금(원)": st.column_config.NumberColumn("잔여금(원)", format="%d"),
        },
    )


def percent_dataframe(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    formatted = df.copy()
    for column in columns:
        formatted[column] = formatted[column].map(lambda value: f"{value * 100:.1f}%")
    return formatted
