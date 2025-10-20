# modules/cause_viz.py
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from shiny import ui

# ---- Plotly -> HTML ----
def fig_html(fig, height=300):
    """Plotly Figure를 Shiny UI에 안전하게 삽입하기 위한 HTML 래퍼"""
    fig.update_layout(height=height)
    return ui.HTML(fig.to_html(full_html=False, include_plotlyjs="cdn"))

# ---- p-관리도 figure ----
def build_p_chart(dff: pd.DataFrame):
    daily = (
        dff.dropna(subset=["date"])
           .groupby("date", as_index=False)
           .agg(d=("d", "sum"), n=("n", "sum"))
    )
    if daily.empty:
        fig = go.Figure()
        fig.add_annotation(text="선택 구간에 데이터가 없습니다.", showarrow=False)
        fig.update_layout(template="plotly_white", height=400)
        return fig

    daily["p"] = np.where(daily["n"] > 0, daily["d"] / daily["n"], 0.0)
    pbar = float(daily["p"].mean()); nbar = float(daily["n"].mean() or 1)
    sigma = np.sqrt(pbar * (1 - pbar) / nbar)
    UCL = pbar + 3 * sigma
    LCL = max(0.0, pbar - 3 * sigma)
    oc_mask = (daily["p"] > UCL) | (daily["p"] < LCL)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["p"], mode="markers+lines",
                             marker=dict(size=6, color="#5DADE2"),
                             line=dict(color="#AED6F1"), name="불량률"))
    fig.add_trace(go.Scatter(x=daily.loc[oc_mask, "date"], y=daily.loc[oc_mask, "p"],
                             mode="markers", marker=dict(size=9, color="#E74C3C"),
                             name="Out-of-control"))
    fig.add_hline(y=pbar, line=dict(color="#F5B041", width=2), annotation_text=f"CL ({pbar:.3f})", annotation_position="right")
    fig.add_hline(y=UCL, line=dict(color="#E74C3C", width=2), annotation_text=f"UCL ({UCL:.3f})", annotation_position="right")
    fig.add_hline(y=LCL, line=dict(color="#2E86C1", width=2), annotation_text=f"LCL ({LCL:.3f})", annotation_position="right")
    fig.update_layout(template="plotly_white", height=420, hovermode="x unified", margin=dict(l=40, r=20, t=40, b=40))
    return fig

# ---- SHAP bar figure ----
def build_shap_bar(dff: pd.DataFrame):
    shap_cols = [c for c in dff.columns if c.lower().startswith("shap_")]
    if len(shap_cols) == 0:
        return None

    base = dff[dff["d"] > 0].copy()
    if base.empty:
        base = dff.copy()
    imp = base[shap_cols].abs().mean().sort_values(ascending=True)
    topk = imp.tail(10)

    fig = go.Figure(go.Bar(
        x=topk.values,
        y=[c.replace("shap_", "") for c in topk.index],
        orientation="h",
        text=np.round(topk.values, 4),
        textposition="auto",
    ))
    fig.update_layout(template="plotly_white", height=420, margin=dict(l=80, r=20, t=20, b=40))
    return fig

# ---- 몰드별 누적 카드 HTML (스크롤 없이 한 줄, 균등폭) ----
def mold_cards_html(df_all: pd.DataFrame):
    """몰드별 누적 카드: 가로 한 줄(줄바꿈 없음), 카드 폭 자동 분할로 한 화면에 모두 표시"""
    if df_all.empty:
        return ui.div("데이터 없음", style="text-align:center; padding:24px;")

    dff = df_all.copy()
    molds = sorted(dff["mold_code"].unique().tolist())
    n = max(1, len(molds))
    GAP = 12  # 카드 간격(px)
    width_css = f"calc((100% - {(n-1)}*{GAP}px)/{n})"

    def _count_oc_anom(sub: pd.DataFrame):
        sub_dated = sub.dropna(subset=["date"])
        if sub_dated.empty:
            return 0, 0
        daily = sub_dated.groupby("date", as_index=False).agg(d=("d","sum"), n=("n","sum"))
        daily["p"] = np.where(daily["n"]>0, daily["d"]/daily["n"], 0.0)
        pbar = float(daily["p"].mean()); nbar = float(daily["n"].mean() or 1)
        sigma = np.sqrt(pbar*(1-pbar)/nbar)
        UCL = pbar + 3*sigma; LCL = max(0.0, pbar - 3*sigma)
        oc = int(((daily["p"]>UCL) | (daily["p"]<LCL)).sum())
        roll = daily["p"].rolling(10, min_periods=6)
        z_anom = (daily["p"] - roll.mean()).abs() > 3*roll.std().replace(0,np.nan)
        anom = int(z_anom.fillna(False).sum())
        return oc, anom

    def _card(mold: str):
        sub = dff[dff["mold_code"] == mold]
        total_n = int(sub["n"].sum()); total_d = int(sub["d"].sum())
        rate = (total_d/total_n*100) if total_n > 0 else 0.0
        oc_cnt, anom_cnt = _count_oc_anom(sub)

        return ui.card(
            ui.card_header(f"몰드 {mold}"),
            ui.div(
                ui.div("누적 불량률", style="text-align:center; color:#6b7280; font-weight:600;"),
                ui.div(f"{rate:,.2f}%", style="text-align:center; font-weight:800; font-size:26px; color:#1f60c4;"),
                ui.div(
                    ui.span(f"누적 이상 {anom_cnt} 건", style="margin-right:12px;"),
                    ui.span(f"누적 관리도 이탈 {oc_cnt} 건"),
                    style="text-align:center; margin-top:8px; font-size:15px; font-weight:700;"
                ),
                style="padding:12px 0;"
            ),
            style=f"flex:0 0 {width_css}; max-width:{width_css}; min-width:160px;"
        )

    cards = [_card(m) for m in molds]

    return ui.div(
        *cards,
        style=(
            f"display:flex; gap:{GAP}px; justify-content:space-between; flex-wrap:nowrap; "
            "max-width:1400px; margin:0 auto;"
        ),
    )
