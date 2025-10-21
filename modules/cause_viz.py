# cause_viz.py
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from shiny import ui

def fig_html(fig, height=300):
    """Plotly Figure를 Shiny UI에 삽입하기 위한 HTML 래퍼"""
    fig.update_layout(height=height)
    return ui.HTML(fig.to_html(full_html=False, include_plotlyjs="cdn"))

# ---- (참고) p-관리도(일자 집계) : 필요시 사용 ----
def build_p_chart(dff: pd.DataFrame, title: str | None = None):
    if dff.empty or dff["date"].notna().sum() == 0:
        fig = go.Figure()
        fig.add_annotation(text="선택 구간에 데이터가 없습니다.", showarrow=False)
        fig.update_layout(template="plotly_white", height=400, title=title or "")
        return fig

    day = dff["date"].dt.floor("D")
    daily = (
        dff.assign(__day=day)
           .dropna(subset=["__day"])
           .groupby("__day", as_index=False)
           .agg(d=("d", "sum"), n=("n", "sum"))
    )
    if daily.empty:
        fig = go.Figure()
        fig.add_annotation(text="선택 구간에 데이터가 없습니다.", showarrow=False)
        fig.update_layout(template="plotly_white", height=400, title=title or "")
        return fig

    daily["p"] = np.where(daily["n"] > 0, daily["d"] / daily["n"], 0.0)
    pbar = float(daily["p"].mean()); nbar = float(daily["n"].mean() or 1)
    sigma = np.sqrt(pbar * (1 - pbar) / nbar)
    UCL = pbar + 3 * sigma
    LCL = max(0.0, pbar - 3 * sigma)
    oc_mask = (daily["p"] > UCL) | (daily["p"] < LCL)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["__day"], y=daily["p"], mode="markers+lines",
                             marker=dict(size=6), name="불량률"))
    fig.add_trace(go.Scatter(x=daily.loc[oc_mask, "__day"], y=daily.loc[oc_mask, "p"],
                             mode="markers", marker=dict(size=9), name="Out-of-control"))
    fig.add_hline(y=pbar, annotation_text=f"CL ({pbar:.3f})", annotation_position="right")
    fig.add_hline(y=UCL,  annotation_text=f"UCL ({UCL:.3f})", annotation_position="right")
    fig.add_hline(y=LCL,  annotation_text=f"LCL ({LCL:.3f})", annotation_position="right")
    fig.update_layout(template="plotly_white", height=420, hovermode="x unified",
                      margin=dict(l=40, r=20, t=60, b=40), title=title or "")
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

# ---- 몰드별 누적 카드 HTML ----
def mold_cards_html(df_all: pd.DataFrame):
    if df_all.empty:
        return ui.div("데이터 없음", style="text-align:center; padding:24px;")

    dff = df_all.copy()
    molds = sorted(dff["mold_code"].unique().tolist())
    n = max(1, len(molds))
    GAP = 12
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
                ui.div(f"{rate:,.2f}%", style="text-align:center; font-weight:800; font-size:26px;"),
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
    return ui.div(*cards,
                  style=f"display:flex; gap:{GAP}px; justify-content:space-between; flex-wrap:nowrap; max-width:1400px; margin:0 auto;")

# ---- 롤링 60샷 p-관리도 figure ----
def build_rolling_p_chart(series: pd.DataFrame, limits: dict, title: str | None = None):
    fig = go.Figure()
    if series.empty or not pd.notna(limits.get("pbar", np.nan)):
        fig.add_annotation(text="데이터/윈도우 부족(60 A-샷 미만)", showarrow=False)
        fig.update_layout(template="plotly_white", height=420, title=title or "")
        return fig

    # 위험도별 레이어
    for label in ["NORMAL", "CAUTION", "CRITICAL"]:
        sub = series[series["risk_level"] == label]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["date"], y=sub["p_hat"], mode="markers+lines", name=label
        ))

    # 한계선
    pbar = limits["pbar"]; UCL2 = limits["UCL2"]; UCL3 = limits["UCL3"]; LCL3 = limits["LCL3"]
    fig.add_hline(y=pbar, annotation_text=f"CL ({pbar:.3f})",  annotation_position="right")
    fig.add_hline(y=UCL2, annotation_text=f"UCL2 ({UCL2:.3f})", annotation_position="right")
    fig.add_hline(y=UCL3, annotation_text=f"UCL3 ({UCL3:.3f})", annotation_position="right")
    fig.add_hline(y=LCL3, annotation_text=f"LCL3 ({LCL3:.3f})", annotation_position="right")

    fig.update_layout(template="plotly_white", height=420, hovermode="x unified",
                      margin=dict(l=40, r=20, t=60, b=40), title=title or "")
    return fig
