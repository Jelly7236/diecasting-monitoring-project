# modules/page_cause.py
from shiny import ui, render
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import timedelta

# =========================================================
# 0) 데이터 로더
#    - CSV/XLSX 모두 지원
#    - 필요한 컬럼: date, mold_code, n(검사수), d(불량수)
#    - 선택 컬럼: passorfail(0/1), rf_flag(bool/int)
# =========================================================
def load_quality_from_file() -> pd.DataFrame:
    candidates = ["/mnt/data/test2.xlsx", "/mnt/data/test2.csv"]
    for p in candidates:
        try:
            if p.endswith(".xlsx"):
                df = pd.read_excel(p)
            elif p.endswith(".csv"):
                df = pd.read_csv(p)
            else:
                continue
            if not df.empty:
                break
        except Exception:
            continue
    else:
        return pd.DataFrame(columns=["date", "mold_code", "n", "d"])

    # 표준화
    rename_map = {
        "Date": "date", "DATE": "date",
        "MOLD": "mold_code", "mold": "mold_code",
        "N": "n", "D": "d", "defect": "d", "Defect": "d", "OK": "g"
    }
    df = df.rename(columns=rename_map)

    # 날짜/타입 정리
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    for c in ["n", "d"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    df["mold_code"] = df["mold_code"].astype(str)
    df["p"] = np.where((df.get("n", 0) > 0), df.get("d", 0) / df.get("n", 1), 0.0)
    return df


# ------------------------ 공통: Plotly → HTML ------------------------
def fig_html(fig, height=300):
    fig.update_layout(height=height)
    # CDN 방식으로 plotly.js 포함 → 위젯/comm 미사용
    return ui.HTML(fig.to_html(full_html=False, include_plotlyjs="cdn"))


# ============================== UI ===============================
def ui_cause():
    row_style = "display:flex; gap:16px; justify-content:space-between; flex-wrap:nowrap;"
    card_style = "flex:1; min-width:220px;"

    return ui.page_fluid(
        ui.h3("🎯 불량 원인 분석"),
        ui.p("상단: 몰드별 불량률 · 중단: p-관리도(날짜/몰드 단일 선택) · 하단: 불량 샘플 감지 로그"),
        ui.hr(),

        # ── 상단: 몰드별 불량률 카드(도넛 + 불량률) ──
        ui.div(
            ui.card(ui.card_header("몰드 8722"), ui.output_ui("donut_8722"),
                    ui.div(ui.output_text("rate_8722"),
                           style="text-align:center; font-weight:700; color:#1f60c4;"),
                    style=card_style),
            ui.card(ui.card_header("몰드 8412"), ui.output_ui("donut_8412"),
                    ui.div(ui.output_text("rate_8412"),
                           style="text-align:center; font-weight:700; color:#1f60c4;"),
                    style=card_style),
            ui.card(ui.card_header("몰드 8573"), ui.output_ui("donut_8573"),
                    ui.div(ui.output_text("rate_8573"),
                           style="text-align:center; font-weight:700; color:#1f60c4;"),
                    style=card_style),
            ui.card(ui.card_header("몰드 8917"), ui.output_ui("donut_8917"),
                    ui.div(ui.output_text("rate_8917"),
                           style="text-align:center; font-weight:700; color:#1f60c4;"),
                    style=card_style),
            ui.card(ui.card_header("몰드 8600"), ui.output_ui("donut_8600"),
                    ui.div(ui.output_text("rate_8600"),
                           style="text-align:center; font-weight:700; color:#1f60c4;"),
                    style=card_style),
            style=f"{row_style} max-width:1400px; margin:0 auto;"
        ),

        ui.hr(),

        # ── 중단: p-관리도(왼쪽) + (옵션)SHAP(오른쪽 자리) ──
        ui.div(
            ui.card(
                ui.card_header(
                    ui.div(
                        ui.div("📊 p-관리도 (일별 불량률)", class_="text-lg font-semibold"),
                        ui.div(
                            ui.input_date("p_date", "기준일", value=None),
                            ui.input_select("p_mold", "몰드", choices=[], multiple=False),
                            style="display:flex; gap:12px; align-items:center;"
                        ),
                        style="display:flex; justify-content:space-between; align-items:center; gap:12px;"
                    )
                ),
                ui.output_ui("p_chart"),
                style="flex:1; min-width:560px;"
            ),
            ui.card(
                ui.card_header("🔥 SHAP 주요 변수 영향도 (모델 연결 시 표시)"),
                ui.p("※ 현재는 파일 기반 분석만 활성화. 모델 연결 후 표시됩니다."),
                style="flex:0.7; min-width:380px;"
            ),
            style=f"{row_style} max-width:1400px; margin:0 auto;"
        ),

        ui.hr(),

        # ── 하단: 감지 로그 ──
        ui.card(
            ui.card_header("🚨 불량 샘플 감지 로그 (이상탐지 / 관리도 / Rule / 랜덤포레스트)"),
            ui.output_table("detect_log"),
            style="max-width:1400px; margin:0 auto;"
        ),
    )


# ============================ SERVER =============================
def server_cause(input, output, session):
    df = load_quality_from_file()

    # 컨트롤 초기화
    if df.empty:
        session.send_input_message("p_mold", {"choices": [], "selected": None})
    else:
        molds = sorted(df["mold_code"].unique().tolist())
        max_date = df["date"].max().date()
        session.send_input_message("p_mold", {"choices": molds, "selected": molds[0]})
        session.send_input_message("p_date", {"value": str(max_date)})

    # ── 공통: 도넛 만들기 ──
    def _donut_fig(N, D, height=220):
        G = max(0, N - D)
        if N == 0:
            fig = go.Figure()
            fig.add_annotation(text="데이터 없음", showarrow=False,
                               font=dict(size=16, color="#808080"))
            fig.update_layout(template="plotly_white",
                              margin=dict(t=40, b=40, l=10, r=10), height=height)
            return fig
        fig = go.Figure(go.Pie(labels=["양품", "불량"],
                               values=[G, D],
                               hole=0.65, textinfo="percent+label"))
        fig.update_layout(showlegend=False, template="plotly_white",
                          margin=dict(t=8, b=8, l=8, r=8), height=height)
        return fig

    # ── 상단 5카드 (출력 ID = 함수명) ──
    @render.ui
    def donut_8722():
        d = df[df["mold_code"] == "8722"]
        return fig_html(_donut_fig(int(d["n"].sum()), int(d["d"].sum())), height=220)

    @output
    @render.text
    def rate_8722():
        d = df[df["mold_code"] == "8722"]
        N, D = int(d["n"].sum()), int(d["d"].sum())
        return f"불량률: {0.0 if N == 0 else D / N * 100:,.1f}%"

    @render.ui
    def donut_8412():
        d = df[df["mold_code"] == "8412"]
        return fig_html(_donut_fig(int(d["n"].sum()), int(d["d"].sum())), height=220)

    @output
    @render.text
    def rate_8412():
        d = df[df["mold_code"] == "8412"]
        N, D = int(d["n"].sum()), int(d["d"].sum())
        return f"불량률: {0.0 if N == 0 else D / N * 100:,.1f}%"

    @render.ui
    def donut_8573():
        d = df[df["mold_code"] == "8573"]
        return fig_html(_donut_fig(int(d["n"].sum()), int(d["d"].sum())), height=220)

    @output
    @render.text
    def rate_8573():
        d = df[df["mold_code"] == "8573"]
        N, D = int(d["n"].sum()), int(d["d"].sum())
        return f"불량률: {0.0 if N == 0 else D / N * 100:,.1f}%"

    @render.ui
    def donut_8917():
        d = df[df["mold_code"] == "8917"]
        return fig_html(_donut_fig(int(d["n"].sum()), int(d["d"].sum())), height=220)

    @output
    @render.text
    def rate_8917():
        d = df[df["mold_code"] == "8917"]
        N, D = int(d["n"].sum()), int(d["d"].sum())
        return f"불량률: {0.0 if N == 0 else D / N * 100:,.1f}%"

    @render.ui
    def donut_8600():
        d = df[df["mold_code"] == "8600"]
        return fig_html(_donut_fig(int(d["n"].sum()), int(d["d"].sum())), height=220)

    @output
    @render.text
    def rate_8600():
        d = df[df["mold_code"] == "8600"]
        N, D = int(d["n"].sum()), int(d["d"].sum())
        return f"불량률: {0.0 if N == 0 else D / N * 100:,.1f}%"

    # ── p-관리도(기준일 포함 최근 21일, 단일 몰드) ──
    @render.ui
    def p_chart():
        if df.empty or input.p_mold() is None or input.p_date() is None:
            fig = go.Figure()
            fig.add_annotation(text="데이터/선택값 없음", showarrow=False)
            fig.update_layout(template="plotly_white", height=400)
            return fig_html(fig, height=400)

        mold = input.p_mold()
        end = pd.to_datetime(input.p_date()).normalize()
        start = end - timedelta(days=20)

        sel = df[(df["mold_code"] == mold) &
                 (df["date"] >= start) & (df["date"] <= end)].copy()

        fig = go.Figure()
        if sel.empty:
            fig.add_annotation(text="선택 구간에 데이터가 없습니다.", showarrow=False)
            fig.update_layout(template="plotly_white", height=400)
            return fig_html(fig, height=400)

        # 일자별 집계
        daily = sel.groupby("date", as_index=False).agg({"d": "sum", "n": "sum"})
        daily["p"] = daily["d"] / daily["n"]

        # CL/UCL/LCL (상수선)
        pbar = daily["p"].mean()
        nbar = daily["n"].mean() or 1
        sigma = np.sqrt(pbar * (1 - pbar) / nbar)
        UCL = pbar + 3 * sigma
        LCL = max(0.0, pbar - 3 * sigma)

        # 관리도 위반 점
        out_mask = (daily["p"] > UCL) | (daily["p"] < LCL)

        fig.add_trace(go.Scatter(
            x=daily["date"], y=daily["p"],
            mode="markers+lines",
            marker=dict(size=6, color="#5DADE2"),
            line=dict(color="#AED6F1"),
            name="불량률"
        ))
        fig.add_trace(go.Scatter(
            x=daily.loc[out_mask, "date"],
            y=daily.loc[out_mask, "p"],
            mode="markers",
            marker=dict(size=9, color="#E74C3C"),
            name="Out-of-control"
        ))

        fig.add_hline(y=pbar, line=dict(color="#F5B041", width=2),
                      annotation_text=f"CL ({pbar:.3f})", annotation_position="right")
        fig.add_hline(y=UCL, line=dict(color="#E74C3C", width=2),
                      annotation_text=f"UCL ({UCL:.3f})", annotation_position="right")
        fig.add_hline(y=LCL, line=dict(color="#2E86C1", width=2),
                      annotation_text=f"LCL ({LCL:.3f})", annotation_position="right")

        fig.update_layout(
            title=f"p-관리도｜몰드 {mold} · {start.date()} ~ {end.date()}",
            template="plotly_white", height=420,
            hovermode="x unified", margin=dict(l=40, r=20, t=60, b=40)
        )
        return fig_html(fig, height=420)

    # ── 불량 샘플 감지 로그 ──
    @output
    @render.table
    def detect_log():
        if df.empty or input.p_mold() is None or input.p_date() is None:
            return pd.DataFrame({"메시지": ["데이터/선택값 없음"]})

        mold = input.p_mold()
        end = pd.to_datetime(input.p_date()).normalize()
        start = end - timedelta(days=20)

        sel = df[(df["mold_code"] == mold) &
                 (df["date"] >= start) & (df["date"] <= end)].copy()
        if sel.empty:
            return pd.DataFrame({"메시지": ["선택 구간에 불량 샘플이 없습니다."]})

        # 일자별 p, CL/UCL/LCL 계산
        daily = sel.groupby("date", as_index=False).agg({"d": "sum", "n": "sum"})
        daily["p"] = daily["d"] / daily["n"]
        pbar = daily["p"].mean()
        nbar = daily["n"].mean() or 1
        sigma = np.sqrt(max(pbar * (1 - pbar) / nbar, 1e-12))
        UCL = pbar + 3 * sigma
        LCL = max(0.0, pbar - 3 * sigma)

        # Rule 기반(간단): 7연속 CL 위/아래
        sign = np.sign(daily["p"] - pbar).replace(0, np.nan)
        run_up = (sign == 1).astype(int).groupby((sign != 1).cumsum()).cumsum()
        run_dn = (sign == -1).astype(int).groupby((sign != -1).cumsum()).cumsum()
        rule_hit_idx = daily.index[(run_up >= 7) | (run_dn >= 7)]

        # 이상탐지(통계): |p - rolling mean| > 3*rolling std (윈도 10)
        roll = daily["p"].rolling(10, min_periods=6)
        z_anom = (daily["p"] - roll.mean()).abs() > 3 * roll.std().replace(0, np.nan)
        z_anom = z_anom.fillna(False)

        # 관리도 위반(UCL/LCL)
        oc = (daily["p"] > UCL) | (daily["p"] < LCL)

        # 랜덤포레스트: 파일에 rf_flag / passorfail 있으면 사용
        rf_col = None
        for c in ["rf_flag", "rf_detect", "rf", "model_flag", "passorfail"]:
            if c in sel.columns:
                rf_col = c
                break

        rows = []
        for i, r in daily.iterrows():
            rows.append({
                "날짜": r["date"].date(),
                "몰드": mold,
                "불량수": int(r["d"]),
                "검사수": int(r["n"]),
                "불량률": round(float(r["p"]), 4),
                "이상탐지": "✅" if bool(z_anom.iloc[i]) else "",
                "관리도": "✅" if bool(oc.iloc[i]) else "",
                "Rule기반": "✅" if i in rule_hit_idx else "",
                "랜덤포레스트": "✅" if (rf_col and sel[sel["date"] == r["date"]][rf_col].astype(int).any()) else "",
            })
        return pd.DataFrame(rows).sort_values("날짜")
