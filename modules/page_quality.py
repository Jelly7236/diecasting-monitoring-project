# modules/page_quality.py
from shiny import ui, render
import plotly.graph_objs as go
import numpy as np
import pandas as pd


# =========================
# 1) 데이터 훅 (여기만 실제 DF로 교체)
#    필요한 컬럼: date(datetime), value(float), mold_code(str)
# =========================
def load_quality_source() -> pd.DataFrame:
    # 샘플: 몰드 5개, 일부 이상치 포함
    rng = pd.date_range("2020-01-01", periods=200, freq="D")
    molds = ["8412", "8573", "8600", "8722", "8917"]
    out = []
    for m in molds:
        vals = np.random.normal(250, 50, len(rng))
        vals[np.random.randint(0, len(rng), 5)] += np.random.randint(300, 700)
        out.append(pd.DataFrame({"date": rng, "value": vals, "mold_code": m}))
    return pd.concat(out, ignore_index=True)


# =========================
# 2) 통계/집계 유틸
# =========================
def calc_control_limits(series: pd.Series):
    mu = series.mean()
    sigma = series.std(ddof=1)
    return mu, mu + 3 * sigma, mu - 3 * sigma, sigma  # CL, UCL, LCL, sigma


def flag_violations(df: pd.DataFrame):
    d = df.copy().sort_values("date")
    cl, ucl, lcl, sigma = calc_control_limits(d["value"])
    d["CL"], d["UCL"], d["LCL"], d["sigma"] = cl, ucl, lcl, sigma
    d["is_out"] = (d["value"] > ucl) | (d["value"] < lcl)
    return d


def monthly_summary(df_flagged: pd.DataFrame):
    tmp = df_flagged.copy()
    tmp["ym"] = tmp["date"].dt.to_period("M").astype(str)
    return (
        tmp.groupby("ym")
        .agg(out_cnt=("is_out", "sum"), avg_val=("value", "mean"))
        .reset_index()
    )


# =========================
# 3) UI
# =========================
def ui_quality():
    df_all = load_quality_source()
    molds = [str(m) for m in sorted(df_all["mold_code"].unique())]
    tabs = [ui.nav_panel(f"Mold {m}", _mold_content_ui(m)) for m in molds]

    return ui.page_fluid(
        ui.h3("품질관리팀 탭 (Quality Control)"),
        ui.navset_tab(*tabs),
    )


def _mold_content_ui(mold_code: str):
    sid = lambda base: f"{base}_{mold_code}"

    # 왼쪽 큰 카드: 관리도 + (같은 카드 내부) KPI 스트립
    left_panel = ui.card(
        ui.card_header(f"관리도 (Mold {mold_code})"),
        ui.output_ui(sid("control_chart")),
        ui.div(  # KPI 스트립 (그래프 아래, 같은 카드)
            ui.output_ui(sid("kpi_strip")),
            style="padding:10px 6px 2px 6px; border-top:1px solid #eee;"
        ),
        full_screen=True,
    )

    # 오른쪽: 월별 막대 + 이상 포인트 테이블
    right_panel = ui.layout_columns(
        ui.card(ui.card_header("한계를 넘은 포인트 (월별)"), ui.output_ui(sid("viol_bar"))),
        ui.card(ui.card_header("한계를 넘은 포인트 (상세)"), ui.output_data_frame(sid("viol_table"))),
        col_widths=(12, 12),
    )

    # 추가 섹션들 (모두 몰드별로 렌더)
    row_2 = ui.layout_columns(
        ui.card(ui.card_header("Cp/Cpk 시각화 (샘플)"), ui.output_ui(sid("cpk_plot"))),
        ui.card(ui.card_header("상관분석 Heatmap (샘플)"), ui.output_ui(sid("heatmap"))),
        col_widths=(6, 6),
    )
    row_3 = ui.layout_columns(
        ui.card(ui.card_header("이상치 탐지 그래프"), ui.output_ui(sid("outlier_detection"))),
        ui.card(
            ui.card_header("이상치 분포"),
            ui.div(
                ui.input_action_button(sid("btn_lookup"), "이상치 분포", class_="btn-primary w-100"),
                ui.output_text(sid("lookup_result")),
                style="padding:10px;"
            ),
        ),
        col_widths=(6, 6),
    )
    row_4 = ui.layout_columns(
        ui.card(ui.card_header("상관관계 그래프"), ui.output_ui(sid("correlation_graph"))),
        ui.card(ui.card_header("SHAP (샘플)"), ui.output_ui(sid("shap_plot"))),
        col_widths=(6, 6),
    )

    return ui.page_fluid(
        ui.layout_columns(left_panel, right_panel, col_widths=(8, 4)),
        ui.br(),
        row_2,
        ui.br(),
        row_3,
        ui.br(),
        row_4,
    )


# =========================
# 4) SERVER
# =========================
def server_quality(input, output, session):
    df_all = load_quality_source()
    molds = [str(m) for m in sorted(df_all["mold_code"].unique())]

    for mold in molds:
        _bind_mold_outputs(mold, df_all[df_all["mold_code"] == mold], output, input)


def _bind_mold_outputs(mold: str, df_mold: pd.DataFrame, output, input):
    sid = lambda base: f"{base}_{mold}"
    df_flag = flag_violations(df_mold)
    mbar = monthly_summary(df_flag)

    # -------- A) 관리도 --------
    @output(id=sid("control_chart"))
    @render.ui
    def _control_chart():
        d = df_flag
        fig = go.Figure()

        # 시계열
        fig.add_trace(go.Scatter(
            x=d["date"], y=d["value"], mode="lines+markers",
            line=dict(width=1), marker=dict(size=5),
            name=f"Mold {mold}"
        ))
        # CL / UCL / LCL (색상: CL=노랑, UCL=빨강, LCL=파랑)
        fig.add_hline(y=float(d["CL"].iloc[0]),  line_color="#fbbf24",
                      annotation_text="CL", annotation_position="top right")
        fig.add_hline(y=float(d["UCL"].iloc[0]), line_color="#ef4444",
                      annotation_text="UCL", annotation_position="top right")
        fig.add_hline(y=float(d["LCL"].iloc[0]), line_color="#3b82f6",
                      annotation_text="LCL", annotation_position="bottom right")

        # 이상 포인트
        outs = d[d["is_out"]]
        if not outs.empty:
            fig.add_trace(go.Scatter(
                x=outs["date"], y=outs["value"], mode="markers",
                marker=dict(size=10, color="#ef4444"), name="Out of Control"
            ))

        fig.update_layout(
            template="plotly_white",
            height=420,
            margin=dict(l=20, r=20, t=40, b=20),
            title=f"관리도 (Individuals, CL±3σ)"
        )
        return ui.HTML(fig.to_html(include_plotlyjs=False, full_html=False))

    # -------- A-2) KPI 스트립 (차트 아래, 같은 카드 내부) --------
    @output(id=sid("kpi_strip"))
    @render.ui
    def _kpi_strip():
        d = df_flag
        k_groups = f"{len(d):,}"
        k_sigma  = f"{d['sigma'].iloc[0]:,.3f}"
        k_cl     = f"{d['CL'].iloc[0]:,.0f}"
        k_ucl    = f"{d['UCL'].iloc[0]:,.0f}"
        k_lcl    = f"{d['LCL'].iloc[0]:,.0f}"
        k_out    = f"{int(d['is_out'].sum()):,}"

        # 타일 스타일
        tile = (
            "flex:1; min-width: 140px; background:#fafafa; border:1px solid #eaeaea;"
            "border-radius:14px; padding:14px 12px; text-align:center; "
            "box-shadow: 0 1px 2px rgba(0,0,0,.04);"
        )
        label = "color:#6b7280; font-size:13px; line-height:16px;"
        val   = "font-size:30px; font-weight:800; color:#111827; margin-top:2px;"
        val_y = "font-size:30px; font-weight:800; color:#f59e0b; margin-top:2px;"  # CL
        val_r = "font-size:30px; font-weight:800; color:#ef4444; margin-top:2px;"  # UCL
        val_b = "font-size:30px; font-weight:800; color:#2563eb; margin-top:2px;"  # LCL

        html = f"""
        <div style="display:flex; gap:14px; align-items:stretch; width:100%; padding:6px 4px;">
          <div style="{tile}"><div style="{label}">Groups</div><div style="{val}">{k_groups}</div></div>
          <div style="{tile}"><div style="{label}">Stddev</div><div style="{val}">{k_sigma}</div></div>
          <div style="{tile}"><div style="{label}">CL</div><div style="{val_y}">{k_cl}</div></div>
          <div style="{tile}"><div style="{label}">UCL</div><div style="{val_r}">{k_ucl}</div></div>
          <div style="{tile}"><div style="{label}">LCL</div><div style="{val_b}">{k_lcl}</div></div>
          <div style="{tile}"><div style="{label}">한계밖 포인트</div><div style="{val}">{k_out}</div></div>
        </div>
        """
        return ui.HTML(html)

    # -------- B) 월별 한계초과 막대 --------
    @output(id=sid("viol_bar"))
    @render.ui
    def _viol_bar():
        mm = mbar
        fig = go.Figure(go.Bar(x=mm["ym"], y=mm["out_cnt"], name="Out-of-control count"))
        fig.update_layout(
            height=240, template="plotly_white",
            margin=dict(l=10, r=10, t=30, b=30),
            title=f"월별 한계초과 개수 (Mold {mold})"
        )
        return ui.HTML(fig.to_html(include_plotlyjs=False, full_html=False))

    # -------- C) 이상 포인트 테이블 --------
    @output(id=sid("viol_table"))
    @render.data_frame
    def _viol_table():
        t = df_flag[df_flag["is_out"]][["date", "value", "UCL", "LCL"]].copy()
        t["date"] = pd.to_datetime(t["date"]).dt.strftime("%Y-%m-%d")
        return render.DataGrid(t, row_selection_mode="none", filters=True)

    # -------- D) Cp/Cpk 시각화 (샘플) --------
    @output(id=sid("cpk_plot"))
    @render.ui
    def _cpk_plot():
        x = np.random.normal(0, 1, 100)
        fig = go.Figure(go.Histogram(x=x, nbinsx=20))
        fig.update_layout(template="plotly_white", height=320, title="Cp/Cpk 분포 (샘플)")
        return ui.HTML(fig.to_html(include_plotlyjs=False, full_html=False))

    # -------- E) 상관분석 Heatmap (샘플) --------
    @output(id=sid("heatmap"))
    @render.ui
    def _heatmap():
        z = np.random.rand(5, 5)
        fig = go.Figure(go.Heatmap(z=z, x=[f"Var{i}" for i in range(1,6)], y=[f"Q{i}" for i in range(1,6)]))
        fig.update_layout(template="plotly_white", height=320, title="상관분석 히트맵 (샘플)")
        return ui.HTML(fig.to_html(include_plotlyjs=False, full_html=False))

    # -------- F) 이상치 탐지 그래프 --------
    @output(id=sid("outlier_detection"))
    @render.ui
    def _outlier_detection():
        fig = go.Figure()
        fig.add_trace(go.Box(y=df_flag["value"], name="Value Distribution",
                             marker_color='lightblue', boxmean='sd'))
        outs = df_flag[df_flag["is_out"]]
        if not outs.empty:
            fig.add_trace(go.Scatter(
                y=outs["value"], x=[0] * len(outs),
                mode="markers",
                marker=dict(size=12, color="red", symbol="x", line=dict(width=2)),
                name="Outliers"
            ))
        fig.update_layout(
            height=280, template="plotly_white",
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=True, title="이상치 분포"
        )
        return ui.HTML(fig.to_html(include_plotlyjs=False, full_html=False))

    # -------- G) 이상치 버튼 결과 --------
    @output(id=sid("lookup_result"))
    @render.text
    def _lookup_result():
        btn_val = getattr(input, sid("btn_lookup"))()
        if btn_val:
            out_cnt = int(df_flag['is_out'].sum())
            return f"✓ 이상치 분포 완료 · 이상치 {out_cnt}개"
        return "버튼을 클릭하여 이상치 분포를 확인하세요"

    # -------- H) 상관관계 그래프 (이동평균 추세) --------
    @output(id=sid("correlation_graph"))
    @render.ui
    def _correlation_graph():
        d = df_flag.copy()
        d["ma7"] = d["value"].rolling(window=7, min_periods=1).mean()
        d["ma30"] = d["value"].rolling(window=30, min_periods=1).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d["date"], y=d["value"], mode="lines",
                                 name="원본값", line=dict(color="lightgray", width=1), opacity=0.5))
        fig.add_trace(go.Scatter(x=d["date"], y=d["ma7"],  mode="lines",
                                 name="MA7",  line=dict(color="blue", width=2)))
        fig.add_trace(go.Scatter(x=d["date"], y=d["ma30"], mode="lines",
                                 name="MA30", line=dict(color="red", width=2, dash="dash")))
        fig.update_layout(
            height=280, template="plotly_white",
            margin=dict(l=20, r=20, t=40, b=20),
            title="이동평균 추세 분석",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return ui.HTML(fig.to_html(include_plotlyjs=False, full_html=False))

    # -------- I) SHAP (샘플: feature importance) --------
    @output(id=sid("shap_plot"))
    @render.ui
    def _shap_plot():
        features = ["온도", "습도", "압력", "속도", "시간"]
        importance = np.array([8.5, 6.2, 5.8, 4.3, 2.1])
        fig = go.Figure(go.Bar(
            x=importance, y=features, orientation='h',
            marker=dict(color=importance, colorscale='Blues', showscale=False),
            text=[f"{v:.1f}" for v in importance], textposition='outside'
        ))
        fig.update_layout(
            height=280, template="plotly_white",
            margin=dict(l=60, r=20, t=40, b=20),
            title="Feature Importance (SHAP, 샘플)",
            xaxis_title="중요도", yaxis_title=""
        )
        return ui.HTML(fig.to_html(include_plotlyjs=False, full_html=False))
