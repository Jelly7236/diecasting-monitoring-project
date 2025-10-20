# modules/page_cause.py
from shiny import ui, render, reactive
import pandas as pd

from .cause_data import load_quality_from_file
from .cause_ui import page_ui
from .cause_viz import fig_html, build_p_chart, build_shap_bar, mold_cards_html
from .cause_service import snapshot_filter, build_log_table, report_csv_bytes

# ============================== UI ===============================
def ui_cause():
    return page_ui()

# ============================ SERVER =============================
def server_cause(input, output, session):
    df_all = load_quality_from_file()

    # 초기 컨트롤
    molds = sorted(df_all["mold_code"].unique().tolist()) if not df_all.empty else []
    session.send_input_message("p_mold", {"choices": molds, "selected": (molds[0] if molds else None)})
    if df_all["date"].notna().any():
        latest = df_all["date"].max().date()
        session.send_input_message("p_date", {"value": str(latest)})

    # 최신 날짜 버튼
    @reactive.effect
    @reactive.event(input.btn_update_date)
    def _update_date_to_latest():
        if df_all["date"].notna().any():
            latest = df_all["date"].max().date()
            session.send_input_message("p_date", {"value": str(latest)})

    # 적용 스냅샷
    @reactive.calc
    @reactive.event(input.btn_apply)
    def filt():
        if df_all.empty or input.p_mold() is None or input.p_date() is None:
            return pd.DataFrame(columns=df_all.columns)
        return snapshot_filter(df_all, input.p_mold(), pd.to_datetime(input.p_date()))

    # 상단: 몰드별 누적 카드 (가로 한 줄)
    @render.ui
    def mold_cards():
        return mold_cards_html(df_all)

    # p-관리도
    @render.ui
    @reactive.event(input.btn_apply)
    def p_chart():
        dff = filt()
        if dff.empty or dff["date"].notna().sum() == 0:
            import plotly.graph_objs as go
            fig = go.Figure(); fig.add_annotation(text="데이터/기간 없음", showarrow=False)
            fig.update_layout(template="plotly_white", height=400)
            return fig_html(fig, height=400)
        return fig_html(build_p_chart(dff), height=420)

    # SHAP
    @render.ui
    @reactive.event(input.btn_apply)
    def shap_plot():
        dff = filt()
        fig = build_shap_bar(dff)
        if fig is None:
            return ui.div("SHAP 컬럼이 없어 표시할 수 없습니다.", style="color:#6b7280; padding:12px;")
        return fig_html(fig, height=420)

    # 불량 샘플 로그
    @output
    @render.table
    @reactive.event(input.btn_apply)
    def detect_log():
        return build_log_table(filt())

    # 리포트 다운로드(CSV)
    @render.download(filename=lambda: f"report_{input.p_mold() or 'ALL'}_{input.p_date() or 'NA'}.csv")
    def btn_report():
        log = build_log_table(filt())
        yield report_csv_bytes(log)
