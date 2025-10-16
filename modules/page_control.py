from shiny import ui, render
import plotly.graph_objs as go
import pandas as pd
import numpy as np


def ui_control():
    return ui.page_fluid(
        ui.h3("공정 관리 상태 분석"),
        ui.p("단변량 및 다변량 관리도를 통해 공정 이상 여부를 모니터링하고, 발생 로그를 확인합니다."),
        ui.hr(),

        ui.layout_columns(
            ui.card(
                ui.card_header("📈 단변량 관리도 (예: 주조압력)"),
                ui.output_plot("univariate_chart", height="320px")
            ),
            ui.card(
                ui.card_header("📊 다변량 관리도 (Hotelling T²)"),
                ui.output_plot("multivariate_chart", height="320px")
            ),
            col_widths=[6, 6]
        ),
        ui.hr(),
        ui.card(
            ui.card_header("⚠️ 이상 발생 로그"),
            ui.output_table("control_log")
        ),
        style="max-width:1300px;margin:0 auto;"
    )


def server_control(input, output, session):
    # 샘플 데이터 생성
    np.random.seed(0)
    x = np.arange(1, 51)
    y = np.random.normal(100, 5, size=50)
    ucl, lcl = 110, 90

    # 단변량 관리도
    @output
    @render.plot
    def univariate_chart():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name='값'))
        fig.add_hline(y=ucl, line_dash="dash", line_color="red", name="UCL")
        fig.add_hline(y=lcl, line_dash="dash", line_color="red", name="LCL")
        fig.update_layout(title="주조압력 관리도", template="plotly_white")
        return fig

    # 다변량 관리도
    @output
    @render.plot
    def multivariate_chart():
        t2 = np.random.chisquare(df=3, size=50)
        ucl = 7.8
        fig = go.Figure()
        fig.add_trace(go.Bar(x=x, y=t2, name='T²'))
        fig.add_hline(y=ucl, line_dash="dash", line_color="red", name="UCL")
        fig.update_layout(title="Hotelling T² 관리도", template="plotly_white")
        return fig

    # 이상 발생 로그
    @output
    @render.table
    def control_log():
        df = pd.DataFrame({
            "발생일시": pd.date_range("2025-10-01", periods=5, freq="D"),
            "변수명": ["주조압력", "용탕온도", "형체력", "하형온도1", "주조속도"],
            "이상유형": ["UCL 초과", "LCL 미만", "급격한 변동", "UCL 초과", "공정불안정"]
        })
        return df
