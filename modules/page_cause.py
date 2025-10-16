from shiny import ui, render
import plotly.graph_objs as go
import pandas as pd
import numpy as np


def ui_cause():
    return ui.page_fluid(
        ui.h3("불량 원인 분석"),
        ui.p("기간을 선택하여 불량 발생 경향, 주요 영향 변수(SHAP), 몰드별 불량율, 룰별 관계를 분석합니다."),
        ui.hr(),

        ui.layout_columns(
            ui.input_date_range("date_range", "📅 분석 기간 선택",
                                start="2025-09-01", end="2025-10-01"),
            col_widths=[12]
        ),
        ui.hr(),
        ui.layout_columns(
            ui.card(
                ui.card_header("📊 p-관리도 (불량률 추이)"),
                ui.output_plot("p_chart", height="300px")
            ),
            ui.card(
                ui.card_header("🔥 SHAP 주요 변수 영향도"),
                ui.output_plot("shap_plot", height="300px")
            ),
            col_widths=[6, 6]
        ),
        ui.hr(),
        ui.layout_columns(
            ui.card(
                ui.card_header("⚙️ 몰드코드별 불량율"),
                ui.output_plot("mold_defect", height="320px")
            ),
            ui.card(
                ui.card_header("📈 불량-룰 관계 분석"),
                ui.output_plot("rule_relation", height="320px")
            ),
            col_widths=[6, 6]
        ),
        style="max-width:1300px;margin:0 auto;"
    )


def server_cause(input, output, session):
    # 샘플 불량률 데이터
    dates = pd.date_range("2025-09-01", periods=30)
    defect_rate = np.random.uniform(0.01, 0.1, size=30)

    @output
    @render.plot
    def p_chart():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=defect_rate, mode='lines+markers', name='불량률'))
        fig.update_layout(title="p관리도 - 일별 불량률", template="plotly_white")
        return fig

    @output
    @render.plot
    def shap_plot():
        variables = ["용탕온도", "형체력", "주조압력", "하형온도1", "슬리브온도"]
        shap_values = np.abs(np.random.randn(5))
        fig = go.Figure(go.Bar(x=shap_values, y=variables, orientation='h', name='SHAP'))
        fig.update_layout(title="SHAP 변수 중요도", template="plotly_white")
        return fig

    @output
    @render.plot
    def mold_defect():
        molds = ["8412", "8573", "8600", "8722", "8917"]
        rates = np.random.uniform(2, 10, size=5)
        fig = go.Figure(go.Bar(x=molds, y=rates, text=[f"{r:.1f}%" for r in rates], textposition='auto'))
        fig.update_layout(title="몰드코드별 불량율", template="plotly_white")
        return fig

    @output
    @render.plot
    def rule_relation():
        x = ["Rule1", "Rule2", "Rule3", "Rule4", "Rule5"]
        y = np.random.randint(10, 100, size=5)
        fig = go.Figure(go.Bar(x=x, y=y))
        fig.update_layout(title="불량과 관리도 룰 간 상관성", template="plotly_white")
        return fig
