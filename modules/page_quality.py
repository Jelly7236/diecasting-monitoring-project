# modules/page_quality.py
from shiny import ui, render
import plotly.graph_objs as go
import numpy as np

def ui_quality():
    return ui.page_fluid(
        ui.h3("품질관리팀 탭 (Quality Control)"),
        ui.layout_columns(
            ui.card(ui.card_header("Cp/Cpk 시각화"), ui.output_ui("cpk_plot")),
            ui.card(ui.card_header("관리도 (Control Chart)"), ui.output_ui("control_chart")),
        ),
        ui.card(ui.card_header("상관분석 Heatmap"), ui.output_ui("heatmap")),
    )

def server_quality(input, output, session):
    @output
    @render.ui
    def cpk_plot():
        x = np.random.normal(0, 1, 100)
        fig = go.Figure(go.Histogram(x=x, nbinsx=20, marker_color="#007bff"))
        fig.update_layout(title="Cp/Cpk 분포 (샘플)", template="plotly_white")
        html = fig.to_html(include_plotlyjs=False, full_html=False)
        return ui.HTML(f"<div style='width:100%;height:400px;'>{html}</div>")

    @output
    @render.ui
    def control_chart():
        y = np.random.normal(50, 3, 30)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y, mode="lines+markers"))
        fig.add_hline(y=np.mean(y), line_color="green", annotation_text="CL")
        fig.add_hline(y=np.mean(y)+3*np.std(y), line_color="red", annotation_text="UCL")
        fig.add_hline(y=np.mean(y)-3*np.std(y), line_color="red", annotation_text="LCL")
        fig.update_layout(title="관리도 (X-bar Chart)", template="plotly_white")
        html = fig.to_html(include_plotlyjs=False, full_html=False)
        return ui.HTML(f"<div style='width:100%;height:400px;'>{html}</div>")

    @output
    @render.ui
    def heatmap():
        z = np.random.rand(5, 5)
        fig = go.Figure(go.Heatmap(z=z, x=[f"Var{i}" for i in range(1,6)], y=[f"Q{i}" for i in range(1,6)]))
        fig.update_layout(title="상관분석 히트맵", template="plotly_white")
        html = fig.to_html(include_plotlyjs=False, full_html=False)
        return ui.HTML(f"<div style='width:100%;height:400px;'>{html}</div>")
