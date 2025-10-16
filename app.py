from shiny import App, ui
from modules import (
    page_operations, page_quality, page_input,
    page_control, page_cause, page_monitoring
)
from pathlib import Path

www_dir = Path(__file__).parent / "www"

# ✅ 최신 버전 Plotly.js (v2.35.2)
plotly_head = ui.tags.head(
    ui.tags.script(src="https://cdn.plot.ly/plotly-2.35.2.min.js")
)

app_ui = ui.page_fluid(
    plotly_head,
    ui.page_navbar(
        ui.nav_panel("실시간 운영 현황", page_operations.ui_operations()),
        # ui.nav_panel("품질 관리 분석", page_quality.ui_quality()),
        ui.nav_panel("공정 관리 상태", page_control.ui_control()),
        ui.nav_panel("불량 원인 분석", page_cause.ui_cause()),
        ui.nav_panel("모델 모니터링", page_monitoring.ui_monitoring()),
        ui.nav_panel("공정 불량 예측", page_input.inputs_layout()),
        title="주조 공정 실시간 데이터 관리 대시보드",
        id="main_nav",
        bg="#2C3E50",
        inverse=True
    )
)

def server(input, output, session):
    page_operations.server_operations(input, output, session)
    page_quality.server_quality(input, output, session)
    page_control.server_control(input, output, session)
    page_cause.server_cause(input, output, session)
    page_monitoring.server_monitoring(input, output, session)
    page_input.page_input_server(input, output, session)

app = App(app_ui, server, static_assets=www_dir)



