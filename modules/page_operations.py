# modules/page_operations.py
from shiny import ui, render, reactive
import pandas as pd
from shared import streaming_df
from utils.real_time_streamer import RealTimeStreamer
from utils.kpi_metrics import calculate_realtime_metrics
from viz.operation_plots import plot_live, plot_oee, plot_mold_pie, plot_mold_ratio


# -----------------------------
# 전역 변수
# -----------------------------
SENSOR_COLS = ["molten_temp", "cast_pressure", "upper_mold_temp1", "sleeve_temperature"]
MOLD_CODES = streaming_df["mold_code"].unique().tolist()
COLUMNS = [
    "mold_code", "passorfail", "working", "tryshot_signal",
    "facility_operation_cycleTime", "production_cycletime"
] + SENSOR_COLS


# -----------------------------
# UI
# -----------------------------
def ui_operations():
    from textwrap import dedent
    return ui.page_fluid(
        # ✅ 폰트 및 CSS
        ui.tags.link(
            href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;600;700&display=swap",
            rel="stylesheet"
        ),
        ui.tags.style(dedent("""
            * { font-family: 'Noto Sans KR', sans-serif; }
            body { background-color: #f5f7fa; padding: 2rem 0; }
            .card { border: 1px solid #e5e7eb; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); background: white; margin-bottom: 1.5rem; }
            .card-header { background-color: #f9fafb; border-bottom: 1px solid #e5e7eb; color: #1f2937; font-weight: 600; padding: 1rem 1.5rem; font-size: 0.95rem; }
            .btn { border-radius: 6px; font-weight: 500; padding: 0.5rem 1rem; font-size: 0.9rem; }
            .kpi-card { border: 1px solid #e5e7eb; border-radius: 8px; padding: 1.5rem; text-align: center; margin-bottom: 1rem; }
            .kpi-value { font-size: 2.5rem; font-weight: 700; color: #1f2937; margin: 0.5rem 0; }
            .kpi-label { font-size: 0.85rem; color: #6b7280; font-weight: 500; }
            .oee-highlight { background: #3b82f6; color: white; border: none; }
            .oee-highlight .kpi-value { color: white; }
            .oee-highlight .kpi-label { color: #e0e7ff; }
            .status-badge { padding: 0.4rem 1rem; border-radius: 6px; font-weight: 600; font-size: 0.85rem; }
        """)),

        ui.div(
            ui.h2("🏭 실시간 공정 모니터링 대시보드", class_="mb-4 text-center"),
            style="max-width: 1400px; margin: 0 auto; padding: 0 1rem;"
        ),

        ui.div(
            ui.h4("실시간 모니터링", class_="mt-4 mb-3"),
            # -----------------------------
            # 상단: 스트리밍 제어 + KPI 그리드
            # -----------------------------
            ui.layout_columns(
                # 좌측: 스트리밍 그래프
                ui.card(
                    ui.card_header("📈 실시간 센서 모니터링"),
                    ui.div(
                        ui.layout_columns(
                            ui.input_selectize(
                                "sensor_select", "센서 선택",
                                SENSOR_COLS, multiple=True,
                                selected=["molten_temp", "cast_pressure"]
                            ),
                            ui.div(
                                ui.div(
                                    ui.input_action_button("start", "▶ 시작", class_="btn-success"),
                                    ui.input_action_button("pause", "⏸ 일시정지", class_="btn-warning mx-2"),
                                    ui.input_action_button("reset", "🔄 리셋", class_="btn-secondary"),
                                    class_="d-flex justify-content-end"
                                ),
                                ui.div(
                                    ui.output_ui("stream_status"),
                                    class_="d-flex justify-content-end mt-2"
                                ),
                            ),
                            col_widths=[6, 6]
                        ),
                        style="padding: 1rem; padding-bottom: 0.5rem;"
                    ),
                    ui.output_plot("live_plot", height="340px")
                ),

                # 우측: KPI 카드 5개
                ui.div(
                    # 1행
                    ui.layout_columns(
                        ui.div(
                            ui.div("⚠️", style="font-size: 1.5rem; margin-bottom: 0.5rem;"),
                            ui.div(ui.output_text("abnormal_count"), class_="kpi-value"),
                            ui.div("이상항목", class_="kpi-label"),
                            class_="kpi-card"
                        ),
                        ui.div(
                            ui.div("✅", style="font-size: 1.5rem; margin-bottom: 0.5rem;"),
                            ui.div(ui.output_text("good_rate"), class_="kpi-value"),
                            ui.div("양품율", class_="kpi-label"),
                            class_="kpi-card"
                        ),
                        col_widths=[6, 6]
                    ),
                    # 2행 (OEE 강조)
                    ui.layout_columns(
                        ui.div(
                            ui.div("📊", style="font-size: 2rem; margin-bottom: 0.5rem;"),
                            ui.div(ui.output_text("oee_value"), class_="kpi-value", style="font-size: 3rem;"),
                            ui.div("OEE (설비 종합 효율)", class_="kpi-label"),
                            class_="kpi-card oee-highlight"
                        ),
                        col_widths=[12]
                    ),
                    # 3행
                    ui.layout_columns(
                        ui.div(
                            ui.div("📦", style="font-size: 1.5rem; margin-bottom: 0.5rem;"),
                            ui.div(ui.output_text("prod_count"), class_="kpi-value"),
                            ui.div("생산량", class_="kpi-label"),
                            class_="kpi-card"
                        ),
                        ui.div(
                            ui.div("⏱", style="font-size: 1.5rem; margin-bottom: 0.5rem;"),
                            ui.div(ui.output_text("cycle_time"), class_="kpi-value"),
                            ui.div("사이클 타임", class_="kpi-label"),
                            class_="kpi-card"
                        ),
                        col_widths=[6, 6]
                    ),
                ),
                col_widths=[7, 5]
            ),

            # -----------------------------
            # 중단: 몰드별 생산 현황
            # -----------------------------
            ui.hr(),
            ui.h4("🎯 몰드별 생산 현황", class_="mt-4 mb-3"),
            ui.div(
                *[
                    ui.card(
                        ui.card_header(f"몰드 {mold}"),
                        ui.output_plot(f"mold_{mold}_pie", height="200px"),
                        ui.output_ui(f"mold_{mold}_info"),
                        style="width:100%;"
                    )
                    for mold in MOLD_CODES
                ],
                # ✅ CSS Grid로 5등분 균등 분할
                style="""
                    display: grid;
                    grid-template-columns: repeat(5, 1fr);
                    gap: 1rem;
                """
            ),

            # -----------------------------
            # 하단: OEE + 몰드별 생산 비율
            # -----------------------------
            ui.hr(),
            ui.h4("📊 생산 분석", class_="mt-4 mb-3"),
            ui.layout_columns(
                ui.card(ui.card_header("⚙️ OEE 구성 요소"), ui.output_plot("oee_chart", height="320px")),
                ui.card(ui.card_header("🥧 몰드별 전체 생산 비율"), ui.output_plot("mold_ratio", height="320px")),
                col_widths=[6, 6]
            ),

            # -----------------------------
            # 최하단: 최근 데이터 로그
            # -----------------------------
            ui.card(
                ui.card_header("🗒 최근 데이터 로그"),
                ui.output_table("recent_data")
            ),

            style="max-width: 1400px; margin: 0 auto; padding: 0 1rem;"
        )
    )


# -----------------------------
# SERVER
# -----------------------------
def server_operations(input, output, session):
    streamer = reactive.value(RealTimeStreamer(streaming_df[COLUMNS]))
    current_data = reactive.value(pd.DataFrame())
    is_streaming = reactive.value(False)

    # ▶ 스트리밍 제어
    @reactive.effect
    @reactive.event(input.start)
    def _(): is_streaming.set(True)

    @reactive.effect
    @reactive.event(input.pause)
    def _(): is_streaming.set(False)

    @reactive.effect
    @reactive.event(input.reset)
    def _():
        s = streamer()
        s.reset_stream()
        current_data.set(pd.DataFrame())
        is_streaming.set(False)

    @reactive.effect
    def _stream_loop():
        reactive.invalidate_later(1)
        if is_streaming():
            s = streamer()
            new_batch = s.get_next_batch(1)
            if new_batch is not None and not new_batch.empty:
                current_data.set(s.get_current_data())

    # -----------------------------
    # 상태 표시
    # -----------------------------
    @output
    @render.ui
    def stream_status():
        color = "#10b981" if is_streaming() else "#ef4444"
        text = "🟢 스트리밍 중" if is_streaming() else "🔴 정지됨"
        return ui.div(text, class_="status-badge text-white", style=f"background:{color};")

    # -----------------------------
    # KPI 계산
    # -----------------------------
    @reactive.calc
    def metrics():
        return calculate_realtime_metrics(current_data(), MOLD_CODES)

    # KPI 출력
    @output 
    @render.text
    def abnormal_count(): return f"{metrics()['abnormal']}"

    @output 
    @render.text
    def good_rate(): return f"{metrics()['good_rate']:.1f}%"

    @output 
    @render.text
    def prod_count(): return f"{metrics()['prod_count']}"

    @output 
    @render.text
    def cycle_time(): return f"{metrics()['cycle_time']:.1f}s"

    @output 
    @render.text
    def oee_value(): return f"{metrics()['oee']*100:.1f}%"

    # -----------------------------
    # 그래프 출력
    # -----------------------------
    @output
    @render.plot
    def live_plot(): return plot_live(current_data(), input.sensor_select())

    @output 
    @render.plot
    def oee_chart(): return plot_oee(metrics())

    @output 
    @render.plot
    def mold_ratio(): return plot_mold_ratio(metrics()["molds"])

    # -----------------------------
    # 몰드별 카드 (파이 + 텍스트)
    # -----------------------------
    for mold in MOLD_CODES:
        @output(id=f"mold_{mold}_pie")
        @render.plot
        def mold_pie(mold=mold):
            return plot_mold_pie(metrics()["molds"][mold])

        @output(id=f"mold_{mold}_info")
        @render.ui
        def mold_info(mold=mold):
            data = metrics()["molds"][mold]
            return ui.div(
                ui.p(f"✅ 양품: {data['good']} EA", class_="mb-1 text-success fw-bold"),
                ui.p(f"❌ 불량: {data['defect']} EA", class_="mb-1 text-danger fw-bold"),
                ui.p(f"📊 생산율: {data['rate']:.1f}%", class_="text-primary fw-bold mb-0"),
                style="text-align:center; padding:0.5rem;"
            )

    # -----------------------------
    # 최근 로그
    # -----------------------------
    @output
    @render.table
    def recent_data():
        df = current_data()
        return df.tail(10).round(2) if not df.empty else pd.DataFrame({"상태": ["데이터 없음"]})
