# modules/page_operations.py
from shiny import ui, render, reactive
import pandas as pd
import matplotlib.pyplot as plt
from shared import streaming_df
from utils.real_time_streamer import RealTimeStreamer


SENSOR_COLS = ["molten_temp", "cast_pressure", "upper_mold_temp1", "sleeve_temperature"]


def ui_operations():
    return ui.page_fluid(
        ui.h3("🏭 현장 운영 담당자 탭 (Real-time Operations)"),

        ui.layout_columns(
            ui.card(
                ui.card_header("스트리밍 제어"),
                ui.input_action_button("start", "▶ 시작", class_="btn-success me-2"),
                ui.input_action_button("pause", "⏸ 일시정지", class_="btn-warning me-2"),
                ui.input_action_button("reset", "🔄 리셋", class_="btn-secondary me-2"),
                ui.br(),
                ui.output_ui("stream_status"),
                style="text-align:center; padding:1rem;"
            ),
        ),

        ui.card(
            ui.card_header("실시간 센서 모니터링"),
            ui.output_plot("live_plot", height="400px")
        ),

        ui.layout_columns(
            ui.card(ui.card_header("최근 데이터 (10개)"), ui.output_table("recent_data"))
        ),
    )


def server_operations(input, output, session):
    # ✅ Reactive 상태 관리 - reactive.value로 수정
    streamer = reactive.value(RealTimeStreamer(streaming_df[SENSOR_COLS]))
    current_data = reactive.value(pd.DataFrame())
    is_streaming = reactive.value(False)

    # ▶ 시작
    @reactive.effect
    @reactive.event(input.start)
    def _start():
        print("[INFO] ▶ Start pressed")
        is_streaming.set(True)

    # ⏸ 일시정지
    @reactive.effect
    @reactive.event(input.pause)
    def _pause():
        print("[INFO] ⏸ Pause pressed")
        is_streaming.set(False)

    # 🔄 리셋
    @reactive.effect
    @reactive.event(input.reset)
    def _reset():
        print("[INFO] 🔄 Reset pressed")
        # streamer 자체를 새로 생성하거나 reset 호출
        current_streamer = streamer()
        current_streamer.reset_stream()
        current_data.set(pd.DataFrame())
        is_streaming.set(False)

    # ✅ 스트리밍 루프: invalidate_later 반복
    @reactive.effect
    def _stream_loop():
        if not is_streaming():
            return

        reactive.invalidate_later(1.0)  # 1초마다 실행

        s = streamer()
        next_batch = s.get_next_batch(1)
        if next_batch is not None and not next_batch.empty:
            df = s.get_current_data()
            current_data.set(df)
            print(f"[LOOP] index={s.current_index}, shape={df.shape}")
        else:
            print("[LOOP] stream ended")
            is_streaming.set(False)

    # ✅ 상태 표시
    @output
    @render.ui
    def stream_status():
        if is_streaming():
            progress = streamer().progress()
            return ui.div(f"🟢 스트리밍 중 ({progress:.3f}%)", class_="fw-bold text-success")
        return ui.div("🔴 정지됨", class_="fw-bold text-danger")

    # ✅ Matplotlib 실시간 그래프
    @output
    @render.plot
    def live_plot():
        # 반드시 reactive 참조
        df = current_data()
        
        fig, ax = plt.subplots(figsize=(10, 4))
        
        if df.empty:
            ax.text(0.5, 0.5, "▶ Start Streaming", ha="center", va="center", fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            # 데이터가 있을 때만 플롯
            for col in SENSOR_COLS[:2]:  # 처음 2개 센서만
                if col in df.columns:
                    ax.plot(df.index, df[col].values, label=col, marker='o', markersize=3)
            
            ax.legend(loc='best')
            ax.set_title("Real Time Sensor Data (1초 간격)")
            ax.set_xlabel("Index")
            ax.set_ylabel("Sensor Value")
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    # ✅ 최근 데이터
    @output
    @render.table
    def recent_data():
        df = current_data()  # 반드시 reactive 참조
        if df.empty:
            return pd.DataFrame({"상태": ["데이터 없음"]})
        return df.tail(10).round(2)