# modules/page_operations.py
from shiny import ui, render, reactive
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from shared import streaming_df
from utils.real_time_streamer import RealTimeStreamer

SENSOR_COLS = ["molten_temp", "cast_pressure", "upper_mold_temp1", "sleeve_temperature"]
MOLD_CODES = streaming_df["mold_code"].unique().tolist()
COLUMNS = ["mold_code", "passorfail"] + SENSOR_COLS


def ui_operations():
    return ui.page_fluid(
        ui.h2("🏭 실시간 공정 모니터링 대시보드", class_="mb-3 fw-bold"),

        # 상단
        ui.layout_columns(
            ui.card(
                ui.card_header("📈 실시간 데이터 스트리밍"),
                ui.input_selectize("sensor_select", "센서 선택", SENSOR_COLS, multiple=True, selected=["molten_temp"]),
                ui.div(
                    ui.input_action_button("start", "▶ 시작", class_="btn-success me-2"),
                    ui.input_action_button("pause", "⏸ 일시정지", class_="btn-warning me-2"),
                    ui.input_action_button("reset", "🔄 리셋", class_="btn-secondary me-2"),
                    class_="mb-2"
                ),
                ui.output_ui("stream_status"),
                ui.output_plot("live_plot", height="350px")
            ),
            ui.layout_columns(
                ui.card(ui.card_header("⚠️ 이상항목"), ui.output_text("abnormal_count")),
                ui.card(ui.card_header("✅ 양품율"), ui.output_text("good_rate")),
                ui.card(ui.card_header("📦 생산량"), ui.output_text("prod_count")),
                ui.card(ui.card_header("⏱ 사이클 타임"), ui.output_text("cycle_time")),
            ),
        ),

        ui.hr(),

        # 중단: 몰드별 카드
        ui.h4("🎯 몰드별 생산 현황", class_="mt-3 mb-2 fw-bold"),
        ui.layout_columns(
            *[
                ui.card(
                    ui.card_header(f"몰드 {mold}"),
                    ui.output_plot(f"mold_{mold}_pie", height="220px"),
                    ui.output_ui(f"mold_{mold}_info")
                )
                for mold in MOLD_CODES
            ]
        ),

        ui.hr(),

        # 하단: OEE + 전체 비율
        ui.layout_columns(
            ui.card(ui.card_header("⚙️ OEE(설비 종합 효율)"), ui.output_plot("oee_chart", height="300px")),
            ui.card(ui.card_header("🥧 몰드별 전체 생산 비율"), ui.output_plot("mold_ratio", height="300px")),
        ),

        ui.hr(),

        # 최하단: 로그
        ui.card(
            ui.card_header("🗒 최근 데이터 로그"),
            ui.output_table("recent_data")
        ),
    )


def server_operations(input, output, session):
    streamer = reactive.value(RealTimeStreamer(streaming_df[COLUMNS]))
    current_data = reactive.value(pd.DataFrame())
    is_streaming = reactive.value(False)

    # ▶ 시작
    @reactive.effect
    @reactive.event(input.start)
    def _start():
        is_streaming.set(True)
        print("[INFO] ▶ Start streaming")

    # ⏸ 일시정지
    @reactive.effect
    @reactive.event(input.pause)
    def _pause():
        is_streaming.set(False)
        print("[INFO] ⏸ Pause streaming")

    # 🔄 리셋
    @reactive.effect
    @reactive.event(input.reset)
    def _reset():
        s = streamer()
        s.reset_stream()
        current_data.set(pd.DataFrame())
        is_streaming.set(False)
        print("[INFO] 🔄 Reset stream")

    # 실시간 루프
    @reactive.effect
    def _stream_loop():
        reactive.invalidate_later(1.0)
        if not is_streaming():
            return
        s = streamer()
        next_batch = s.get_next_batch(1)
        if next_batch is not None and not next_batch.empty:
            current_data.set(s.get_current_data())

    # 상태 표시
    @output
    @render.ui
    def stream_status():
        return ui.div(
            "🟢 스트리밍 중" if is_streaming() else "🔴 정지됨",
            class_="fw-bold text-success" if is_streaming() else "fw-bold text-danger"
        )

    # 실시간 그래프
    @output
    @render.plot
    def live_plot():
        df = current_data()
        cols = input.sensor_select()
        fig, ax = plt.subplots(figsize=(10, 4))
        if df.empty:
            ax.text(0.5, 0.5, "▶ Start Streaming", ha="center", va="center", fontsize=14)
            ax.axis("off")
        else:
            for col in cols:
                if col in df.columns:
                    ax.plot(df.index, df[col], label=col, lw=1.5)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_title("실시간 센서 데이터")
        plt.tight_layout()
        return fig

    # ✅ 누적 기반 메트릭 계산 (passorfail=1 → 불량 / 0 → 양품)
    @reactive.calc
    def get_realtime_metrics():
        df = current_data()
        if df.empty:
            molds_init = {m: {"good": 0, "defect": 0, "rate": 0.0} for m in MOLD_CODES}
            return {"abnormal": 0, "good_rate": 0.0, "prod_count": 0, "cycle_time": 0.0, "molds": molds_init}

        n = len(df)
        abnormal = int(np.sum(df["molten_temp"] > 700)) if "molten_temp" in df.columns else 0
        good_count = int(np.sum(df["passorfail"] == 0))
        defect_count = int(np.sum(df["passorfail"] == 1))
        good_rate = (good_count / n) * 100
        prod_count = n
        cycle_time = df["cast_pressure"].mean() / 10 if "cast_pressure" in df.columns else 50.0

        # 몰드별 누적 통계
        mold_data = {}
        mold_group = df.groupby("mold_code")["passorfail"].value_counts().unstack(fill_value=0)
        for mold in MOLD_CODES:
            if mold in mold_group.index:
                good = mold_group.loc[mold].get(0.0, 0)  # ✅ 0이 양품
                defect = mold_group.loc[mold].get(1.0, 0)  # ✅ 1이 불량
                total = good + defect
                rate = (good / total * 100) if total > 0 else 0.0
                mold_data[mold] = {"good": good, "defect": defect, "rate": rate}
            else:
                mold_data[mold] = {"good": 0, "defect": 0, "rate": 0.0}

        return {
            "abnormal": abnormal,
            "good_rate": good_rate,
            "prod_count": prod_count,
            "cycle_time": cycle_time,
            "molds": mold_data
        }

    # 상단 KPI
    @output
    @render.text
    def abnormal_count():
        return f"{get_realtime_metrics()['abnormal']} 건"

    @output
    @render.text
    def good_rate():
        return f"{get_realtime_metrics()['good_rate']:.1f} %"

    @output
    @render.text
    def prod_count():
        return f"{get_realtime_metrics()['prod_count']} EA"

    @output
    @render.text
    def cycle_time():
        return f"{get_realtime_metrics()['cycle_time']:.1f} sec"

    # 몰드별 카드
    for mold in MOLD_CODES:

        @output(id=f"mold_{mold}_pie")
        @render.plot
        def mold_pie(mold=mold):
            metrics = get_realtime_metrics()
            data = metrics["molds"].get(mold, {"good": 0, "defect": 0})
            fig, ax = plt.subplots(figsize=(3, 3))
            sizes = [data["good"], data["defect"]]
            colors = ["#28a745", "#dc3545"]
            labels = ["양품", "불량"]

            if sum(sizes) == 0:
                ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center")
                ax.axis("off")
            else:
                ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
                ax.set_title(f"{mold} 생산 현황")
            plt.tight_layout()
            return fig

        @output(id=f"mold_{mold}_info")
        @render.ui
        def mold_info(mold=mold):
            metrics = get_realtime_metrics()
            data = metrics["molds"].get(mold, {"good": 0, "defect": 0, "rate": 0.0})
            return ui.div(
                ui.p(f"✅ 양품: {data['good']} EA", class_="mb-1"),
                ui.p(f"❌ 불량: {data['defect']} EA", class_="mb-1"),
                ui.p(f"📊 생산율: {data['rate']:.1f}%", class_="fw-bold text-primary mb-0"),
                style="text-align:center; padding:0.5rem;"
            )

    # OEE
    @output
    @render.plot
    def oee_chart():
        metrics = get_realtime_metrics()
        good_rate = metrics["good_rate"]
        availability = min(1.0, good_rate / 100 + np.random.uniform(0, 0.05))
        performance = min(1.0, good_rate / 100 + np.random.uniform(-0.05, 0.05))
        quality = good_rate / 100

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(["가동률", "성능", "품질"], [availability, performance, quality],
                      color=["#007bff", "#ffc107", "#28a745"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("비율")
        ax.set_title("OEE 구성 요소")
        ax.grid(axis='y', alpha=0.3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h, f'{h:.1%}', ha='center', va='bottom')
        plt.tight_layout()
        return fig

    # 전체 몰드 생산 비율
    @output
    @render.plot
    def mold_ratio():
        metrics = get_realtime_metrics()
        molds = metrics["molds"]
        labels = list(molds.keys())
        sizes = [molds[m]["good"] + molds[m]["defect"] for m in labels]

        fig, ax = plt.subplots(figsize=(6, 6))
        if sum(sizes) == 0:
            ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center")
            ax.axis("off")
        else:
            ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
            ax.set_title("몰드별 전체 생산 비율")
        plt.tight_layout()
        return fig

    # 최근 데이터 로그
    @output
    @render.table
    def recent_data():
        df = current_data()
        return df.tail(10).round(2) if not df.empty else pd.DataFrame({"상태": ["데이터 없음"]})
