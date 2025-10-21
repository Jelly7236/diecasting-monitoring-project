from shiny import ui, render, reactive
import pandas as pd
import numpy as np
from shared import streaming_df, iso_models, iso_features, current_state, prediction_state
from utils.real_time_streamer import RealTimeStreamer
from utils.kpi_metrics import calculate_realtime_metrics
from utils.realtime_predictor import predict_quality
from viz.operation_plots import plot_live, plot_oee, plot_mold_pie, plot_mold_ratio


# -----------------------------
# 전역 변수
# -----------------------------
EXCLUDE_COLS = ["id", "count", "mold_code", "passorfail", "working", "tryshot_signal"]
SENSOR_COLS = [
    col for col in streaming_df.select_dtypes(include=np.number).columns
    if col not in EXCLUDE_COLS
]

MOLD_CODES = streaming_df["mold_code"].unique().tolist()
COLUMNS = ["datetime", "mold_code", "passorfail", "working", "tryshot_signal", "count"] + SENSOR_COLS


# -----------------------------
# UI
# -----------------------------
def ui_operations():
    from textwrap import dedent
    return ui.page_fluid(
        ui.tags.link(
            href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;600;700&display=swap",
            rel="stylesheet"
        ),
        ui.tags.style(dedent("""
            * { font-family: 'Noto Sans KR', sans-serif; }
            body { background-color: #f5f7fa; padding: 2rem 0; }
            .card { border: 1px solid #e5e7eb; border-radius: 8px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.05); background: white; margin-bottom: 1.5rem; }
            .card-header { background-color: #f9fafb; border-bottom: 1px solid #e5e7eb;
                           color: #1f2937; font-weight: 600; padding: 1rem 1.5rem; font-size: 0.95rem; }
            .btn { border-radius: 6px; font-weight: 500; padding: 0.5rem 1rem; font-size: 0.9rem; }
            .kpi-card { border: 1px solid #e5e7eb; border-radius: 8px; padding: 1.5rem;
                        text-align: center; margin-bottom: 1rem; }
            .kpi-value { font-size: 2.5rem; font-weight: 700; color: #1f2937; margin: 0.5rem 0; }
            .kpi-label { font-size: 0.85rem; color: #6b7280; font-weight: 500; }
            .oee-highlight { background: #3b82f6; color: white; border: none; }
            .oee-highlight .kpi-value { color: white; }
            .oee-highlight .kpi-label { color: #e0e7ff; }
            .status-badge { padding: 0.4rem 1rem; border-radius: 6px;
                            font-weight: 600; font-size: 0.85rem; }
        """)),

        ui.div(
            ui.h2("🏭 실시간 공정 모니터링 대시보드", class_="mb-4 text-center"),
            style="max-width: 1400px; margin: 0 auto; padding: 0 1rem;"
        ),

        ui.div(
            ui.h4("실시간 모니터링", class_="mt-4 mb-3"),
            ui.layout_columns(
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
                                ui.div(ui.output_ui("stream_status"),
                                       class_="d-flex justify-content-end mt-2"),
                            ),
                            col_widths=[6, 6]
                        ),
                        style="padding: 1rem;"
                    ),
                    ui.output_plot("live_plot", height="340px")
                ),
                ui.div(
                    ui.layout_columns(
                        ui.div(
                            ui.div("⚠️"), ui.div(ui.output_text("abnormal_count"), class_="kpi-value"),
                            ui.div("이상항목", class_="kpi-label"), class_="kpi-card"
                        ),
                        ui.div(
                            ui.div("✅"), ui.div(ui.output_text("good_rate"), class_="kpi-value"),
                            ui.div("양품율", class_="kpi-label"), class_="kpi-card"
                        ),
                        col_widths=[6, 6]
                    ),
                    ui.layout_columns(
                        ui.div(
                            ui.div("📊"), ui.div(ui.output_text("oee_value"), class_="kpi-value", style="font-size: 3rem;"),
                            ui.div("OEE (설비 종합 효율)", class_="kpi-label"), class_="kpi-card oee-highlight"
                        ),
                        col_widths=[12]
                    ),
                    ui.layout_columns(
                        ui.div(
                            ui.div("📦"), ui.div(ui.output_text("prod_count"), class_="kpi-value"),
                            ui.div("생산량", class_="kpi-label"), class_="kpi-card"
                        ),
                        ui.div(
                            ui.div("⏱"), ui.div(ui.output_text("cycle_time"), class_="kpi-value"),
                            ui.div("사이클 타임", class_="kpi-label"), class_="kpi-card"
                        ),
                        col_widths=[6, 6]
                    ),
                ),
                col_widths=[7, 5]
            ),

            ui.hr(),
            ui.card(ui.card_header("🤖 실시간 불량 예측 결과 (실제값 비교)"), ui.output_table("recent_prediction")),

            ui.hr(),
            ui.h4("🎯 몰드별 생산 현황", class_="mt-4 mb-3"),
            ui.div(
                *[
                    ui.card(
                        ui.card_header(f"몰드 {mold}"),
                        ui.output_plot(f"mold_{mold}_pie", height="200px"),
                        ui.output_ui(f"mold_{mold}_info"),
                        style="width:100%;"
                    ) for mold in MOLD_CODES
                ],
                style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 1rem;"
            ),

            ui.hr(),
            ui.h4("📊 생산 분석", class_="mt-4 mb-3"),
            ui.layout_columns(
                ui.card(ui.card_header("⚙️ OEE 구성 요소"), ui.output_plot("oee_chart", height="320px")),
                ui.card(ui.card_header("🥧 몰드별 전체 생산 비율"), ui.output_plot("mold_ratio", height="320px")),
                col_widths=[6, 6]
            ),

            ui.hr(),
            ui.card(ui.card_header("🗒 최근 데이터 로그"), ui.output_table("recent_data")),
            ui.card(ui.card_header("⚠️ 최근 이상치 로그"), ui.output_table("recent_abnormal")),

            style="max-width: 1400px; margin: 0 auto; padding: 0 1rem;"
        )
    )


# -----------------------------
# SERVER
# -----------------------------
def server_operations(input, output, session):
    streamer = reactive.value(RealTimeStreamer(streaming_df[COLUMNS]))
    current_data = reactive.value(pd.DataFrame())
    detected_data = reactive.value(pd.DataFrame())
    prediction_data = reactive.value(pd.DataFrame())
    is_streaming = reactive.value(False)

    # ▶ 제어
    @reactive.effect
    @reactive.event(input.start)
    def _(): is_streaming.set(True)

    @reactive.effect
    @reactive.event(input.pause)
    def _(): is_streaming.set(False)

    @reactive.effect
    @reactive.event(input.reset)
    def _():
        s = streamer(); s.reset_stream()
        current_data.set(pd.DataFrame())
        detected_data.set(pd.DataFrame())
        prediction_data.set(pd.DataFrame())
        current_state.set(pd.DataFrame())   # ✅ 전역 current_state 초기화
        prediction_state.set(pd.DataFrame())
        is_streaming.set(False)

    # -----------------------------
    # 실시간 스트리밍 루프
    # -----------------------------
    @reactive.effect
    def _stream_loop():
        reactive.invalidate_later(1)
        if is_streaming():
            s = streamer()
            new_batch = s.get_next_batch(1)
            if new_batch is not None and not new_batch.empty:
                df_now = s.get_current_data()
                current_data.set(df_now)
                current_state.set(df_now)  # ✅ 전역 동기화 (shared에서 사용 가능)

    # -----------------------------
    # 스트리밍 상태
    # -----------------------------
    @output
    @render.ui
    def stream_status():
        color = "#10b981" if is_streaming() else "#ef4444"
        text = "🟢 스트리밍 중" if is_streaming() else "🔴 정지됨"
        return ui.div(text, class_="status-badge text-white", style=f"background:{color};")

    # -----------------------------
    # 이상치 탐지
    # -----------------------------
    def detect_anomalies(df):
        if df.empty or not iso_models or not iso_features:
            return df
        df = df.copy()
        df["anomaly"] = 0
        df["anomaly_score"] = np.nan
        for mold, group in df.groupby("mold_code"):
            model = iso_models.get(str(mold))
            if model is None:
                continue
            X = group.copy()
            for c in [c for c in iso_features if c not in X.columns]:
                X[c] = 0
            X = X[iso_features]
            preds = model.predict(X)
            scores = model.decision_function(X)
            df.loc[group.index, "anomaly"] = preds
            df.loc[group.index, "anomaly_score"] = scores
        return df

    # -----------------------------
    # KPI 계산
    # -----------------------------
    @reactive.calc
    def metrics():
        df = current_data()
        if df.empty:
            detected_data.set(pd.DataFrame())
            return {"abnormal": 0, "good_rate": 0, "prod_count": 0, "cycle_time": 0,
                    "oee": 0, "availability": 0, "performance": 0, "quality": 0, "molds": {}}
        df_detected = detect_anomalies(df)
        detected_data.set(df_detected)
        abnormal_count = (df_detected["anomaly"] == -1).sum()
        base_metrics = calculate_realtime_metrics(df_detected, MOLD_CODES)
        for k in ["availability", "performance", "quality"]:
            base_metrics.setdefault(k, 0)
        base_metrics["abnormal"] = abnormal_count
        return base_metrics

    # -----------------------------
    # ✅ 실시간 불량 예측 (실제값 비교)
    # -----------------------------
    @reactive.effect
    @reactive.event(current_data)
    def _predict_and_store():
        df = current_data()
        if df.empty:
            return
        latest_row = df.iloc[-1]
        mold = str(latest_row["mold_code"])
        one_row_df = pd.DataFrame([latest_row])
        result, err = predict_quality(one_row_df, mold)
        print(f"🔍 새 데이터 예측: mold={mold}, result={result}, err={err}")

        if result is not None and err is None:
            actual = int(latest_row.get("passorfail", np.nan)) if "passorfail" in latest_row else np.nan
            hist = prediction_data()
            new_row = pd.DataFrame([{
                "datetime": latest_row.get("datetime", pd.Timestamp.now()),
                "mold": result["mold"],
                "pred": result["pred"],
                "prob": result["prob"],
                "actual": actual
            }])
            updated = pd.concat([hist, new_row], ignore_index=True)
            prediction_data.set(updated)
            prediction_state.set(updated)

    # -----------------------------
    # 출력
    # -----------------------------
    @output
    @render.table
    def recent_prediction():
        df = prediction_data()
        if df.empty:
            return pd.DataFrame({"상태": ["예측 결과 없음"]})
        df = df.copy()
        df["예측결과"] = df["pred"].map({0: "양품", 1: "불량"})
        df["실제결과"] = df["actual"].map({0: "양품", 1: "불량"})
        df["일치여부"] = np.where(df["pred"] == df["actual"], "✅ 일치", "❌ 불일치")
        df["불량확률(%)"] = (df["prob"] * 100).round(1)
        return df[["datetime", "mold", "실제결과", "예측결과", "일치여부", "불량확률(%)"]].tail(10)

    @output
    @render.text
    def abnormal_count(): return f"{metrics()['abnormal']}"

    @output
    @render.text
    def good_rate(): return f"{metrics().get('good_rate', 0):.1f}%"

    @output
    @render.text
    def prod_count(): return f"{metrics().get('prod_count', 0)}"

    @output
    @render.text
    def cycle_time(): return f"{metrics().get('cycle_time', 0):.1f}s"

    @output
    @render.text
    def oee_value(): return f"{metrics().get('oee', 0)*100:.1f}%"

    @output
    @render.plot
    def live_plot(): return plot_live(current_data(), input.sensor_select())

    @output
    @render.plot
    def oee_chart(): return plot_oee(metrics())

    @output
    @render.plot
    def mold_ratio(): return plot_mold_ratio(metrics()["molds"])

    for mold in MOLD_CODES:
        @output(id=f"mold_{mold}_pie")
        @render.plot
        def mold_pie(mold=mold):
            data = metrics()["molds"].get(mold, {"good": 0, "defect": 0, "rate": 0})
            return plot_mold_pie(data)

        @output(id=f"mold_{mold}_info")
        @render.ui
        def mold_info(mold=mold):
            data = metrics()["molds"].get(mold, {"good": 0, "defect": 0, "rate": 0})
            return ui.div(
                ui.p(f"✅ 양품: {data['good']} EA", class_="mb-1 text-success fw-bold"),
                ui.p(f"❌ 불량: {data['defect']} EA", class_="mb-1 text-danger fw-bold"),
                ui.p(f"📊 생산율: {data['rate']:.1f}%", class_="text-primary fw-bold mb-0"),
                style="text-align:center; padding:0.5rem;"
            )

    @output
    @render.table
    def recent_data():
        df = current_data()
        if df.empty:
            return pd.DataFrame({"상태": ["데이터 없음"]})
        return df.tail(10).round(2)

    @output
    @render.table
    def recent_abnormal():
        df = detected_data()
        if df.empty or "anomaly" not in df.columns:
            return pd.DataFrame({"상태": ["최근 이상치 없음"]})
        abn = df[df["anomaly"] == -1]
        if abn.empty:
            return pd.DataFrame({"상태": ["최근 이상치 없음"]})
        return abn.tail(10).round(2)
