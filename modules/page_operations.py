# modules/page_operations.py
from shiny import ui, render, reactive
import pandas as pd
import numpy as np
from shared import streaming_df, iso_models, iso_features, current_state, prediction_state
from utils.real_time_streamer import RealTimeStreamer
from utils.kpi_metrics import calculate_realtime_metrics
from utils.realtime_predictor import predict_quality
from viz.operation_plots import plot_live, plot_oee, plot_mold_pie, plot_mold_ratio

# -----------------------------
# 전역
# -----------------------------
EXCLUDE_COLS = ["id", "count", "mold_code", "passorfail", "working", "tryshot_signal"]
SENSOR_COLS = [c for c in streaming_df.select_dtypes(include=np.number).columns if c not in EXCLUDE_COLS]
MOLD_CODES = streaming_df["mold_code"].unique().tolist()
COLUMNS = ["datetime", "mold_code", "passorfail", "working", "tryshot_signal", "count"] + SENSOR_COLS

# 연속 경보 판단 기준
N_CONSECUTIVE = 3

# 임계값(간단 예시)
THRESHOLDS = {
    "oee_min": 0.65,          # OEE 65% 미만
    "good_rate_min": 95.0,    # 양품률 95% 미만
    "cycle_time_max": 120.0,  # 사이클타임 120초 초과
}

# -----------------------------
# UI
# -----------------------------
def ui_operations():
    return ui.page_fluid(
        ui.tags.link(
            href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;600;800&display=swap",
            rel="stylesheet",
        ),
        ui.tags.link(rel="stylesheet", href="css/operations.css"),

        # 헤더
        ui.div(
            ui.h2("실시간 공정 모니터링 대시보드", class_="title"),
            ui.div(
                ui.h4("전자교반 3라인 2호기 TM Carrier RH", class_="machine"),
                ui.div(ui.output_ui("working_badge"), ui.output_ui("tryshot_badge"), class_="badge-row"),
                class_="machine-row",
            ),
            class_="header",
        ),

        # 1단: 센서(좌) + KPI(우)
        ui.div(
            ui.layout_columns(
                ui.card(
                    ui.card_header("센서"),
                    ui.layout_columns(
                        ui.input_selectize(
                            "sensor_select", None, SENSOR_COLS, multiple=True,
                            selected=["molten_temp", "cast_pressure"]
                        ),
                        ui.div(
                            ui.input_action_button("start", "▶ 시작", class_="btn btn-start"),
                            ui.input_action_button("pause", "⏸ 일시정지", class_="btn btn-pause"),
                            ui.input_action_button("reset", "🔄 리셋", class_="btn btn-reset"),
                            class_="controls"
                        ),
                        col_widths=[6, 6]
                    ),
                    ui.div(ui.output_ui("stream_status"), class_="stream-chip"),
                    ui.output_plot("live_plot", height="360px"),
                    class_="panel"
                ),
                ui.div(
                    ui.div(
                        ui.div(
                            _kpi_card("✅", "양품율", ui.output_text("good_rate"), accent="green"),
                            _kpi_card("⚠️", "이상항목", ui.output_text("abnormal_count"), accent="red"),
                            _kpi_card("📦", "생산량", ui.output_text("prod_count"), accent="indigo"),
                            class_="kpi-row"
                        ),
                        ui.div(
                            _kpi_card("⏱", "사이클타임", ui.output_text("cycle_time"), accent="cyan"),
                            _kpi_card_oee("💡", "OEE (설비 종합 효율)", ui.output_text("oee_value")),
                            class_="kpi-row second"
                        ),
                        class_="kpi-grid"
                    ),
                    class_="panel panel-right"
                ),
                col_widths=[7, 5]
            ),
            class_="section"
        ),

        ui.hr(class_="divider"),

        # 상태 요약(불량예측/이상탐지/관리도/임계) — 단일 출력(그리기 전용)
        ui.h4("상태 요약", class_="section-title"),
        ui.output_ui("status_overview"),

        # 로그(최근 5개)
        ui.card(ui.card_header("실시간 알림 메시지 (최근 5개)"),
                ui.output_table("event_feed"), class_="panel"),

        ui.hr(class_="divider"),

        # 몰드별 5분할 그리드
        ui.h4("몰드별 생산 현황", class_="section-title"),
        ui.div(
            *[
                ui.card(
                    ui.card_header(f"몰드 {mold}"),
                    ui.output_plot(f"mold_{mold}_pie", height="170px"),
                    ui.output_ui(f"mold_{mold}_info"),
                    class_="mold-card"
                ) for mold in MOLD_CODES
            ],
            class_="mold-grid-5"
        ),

        ui.hr(class_="divider"),

        # 분석
        ui.h4("생산 분석", class_="section-title"),
        ui.layout_columns(
            ui.card(ui.card_header("⚙️ OEE 구성 요소"), ui.output_plot("oee_chart", height="300px"), class_="panel"),
            ui.card(ui.card_header("🥧 몰드별 전체 생산 비율"), ui.output_plot("mold_ratio", height="300px"), class_="panel"),
            col_widths=[6, 6]
        ),

        ui.hr(class_="divider"),

        # 로그 2종
        ui.layout_columns(
            ui.card(ui.card_header("🗒 최근 데이터 로그"), ui.output_table("recent_data"), class_="panel"),
            ui.card(ui.card_header("⚠️ 최근 이상치 로그"), ui.output_table("recent_abnormal"), class_="panel"),
            col_widths=[6, 6]
        ),

        style="max-width:1400px; margin:0 auto; padding:0 1rem 2rem;"
    )


# ---- KPI 카드 컴포넌트 ----
def _kpi_card(icon, label, value_output, accent="indigo"):
    return ui.div(
        ui.div(class_=f"accent accent--{accent}"),
        ui.div(icon, class_="kpi-icon"),
        ui.div(label, class_="kpi-label"),
        ui.div(value_output, class_="kpi-value"),
        class_="kpi-card"
    )

def _kpi_card_oee(icon, label, value_output):
    return ui.div(
        ui.div(class_="accent accent--oee"),
        ui.div(icon, class_="kpi-icon oee"),
        ui.div(label, class_="kpi-label oee"),
        ui.div(value_output, class_="kpi-value oee"),
        class_="kpi-card kpi-card--oee"
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

    # 연속 카운터(팝업 없음·카드 색만 변경)
    anomaly_streak = reactive.value(0)
    defect_streak = reactive.value(0)

    # ✅ 상태 요약 사전 계산 값을 담아두는 컨테이너
    overview_state = reactive.value({
        "pred_active": False, "pred_prob": "",
        "anom_active": False, "anom_score": "",
        "spc_breach": False, "thresh_breach": False
    })

    # ▶ 제어
    @reactive.effect
    @reactive.event(input.start)
    def _():
        is_streaming.set(True)

    @reactive.effect
    @reactive.event(input.pause)
    def _():
        is_streaming.set(False)

    @reactive.effect
    @reactive.event(input.reset)
    def _():
        s = streamer(); s.reset_stream()
        current_data.set(pd.DataFrame()); detected_data.set(pd.DataFrame()); prediction_data.set(pd.DataFrame())
        current_state.set(pd.DataFrame()); prediction_state.set(pd.DataFrame())
        anomaly_streak.set(0); defect_streak.set(0)
        overview_state.set({
            "pred_active": False, "pred_prob": "",
            "anom_active": False, "anom_score": "",
            "spc_breach": False, "thresh_breach": False
        })
        is_streaming.set(False)

    # 실시간 루프
    @reactive.effect
    def _stream_loop():
        reactive.invalidate_later(2)  # 갱신 주기(초) — 필요시 조정
        if is_streaming():
            s = streamer()
            new_batch = s.get_next_batch(1)
            if new_batch is not None and not new_batch.empty:
                df_now = s.get_current_data()
                current_data.set(df_now)
                current_state.set(df_now)

    # 스트리밍 상태칩
    @output
    @render.ui
    def stream_status():
        text = "🟢 스트리밍 중" if is_streaming() else "🔴 정지됨"
        color = "#10b981" if is_streaming() else "#ef4444"
        return ui.div(text, class_="chip", style=f"background:{color}; color:white;")

    # 작동/시험샷 배지
    def _is_running(v):
        if pd.isna(v): return False
        s = str(v).strip().lower()
        return s in {"1", "true", "가동", "작동", "run", "running", "yes"}

    def _is_tryshot(v):
        if pd.isna(v): return False
        return str(v).strip().upper() == "D"  # D=시험샷, A=정상

    @output
    @render.ui
    def working_badge():
        df = current_data()
        if df.empty: return ui.span("대기", class_="badge idle")
        return ui.span("작동", class_="badge run") if _is_running(df.iloc[-1].get("working")) else ui.span("정지", class_="badge stop")

    @output
    @render.ui
    def tryshot_badge():
        df = current_data()
        if df.empty: return ui.span("정보없음", class_="badge idle")
        return ui.span("시험샷", class_="badge trial") if _is_tryshot(df.iloc[-1].get("tryshot_signal")) else ui.span("정상", class_="badge normal")

    # 이상치 탐지
    def detect_anomalies(df):
        if df.empty or not iso_models or not iso_features:
            return df
        df = df.copy()
        df["anomaly"] = 0
        df["anomaly_score"] = np.nan
        for mold, group in df.groupby("mold_code"):
            model = iso_models.get(str(mold))
            if model is None: continue
            X = group.copy()
            for c in [c for c in iso_features if c not in X.columns]:
                X[c] = 0
            X = X[iso_features]
            try:
                preds = model.predict(X)
                scores = model.decision_function(X)
                df.loc[group.index, "anomaly"] = preds
                df.loc[group.index, "anomaly_score"] = scores
            except Exception as e:
                print(f"[WARN] anomaly detection failed for mold {mold}: {e}")
        return df

    # KPI 계산 + 연속 이상치 카운트(팝업 X)
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
        base = calculate_realtime_metrics(df_detected, MOLD_CODES)
        for k in ["availability", "performance", "quality"]:
            base.setdefault(k, 0)
        base["abnormal"] = abnormal_count

        # 연속 이상치 카운트만 관리
        if abnormal_count > 0:
            anomaly_streak.set(anomaly_streak() + 1)
        else:
            anomaly_streak.set(0)

        return base

    # 실시간 예측 + 연속 카운트(팝업 X)
    @reactive.effect
    @reactive.event(current_data)
    def _predict_and_store():
        df = current_data()
        if df.empty: return
        latest = df.iloc[-1]
        mold = str(latest["mold_code"])
        res, err = predict_quality(pd.DataFrame([latest]), mold)
        if res is not None and err is None:
            actual = int(latest.get("passorfail", np.nan)) if "passorfail" in latest else np.nan
            hist = prediction_data()
            new = pd.DataFrame([{
                "datetime": latest.get("datetime", pd.Timestamp.now()),
                "mold": res["mold"], "pred": res["pred"], "prob": res["prob"], "actual": actual
            }])
            merged = pd.concat([hist, new], ignore_index=True)
            prediction_data.set(merged)
            prediction_state.set(merged)

            # 연속 불량 예측 카운트만 관리
            if res["pred"] == 1:
                defect_streak.set(defect_streak() + 1)
            else:
                defect_streak.set(0)

    # ✅ 상태 요약 사전 계산(계산 전용 이펙트)
    @reactive.effect
    @reactive.event(current_data)   # 새 배치 들어올 때만 갱신 (필요시 prediction_data/detected_data도 이벤트로 추가 가능)
    def _update_overview_state():
        # 연속 카운터 기준 활성화
        pred_active = defect_streak() >= N_CONSECUTIVE
        anom_active = anomaly_streak() >= N_CONSECUTIVE

        # 보조 텍스트: 최근 값
        prob_txt = ""
        pred_df = prediction_data()
        if not pred_df.empty and not pd.isna(pred_df.iloc[-1].get("prob", np.nan)):
            prob_txt = f"{float(pred_df.iloc[-1]['prob']) * 100:.1f}%"

        score_txt = ""
        det_df = detected_data()
        if not det_df.empty and "anomaly_score" in det_df.columns:
            try:
                score_txt = f"{det_df[det_df['anomaly'] == -1]['anomaly_score'].iloc[-1]:.3f}"
            except Exception:
                pass

        # SPC/임계 (단순 규칙) — metrics()는 여기서 한 번만 호출
        m = metrics()
        spc_breach = (m.get("oee", 0) < 0.6) or (m.get("good_rate", 100) < 92.0)
        thresh_breach = (
            (m.get("oee", 0) < THRESHOLDS["oee_min"]) or
            (m.get("good_rate", 100) < THRESHOLDS["good_rate_min"]) or
            (m.get("cycle_time", 0) > THRESHOLDS["cycle_time_max"])
        )

        overview_state.set({
            "pred_active": pred_active, "pred_prob": prob_txt,
            "anom_active": anom_active, "anom_score": score_txt,
            "spc_breach": spc_breach, "thresh_breach": thresh_breach
        })

    # ---- 상태 요약 4칸(렌더 전용) ----
    def _flag_card(title, active, desc, icon="•"):
        klass = "flag-card danger" if active else "flag-card ok"
        return ui.div(
            ui.div(class_="flag-accent danger" if active else "flag-accent ok"),
            ui.div(icon, class_="flag-icon"),
            ui.div(title, class_="flag-title"),
            ui.div(desc, class_="flag-desc"),
            class_=klass
        )

    @output
    @render.ui
    def status_overview():
        s = overview_state()  # ✅ 계산된 값만 읽음
        return ui.div(
            _flag_card("불량예측", s["pred_active"],
                       f"불량확률: {s['pred_prob']}" if s["pred_prob"] else "최근 예측 정상", icon="🧪"),
            _flag_card("이상탐지", s["anom_active"],
                       f"Anomaly score: {s['anom_score']}" if (s["anom_active"] and s["anom_score"]) else "최근 이상 없음", icon="⚠️"),
            _flag_card("관리도 이탈", s["spc_breach"],
                       "Rule breach 감지" if s["spc_breach"] else "정상", icon="📏"),
            _flag_card("임계값", s["thresh_breach"],
                       "임계 초과/미달" if s["thresh_breach"] else "정상", icon="🎚️"),
            class_="flag-row"
        )

    # ---- 기존 출력 ----
    @output
    @render.text
    def abnormal_count():
        return f"{metrics()['abnormal']}"

    @output
    @render.text
    def good_rate():
        return f"{metrics().get('good_rate', 0):.1f}%"

    @output
    @render.text
    def prod_count():
        return f"{metrics().get('prod_count', 0)}"

    @output
    @render.text
    def cycle_time():
        return f"{metrics().get('cycle_time', 0):.1f}s"

    @output
    @render.text
    def oee_value():
        return f"{metrics().get('oee', 0)*100:.1f}%"

    @output
    @render.plot
    def live_plot():
        return plot_live(current_data(), input.sensor_select())

    @output
    @render.plot
    def oee_chart():
        return plot_oee(metrics())

    @output
    @render.plot
    def mold_ratio():
        return plot_mold_ratio(metrics()["molds"])

    # 몰드 카드(5분할 그리드용)
    for mold in MOLD_CODES:
        @output(id=f"mold_{mold}_pie")
        @render.plot
        def mold_pie(mold=mold):
            data = metrics()["molds"].get(mold, {"good": 0, "defect": 0, "rate": 0})
            return plot_mold_pie(data)

        @output(id=f"mold_{mold}_info")
        @render.ui
        def mold_info(mold=mold):
            m = metrics()["molds"].get(mold, {"good": 0, "defect": 0, "rate": 0})
            return ui.div(
                ui.p(f"✅ 양품: {m['good']} EA", class_="mb-1 text-success fw-bold"),
                ui.p(f"❌ 불량: {m['defect']} EA", class_="mb-1 text-danger fw-bold"),
                ui.p(f"📊 생산율: {m['rate']:.1f}%", class_="text-primary fw-bold mb-0"),
                class_="mold-info"
            )

    @output
    @render.table
    def recent_data():
        df = current_data()
        return df.tail(5).round(2) if not df.empty else pd.DataFrame({"상태": ["데이터 없음"]})

    @output
    @render.table
    def recent_abnormal():
        df = detected_data()
        if df.empty or "anomaly" not in df.columns:
            return pd.DataFrame({"상태": ["최근 이상치 없음"]})
        abn = df[df["anomaly"] == -1]
        return abn.tail(5).round(2) if not abn.empty else pd.DataFrame({"상태": ["최근 이상치 없음"]})

    # 이벤트 피드(최근 5개) — status_overview와 완전히 분리
    @output
    @render.table
    @reactive.event(current_data)  # 새 배치 들어올 때만 테이블 갱신
    def event_feed():
        rows = []
        # 불량예측
        pred = prediction_data()
        if not pred.empty:
            for _, r in pred.sort_values("datetime").tail(10).iterrows():
                rows.append({
                    "시간": r.get("datetime", ""),
                    "유형": "불량예측",
                    "몰드": r.get("mold", ""),
                    "메시지": f"{'불량' if int(r.get('pred',0))==1 else '양품'} (확률 {(float(r.get('prob',0))*100):.1f}%)"
                })
        # 이상탐지
        det = detected_data()
        if not det.empty and "anomaly" in det.columns:
            for _, r in det[det["anomaly"] == -1].tail(10).iterrows():
                score = r.get("anomaly_score", np.nan)
                msg = f"이상탐지 발생 (score={score:.3f})" if not pd.isna(score) else "이상탐지 발생"
                rows.append({"시간": r.get("datetime",""), "유형":"이상탐지", "몰드": r.get("mold_code",""), "메시지": msg})
        # 관리도/임계(요약) — overview_state/metrics 직접 사용
        m = metrics()
        spc_breach = (m.get("oee", 0) < 0.6) or (m.get("good_rate", 100) < 92.0)
        thresh_breach = (
            (m.get("oee", 0) < THRESHOLDS["oee_min"]) or
            (m.get("good_rate", 100) < THRESHOLDS["good_rate_min"]) or
            (m.get("cycle_time", 0) > THRESHOLDS["cycle_time_max"])
        )
        if spc_breach:
            rows.append({"시간": pd.Timestamp.now(), "유형":"관리도", "몰드":"-", "메시지":"Rule breach 감지"})
        if thresh_breach:
            rows.append({"시간": pd.Timestamp.now(), "유형":"임계값", "몰드":"-", "메시지":"임계 초과/미달"})

        if not rows:
            return pd.DataFrame({"정보": ["최근 알림 없음"]})
        feed = pd.DataFrame(rows).sort_values("시간").tail(5)
        return feed
