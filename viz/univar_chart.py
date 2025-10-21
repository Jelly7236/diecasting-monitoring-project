import numpy as np
import pandas as pd
import plotly.graph_objs as go
from shiny import ui
from utils.control_utils import check_nelson_rules
from shared import arima_models, xr_limits


# ====================== 단변량 카드 ======================
def make_univar_cards(input, df_view, df_baseline, PROCESS_GROUPS):
    process = input.process_select()
    var_list = PROCESS_GROUPS[process]
    df = df_view()
    base = df_baseline()
    cards = []

    mold = input.mold() or "(전체)"

    for var in var_list:
        key = f"{mold}_{var}"
        series = df[var].dropna()
        if len(series) < 5:
            continue

        # -------------------------
        # 1️⃣ ARIMA 기반 관리도 (잔차 기준)
        # -------------------------
        if key in arima_models:
            info = arima_models[key]
            model = info["model"]
            sigma = info["sigma"]

            try:
                y_pred = np.asarray(model.forecast(steps=len(series)))
                residuals = np.asarray(series) - y_pred
                mu0, sd0, ucl, lcl = 0.0, sigma, 3 * sigma, -3 * sigma
                current_val = residuals[-1]
                target_array = residuals
            except Exception as e:
                print(f"⚠️ ARIMA 계산 실패 ({key}):", e)
                mu0, sd0 = series.mean(), series.std(ddof=1)
                ucl, lcl = mu0 + 3 * sd0, mu0 - 3 * sd0
                current_val = series.iloc[-1]
                target_array = series.to_numpy()

        # -------------------------
        # 2️⃣ X–R 관리도 기반
        # -------------------------
        elif key in xr_limits:
            info = xr_limits[key]
            mu0 = info["CL_X"]
            ucl = info["UCL_X"]
            lcl = info["LCL_X"]
            sd0 = (ucl - mu0) / 3  # 근사값
            current_val = series.iloc[-1]
            target_array = series.to_numpy()

        # -------------------------
        # 3️⃣ 기본값 (데이터 기반)
        # -------------------------
        else:
            mu0, sd0 = series.mean(), series.std(ddof=1)
            ucl, lcl = mu0 + 3 * sd0, mu0 - 3 * sd0
            current_val = series.iloc[-1]
            target_array = series.to_numpy()

        # -------------------------
        # ✅ Nelson Rule 기반 이상 판정
        # -------------------------
        try:
            violations = check_nelson_rules(target_array, mu0, ucl, lcl, sd0)
            violated_rules = [v[1] for v in violations]
        except Exception as e:
            print(f"⚠️ Nelson Rule 계산 실패 ({key}):", e)
            violated_rules = []

        if any("Rule 1" in r for r in violated_rules):
            status_class = "alert"     # 관리한계 초과
        elif any(r in ["Rule 2", "Rule 3", "Rule 5"] for r in violated_rules):
            status_class = "warning"   # 추세/집단 이상
        else:
            status_class = ""          # 정상

        status_text = (
            "경고" if status_class == "alert"
            else "주의" if status_class == "warning"
            else "정상"
        )

        # -------------------------
        # 카드 HTML
        # -------------------------
        card_html = f"""
        <div class="var-card {status_class}" onclick="Shiny.setInputValue('card_click','{var}',{{priority:'event'}})">
            <div class="var-card-header">
                <div class="var-name">{var}</div>
                <div class="var-status {status_class}">{status_text}</div>
            </div>
            <div class="var-value {status_class}">{current_val:.2f}</div>
        </div>
        """
        cards.append(card_html)

    return ui.HTML(f'<div class="var-cards-grid">{"".join(cards)}</div>')


# ====================== 단변량 모달 ======================
def make_univar_modal(input, df_view, df_baseline):
    import plotly.graph_objects as go
    import numpy as np
    import pandas as pd
    from shiny import ui
    from utils.control_utils import check_nelson_rules
    from shared import arima_models, xr_limits

    var = input.card_click()
    df = df_view()
    mold = input.mold() or "(전체)"
    key = f"{mold}_{var}"

    if var not in df.columns:
        ui.notification_show(f"'{var}' 변수를 찾을 수 없습니다.", type="error")
        return

    x = df[var].dropna().to_numpy()
    if len(x) < 10:
        ui.notification_show("표본이 부족합니다.", type="warning")
        return

    # ======================
    # 1️⃣ ARIMA 모델 기반 (잔차 관리도)
    # ======================
    if key in arima_models:
        info = arima_models[key]
        model = info["model"]
        sigma = info["sigma"]

        try:
            y_pred = np.asarray(model.forecast(steps=len(x)))
            residuals = np.asarray(x) - y_pred
        except Exception as e:
            print(f"⚠️ ARIMA 예측 실패 ({key}):", e)
            residuals = np.asarray(x) - np.mean(x)

        # ✅ 잔차 관리도는 CL=0 기준으로 설정
        cl = 0.0
        ucl = 3 * sigma
        lcl = -3 * sigma

        vio = check_nelson_rules(np.array(residuals), cl, ucl, lcl, sigma)
        violation_indices = [v[0] for v in vio]
        colors = ["red" if i + 1 in violation_indices else "#3b82f6" for i in range(len(residuals))]

        y_plot = residuals
        title_suffix = "ARIMA 잔차 관리도"
        y_label = "잔차 (Residual)"

    # ======================
    # 2️⃣ X–R 관리도 기반
    # ======================
    elif key in xr_limits:
        info = xr_limits[key]
        cl = info["CL_X"]
        ucl = info["UCL_X"]
        lcl = info["LCL_X"]
        sigma = (ucl - cl) / 3

        vio = check_nelson_rules(np.array(x), cl, ucl, lcl, sigma)
        violation_indices = [v[0] for v in vio]
        colors = ["red" if i + 1 in violation_indices else "#3b82f6" for i in range(len(x))]

        y_plot = x
        title_suffix = "X–R 관리도"
        y_label = "측정값"

    # ======================
    # 3️⃣ 기본 데이터 기반 (백업)
    # ======================
    else:
        cl = np.mean(x)
        sigma = np.std(x, ddof=1)
        ucl = cl + 3 * sigma
        lcl = cl - 3 * sigma

        vio = check_nelson_rules(np.array(x), cl, ucl, lcl, sigma)
        violation_indices = [v[0] for v in vio]
        colors = ["red" if i + 1 in violation_indices else "#3b82f6" for i in range(len(x))]

        y_plot = x
        title_suffix = "기초 통계 기반"
        y_label = "측정값"

    # ======================
    # Plotly 관리도 시각화
    # ======================
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(y_plot) + 1)),
            y=y_plot,
            mode="lines+markers",
            name="값",
            line=dict(color="#3b82f6", width=2),
            marker=dict(color=colors, size=5),
        )
    )

    fig.add_hline(y=cl, line_dash="solid", line_color="#10b981", annotation_text="CL", annotation_position="right")
    fig.add_hline(y=ucl, line_dash="dash", line_color="#ef4444", annotation_text="UCL", annotation_position="right")
    fig.add_hline(y=lcl, line_dash="dash", line_color="#ef4444", annotation_text="LCL", annotation_position="right")

    fig.update_layout(
        title=f"{var} ({title_suffix})",
        xaxis_title="샘플 번호",
        yaxis_title=y_label,
        template="plotly_white",
        height=400,
        hovermode="x unified",
    )

    chart_html = fig.to_html(include_plotlyjs="cdn", div_id="modal_chart_div")

    # ======================
    # 통계값 표시
    # ======================
    stats_html = f"""
    <div style='display:flex; justify-content:space-between; background:#f9fafb; border-radius:10px;
                padding:1rem; margin-top:1rem;'>
        <div style='text-align:center; flex:1;'><div style='color:#6b7280;'>CL</div><div style='font-size:1.3rem;'>{cl:.2f}</div></div>
        <div style='text-align:center; flex:1;'><div style='color:#6b7280;'>σ</div><div style='font-size:1.3rem;'>{sigma:.2f}</div></div>
        <div style='text-align:center; flex:1;'><div style='color:#6b7280;'>UCL</div><div style='font-size:1.3rem;color:#ef4444;'>{ucl:.2f}</div></div>
        <div style='text-align:center; flex:1;'><div style='color:#6b7280;'>LCL</div><div style='font-size:1.3rem;color:#ef4444;'>{lcl:.2f}</div></div>
    </div>
    """

    # ======================
    # 로그 테이블
    # ======================
    if not vio:
        log_html = "<p style='text-align:center; color:#6b7280; padding:1rem;'>✅ 이상 없음</p>"
    else:
        log_df = pd.DataFrame(vio, columns=["샘플", "룰", "설명", "값"])
        log_df["값"] = log_df["값"].round(3)
        log_html = log_df.to_html(index=False, classes="table table-striped table-sm", border=0)

    # ======================
    # 모달 출력
    # ======================
    m = ui.modal(
        ui.h4(f"{var} 상세 관리도", class_="mb-3"),
        ui.HTML(chart_html),
        ui.HTML(stats_html),
        ui.h5("🚨 이상 패턴 로그", class_="mt-3 mb-2"),
        ui.HTML(f"<div class='scroll-table' style='max-height:250px;'>{log_html}</div>"),
        size="xl",
        easy_close=True,
        footer=None,
    )
    ui.modal_show(m)
