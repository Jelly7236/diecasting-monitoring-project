# viz/univar_chart.py
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
        # 1️⃣ ARIMA 기반 관리도
        # -------------------------
        if key in arima_models:
            info = arima_models[key]
            mu0 = info["cl"]
            sd0 = info["sigma"]
            ucl = info["ucl"]
            lcl = info["lcl"]

        # -------------------------
        # 2️⃣ X–R 관리도 기반
        # -------------------------
        elif key in xr_limits:
            info = xr_limits[key]
            mu0 = info["CL_X"]
            ucl = info["UCL_X"]
            lcl = info["LCL_X"]
            sd0 = (ucl - mu0) / 3  # 근사값

        # -------------------------
        # 3️⃣ 기본값 (데이터 기반)
        # -------------------------
        else:
            mu0, sd0 = series.mean(), series.std(ddof=1)
            ucl, lcl = mu0 + 3 * sd0, mu0 - 3 * sd0

        current_val = series.iloc[-1]

        status_class = (
            "alert" if (current_val > ucl or current_val < lcl)
            else "warning" if (current_val < mu0 - 2 * sd0 or current_val > mu0 + 2 * sd0)
            else ""
        )
        status_text = "경고" if status_class == "alert" else "주의" if status_class == "warning" else "정상"

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
    import pandas as pd
    import numpy as np
    from shiny import ui
    from utils.control_utils import check_nelson_rules

    var = input.card_click()
    df = df_view()
    base = df_baseline()
    mold = input.mold() or "(전체)"
    key = f"{mold}_{var}"
    print("현재 mold 선택:", input.mold())
    print("xr_limits key 샘플:", list(xr_limits.keys())[:5])

    x = df[var].dropna().to_numpy()
    if len(x) < 10:
        ui.notification_show("표본이 부족합니다.", type="warning")
        return

    # ======================
    # 1️⃣ ARIMA 모델 있는 경우
    # ======================
    if key in arima_models:
        info = arima_models[key]
        mu = info["cl"]
        sd = info["sigma"]
        ucl = info["ucl"]
        lcl = info["lcl"]

    # ======================
    # 2️⃣ X–R 관리도 기반
    # ======================
    elif key in xr_limits:
        info = xr_limits[key]
        mu = info["CL_X"]
        ucl = info["UCL_X"]
        lcl = info["LCL_X"]
        sd = (ucl - mu) / 3  # 근사값

    # ======================
    # 3️⃣ 데이터 기반 (백업)
    # ======================
    else:
        mu = np.mean(x)
        sd = np.std(x, ddof=1)
        ucl = mu + 3 * sd
        lcl = mu - 3 * sd

    vio = check_nelson_rules(x, mu, ucl, lcl, sd)
    violation_indices = [v[0] for v in vio]
    colors = ['red' if i+1 in violation_indices else '#3b82f6' for i in range(len(x))]

    # ======================
    # Plotly 관리도
    # ======================
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(x)+1)),
        y=x,
        mode='lines+markers',
        name='측정값',
        line=dict(color='#3b82f6', width=2),
        marker=dict(color=colors, size=5)
    ))
    fig.add_hline(y=mu, line_dash="solid", line_color="#10b981", annotation_text="CL", annotation_position="right")
    fig.add_hline(y=ucl, line_dash="dash", line_color="#ef4444", annotation_text="UCL", annotation_position="right")
    fig.add_hline(y=lcl, line_dash="dash", line_color="#ef4444", annotation_text="LCL", annotation_position="right")

    fig.update_layout(
        title=f"{var} 관리도 ({'ARIMA' if key in arima_models else 'X–R' if key in xr_limits else '기초통계'})",
        xaxis_title="샘플 번호",
        yaxis_title="측정값",
        template="plotly_white",
        height=400,
        hovermode='x unified'
    )

    chart_html = fig.to_html(include_plotlyjs='cdn', div_id="modal_chart_div")

    # ======================
    # 통계값 표시
    # ======================
    stats_html = f"""
    <div style='display:flex; justify-content:space-between; background:#f9fafb; border-radius:10px;
                padding:1rem; margin-top:1rem;'>
        <div style='text-align:center; flex:1;'><div style='color:#6b7280;'>평균</div><div style='font-size:1.3rem;'>{mu:.2f}</div></div>
        <div style='text-align:center; flex:1;'><div style='color:#6b7280;'>표준편차</div><div style='font-size:1.3rem;'>{sd:.2f}</div></div>
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

    m = ui.modal(
        ui.h4(f"{var} 상세 관리도", class_="mb-3"),
        ui.HTML(chart_html),
        ui.HTML(stats_html),
        ui.h5("🚨 이상 패턴 로그", class_="mt-3 mb-2"),
        ui.HTML(f"<div class='scroll-table' style='max-height:250px;'>{log_html}</div>"),
        size="xl",
        easy_close=True,
        footer=None
    )
    ui.modal_show(m)
