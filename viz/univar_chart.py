# viz/univar_chart
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from shiny import ui
from utils.control_utils import check_nelson_rules

def make_univar_cards(input, df_view, df_baseline, PROCESS_GROUPS):
    process = input.process_select()
    var_list = PROCESS_GROUPS[process]
    df = df_view()
    base = df_baseline()
    cards = []

    for var in var_list:
        series = df[var].dropna()
        if len(series) < 5:
            continue
        if base is None or var not in base.columns or len(base) < 5:
            mu0, sd0 = series.mean(), series.std(ddof=1)
        else:
            mu0, sd0 = base[var].mean(), base[var].std(ddof=1)
        current_val = series.iloc[-1]
        ucl, lcl = mu0 + 3 * sd0, mu0 - 3 * sd0
        status_class = (
            "alert" if (current_val > ucl or current_val < lcl)
            else "warning" if (current_val < mu0 - 2 * sd0 or current_val > mu0 + 2 * sd0)
            else ""
        )
        status_text = "경고" if status_class == "alert" else "주의" if status_class == "warning" else "정상"
        card_html = f"""
        <div class="var-card {status_class}" onclick="Shiny.setInputValue('card_click','{var}',{{priority:'event'}})">
            <div class="var-card-header"><div class="var-name">{var}</div><div class="var-status {status_class}">{status_text}</div></div>
            <div class="var-value {status_class}">{current_val:.1f}</div>
        </div>
        """
        cards.append(card_html)
    return ui.HTML(f'<div class="var-cards-grid">{"".join(cards)}</div>')

def make_univar_modal(input, df_view, df_baseline):
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np
    from shiny import ui
    from utils.control_utils import check_nelson_rules

    var = input.card_click()
    df = df_view()
    base = df_baseline()

    x = df[var].dropna().to_numpy()
    if len(x) < 10:
        ui.notification_show("표본이 부족합니다.", type="warning")
        return

    mu = (base[var].mean() if base is not None and var in base.columns and len(base) > 5 else np.mean(x))
    sd = (base[var].std(ddof=1) if base is not None and var in base.columns and len(base) > 5 else np.std(x, ddof=1))
    ucl, lcl = mu + 3*sd, mu - 3*sd

    vio = check_nelson_rules(x, mu, ucl, lcl, sd)
    violation_indices = [v[0] for v in vio]

    colors = ['red' if i+1 in violation_indices else '#3b82f6' for i in range(len(x))]

    # plotly 관리도
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

    fig.add_hrect(y0=mu-sd, y1=mu+sd, fillcolor="#dbeafe", opacity=0.2)
    fig.add_hrect(y0=mu-2*sd, y1=mu+2*sd, fillcolor="#bfdbfe", opacity=0.15)

    fig.update_layout(
        title=f"{var} 관리도 (n={len(x)})",
        xaxis_title="샘플 번호",
        yaxis_title="측정값",
        template="plotly_white",
        height=400,
        hovermode='x unified'
    )

    chart_html = fig.to_html(include_plotlyjs='cdn', div_id="modal_chart_div")

    # ================= 통계 영역 =================
    stats_html = f"""
    <div style='display:flex; justify-content:space-between; background:#f9fafb; border-radius:10px;
                padding:1rem; margin-top:1rem;'>
        <div style='text-align:center; flex:1;'>
            <div style='color:#6b7280; font-size:0.85rem;'>현재 평균</div>
            <div style='font-size:1.3rem; font-weight:700;'>{np.mean(x):.2f}</div>
        </div>
        <div style='text-align:center; flex:1;'>
            <div style='color:#6b7280; font-size:0.85rem;'>표준편차</div>
            <div style='font-size:1.3rem; font-weight:700;'>{np.std(x, ddof=1):.2f}</div>
        </div>
        <div style='text-align:center; flex:1;'>
            <div style='color:#6b7280; font-size:0.85rem;'>기준선 평균</div>
            <div style='font-size:1.3rem; font-weight:700; color:#10b981;'>{mu:.2f}</div>
        </div>
        <div style='text-align:center; flex:1;'>
            <div style='color:#6b7280; font-size:0.85rem;'>UCL</div>
            <div style='font-size:1.3rem; font-weight:700; color:#ef4444;'>{ucl:.2f}</div>
        </div>
        <div style='text-align:center; flex:1;'>
            <div style='color:#6b7280; font-size:0.85rem;'>LCL</div>
            <div style='font-size:1.3rem; font-weight:700; color:#ef4444;'>{lcl:.2f}</div>
        </div>
    </div>
    """

    # ================= 로그 테이블 =================
    if not vio:
        log_html = "<p style='text-align:center; color:#6b7280; padding:1rem;'>✅ 이상 없음</p>"
    else:
        log_df = pd.DataFrame(vio, columns=["샘플", "룰", "설명", "값"])
        log_df["값"] = log_df["값"].round(3)
        log_html = log_df.to_html(index=False, classes="table table-striped table-sm", border=0)

    # ================= 모달 =================
    m = ui.modal(
        ui.h4(f"{var} 상세 관리도", class_="mb-3"),
        ui.HTML(chart_html),
        ui.HTML(stats_html),
        ui.h5("🔴 이상 패턴 로그", class_="mt-3 mb-2"),
        ui.HTML(f"<div class='scroll-table' style='max-height:250px;'>{log_html}</div>"),
        size="xl",
        easy_close=True,
        footer=None
    )

    ui.modal_show(m)
