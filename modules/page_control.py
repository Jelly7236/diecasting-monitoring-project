from shiny import ui, render, reactive
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objs as go
from shared import streaming_df


PROCESS_GROUPS = {
    "1) 용탕 준비 및 가열": ["molten_temp", "molten_volume"],
    "2) 반고체 슬러리 제조": ["sleeve_temperature", "EMS_operation_time"],
    "3) 사출 & 금형 충전": [
        "cast_pressure", "low_section_speed", "high_section_speed",
        "physical_strength", "biscuit_thickness"
    ],
    "4) 응고": [
        "upper_mold_temp1", "upper_mold_temp2",
        "lower_mold_temp1", "lower_mold_temp2", "Coolant_temperature"
    ]
}

FEATURES_ALL = [v for vars_ in PROCESS_GROUPS.values() for v in vars_]


def check_nelson_rules(data, mean, ucl, lcl, sigma):
    violations = []
    n = len(data)
    for i in range(n):
        if data[i] > ucl:
            violations.append((i + 1, "Rule 1", "UCL 초과", data[i]))
        elif data[i] < lcl:
            violations.append((i + 1, "Rule 1", "LCL 미만", data[i]))
        if i >= 8:
            if all(data[i-j] > mean for j in range(9)):
                violations.append((i+1, "Rule 2", "연속 9개 점이 중심선 위", data[i]))
            elif all(data[i-j] < mean for j in range(9)):
                violations.append((i+1, "Rule 2", "연속 9개 점이 중심선 아래", data[i]))
    return violations


def ui_control():
    return ui.page_fluid(
        ui.head_content(
            ui.tags.link(rel="stylesheet", href="/css/control.css")
        ),
        ui.h3("📊 공정 관리 상태 분석", class_="text-center mb-3"),

        # 설정 카드
        ui.card(
            ui.card_header("⚙️ 분석 설정"),
            ui.layout_columns(
                ui.input_select(
                    "process_select",
                    "공정 선택",
                    choices={k: k for k in PROCESS_GROUPS.keys()},
                    selected=list(PROCESS_GROUPS.keys())[0],
                ),
                ui.output_ui("mold_select"),
                ui.input_numeric("win", "윈도우(샘플 수)", 200, min=50, max=5000, step=50),
                col_widths=[4, 4, 4],
            ),
        ),

        # 단변량 관리 카드
        ui.card(
            ui.card_header("📈 단변량 관리도 (클릭하여 상세 보기)"),
            ui.output_ui("variable_cards")
        ),

        # 전체 로그
        ui.card(
            ui.card_header("🕒 전체 이탈 로그 (단변량 + 다변량 통합)"),
            ui.div(ui.output_table("timeline_table"), class_="scroll-table", style="max-height: 400px;")
        ),

        style="max-width:1600px; margin:0 auto; padding:0 1rem;"
    )


def server_control(input, output, session):
    @output
    @render.ui
    def mold_select():
        df = streaming_df
        choices = ["(전체)"]
        if "mold_code" in df:
            choices += [str(m) for m in sorted(df["mold_code"].dropna().unique())]
        return ui.input_select("mold", "몰드 선택", choices=choices, selected="(전체)")

    @reactive.calc
    def df_view():
        df = streaming_df.copy()
        if "id" in df:
            df = df.sort_values("id")
        df = df.tail(int(input.win()))
        if "mold_code" in df and input.mold() not in (None, "", "(전체)"):
            df = df[df["mold_code"].astype(str) == str(input.mold())]
        return df.reset_index(drop=True)

    @reactive.calc
    def df_baseline():
        df = streaming_df.copy()
        if "id" in df:
            df = df.sort_values("id")
        mask = (df["passorfail"] == 0) if "passorfail" in df else np.ones(len(df), bool)
        base = df.loc[mask, FEATURES_ALL].dropna()
        return base if len(base) >= 50 else None

    # ==================== 단변량 카드 ====================
    @output
    @render.ui
    def variable_cards():
        df, base = df_view(), df_baseline()
        process = input.process_select()
        var_list = PROCESS_GROUPS[process]
        cards = []

        for var in var_list:
            s = df[var].dropna()
            if len(s) < 5:
                continue
            mu = base[var].mean() if base is not None and var in base else s.mean()
            sd = base[var].std(ddof=1) if base is not None and var in base else s.std(ddof=1)
            ucl, lcl = mu + 3*sd, mu - 3*sd
            val = s.iloc[-1]
            cv = (sd / mu * 100) if mu != 0 else 0

            if len(s) >= 10:
                recent = s.tail(10)
                diff = recent.iloc[-1] - recent.iloc[0]
                trend = "안정"
                if abs(diff) >= sd * 0.5:
                    trend = "상승" if diff > 0 else "하락"
            else:
                trend = "안정"

            if val > ucl or val < lcl:
                css, status = "alert", "경고"
            elif val > mu + 2*sd or val < mu - 2*sd:
                css, status = "warning", "주의"
            else:
                css, status = "normal", "정상"

            card_html = f"""
            <div class="var-card {css}" onclick="Shiny.setInputValue('card_click', '{var}', {{priority:'event'}})">
                <div class="var-card-header">
                    <div class="var-name">{var}</div>
                    <div class="var-status {css}">{status}</div>
                </div>
                <div class="var-value">{val:.1f} <span class="trend">{'━' if trend=='안정' else ('↗' if trend=='상승' else '↘')}</span></div>
                <div class="var-stats">
                    <div><span class="stat-label">평균</span><span class="stat-value">{mu:.1f}</span></div>
                    <div><span class="stat-label">UCL</span><span class="stat-value">{ucl:.1f}</span></div>
                    <div><span class="stat-label">CV</span><span class="stat-value">{cv:.1f}%</span></div>
                    <div><span class="stat-label">추세</span><span class="stat-value">{trend}</span></div>
                </div>
            </div>
            """
            cards.append(card_html)

        return ui.HTML(f'<div class="var-cards-grid">{"".join(cards)}</div>')

    # ==================== 모달 관리도 ====================
    @reactive.effect
    @reactive.event(input.card_click)
    def _():
        var = input.card_click()
        df, base = df_view(), df_baseline()
        x = df[var].dropna().to_numpy()
        if len(x) < 10:
            ui.notification_show("표본이 부족합니다.", type="warning")
            return

        mu = base[var].mean() if base is not None and var in base else np.mean(x)
        sd = base[var].std(ddof=1) if base is not None and var in base else np.std(x, ddof=1)
        ucl, lcl = mu + 3 * sd, mu - 3 * sd
        vio = check_nelson_rules(x, mu, ucl, lcl, sd)
        colors = ['#3b82f6' if (i+1) not in [v[0] for v in vio] else 'red' for i in range(len(x))]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(1, len(x)+1)), y=x,
                                 mode='lines+markers',
                                 line=dict(color='#3b82f6', width=2),
                                 marker=dict(color=colors, size=5)))
        fig.add_hline(y=mu, line_dash="solid", line_color="#10b981", annotation_text="CL")
        fig.add_hline(y=ucl, line_dash="dash", line_color="#ef4444", annotation_text="UCL")
        fig.add_hline(y=lcl, line_dash="dash", line_color="#ef4444", annotation_text="LCL")
        fig.update_layout(title=f"{var} 관리도 (n={len(x)})",
                          xaxis_title="샘플 번호", yaxis_title="측정값",
                          template="plotly_white", height=400)

        chart_html = fig.to_html(include_plotlyjs='cdn')

        mu_current, sd_current = np.mean(x), np.std(x, ddof=1)
        stats_html = f"""
        <div class='modal-stats'>
            <div><div class='stat-label'>현재 평균</div><div class='stat-value'>{mu_current:.2f}</div></div>
            <div><div class='stat-label'>표준편차</div><div class='stat-value'>{sd_current:.2f}</div></div>
            <div><div class='stat-label'>기준선 평균</div><div class='stat-value base'>{mu:.2f}</div></div>
            <div><div class='stat-label'>UCL</div><div class='stat-value alert'>{ucl:.2f}</div></div>
            <div><div class='stat-label'>LCL</div><div class='stat-value alert'>{lcl:.2f}</div></div>
        </div>
        """

        if not vio:
            log_html = "<p class='no-viol'>✅ 이상 없음</p>"
        else:
            df_vio = pd.DataFrame(vio, columns=["샘플", "룰", "설명", "값"])
            log_html = df_vio.to_html(index=False, classes="table table-sm")

        modal = ui.modal(
            ui.h4(f"{var} 상세 관리도"),
            ui.HTML(chart_html),
            ui.HTML(stats_html),
            ui.h5("🚨 이상 패턴 로그", class_="mt-3 mb-2"),
            ui.HTML(f"<div class='scroll-table'>{log_html}</div>"),
            size="xl", easy_close=True
        )
        ui.modal_show(modal)

    # ==================== 전체 로그 ====================
    @output
    @render.table
    def timeline_table():
        df = df_view()
        base = df_baseline()
        out_rows = []

        for var in FEATURES_ALL:
            s = df[var].dropna()
            if len(s) < 10:
                continue
            mu = base[var].mean() if base is not None and var in base else s.mean()
            sd = base[var].std(ddof=1) if base is not None and var in base else s.std(ddof=1)
            vio = check_nelson_rules(s.to_numpy(), mu, mu+3*sd, mu-3*sd, sd)
            for (idx, r, desc, val) in vio[-20:]:
                out_rows.append({
                    "변수": var,
                    "룰": r,
                    "설명": desc,
                    "값": round(val, 3)
                })

        return pd.DataFrame(out_rows) if out_rows else pd.DataFrame({"상태": ["최근 이상 없음"]})
