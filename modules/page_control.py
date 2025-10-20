# modules/page_control.py
from shiny import ui, render, reactive
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objs as go

from shared import streaming_df

# ==================== 공정별 변수 정의 ====================
PROCESS_GROUPS = {
    "1) 용탕 준비 및 가열": ["molten_temp", "molten_volume"],
    "2) 반고체 슬러리 제조": ["sleeve_temperature", "EMS_operation_time"],
    "3) 사출 & 금형 충전": ["cast_pressure", "low_section_speed", "high_section_speed", 
                        "physical_strength", "biscuit_thickness"],
    "4) 응고": ["upper_mold_temp1", "upper_mold_temp2", "lower_mold_temp1", 
              "lower_mold_temp2", "Coolant_temperature"]
}

FEATURES_ALL = [
    "molten_temp", "molten_volume", "sleeve_temperature", "EMS_operation_time",
    "cast_pressure", "low_section_speed", "high_section_speed", "physical_strength", 
    "biscuit_thickness", "upper_mold_temp1", "upper_mold_temp2", "lower_mold_temp1", 
    "lower_mold_temp2", "Coolant_temperature"
]

# 규격 한계
SPEC_LIMITS = {
    "molten_temp": {"usl": 750, "lsl": 650},
    "cast_pressure": {"usl": 370, "lsl": 250},
    "upper_mold_temp1": {"usl": 250, "lsl": 150},
    "sleeve_temperature": {"usl": 500, "lsl": 400},
    "Coolant_temperature": {"usl": 45, "lsl": 35},
    "physical_strength": {"usl": 750, "lsl": 600}
}

# ==================== 통계 함수 ====================
def check_nelson_rules(data, mean, ucl, lcl, sigma):
    violations = []
    n = len(data)
    
    for i in range(n):
        if data[i] > ucl:
            violations.append((i+1, "Rule 1", "UCL 초과", data[i]))
        elif data[i] < lcl:
            violations.append((i+1, "Rule 1", "LCL 미만", data[i]))
        
        if i >= 8:
            if all(data[i-j] > mean for j in range(9)):
                violations.append((i+1, "Rule 2", "연속 9개 점이 중심선 위", data[i]))
            elif all(data[i-j] < mean for j in range(9)):
                violations.append((i+1, "Rule 2", "연속 9개 점이 중심선 아래", data[i]))
        
        if i >= 5:
            increasing = all(data[i-j] < data[i-j+1] for j in range(5, 0, -1))
            decreasing = all(data[i-j] > data[i-j+1] for j in range(5, 0, -1))
            if increasing:
                violations.append((i+1, "Rule 3", "연속 6개 점 증가 추세", data[i]))
            elif decreasing:
                violations.append((i+1, "Rule 3", "연속 6개 점 감소 추세", data[i]))
        
        if i >= 2:
            zone2_upper = mean + 2*sigma
            zone2_lower = mean - 2*sigma
            count = sum(1 for j in range(3) if data[i-j] > zone2_upper or data[i-j] < zone2_lower)
            if count >= 2:
                violations.append((i+1, "Rule 5", "3개 중 2개가 2σ 영역 밖", data[i]))
    
    return violations

def calculate_hotelling_t2(data_matrix, mean_vector, inv_cov):
    t2_values = []
    for i in range(len(data_matrix)):
        diff = data_matrix[i] - mean_vector
        t2 = diff @ inv_cov @ diff.T
        t2_values.append(t2)
    return np.array(t2_values)

def phaseII_ucl_t2(n, p, alpha=0.01):
    return (p * (n-1) * (n+1) / (n * (n-p))) * stats.f.ppf(1-alpha, p, n-p)

def to_datetime_safe(df):
    if "tryshot_time" in df.columns:
        return pd.to_datetime(df["tryshot_time"], errors='coerce')
    return None

# ==================== UI ====================
def ui_control():
    return ui.page_fluid(
        ui.tags.style("""
            * { font-family: 'Noto Sans KR', sans-serif; }
            body { background-color: #f5f7fa; padding: 1rem 0; }
            .card {
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                background: white;
                margin-bottom: 1rem;
            }
            .card-header {
                background-color: #f9fafb;
                border-bottom: 1px solid #e5e7eb;
                color: #1f2937;
                font-weight: 600;
                padding: 0.75rem 1rem;
                font-size: 0.9rem;
            }
            
            /* 변수 카드 그리드 */
            .var-cards-grid {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 1rem;
                margin: 1rem 0;
            }
            
            .var-card {
                background: white;
                border: 2px solid #e5e7eb;
                border-radius: 8px;
                padding: 1rem;
                cursor: pointer;
                transition: all 0.3s ease;
                height: 180px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            }
            
            .var-card:hover {
                transform: translateY(-3px);
                border-color: #3b82f6;
                box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
            }
            
            .var-card.alert {
                border-color: #ef4444;
                background: #fef2f2;
            }
            
            .var-card.warning {
                border-color: #f59e0b;
                background: #fffbeb;
            }
            
            .var-card-header {
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                margin-bottom: 0.75rem;
            }
            
            .var-name {
                font-size: 0.9rem;
                font-weight: 600;
                color: #1f2937;
            }
            
            .var-status {
                font-size: 0.7rem;
                padding: 0.2rem 0.5rem;
                border-radius: 4px;
                background: #10b981;
                color: white;
                font-weight: 600;
            }
            
            .var-status.alert { background: #ef4444; }
            .var-status.warning { background: #f59e0b; }
            
            .var-value {
                font-size: 2rem;
                font-weight: 700;
                color: #1f2937;
                text-align: center;
                margin: 0.5rem 0;
            }
            
            .var-value.alert { color: #ef4444; }
            .var-value.warning { color: #f59e0b; }
            
            .var-stats {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 0.5rem;
                font-size: 0.7rem;
                color: #6b7280;
                margin-top: 0.75rem;
                padding-top: 0.75rem;
                border-top: 1px solid #e5e7eb;
            }
            
            .stat-item {
                text-align: center;
            }
            
            .stat-label {
                display: block;
                margin-bottom: 0.2rem;
                font-size: 0.65rem;
            }
            
            .stat-value {
                font-weight: 600;
                color: #1f2937;
                font-size: 0.75rem;
            }
            
            .trend-indicator {
                font-size: 1.2rem;
                display: inline-block;
                margin-left: 0.3rem;
            }
            
            .trend-up { color: #ef4444; }
            .trend-down { color: #10b981; }
            .trend-stable { color: #6b7280; }
            
            .scroll-table { max-height: 250px; overflow-y: auto; }
            h3 { color: #1f2937; font-weight: 700; margin-bottom: 1rem; font-size: 1.5rem; }
            h5 { color: #374151; font-weight: 600; margin-bottom: 0.75rem; font-size: 1rem; }
            
            /* 모달 스타일 */
            .modal-content {
                border-radius: 12px;
            }
            
            .modal-header {
                background-color: #f9fafb;
                border-bottom: 1px solid #e5e7eb;
            }
        """),

        ui.div(
            ui.h3("📊 공정 관리 상태 분석", class_="text-center mb-3"),

            # ==================== 1. 상단 컨트롤 ====================
            ui.card(
                ui.card_header("⚙️ 분석 설정"),
                ui.layout_columns(
                    ui.input_select(
                        "process_select",
                        "공정 선택",
                        choices={k: k for k in PROCESS_GROUPS.keys()},
                        selected=list(PROCESS_GROUPS.keys())[0]
                    ),
                    ui.output_ui("mold_select"),
                    ui.input_numeric("win", "윈도우(샘플 수)", 200, min=50, max=5000, step=50),
                    col_widths=[4, 4, 4]
                )
            ),

            # ==================== 2. 다변량 관리도 ====================
            ui.card(
                ui.card_header(ui.output_text("multivar_title")),
                ui.layout_columns(
                    ui.output_ui("t2_plot"),
                    ui.div(
                        ui.h5("📄 T² 이탈 로그", class_="mb-2"),
                        ui.div(ui.output_table("t2_table"), class_="scroll-table")
                    ),
                    col_widths=[7, 5]
                )
            ),

            # ==================== 3. 단변량 카드 그리드 ====================
            ui.card(
                ui.card_header("📈 단변량 관리도 (클릭하여 상세 차트 보기)"),
                ui.output_ui("variable_cards")
            ),

            # ==================== 5. 전체 이탈 로그 ====================
            ui.card(
                ui.card_header("🕒 전체 이탈 로그 (단변량 + 다변량 통합)"),
                ui.div(ui.output_table("timeline_table"), class_="scroll-table", style="max-height: 400px;")
            ),

            style="max-width: 1600px; margin: 0 auto; padding: 0 0.75rem;"
        )
    )


# ==================== SERVER ====================
def server_control(input, output, session):
    
    # 선택된 변수 저장
    selected_var = reactive.value(None)
    
    # 동적 몰드 선택
    @output
    @render.ui
    def mold_select():
        df = streaming_df
        choices = ["(전체)"]
        if "mold_code" in df:
            choices += [str(m) for m in sorted(df["mold_code"].dropna().unique())]
        return ui.input_select("mold", "몰드 선택", choices=choices, selected="(전체)")
    
    # 공통 뷰
    @reactive.calc
    def df_view():
        df = streaming_df.copy()
        if "id" in df:
            df = df.sort_values("id")
        df = df.tail(int(input.win()))
        
        if "mold_code" in df and input.mold() not in (None, "", "(전체)"):
            try:
                sel = int(input.mold())
                df = df[df["mold_code"] == sel]
            except:
                df = df[df["mold_code"].astype(str) == str(input.mold())]
        
        dt = to_datetime_safe(df)
        df["__dt__"] = dt if dt is not None else pd.RangeIndex(len(df)).astype(float)
        return df.reset_index(drop=True)

    # 기준선
    @reactive.calc
    def df_baseline():
        df = streaming_df.copy()
        if "id" in df:
            df = df.sort_values("id")
        
        if "mold_code" in df and input.mold() not in (None, "", "(전체)"):
            try:
                sel = int(input.mold())
                df = df[df["mold_code"] == sel]
            except:
                df = df[df["mold_code"].astype(str) == str(input.mold())]
        
        mask = (df["passorfail"] == 0) if "passorfail" in df else np.ones(len(df), dtype=bool)
        base = df.loc[mask, FEATURES_ALL].dropna()
        
        if len(base) < 50:
            return None
        return base

    # ==================== 다변량 타이틀 ====================
    @output
    @render.text
    def multivar_title():
        process = input.process_select()
        var_list = PROCESS_GROUPS[process]
        return f"🔬 다변량 관리도 (Hotelling T²) - {process} [{', '.join(var_list)}]"

    # ==================== 다변량 T² 차트 ====================
    @output
    @render.ui
    def t2_plot():
        process = input.process_select()
        var_list = PROCESS_GROUPS[process]
        df = df_view()
        base = df_baseline()
        
        X = df[var_list].dropna().to_numpy()
        p = len(var_list)
        
        if X.shape[0] < max(30, p + 5):
            return ui.p("표본이 부족합니다.", style="color: #6b7280; padding: 2rem; text-align: center;")
        
        base_df = base[var_list].dropna() if (base is not None and set(var_list).issubset(base.columns)) else df[var_list].dropna()
        mu = base_df.mean().to_numpy()
        cov = np.cov(base_df.to_numpy().T)
        
        try:
            inv_cov = np.linalg.inv(cov)
        except:
            inv_cov = np.linalg.pinv(cov)
        
        t2 = calculate_hotelling_t2(X, mu, inv_cov)
        ucl = phaseII_ucl_t2(X.shape[0], p, alpha=0.01)
        viol_idx = np.where(t2 > ucl)[0]
        
        # Plotly 차트
        colors = ['red' if i in viol_idx else '#3b82f6' for i in range(len(t2))]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(t2)+1)),
            y=t2,
            mode='lines+markers',
            name='T² 통계량',
            line=dict(color='#3b82f6', width=2),
            marker=dict(color=colors, size=5)
        ))
        
        fig.add_hline(y=ucl, line_dash="dash", line_color="#ef4444",
                     annotation_text=f"UCL ({ucl:.2f})", annotation_position="right")
        
        fig.update_layout(
            title=f"Hotelling T² (n={X.shape[0]})",
            xaxis_title="샘플 번호",
            yaxis_title="T² 값",
            template="plotly_white",
            height=350,
            hovermode='x unified'
        )
        
        return ui.HTML(fig.to_html(include_plotlyjs='cdn', div_id="t2_chart"))

    # ==================== 다변량 T² 로그 ====================
    @output
    @render.table
    def t2_table():
        process = input.process_select()
        var_list = PROCESS_GROUPS[process]
        df = df_view()
        base = df_baseline()
        
        X = df[var_list].dropna().to_numpy()
        p = len(var_list)
        
        if X.shape[0] < max(30, p + 5):
            return pd.DataFrame({"상태": ["표본 부족"]})
        
        base_df = base[var_list].dropna() if (base is not None and set(var_list).issubset(base.columns)) else df[var_list].dropna()
        mu = base_df.mean().to_numpy()
        cov = np.cov(base_df.to_numpy().T)
        
        try:
            inv_cov = np.linalg.inv(cov)
        except:
            inv_cov = np.linalg.pinv(cov)
        
        t2 = calculate_hotelling_t2(X, mu, inv_cov)
        ucl = phaseII_ucl_t2(X.shape[0], p, alpha=0.01)
        viol = np.where(t2 > ucl)[0]
        
        if len(viol) == 0:
            return pd.DataFrame({"상태": ["✅ 관리 상태 양호"]})
        
        log = pd.DataFrame({
            "샘플": viol + 1,
            "T²": t2[viol].round(3),
            "UCL": np.round(ucl, 3),
            "유형": ["T² 초과"] * len(viol)
        })
        return log.tail(50)

    # ==================== 단변량 카드 그리드 ====================
    @output
    @render.ui
    def variable_cards():
        process = input.process_select()
        var_list = PROCESS_GROUPS[process]
        df = df_view()
        base = df_baseline()
        
        cards = []
        
        for var in var_list:
            series = df[var].dropna()
            if len(series) < 5:
                continue
            
            # 기준선
            if base is None or var not in base.columns or len(base) < 5:
                mu0, sd0 = series.mean(), series.std(ddof=1)
            else:
                mu0, sd0 = base[var].mean(), base[var].std(ddof=1)
            
            current_val = series.iloc[-1]
            ucl = mu0 + 3 * sd0
            lcl = mu0 - 3 * sd0
            
            # 추세 계산 (최근 10개 데이터)
            if len(series) >= 10:
                recent = series.tail(10)
                trend_diff = recent.iloc[-1] - recent.iloc[0]
                trend_pct = (trend_diff / recent.iloc[0]) * 100 if recent.iloc[0] != 0 else 0
                
                if abs(trend_pct) < 1:
                    trend_symbol = "━"
                    trend_class = "trend-stable"
                    trend_text = "안정"
                elif trend_pct > 0:
                    trend_symbol = "↗"
                    trend_class = "trend-up"
                    trend_text = f"+{trend_pct:.1f}%"
                else:
                    trend_symbol = "↘"
                    trend_class = "trend-down"
                    trend_text = f"{trend_pct:.1f}%"
            else:
                trend_symbol = "━"
                trend_class = "trend-stable"
                trend_text = "—"
            
            # 변동계수 (CV)
            cv = (sd0 / mu0 * 100) if mu0 != 0 else 0
            
            # 상태 판정
            if current_val > ucl or current_val < lcl:
                status_class = "alert"
                status_text = "경고"
            elif current_val < mu0 - 2*sd0 or current_val > mu0 + 2*sd0:
                status_class = "warning"
                status_text = "주의"
            else:
                status_class = ""
                status_text = "정상"
            
            # 카드 생성
            card_html = f"""
            <div class="var-card {status_class}" onclick="Shiny.setInputValue('card_click', '{var}', {{priority: 'event'}})">
                <div class="var-card-header">
                    <div class="var-name">{var}</div>
                    <div class="var-status {status_class}">{status_text}</div>
                </div>
                <div class="var-value {status_class}">
                    {current_val:.1f}
                    <span class="trend-indicator {trend_class}">{trend_symbol}</span>
                </div>
                <div class="var-stats">
                    <div class="stat-item">
                        <span class="stat-label">평균</span>
                        <span class="stat-value">{mu0:.1f}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">UCL</span>
                        <span class="stat-value">{ucl:.1f}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">CV</span>
                        <span class="stat-value">{cv:.1f}%</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">추세</span>
                        <span class="stat-value {trend_class}">{trend_text}</span>
                    </div>
                </div>
            </div>
            """
            cards.append(card_html)
        
        return ui.HTML(f'<div class="var-cards-grid">{"".join(cards)}</div>')

    # ==================== 카드 클릭 이벤트 ====================
    @reactive.effect
    @reactive.event(input.card_click)
    def _():
        var = input.card_click()
        selected_var.set(var)
        
        # 모달 내용 생성
        df = df_view()
        base = df_baseline()
        
        x = df[var].dropna().to_numpy()
        if len(x) < 10:
            ui.notification_show("표본이 부족합니다.", type="warning")
            return
        
        mu = (base[var].mean() if base is not None and var in base.columns and len(base) > 5 else np.mean(x))
        sd = (base[var].std(ddof=1) if base is not None and var in base.columns and len(base) > 5 else np.std(x, ddof=1))
        
        ucl = mu + 3*sd
        lcl = mu - 3*sd
        
        vio = check_nelson_rules(x, mu, ucl, lcl, sd)
        violation_indices = [v[0] for v in vio]
        
        colors = ['red' if i+1 in violation_indices else '#3b82f6' for i in range(len(x))]
        
        # 차트 생성
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(x)+1)),
            y=x,
            mode='lines+markers',
            name='데이터',
            line=dict(color='#3b82f6', width=2),
            marker=dict(color=colors, size=5)
        ))
        
        fig.add_hline(y=mu, line_dash="solid", line_color="#10b981", 
                     annotation_text="CL", annotation_position="right")
        fig.add_hline(y=ucl, line_dash="dash", line_color="#ef4444", 
                     annotation_text="UCL", annotation_position="right")
        fig.add_hline(y=lcl, line_dash="dash", line_color="#ef4444", 
                     annotation_text="LCL", annotation_position="right")
        
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
        
        # 통계
        series = df[var].dropna()
        mu_current = series.mean()
        sd_current = series.std(ddof=1)
        
        mu0 = mu
        sd0 = sd
        
        stats_html = f"""
        <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 1rem; padding: 1rem; background: #f9fafb; border-radius: 8px; margin: 1rem 0;">
            <div style="text-align: center;">
                <div style="font-size: 0.75rem; color: #6b7280; margin-bottom: 0.4rem;">현재 평균</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #1f2937;">{mu_current:.2f}</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 0.75rem; color: #6b7280; margin-bottom: 0.4rem;">표준편차</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #1f2937;">{sd_current:.2f}</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 0.75rem; color: #6b7280; margin-bottom: 0.4rem;">기준선 평균</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #10b981;">{mu0:.2f}</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 0.75rem; color: #6b7280; margin-bottom: 0.4rem;">UCL</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #ef4444;">{ucl:.2f}</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 0.75rem; color: #6b7280; margin-bottom: 0.4rem;">LCL</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #ef4444;">{lcl:.2f}</div>
            </div>
        </div>
        """
        
        # 로그 테이블
        if not vio:
            log_html = "<p style='text-align: center; color: #6b7280; padding: 2rem;'>✅ 이상 없음</p>"
        else:
            log_df = pd.DataFrame(vio, columns=["샘플", "룰", "설명", "값"])
            log_df["값"] = log_df["값"].round(3)
            log_html = log_df.tail(30).to_html(index=False, classes="table table-sm")
        
        # 모달 표시
        m = ui.modal(
            ui.h4(f"{var} 상세 관리도", class_="mb-3"),
            ui.HTML(chart_html),
            ui.HTML(stats_html),
            ui.h5("🚨 이상 패턴 로그", class_="mt-3 mb-2"),
            ui.HTML(f'<div class="scroll-table" style="max-height: 250px; overflow-y: auto;">{log_html}</div>'),
            title=None,
            size="xl",
            easy_close=True,
            footer=None
        )
        
        ui.modal_show(m)

    # ==================== 전체 타임라인 ====================
    @output
    @render.table
    def timeline_table():
        df = df_view()
        base = df_baseline()
        out_rows = []
        dtcol = "__dt__" if "__dt__" in df.columns else None

    # ==================== 전체 타임라인 ====================
    @output
    @render.table
    def timeline_table():
        df = df_view()
        base = df_baseline()
        out_rows = []
        dtcol = "__dt__" if "__dt__" in df.columns else None

        # 단변량
        for var in FEATURES_ALL:
            s = df[var].dropna()
            if len(s) < 10:
                continue
            
            if base is None or var not in base.columns or len(base) < 5:
                mu0, sd0 = s.mean(), s.std(ddof=1)
            else:
                mu0, sd0 = base[var].mean(), base[var].std(ddof=1)
            
            vio = check_nelson_rules(s.to_numpy(), mu0, mu0 + 3*sd0, mu0 - 3*sd0, sd0)
            
            for (idx, r, desc, val) in vio[-20:]:
                ts = df.iloc[s.index.min() + idx - 1][dtcol] if dtcol else np.nan
                out_rows.append({
                    "시각": ts,
                    "유형": "단변량",
                    "변수": var,
                    "룰": r,
                    "설명": desc,
                    "값": round(val, 3)
                })

        # 다변량
        for key, vars_ in PROCESS_GROUPS.items():
            sub = df[vars_].dropna()
            p = len(vars_)
            if sub.shape[0] < max(30, p + 5):
                continue
            
            base_df = base[vars_].dropna() if (base is not None and set(vars_).issubset(base.columns)) else sub
            mu = base_df.mean().to_numpy()
            cov = np.cov(base_df.to_numpy().T)
            
            try:
                inv_cov = np.linalg.inv(cov)
            except:
                inv_cov = np.linalg.pinv(cov)
            
            t2 = calculate_hotelling_t2(sub.to_numpy(), mu, inv_cov)
            ucl = phaseII_ucl_t2(len(sub), p, 0.01)
            viol_idx = np.where(t2 > ucl)[0][-20:]
            
            for idx in viol_idx:
                orig_idx = sub.index[idx]
                ts = df.loc[orig_idx, dtcol] if dtcol else np.nan
                out_rows.append({
                    "시각": ts,
                    "유형": "다변량",
                    "공정": key,
                    "룰": "T²",
                    "설명": "T² 초과",
                    "값": round(t2[idx], 3)
                })

        if not out_rows:
            return pd.DataFrame({"상태": ["최근 이상 없음"]})
        
        timeline = pd.DataFrame(out_rows)
        if "시각" in timeline.columns and timeline["시각"].notna().any():
            timeline = timeline.sort_values("시각", ascending=False)
        
        return timeline.head(200)