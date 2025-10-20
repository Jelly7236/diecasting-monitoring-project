# modules/page_control.py
from shiny import ui, render, reactive
import pandas as pd
import numpy as np
from scipy import stats

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

# 규격 한계 정의
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
    """넬슨 핵심 4가지 룰 검정"""
    violations = []
    n = len(data)
    
    for i in range(n):
        # Rule 1: UCL/LCL 초과
        if data[i] > ucl:
            violations.append((i+1, "Rule 1", "UCL 초과", data[i]))
        elif data[i] < lcl:
            violations.append((i+1, "Rule 1", "LCL 미만", data[i]))
        
        # Rule 2: 연속 9개 점이 중심선 한쪽
        if i >= 8:
            if all(data[i-j] > mean for j in range(9)):
                violations.append((i+1, "Rule 2", "연속 9개 점이 중심선 위", data[i]))
            elif all(data[i-j] < mean for j in range(9)):
                violations.append((i+1, "Rule 2", "연속 9개 점이 중심선 아래", data[i]))
        
        # Rule 3: 연속 6개 점이 증가/감소
        if i >= 5:
            increasing = all(data[i-j] < data[i-j+1] for j in range(5, 0, -1))
            decreasing = all(data[i-j] > data[i-j+1] for j in range(5, 0, -1))
            if increasing:
                violations.append((i+1, "Rule 3", "연속 6개 점 증가 추세", data[i]))
            elif decreasing:
                violations.append((i+1, "Rule 3", "연속 6개 점 감소 추세", data[i]))
        
        # Rule 5: 3개 중 2개가 2σ 밖
        if i >= 2:
            zone2_upper = mean + 2*sigma
            zone2_lower = mean - 2*sigma
            count = sum(1 for j in range(3) if data[i-j] > zone2_upper or data[i-j] < zone2_lower)
            if count >= 2:
                violations.append((i+1, "Rule 5", "3개 중 2개가 2σ 영역 밖", data[i]))
    
    return violations

def calculate_hotelling_t2(data_matrix, mean_vector, inv_cov):
    """Hotelling T² 통계량 계산"""
    t2_values = []
    for i in range(len(data_matrix)):
        diff = data_matrix[i] - mean_vector
        t2 = diff @ inv_cov @ diff.T
        t2_values.append(t2)
    return np.array(t2_values)

def phaseII_ucl_t2(n, p, alpha=0.01):
    """Phase II UCL 계산"""
    return (p * (n-1) * (n+1) / (n * (n-p))) * stats.f.ppf(1-alpha, p, n-p)

def calculate_cp_cpk(data, usl, lsl):
    """공정능력지수 계산"""
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    
    cp = (usl - lsl) / (6 * std) if std > 0 else 0
    cpu = (usl - mean) / (3 * std) if std > 0 else 0
    cpl = (mean - lsl) / (3 * std) if std > 0 else 0
    cpk = min(cpu, cpl)
    
    return cp, cpk, cpu, cpl, mean, std

def to_datetime_safe(df):
    """날짜 변환"""
    if "tryshot_time" in df.columns:
        return pd.to_datetime(df["tryshot_time"], errors='coerce')
    return None

def build_univar_figure(x, mu, sd, violations, title=""):
    """단변량 관리도 그래프"""
    import plotly.graph_objs as go
    
    ucl = mu + 3*sd
    lcl = mu - 3*sd
    violation_indices = [v[0] for v in violations]
    
    fig = go.Figure()
    
    colors = ['red' if i+1 in violation_indices else '#3b82f6' for i in range(len(x))]
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
        title=title,
        xaxis_title="샘플 번호",
        yaxis_title="측정값",
        template="plotly_white",
        height=350,
        hovermode='x unified',
        showlegend=False
    )
    
    return fig

def build_t2_figure(t2_values, ucl, title="", viol_idx=None):
    """다변량 T² 관리도 그래프"""
    import plotly.graph_objs as go
    
    colors = ['red' if i in viol_idx else '#3b82f6' for i in range(len(t2_values))] if viol_idx is not None else ['#3b82f6']*len(t2_values)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(1, len(t2_values)+1)),
        y=t2_values,
        mode='lines+markers',
        name='T² 통계량',
        line=dict(color='#3b82f6', width=2),
        marker=dict(color=colors, size=5)
    ))
    
    fig.add_hline(y=ucl, line_dash="dash", line_color="#ef4444",
                 annotation_text=f"UCL", annotation_position="right")
    
    fig.update_layout(
        title=title,
        xaxis_title="샘플 번호",
        yaxis_title="T² 값",
        template="plotly_white",
        height=350,
        hovermode='x unified'
    )
    
    return fig

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
            .kpi-row {
                display: grid;
                grid-template-columns: repeat(5, 1fr);
                gap: 0.75rem;
                padding: 1rem;
            }
            .kpi-box {
                background: #fafbfc;
                border: 1px solid #e5e7eb;
                border-radius: 6px;
                padding: 0.75rem;
                text-align: center;
            }
            .kpi-title {
                font-size: 0.75rem;
                color: #6b7280;
                font-weight: 500;
                margin-bottom: 0.4rem;
            }
            .kpi-main-value {
                font-size: 1.8rem;
                font-weight: 700;
                color: #1f2937;
                margin-bottom: 0.3rem;
                line-height: 1;
            }
            .kpi-sub-value {
                font-size: 0.7rem;
                color: #9ca3af;
                font-weight: 400;
            }
            .badge {
                display: inline-block;
                padding: 0.25rem 0.6rem;
                border-radius: 4px;
                font-size: 0.75rem;
                font-weight: 600;
                margin-right: 0.4rem;
            }
            .b-red { background: #fee2e2; color: #991b1b; }
            .b-amber { background: #fef3c7; color: #92400e; }
            .b-blue { background: #dbeafe; color: #1e40af; }
            .b-gray { background: #f3f4f6; color: #374151; }
            .scroll-table { max-height: 250px; overflow-y: auto; }
            .section-divider {
                margin: 1.5rem 0;
                border-top: 2px solid #e5e7eb;
                padding-top: 1rem;
            }
            h3 { color: #1f2937; font-weight: 700; margin-bottom: 1rem; font-size: 1.5rem; }
            h6 { color: #4b5563; font-weight: 600; margin-bottom: 0.5rem; font-size: 0.9rem; }
            .nav-tabs .nav-link { 
                color: #6b7280; 
                font-weight: 500; 
                padding: 0.5rem 1rem;
                font-size: 0.9rem;
            }
            .nav-tabs .nav-link.active { 
                color: #1f2937; 
                font-weight: 600; 
                border-bottom: 3px solid #3b82f6; 
            }
        """),

        ui.div(
            ui.h3("📊 공정 관리 상태 분석", class_="text-center mb-3"),

            # ==================== 컨트롤 바 ====================
            ui.card(
                ui.card_header("⚙️ 분석 설정"),
                ui.div(
                    ui.layout_columns(
                        ui.output_ui("mold_select"),
                        ui.input_numeric("win", "윈도우(샘플 수)", 200, min=50, max=5000, step=50),
                        ui.input_switch("phase_guard", "Phase I 기준선 (정상만)", True),
                        col_widths=[4, 4, 4]
                    ),
                    style="padding: 0.5rem;"
                )
            ),

            ui.hr(style="margin: 1.5rem 0; border-color: #d1d5db;"),

            # ==================== 공정별 탭 ====================
            ui.navset_tab(
                *[
                    ui.nav_panel(
                        process_name,
                        ui.div(
                            # 다변량 관리도
                            ui.card(
                                ui.card_header(f"🔬 다변량 관리도 (Hotelling T²) - 변수: {', '.join(var_list)}"),
                                ui.layout_columns(
                                    ui.output_ui(f"t2_plot_{i}"),
                                    ui.div(
                                        ui.h6("📄 T² 초과 로그", class_="mb-2"),
                                        ui.div(ui.output_table(f"t2_table_{i}"), class_="scroll-table")
                                    ),
                                    col_widths=[7, 5]
                                )
                            ),
                            
                            # 단변량 관리도 섹션
                            ui.div(
                                # KPI + 변수 선택 통합
                                ui.card(
                                    ui.card_header(
                                        ui.div(
                                            ui.div("📈 단변량 관리도 (Nelson Rules)", style="display: inline-block; margin-right: 2rem;"),
                                            ui.div(
                                                ui.input_select(
                                                    f"var_select_{i}",
                                                    "",
                                                    choices={v: v for v in var_list},
                                                    selected=var_list[0]
                                                ),
                                                style="display: inline-block; min-width: 200px;"
                                            ),
                                            style="display: flex; align-items: center; justify-content: space-between;"
                                        )
                                    ),
                                    ui.output_ui(f"kpi_bar_{i}")
                                ),
                                
                                # 단변량 차트 + 로그
                                ui.card(
                                    ui.layout_columns(
                                        ui.div(
                                            ui.output_ui(f"uni_plot_{i}"),
                                            ui.output_ui(f"nelson_badges_{i}"),
                                        ),
                                        ui.div(
                                            ui.h6("🚨 이상 패턴 로그", class_="mb-2"),
                                            ui.div(ui.output_table(f"nelson_table_{i}"), class_="scroll-table")
                                        ),
                                        col_widths=[7, 5]
                                    )
                                ),
                                
                                class_="section-divider"
                            ),
                            
                            class_="mt-3"
                        )
                    )
                    for i, (process_name, var_list) in enumerate(PROCESS_GROUPS.items(), start=1)
                ],
                id="process_tabs"
            ),

            # ==================== 타임라인 (최하단) ====================
            ui.div(
                ui.card(
                    ui.card_header("🕒 최근 이상 타임라인 (전체 통합)"),
                    ui.div(ui.output_table("timeline_table"), class_="scroll-table", style="max-height: 350px;")
                ),
                class_="section-divider"
            ),

            style="max-width: 1600px; margin: 0 auto; padding: 0 0.75rem;"
        )
    )


# ==================== SERVER ====================
def server_control(input, output, session):
    
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

    # 기준선 (Phase I)
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

    # ==================== 공정별 다변량 + 단변량 + KPI ====================
    for i, (process_name, var_list) in enumerate(PROCESS_GROUPS.items(), start=1):
        
        # 다변량 T² 차트
        @output(id=f"t2_plot_{i}")
        @render.ui
        def t2_plot(i=i, var_list=var_list):
            df = df_view()
            base = df_baseline() if input.phase_guard() else None
            
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
            
            fig = build_t2_figure(t2, ucl, title=f"Hotelling T² (n={X.shape[0]})", viol_idx=viol_idx)
            return ui.HTML(fig.to_html(include_plotlyjs='cdn', div_id=f"t2_{i}"))
        
        # 다변량 T² 로그
        @output(id=f"t2_table_{i}")
        @render.table
        def t2_table(i=i, var_list=var_list):
            df = df_view()
            base = df_baseline() if input.phase_guard() else None
            
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
        
        # KPI 바 (선택된 변수)
        @output(id=f"kpi_bar_{i}")
        @render.ui
        def kpi_bar(i=i):
            var = input[f"var_select_{i}"]()
            df = df_view()
            base = df_baseline() if input.phase_guard() else None
            
            series = df[var].dropna()
            if len(series) < 5:
                return ui.p("표본이 부족합니다.", style="color: #6b7280; text-align: center; padding: 2rem;")
            
            # 기준선 평균/표준편차
            if base is None or var not in base.columns or len(base) < 5:
                mu0, sd0 = series.mean(), series.std(ddof=1)
            else:
                mu0, sd0 = base[var].mean(), base[var].std(ddof=1)
            
            # 현재 평균/표준편차
            mu_current = series.mean()
            sd_current = series.std(ddof=1)
            
            # UCL/LCL
            ucl = mu0 + 3 * sd0
            lcl = mu0 - 3 * sd0
            
            # Cp/Cpk
            if var in SPEC_LIMITS:
                cp, cpk, *_ = calculate_cp_cpk(series.to_numpy(), 
                                              SPEC_LIMITS[var]["usl"], 
                                              SPEC_LIMITS[var]["lsl"])
                cp_text = f"{cp:.2f}"
                cpk_text = f"{cpk:.2f}"
            else:
                cp_text = "—"
                cpk_text = "—"
            
            return ui.div(
                # 변수명
                ui.div(
                    ui.div("변수", class_="kpi-title"),
                    ui.div(var, class_="kpi-main-value", style="font-size: 1.5rem;"),
                    class_="kpi-box"
                ),
                # 평균(μ)
                ui.div(
                    ui.div("평균(μ)", class_="kpi-title"),
                    ui.div(f"{mu_current:.2f}", class_="kpi-main-value"),
                    ui.div(f"기준선 μ={mu0:.2f}", class_="kpi-sub-value"),
                    class_="kpi-box"
                ),
                # 표준편차(σ)
                ui.div(
                    ui.div("표준편차(σ)", class_="kpi-title"),
                    ui.div(f"{sd_current:.2f}", class_="kpi-main-value"),
                    ui.div(f"기준선 σ={sd0:.2f}", class_="kpi-sub-value"),
                    class_="kpi-box"
                ),
                # UCL/LCL
                ui.div(
                    ui.div("UCL/LCL(±3σ)", class_="kpi-title"),
                    ui.div(f"{ucl:.2f} / {lcl:.2f}", class_="kpi-main-value", style="font-size: 1.3rem;"),
                    class_="kpi-box"
                ),
                # Cp/Cpk
                ui.div(
                    ui.div("Cp / Cpk", class_="kpi-title"),
                    ui.div(f"{cp_text} / {cpk_text}", class_="kpi-main-value", style="font-size: 1.3rem;"),
                    class_="kpi-box"
                ),
                class_="kpi-row"
            )
        
        # 단변량 차트 (선택된 변수)
        @output(id=f"uni_plot_{i}")
        @render.ui
        def uni_plot(i=i):
            var = input[f"var_select_{i}"]()
            df = df_view()
            base = df_baseline() if input.phase_guard() else None
            
            x = df[var].dropna().to_numpy()
            if len(x) < 10:
                return ui.p("표본 부족", style="color: #9ca3af; padding: 2rem; text-align: center;")
            
            mu = (base[var].mean() if base is not None and var in base.columns and len(base) > 5 else np.mean(x))
            sd = (base[var].std(ddof=1) if base is not None and var in base.columns and len(base) > 5 else np.std(x, ddof=1))
            
            vio = check_nelson_rules(x, mu, mu + 3*sd, mu - 3*sd, sd)
            fig = build_univar_figure(x, mu, sd, vio, title=f"{var} 관리도 (n={len(x)})")
            return ui.HTML(fig.to_html(include_plotlyjs='cdn', div_id=f"uni_{i}"))
        
        # 넬슨 룰 배지
        @output(id=f"nelson_badges_{i}")
        @render.ui
        def nelson_badges(i=i):
            var = input[f"var_select_{i}"]()
            df = df_view()
            base = df_baseline() if input.phase_guard() else None
            
            x = df[var].dropna().to_numpy()
            if len(x) < 10:
                return ui.div()
            
            mu = (base[var].mean() if base is not None and var in base.columns and len(base) > 5 else np.mean(x))
            sd = (base[var].std(ddof=1) if base is not None and var in base.columns and len(base) > 5 else np.std(x, ddof=1))
            
            vio = check_nelson_rules(x, mu, mu + 3*sd, mu - 3*sd, sd)
            counts = {"Rule 1": 0, "Rule 2": 0, "Rule 3": 0, "Rule 5": 0}
            for _, r, _, _ in vio:
                if r in counts:
                    counts[r] += 1
            
            return ui.div(
                ui.span(f"Rule 1: {counts['Rule 1']}", class_="badge b-red"),
                ui.span(f"Rule 2: {counts['Rule 2']}", class_="badge b-amber"),
                ui.span(f"Rule 3: {counts['Rule 3']}", class_="badge b-blue"),
                ui.span(f"Rule 5: {counts['Rule 5']}", class_="badge b-gray"),
                style="margin-top: 0.5rem; padding: 0 1rem;"
            )
        
        # 넬슨 룰 로그
        @output(id=f"nelson_table_{i}")
        @render.table
        def nelson_table(i=i):
            var = input[f"var_select_{i}"]()
            df = df_view()
            base = df_baseline() if input.phase_guard() else None
            
            x = df