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
            sigma = info["std"]

            try:
                y_pred = np.asarray(model.forecast(steps=len(series)))
                residuals = np.asarray(series) - y_pred
                target_array = residuals
                current_val = residuals[-1]
            except Exception as e:
                print(f"⚠️ ARIMA 계산 실패 ({key}):", e)
                target_array = series.to_numpy()
                current_val = series.iloc[-1]

            # ✅ molten_temp만 표준화 한계 사용
            if var == "molten_temp":
                cl = info.get("cl", 0.0)
                ucl = info.get("ucl_standardized", info.get("ucl", 3 * sigma))
                lcl = info.get("lcl_standardized", info.get("lcl", -3 * sigma))
            else:
                cl = info.get("cl", 0.0)
                ucl = info.get("ucl_individual", 3 * sigma)
                lcl = info.get("lcl_individual", -3 * sigma)

        # -------------------------
        # 2️⃣ X–R 관리도 기반
        # -------------------------
        elif key in xr_limits:
            info = xr_limits[key]
            n = info["n"]  # 부분군 크기
            
            # ✅ 부분군별로 X̄ 계산
            x_array = series.to_numpy()
            n_subgroups = len(x_array) // n
            
            if n_subgroups == 0:
                continue
                
            x_bars = np.array([
                x_array[i*n:(i+1)*n].mean() 
                for i in range(n_subgroups)
            ])
            
            # ✅ 실시간 데이터 기준으로 관리한계 계산
            from shared import XR_CONSTANTS
            if n in XR_CONSTANTS:
                A2 = XR_CONSTANTS[n]["A2"]
            else:
                A2 = 0.577  # 기본값 (n=5)
            
            # 초기 10개 부분군 기준으로 X̄̄ 계산
            n_base = min(10, n_subgroups)
            Xbar_bar = np.mean(x_bars[:n_base])
            
            # shared의 R̄ 값 사용
            Rbar = info["CL_R"]
            
            # 실시간 관리한계 계산
            cl = Xbar_bar
            ucl = Xbar_bar + A2 * Rbar
            lcl = Xbar_bar - A2 * Rbar
            sigma = (ucl - cl) / 3
            
            target_array = x_bars
            current_val = x_bars[-1]  # 마지막 부분군의 평균

        # -------------------------
        # 3️⃣ 기본값 (데이터 기반)
        # -------------------------
        else:
            cl = series.mean()
            sigma = series.std(ddof=1)
            ucl = cl + 3 * sigma
            lcl = cl - 3 * sigma
            target_array = series.to_numpy()
            current_val = series.iloc[-1]

        # -------------------------
        # ✅ Nelson Rule 기반 이상 판정
        # -------------------------
        try:
            window_size = 20
            recent_data = target_array[-window_size:] if len(target_array) > window_size else target_array

            violations = check_nelson_rules(recent_data, cl, ucl, lcl, sigma)
            violated_rules = [v[1] for v in violations]
        except Exception as e:
            print(f"⚠️ Nelson Rule 계산 실패 ({key}):", e)
            violated_rules = []

        # ✅ 정확히 일치 비교 (Rule 1만 alert로)
        if "Rule 1" in violated_rules:
            status_class = "alert"       # 관리한계 초과
        elif any(r in ["Rule 2", "Rule 3", "Rule 5"] for r in violated_rules):
            status_class = "warning"     # 추세/집단 이상
        else:
            status_class = ""            # 정상

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
    from shiny import ui
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
        sigma = info["std"]

        try:
            y_pred = np.asarray(model.forecast(steps=len(x)))
            residuals = np.asarray(x) - y_pred
        except Exception as e:
            print(f"⚠️ ARIMA 예측 실패 ({key}):", e)
            residuals = np.asarray(x) - np.mean(x)

        # ✅ molten_temp만 표준화 한계 사용
        if var == "molten_temp":
            cl = info.get("cl", 0.0)
            ucl = info.get("ucl_standardized", info.get("ucl", 3 * sigma))
            lcl = info.get("lcl_standardized", info.get("lcl", -3 * sigma))
        else:
            cl = info.get("cl", 0.0)
            ucl = info.get("ucl_individual", 3 * sigma)
            lcl = info.get("lcl_individual", -3 * sigma)

        vio = check_nelson_rules(np.array(residuals), cl, ucl, lcl, sigma)
        violation_indices = [v[0] for v in vio]
        colors = ["red" if i + 1 in violation_indices else "#3b82f6" for i in range(len(residuals))]

        y_plot = residuals
        x_axis = list(range(1, len(residuals) + 1))
        title_suffix = "ARIMA 잔차 관리도"
        y_label = "잔차 (Residual)"
        x_label = "샘플 번호"

    # ====================== 2️⃣ X–R 관리도 기반 ======================
    elif key in xr_limits:
        info = xr_limits[key]
        n = info["n"]  # 부분군 크기

        # ✅ shared에서 관리도 상수 자동 로드
        from shared import XR_CONSTANTS
        if n in XR_CONSTANTS:
            A2 = XR_CONSTANTS[n]["A2"]
            D3 = XR_CONSTANTS[n]["D3"]
            D4 = XR_CONSTANTS[n]["D4"]
        else:
            A2, D3, D4 = 0.577, 0.0, 2.114
            print(f"⚠️ XR_CONSTANTS에서 n={n} 값 없음 → 기본 상수 사용")

        # ✅ 부분군 나누기
        n_subgroups = len(x) // n
        if n_subgroups < 2:
            ui.notification_show(
                f"X–R 관리도를 그리기엔 부분군이 부족합니다 (최소 2개 필요, 현재 {n_subgroups}개)",
                type="warning"
            )
            return

        # ✅ X̄와 R 계산
        x_bars = np.array([x[i*n:(i+1)*n].mean() for i in range(n_subgroups)])
        ranges = np.array([x[i*n:(i+1)*n].max() - x[i*n:(i+1)*n].min() for i in range(n_subgroups)])

        # ✅ 초기 10개 부분군 기준으로 X̄̄ 계산
        n_base = min(10, n_subgroups)
        base_xbar = x_bars[:n_base]
        Xbar_bar = np.mean(base_xbar)

        # ✅ shared 기준의 Rbar 사용
        Rbar = info["CL_R"]

        # ✅ X̄ 관리도 한계선 (실시간 Xbar 기준 + shared Rbar)
        UCL_X = Xbar_bar + A2 * Rbar
        CL_X = Xbar_bar
        LCL_X = Xbar_bar - A2 * Rbar

        # ✅ R 관리도 한계선 (shared 기준 그대로)
        UCL_R = info["UCL_R"]
        CL_R = info["CL_R"]
        LCL_R = info["LCL_R"]

        # ✅ Nelson Rules 체크 (X̄ 관리도 기준)
        sigma_x = (UCL_X - CL_X) / 3
        vio_x = check_nelson_rules(x_bars, CL_X, UCL_X, LCL_X, sigma_x)
        violation_indices_x = [v[0] for v in vio_x]
        colors_x = ["red" if i + 1 in violation_indices_x else "#3b82f6" for i in range(len(x_bars))]

        # ✅ Plotly 서브플롯 구성
        from plotly.subplots import make_subplots
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            subplot_titles=("X̄ 관리도", "R 관리도"),
            vertical_spacing=0.12
        )

        # --------------------------
        # X̄ 관리도
        # --------------------------
        fig.add_trace(
            go.Scatter(
                x=np.arange(1, len(x_bars)+1),
                y=x_bars,
                mode="lines+markers",
                name="X̄",
                marker=dict(color=colors_x, size=6),
                line=dict(color="#3b82f6", width=2)
            ),
            row=1, col=1
        )
        fig.add_hline(y=UCL_X, line_dash="dash", line_color="#ef4444",
                    annotation_text="UCL", annotation_position="right", row=1, col=1)
        fig.add_hline(y=CL_X, line_dash="solid", line_color="#10b981",
                    annotation_text="CL", annotation_position="right", row=1, col=1)
        fig.add_hline(y=LCL_X, line_dash="dash", line_color="#ef4444",
                    annotation_text="LCL", annotation_position="right", row=1, col=1)

        # --------------------------
        # R 관리도
        # --------------------------
        fig.add_trace(
            go.Scatter(
                x=np.arange(1, len(ranges)+1),
                y=ranges,
                mode="lines+markers",
                name="R",
                marker=dict(color="#f59e0b", size=6),
                line=dict(color="#f59e0b", width=2)
            ),
            row=2, col=1
        )
        fig.add_hline(y=UCL_R, line_dash="dash", line_color="#ef4444",
                    annotation_text="UCL", annotation_position="right", row=2, col=1)
        fig.add_hline(y=CL_R, line_dash="solid", line_color="#10b981",
                    annotation_text="CL", annotation_position="right", row=2, col=1)
        fig.add_hline(y=LCL_R, line_dash="dash", line_color="#ef4444",
                    annotation_text="LCL", annotation_position="right", row=2, col=1)

        fig.update_layout(
            title=f"{var} X–R 관리도 (초기 10개 기준, n={n})",
            xaxis2_title="부분군 번호",
            height=700,
            template="plotly_white",
            showlegend=False
        )

        chart_html = fig.to_html(include_plotlyjs="cdn", div_id="modal_chart_div")

        # ✅ 통계값 표시
        stats_html = f"""
        <div style='display:flex; flex-direction:column; background:#f9fafb; border-radius:10px;
                    padding:1rem; margin-top:1rem;'>
            <div style='text-align:center; font-weight:bold;'>초기 10개 기준값</div>
            <div style='display:flex; justify-content:space-around; margin-bottom:0.5rem;'>
                <div>X̄̄ = {Xbar_bar:.2f}</div><div>R̄ = {Rbar:.2f}</div>
            </div>
            <div style='text-align:center; font-weight:bold;'>X̄ 관리도</div>
            <div style='display:flex; justify-content:space-around; margin-bottom:0.5rem;'>
                <div>CL={CL_X:.2f}</div><div>UCL={UCL_X:.2f}</div><div>LCL={LCL_X:.2f}</div>
            </div>
            <div style='text-align:center; font-weight:bold;'>R 관리도</div>
            <div style='display:flex; justify-content:space-around;'>
                <div>CL={CL_R:.2f}</div><div>UCL={UCL_R:.2f}</div><div>LCL={LCL_R:.2f}</div>
            </div>
        </div>
        """

        # ✅ 이상 패턴 로그 (X̄ 관리도 기준)
        if not vio_x:
            log_html = "<p style='text-align:center; color:#6b7280; padding:1rem;'>✅ 이상 없음</p>"
        else:
            log_df = pd.DataFrame(vio_x, columns=["샘플", "룰", "설명", "값"])
            log_df["값"] = log_df["값"].round(3)
            log_html = log_df.to_html(index=False, classes="table table-striped table-sm", border=0)

        # ✅ 모달 바로 반환
        m = ui.modal(
            ui.h4(f"{var} 상세 관리도", class_="mb-3"),
            ui.HTML(chart_html),
            ui.HTML(stats_html),
            ui.h5("🚨 이상 패턴 로그 (X̄ 관리도)", class_="mt-3 mb-2"),
            ui.HTML(f"<div class='scroll-table' style='max-height:250px;'>{log_html}</div>"),
            size="xl",
            easy_close=True,
            footer=None,
        )
        ui.modal_show(m)
        return   # ← X-R 관리도는 여기서 종료

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
        x_axis = list(range(1, len(x) + 1))
        title_suffix = "기초 통계 기반"
        y_label = "측정값"
        x_label = "샘플 번호"

    # ======================
    # Plotly 관리도 시각화
    # ======================
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=y_plot,
            mode="lines+markers",
            name="값",
            line=dict(color="#3b82f6", width=2),
            marker=dict(color=colors, size=6),
        )
    )

    fig.add_hline(y=cl, line_dash="solid", line_color="#10b981", annotation_text="CL", annotation_position="right")
    fig.add_hline(y=ucl, line_dash="dash", line_color="#ef4444", annotation_text="UCL", annotation_position="right")
    fig.add_hline(y=lcl, line_dash="dash", line_color="#ef4444", annotation_text="LCL", annotation_position="right")

    fig.update_layout(
        title=f"{var} ({title_suffix})",
        xaxis_title=x_label,
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