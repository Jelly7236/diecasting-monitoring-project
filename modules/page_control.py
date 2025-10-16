from shiny import ui, render, reactive
import plotly.graph_objs as go
import pandas as pd
import numpy as np

# -----------------------------
# UI
# -----------------------------
def ui_control():
    return ui.page_fluid(
        ui.h3("공정 관리 상태 분석"),
        ui.p("단변량 및 다변량 관리도를 통해 공정 이상 여부를 모니터링하고, 발생 로그 및 공정 능력(Cp, Cpk)을 확인합니다."),
        ui.hr(),

        # -----------------------------
        # 단변량 관리도
        # -----------------------------
        ui.card(
            ui.card_header("📈 단변량 관리도"),
            ui.layout_columns(
                ui.input_select(
                    "uni_var",
                    "변수 선택",
                    choices=["주조압력", "용탕온도", "형체력", "하형온도1", "주조속도"],
                    selected="주조압력"
                ),
                col_widths=[12]
            ),
            ui.output_plot("univariate_chart", height="350px"),
            ui.output_table("univariate_log")
        ),

        ui.hr(),

        # -----------------------------
        # 다변량 관리도 (공정 단계별 카드)
        # -----------------------------
        ui.h4("공정 단계별 다변량 관리도"),
        ui.div(
            *[
                ui.card(
                    ui.card_header(f"📊 {process}"),
                    ui.output_plot(f"multi_{i}_chart", height="300px"),
                    ui.output_table(f"multi_{i}_log")
                )
                for i, process in enumerate([
                    "용탕 준비 및 가열",
                    "반고체 슬러리 제조",
                    "사출 & 금형 충전",
                    "응고"
                ], start=1)
            ],
            style="display:grid;grid-template-columns:repeat(2,1fr);gap:1rem;"
        ),

        ui.hr(),

        # -----------------------------
        # Cp / Cpk 분석 섹션
        # -----------------------------
        ui.h4("공정능력 분석 (Cp / Cpk)"),
        ui.p("공정이 규격 한계 내에서 얼마나 안정적으로 운영되는지를 Cp, Cpk로 평가합니다."),
        ui.layout_columns(
            ui.card(
                ui.card_header("변수별 Cp / Cpk 분석"),
                ui.input_select(
                    "cp_var",
                    "분석 변수 선택",
                    choices=["주조압력", "용탕온도", "형체력", "하형온도1", "주조속도"],
                    selected="주조압력"
                ),
                ui.output_plot("cpk_plot", height="320px"),
                ui.output_table("cpk_table")
            ),
            col_widths=[12]
        ),

        style="max-width:1300px;margin:0 auto;"
    )


# -----------------------------
# SERVER
# -----------------------------
def server_control(input, output, session):
    np.random.seed(0)
    n = 50
    x = np.arange(1, n + 1)

    # -----------------------------
    # 1. 단변량 관리도
    # -----------------------------
    @output
    @render.plot
    def univariate_chart():
        var = input.uni_var()
        mean_val = np.random.uniform(80, 120)
        data = np.random.normal(mean_val, 5, size=n)
        ucl, lcl = mean_val + 10, mean_val - 10

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=data, mode="lines+markers", name=var))
        fig.add_hline(y=ucl, line_dash="dash", line_color="red", name="UCL")
        fig.add_hline(y=lcl, line_dash="dash", line_color="red", name="LCL")
        fig.update_layout(
            title=f"{var} 관리도",
            xaxis_title="샘플 번호",
            yaxis_title="값",
            template="plotly_white"
        )
        return fig

    @output
    @render.table
    def univariate_log():
        var = input.uni_var()
        df = pd.DataFrame({
            "샘플번호": np.arange(1, 11),
            "변수명": [var] * 10,
            "이상유형": np.random.choice(
                ["UCL 초과", "LCL 미만", "급격한 변동", "공정불안정"], size=10
            ),
            "값": np.round(np.random.uniform(80, 120, size=10), 2)
        })
        return df

    # -----------------------------
    # 2. 다변량 관리도 (공정별 카드)
    # -----------------------------
    process_groups = {
        1: ["용탕온도", "용탕부피", "슬리브온도"],
        2: ["저속속도", "고속속도", "형체력"],
        3: ["주조압력", "주조속도", "비스킷두께"],
        4: ["상형온도1", "하형온도1", "냉각수온도"]
    }

    for i, vars_ in process_groups.items():
        @output(id=f"multi_{i}_chart")
        @render.plot
        def multivariate_chart(i=i, vars_=vars_):
            # Hotelling T² 샘플 생성
            t2_values = np.random.chisquare(df=len(vars_), size=n)
            ucl = np.percentile(t2_values, 95)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=t2_values, mode="lines+markers", name=f"T² ({i})"))
            fig.add_hline(y=ucl, line_dash="dash", line_color="red", name="UCL")
            fig.update_layout(
                title=f"{', '.join(vars_)} (공정 {i}) Hotelling T² 관리도",
                xaxis_title="샘플 번호",
                yaxis_title="T² 값",
                template="plotly_white"
            )
            return fig

        @output(id=f"multi_{i}_log")
        @render.table
        def multivariate_log(i=i, vars_=vars_):
            df = pd.DataFrame({
                "샘플번호": np.random.randint(1, n, 5),
                "공정단계": [f"공정 {i}"] * 5,
                "관련변수": [", ".join(np.random.choice(vars_, size=2, replace=False)) for _ in range(5)],
                "이상유형": np.random.choice(["T² 초과", "급격한 변동", "이상치"], size=5),
                "T²값": np.round(np.random.uniform(5, 15, size=5), 2)
            })
            return df
