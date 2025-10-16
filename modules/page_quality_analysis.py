from shiny import ui, render, reactive
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from shared import streaming_df, rf_models, rf_explainers, iso_model, iso_features
from utils import model_utils, rule_engine
from viz import (
    control_plots as cp,
    correlation_plots as corrp,
    model_perf_plots as mp,
    shap_plots2 as sp
)

# ============================
# 🔤 Font & Matplotlib 설정
# ============================
plt.rcParams["font.family"] = ["Malgun Gothic", "Segoe UI Emoji", "Apple Color Emoji"]
plt.rcParams["axes.unicode_minus"] = False

# ============================
# 🌈 Custom Style
# ============================
STYLE = """
    * { font-family: 'Noto Sans KR', sans-serif; }
    body { background-color: #f5f7fa; }
    h2, h3, h4 { color: #1f2937; font-weight: 700; }
    .section-card {
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
        padding: 1rem;
        background: white;
    }
    .sub-note {
        color: #6b7280;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
"""


# ============================
# 📊 UI
# ============================
def ui_quality_analysis():
    return ui.page_fluid(
        ui.tags.style(STYLE),
        ui.h2("⚙️ 품질 관리·분석 대시보드", class_="text-center mb-4"),

        # 📅 기간 필터
        ui.card(
            ui.card_header("📅 분석 기간 선택"),
            ui.layout_columns(
                ui.input_date_range(
                    "date_range", "기간 선택",
                    start="2025-01-01", end="2025-12-31"
                ),
                ui.div("선택한 기간의 데이터만 분석에 반영됩니다.", class_="sub-note mt-2"),
                col_widths=[6, 6]
            ),
            class_="section-card"
        ),

        # ① 공정 상태 진단
        ui.div(
            ui.h4("📈 ① 공정 상태 진단"),
            ui.div("공정 변수별 이상 여부와 Rule 기반 탐지 결과를 확인합니다.", class_="sub-note"),
            ui.layout_columns(
                ui.div(
                    ui.input_select("mold_select", "금형 선택", list(rf_models.keys())),
                    ui.input_select(
                        "feature_select", "공정 변수 선택",
                        ["molten_temp", "cast_pressure", "sleeve_temperature", "upper_mold_temp1"],
                        selected="molten_temp"
                    ),
                    ui.output_plot("control_chart", height="260px"),
                    ui.output_plot("anomaly_plot", height="260px"),
                    class_="p-3"
                ),
                ui.div(
                    ui.h5("⚠️ Rule 기반 탐지 결과", class_="fw-bold mb-2"),
                    ui.output_table("rule_table"),
                    class_="p-3"
                ),
                col_widths=[8, 4]
            ),
            class_="section-card"
        ),

        # ② 이상 원인 분석
        ui.div(
            ui.h4("🔍 ② 이상 원인 분석"),
            ui.div("불량 발생에 영향을 미치는 주요 변수와 영향 방향을 시각화합니다.", class_="sub-note"),
            ui.layout_columns(
                ui.div(
                    ui.output_plot("feature_importance", height="300px"),
                    ui.div("📊 RandomForest 기반 변수 중요도", class_="sub-note")
                ),
                ui.div(
                    ui.output_plot("shap_summary", height="300px"),
                    ui.div("💡 SHAP Summary Plot — 변수 영향력 해석", class_="sub-note")
                ),
                col_widths=[6, 6]
            ),
            ui.card(
                ui.card_header("⚡ 개별 케이스 해석 (SHAP Force Plot)"),
                ui.input_slider("sample_index", "샘플 선택", min=0, max=200, value=0),
                ui.output_ui("shap_force_plot"),
                class_="p-3 mt-3"
            ),
            class_="section-card"
        ),

        # ③ 품질 영향 관계 분석
        ui.div(
            ui.h4("🔄 ③ 품질 영향 관계 분석"),
            ui.div("변수 간의 연관성과 패턴을 통해 품질 특성을 파악합니다.", class_="sub-note"),
            ui.layout_columns(
                ui.div(
                    ui.output_plot("corr_plot", height="300px"),
                    ui.div("🔗 변수 간 상관관계 Heatmap", class_="sub-note")
                ),
                ui.div(
                    ui.output_plot("scatter_plot", height="300px"),
                    ui.div("📈 주요 변수 간 관계 (산점도)", class_="sub-note")
                ),
                col_widths=[6, 6]
            ),
            class_="section-card"
        ),

        # ④ 모델 성능 및 개선 검증
        ui.div(
            ui.h4("🧠 ④ 모델 성능 및 개선 검증"),
            ui.div("모델의 예측 성능을 확인하고 개선 포인트를 도출합니다.", class_="sub-note"),
            ui.layout_columns(
                ui.div(ui.output_plot("confusion_matrix", height="280px")),
                ui.div(ui.output_plot("roc_curve", height="280px")),
                col_widths=[6, 6]
            ),
            ui.div(
                ui.h5("🗒 최근 예측 결과 샘플", class_="fw-bold mt-3 mb-2"),
                ui.output_table("recent_table"),
                class_="p-3"
            ),
            class_="section-card"
        ),
    )


# ============================
# 🧠 SERVER
# ============================
def server_quality_analysis(input, output, session):

    # ✅ 기간 필터링
    @reactive.calc
    def filtered_df():
        df = streaming_df.copy()
        if "datetime" in df.columns:
            start, end = input.date_range()
            df = df[
                (df["datetime"] >= pd.to_datetime(start)) &
                (df["datetime"] <= pd.to_datetime(end))
            ]
        return df

    # ① 관리도
    @output
    @render.plot
    def control_chart():
        df = filtered_df()
        feature = input.feature_select()
        if feature not in df.columns or df[feature].empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "⚠️ 데이터 없음", ha="center", va="center")
            ax.axis("off")
            return fig
        return cp.plot_xbar_r_chart(df[feature])

    # ② 이상탐지
    @output
    @render.plot
    def anomaly_plot():
        df = filtered_df()
        feature = input.feature_select()
        if iso_model is None or feature not in iso_features:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "⚠️ Isolation Forest 모델이 없습니다.", ha="center", va="center")
            ax.axis("off")
            return fig

        subset = df[iso_features].dropna()
        if subset.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "⚠️ 선택한 기간 내 데이터가 없습니다.", ha="center", va="center")
            ax.axis("off")
            return fig

        try:
            preds = iso_model.predict(subset)
        except Exception:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "⚠️ 예측 불가 (모델 입력 오류)", ha="center", va="center")
            ax.axis("off")
            return fig

        outlier_idx = subset[preds == -1].index
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(df.index, df[feature], label="정상", color="#3b82f6", lw=1.5)
        ax.scatter(outlier_idx, df.loc[outlier_idx, feature], color="#ef4444", s=25, label="이상치")
        ax.legend()
        ax.set_title(f"이상탐지 결과 ({feature})", fontsize=11)
        plt.tight_layout()
        return fig

    @output
    @render.table
    def rule_table():
        df = filtered_df()
        return rule_engine.apply_rules(df)

    # ③ 모델 로드 및 SHAP
    @reactive.calc
    def get_model():
        mold_code = input.mold_select()
        model, explainer = model_utils.load_model_and_shap(mold_code)
        return model, explainer

    @output
    @render.plot
    def feature_importance():
        model, _ = get_model()
        return model_utils.plot_feature_importance(model)

    @output
    @render.plot
    def shap_summary():
        df = filtered_df()
        model, explainer = get_model()
        sample = df.select_dtypes(include=["float64", "int64"]).head(200)
        try:
            shap_values = explainer(model.named_steps["preprocess"].transform(sample))
            fig = plt.figure(figsize=(6, 4))
            sp.plot_shap_summary(shap_values, sample)
            plt.tight_layout()
            return fig
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"⚠️ SHAP 계산 불가: {e}", ha="center", va="center")
            ax.axis("off")
            return fig

    @output
    @render.ui
    def shap_force_plot():
        df = filtered_df()
        model, explainer = get_model()
        sample = df.select_dtypes(include=["float64", "int64"]).head(200)
        idx = input.sample_index()
        if sample.empty or idx >= len(sample):
            return ui.div("⚠️ 데이터 부족", class_="text-muted")
        try:
            shap_values = explainer(model.named_steps["preprocess"].transform(sample))
            shap_html = shap.force_plot(
                explainer.expected_value,
                shap_values.values[idx, :],
                sample.iloc[idx, :],
                matplotlib=False
            ).html()
            return ui.HTML(shap_html)
        except Exception as e:
            return ui.div(f"⚠️ SHAP Force Plot 오류: {e}", class_="text-muted")

    # ④ 상관관계 / 성능
    @output
    @render.plot
    def corr_plot():
        return corrp.plot_corr(filtered_df())

    @output
    @render.plot
    def scatter_plot():
        df = filtered_df()
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
        if len(numeric_cols) < 2:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "⚠️ 수치형 변수가 부족합니다.", ha="center", va="center")
            ax.axis("off")
            return fig
        x, y = numeric_cols[:2]
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x=df[x], y=df[y], s=15, color="#3b82f6", alpha=0.6, ax=ax)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"{x} vs {y}", fontsize=11)
        plt.tight_layout()
        return fig

    @output
    @render.plot
    def confusion_matrix():
        df = filtered_df()
        mold_code = input.mold_select()
        model, _ = model_utils.load_model_and_shap(mold_code)
        sub = df.dropna(subset=["passorfail"])
        if sub.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "⚠️ 데이터 없음", ha="center", va="center")
            ax.axis("off")
            return fig
        return mp.plot_confusion_matrix(sub, model)

    @output
    @render.plot
    def roc_curve():
        df = filtered_df()
        mold_code = input.mold_select()
        model, _ = model_utils.load_model_and_shap(mold_code)
        sub = df.dropna(subset=["passorfail"])
        if sub.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "⚠️ 데이터 없음", ha="center", va="center")
            ax.axis("off")
            return fig
        return mp.plot_roc_curve(sub, model)

    @output
    @render.table
    def recent_table():
        return filtered_df().tail(10).round(2)
