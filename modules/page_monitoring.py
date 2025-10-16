from shiny import ui, render
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc


def ui_monitoring():
    return ui.page_fluid(
        ui.h3("모델 모니터링 및 성능 분석"),
        ui.p("실시간 예측 결과와 실제 결과를 비교하여 모델의 성능을 평가합니다."),
        ui.hr(),

        ui.layout_columns(
            ui.card(
                ui.card_header("✅ 실시간 성능 지표"),
                ui.output_table("model_metrics")
            ),
            ui.card(
                ui.card_header("📉 혼동 행렬"),
                ui.output_plot("confusion_plot", height="300px")
            ),
            col_widths=[5, 7]
        ),
        ui.hr(),
        ui.layout_columns(
            ui.card(
                ui.card_header("📊 ROC 곡선"),
                ui.output_plot("roc_plot", height="320px")
            ),
            ui.card(
                ui.card_header("📈 Precision-Recall 곡선"),
                ui.output_plot("pr_plot", height="320px")
            ),
            col_widths=[6, 6]
        ),
        style="max-width:1300px;margin:0 auto;"
    )


def server_monitoring(input, output, session):
    # 샘플 예측/실제 데이터
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 200)
    y_pred = np.random.randint(0, 2, 200)
    y_prob = np.random.rand(200)

    # 성능지표
    acc = (y_true == y_pred).mean()
    precision = np.sum((y_true == 1) & (y_pred == 1)) / max(np.sum(y_pred == 1), 1)
    recall = np.sum((y_true == 1) & (y_pred == 1)) / max(np.sum(y_true == 1), 1)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    @output
    @render.table
    def model_metrics():
        return pd.DataFrame({
            "지표": ["정확도", "정밀도", "재현율", "F1-score"],
            "값": [f"{acc:.3f}", f"{precision:.3f}", f"{recall:.3f}", f"{f1:.3f}"]
        })

    @output
    @render.plot
    def confusion_plot():
        cm = confusion_matrix(y_true, y_pred)
        fig = go.Figure(data=go.Heatmap(z=cm, x=["예측:양품", "예측:불량"],
                                        y=["실제:양품", "실제:불량"], colorscale="Blues", text=cm, texttemplate="%{text}"))
        fig.update_layout(title="혼동 행렬", template="plotly_white")
        return fig

    @output
    @render.plot
    def roc_plot():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = auc(fpr, tpr)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"AUC={auc_score:.2f}"))
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
        fig.update_layout(title="ROC 곡선", template="plotly_white")
        return fig

    @output
    @render.plot
    def pr_plot():
        precision_curve = np.linspace(0.5, 1, 20)
        recall_curve = np.linspace(1, 0.5, 20)
        fig = go.Figure(go.Scatter(x=recall_curve, y=precision_curve, mode='lines+markers'))
        fig.update_layout(title="Precision-Recall 곡선", xaxis_title="Recall", yaxis_title="Precision", template="plotly_white")
        return fig
