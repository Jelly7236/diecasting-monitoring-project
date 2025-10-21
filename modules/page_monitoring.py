# modules/page_monitoring.py
from shiny import ui, render, reactive
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
# 🔹 추가: 실시간 예측 결과 공유용
from shared import prediction_state


# ==================== UI ====================
def ui_monitoring():
    return ui.page_fluid(
        ui.tags.style("""
            *{font-family:'Noto Sans KR',sans-serif}
            body{background:#f6f7fb}
            .container{max-width:1300px;margin:0 auto;padding:0 12px}
            .card{border:1px solid #e5e7eb;border-radius:14px;box-shadow:0 2px 6px rgba(0,0,0,.05);background:#fff}
            .card-header{background:#fafbfc;border-bottom:1px solid #eef0f3;padding:.9rem 1.1rem;font-weight:800;color:#111827}
            .section{margin-bottom:18px}
            .kpi-row{display:grid;grid-auto-flow:column;grid-auto-columns:minmax(180px,1fr);gap:12px;overflow-x:auto;padding:12px;align-items:stretch}
            .kcard{border:1px solid #e5e7eb;border-radius:12px;background:#fff}
            .kcard .title{color:#6b7280;font-size:.85rem;font-weight:700}
            .kcard .value{font-size:1.35rem;font-weight:900;color:#111827}
            .muted{color:#6b7280}
            .scroll-table{max-height:340px;overflow:auto;border:1px solid #eef0f3;border-radius:8px;background:#fff}
            .scroll-table table{width:100%}
            .scroll-table thead th{position:sticky;top:0;background:#fafbfc;z-index:1}
        """),

        ui.div(
            ui.h3("모델 모니터링 및 성능 분석"),
            ui.p("실시간 예측 결과와 실제 결과를 비교하여 모델의 성능을 평가합니다.", class_="muted"),
            class_="container section"
        ),

        # 컨트롤 바
        ui.div(
            ui.card(
                ui.card_header("⚙️ 컨트롤"),
                ui.layout_columns(
                    ui.input_slider("mon_thr", "임곗값 (τ)", min=0.0, max=1.0, value=0.5, step=0.01),
                    ui.input_select("mon_nshow", "표시할 샘플 수",
                                    choices={"50": "50", "100": "100", "200": "200"},
                                    selected="100"),
                    col_widths=[8, 4]
                ),
            ),
            class_="container section"
        ),

        # KPI 한 줄
        ui.div(
            ui.card(
                ui.card_header("📌 실시간 성능 지표"),
                ui.output_ui("mon_kpi_bar")
            ),
            class_="container section"
        ),

        # 1행: 혼동행렬 + 샘플 테이블
        ui.div(
            ui.layout_columns(
                ui.card(
                    ui.card_header("📉 혼동 행렬"),
                    ui.output_ui("mon_confusion_plot")
                ),
                ui.card(
                    ui.card_header("🧪 샘플(최근)"),
                    ui.div(ui.output_table("mon_sample_table"), class_="scroll-table")
                ),
                col_widths=[6, 6]
            ),
            class_="container section"
        ),

        # 2행: ROC + PR
        ui.div(
            ui.layout_columns(
                ui.card(
                    ui.card_header("📊 ROC 곡선"),
                    ui.output_ui("mon_roc_plot")
                ),
                ui.card(
                    ui.card_header("📈 Precision–Recall 곡선"),
                    ui.output_ui("mon_pr_plot")
                ),
                col_widths=[6, 6]
            ),
            class_="container section"
        ),
    )


# ==================== SERVER ====================
def server_monitoring(input, output, session):

    # --- prediction_state 기반으로 실시간 데이터 반영 ---
    @reactive.Calc
    def view_df():
        df = prediction_state()
        if df.empty or not {"pred", "prob", "actual"}.issubset(df.columns):
            return pd.DataFrame()

        nshow = int(input.mon_nshow())
        thr = float(input.mon_thr())

        df = df.tail(nshow).copy()
        df["y_true"] = df["actual"]
        df["y_prob"] = df["prob"]
        df["y_pred(τ)"] = (df["y_prob"] >= thr).astype(int)
        df["sample_id"] = np.arange(1, len(df) + 1)
        return df[["sample_id", "y_true", "y_prob", "y_pred(τ)"]]

    # --- 성능 지표 계산 ---
    @reactive.Calc
    def metrics():
        df = view_df()
        if df.empty:
            return {"acc": 0, "precision": 0, "recall": 0, "f1": 0, "tp": 0, "fp": 0, "fn": 0}
        y_t = df["y_true"].to_numpy()
        y_p = df["y_pred(τ)"].to_numpy()
        acc = float((y_t == y_p).mean())
        tp = int(np.sum((y_t == 1) & (y_p == 1)))
        fp = int(np.sum((y_t == 0) & (y_p == 1)))
        fn = int(np.sum((y_t == 1) & (y_p == 0)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (2 * precision * recall) / max(precision + recall, 1e-9)
        return {"acc": acc, "precision": precision, "recall": recall, "f1": f1,
                "tp": tp, "fp": fp, "fn": fn}

    # --- KPI 바 ---
    @output(id="mon_kpi_bar")
    @render.ui
    def _mon_kpi_bar():
        m = metrics()
        def kcard(title, value):
            return ui.div(
                ui.div(
                    ui.div(title, class_="title"),
                    ui.div(value, class_="value"),
                    class_="p-3"
                ),
                class_="kcard"
            )
        return ui.div(
            kcard("정확도", f"{m['acc']:.3f}"),
            kcard("정밀도", f"{m['precision']:.3f}"),
            kcard("재현율", f"{m['recall']:.3f}"),
            kcard("F1-score", f"{m['f1']:.3f}"),
            class_="kpi-row"
        )

    # --- 혼동 행렬 ---
    @output(id="mon_confusion_plot")
    @render.ui
    def _mon_confusion_plot():
        df = view_df()
        if df.empty:
            return ui.p("데이터 없음", class_="text-muted")

        y_t = df["y_true"].to_numpy()
        y_p = df["y_pred(τ)"].to_numpy()

        # ✅ 모든 클래스(0,1)가 포함되도록 보정
        labels = [0, 1]
        cm = confusion_matrix(y_t, y_p, labels=labels)
        if cm.shape != (2, 2):  # 불량이 전혀 없는 경우
            full_cm = np.zeros((2, 2), dtype=int)
            for i, lab_i in enumerate(np.unique(y_t)):
                for j, lab_j in enumerate(np.unique(y_p)):
                    full_cm[lab_i, lab_j] = cm[i, j]
            cm = full_cm

        text = np.array([[f"TN<br>{cm[0,0]}", f"FP<br>{cm[0,1]}"],
                        [f"FN<br>{cm[1,0]}", f"TP<br>{cm[1,1]}"]])

        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=["예측:정상(0)", "예측:불량(1)"],
            y=["실제:정상(0)", "실제:불량(1)"],
            colorscale="Blues",
            text=text,
            texttemplate="%{text}",
            hoverongaps=False
        ))
        fig.update_layout(template="plotly_white", height=360,
                        margin=dict(l=50, r=20, t=10, b=40))
        return ui.HTML(fig.to_html(include_plotlyjs='cdn', div_id="mon_conf_mat"))

        # --- 샘플 테이블 (최근) ---
    @output(id="mon_sample_table")
    @render.table
    def _mon_sample_table():
        df = view_df().copy()
        if df.empty:
            return pd.DataFrame({"상태": ["예측 결과 없음"]})
        
        # 최근 데이터가 위로 오게 정렬
        df = df.sort_values("sample_id", ascending=False).reset_index(drop=True)

        # 예측 결과 비교 플래그 컬럼 추가
        df["flag"] = np.where(
            (df["y_true"] == 1) & (df["y_pred(τ)"] == 0), "❗ FN",
            np.where(
                (df["y_true"] == 0) & (df["y_pred(τ)"] == 1), "⚠️ FP",
                np.where(
                    (df["y_true"] == 1) & (df["y_pred(τ)"] == 1), "✅ TP", "✅ TN"
                )
            )
        )

        # 보기 좋은 컬럼 순서로 정리
        df = df[["sample_id", "y_true", "y_prob", "y_pred(τ)", "flag"]]
        df.rename(columns={
            "sample_id": "샘플ID",
            "y_true": "실제",
            "y_prob": "불량확률",
            "y_pred(τ)": "예측(τ)",
            "flag": "판정"
        }, inplace=True)

        return df.head(20)  # 최근 50개까지만 표시
    
    # --- ROC 곡선 ---
    @output(id="mon_roc_plot")
    @render.ui
    def _mon_roc_plot():
        df = view_df()
        if df.empty:
            return ui.p("데이터 없음", class_="text-muted")
        y_t = df["y_true"].to_numpy()
        y_prob = df["y_prob"].to_numpy()

        # ✅ 불량이 없을 경우 기본 직선 표시
        if len(np.unique(y_t)) < 2:
            fig = go.Figure()
            fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
            fig.update_layout(template="plotly_white", height=360,
                            margin=dict(l=50, r=20, t=10, b=40),
                            xaxis_title="False Positive Rate",
                            yaxis_title="True Positive Rate",
                            title="불량 데이터 없음 (기본 ROC 표시)")
            return ui.HTML(fig.to_html(include_plotlyjs='cdn', div_id="mon_roc_curve"))

        fpr, tpr, _ = roc_curve(y_t, y_prob)
        auc_score = auc(fpr, tpr)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                name=f"AUC={auc_score:.3f}"))
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                    line=dict(dash="dash"))
        fig.update_layout(template="plotly_white", height=360,
                        margin=dict(l=50, r=20, t=10, b=40),
                        xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate")
        return ui.HTML(fig.to_html(include_plotlyjs='cdn', div_id="mon_roc_curve"))


    # --- Precision–Recall 곡선 ---
    @output(id="mon_pr_plot")
    @render.ui
    def _mon_pr_plot():
        df = view_df()
        if df.empty:
            return ui.p("데이터 없음", class_="text-muted")
        y_t = df["y_true"].to_numpy()
        y_prob = df["y_prob"].to_numpy()

        # ✅ 불량이 없을 경우 기본 1.0 라인 표시
        if len(np.unique(y_t)) < 2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[1, 1], mode='lines', name='기본 PR'))
            fig.update_layout(template="plotly_white", height=360,
                            margin=dict(l=50, r=20, t=10, b=40),
                            xaxis_title="Recall", yaxis_title="Precision",
                            title="불량 데이터 없음 (기본 PR 표시)")
            return ui.HTML(fig.to_html(include_plotlyjs='cdn', div_id="mon_pr_curve"))

        precision, recall, _ = precision_recall_curve(y_t, y_prob)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines',
                                name='PR curve'))
        fig.update_layout(template="plotly_white", height=360,
                        margin=dict(l=50, r=20, t=10, b=40),
                        xaxis_title="Recall", yaxis_title="Precision")
        return ui.HTML(fig.to_html(include_plotlyjs='cdn', div_id="mon_pr_curve"))
