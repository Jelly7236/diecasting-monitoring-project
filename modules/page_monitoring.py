# modules/page_monitoring.py
from shiny import ui, render, reactive
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

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
            .kcard .sub{color:#6b7280;font-size:.8rem}
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
                    ui.input_select("mon_nshow", "표시할 샘플 수", choices={"50":"50","100":"100","200":"200"}, selected="100"),
                    col_widths=[8,4]
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
                col_widths=[6,6]
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
                col_widths=[6,6]
            ),
            class_="container section"
        ),
    )

# ==================== SERVER ====================
def server_monitoring(input, output, session):
    # --- 샘플 데이터 시뮬레이션 (데모용) ---
    np.random.seed(42)
    N = 200
    y_true_base = np.random.randint(0, 2, N)                 # 실제 레이블
    y_prob_base = np.clip(np.random.beta(2, 5, N), 0, 1)     # 예측 확률(불량일 확률 가정)
    y_prob_base[y_true_base == 1] = np.clip(y_prob_base[y_true_base == 1] + 0.2, 0, 1)
    idx = np.arange(N)

    # --- 리액티브: 컨트롤 반영 데이터뷰 ---
    @reactive.Calc
    def view_df():
        nshow = int(input.mon_nshow())
        thr = float(input.mon_thr())
        take = min(N, nshow)
        sel = idx[-take:]
        y_true = y_true_base[sel]
        y_prob = y_prob_base[sel]
        y_pred = (y_prob >= thr).astype(int)
        df = pd.DataFrame({
            "sample_id": sel + 1,
            "y_true": y_true,
            "y_prob": np.round(y_prob, 4),
            "y_pred(τ)": y_pred
        })
        return df

    # --- 리액티브: 성능지표 계산 ---
    @reactive.Calc
    def metrics():
        df = view_df()
        y_t = df["y_true"].to_numpy()
        y_p = df["y_pred(τ)"].to_numpy()
        acc = float((y_t == y_p).mean())
        tp = int(np.sum((y_t == 1) & (y_p == 1)))
        fp = int(np.sum((y_t == 0) & (y_p == 1)))
        fn = int(np.sum((y_t == 1) & (y_p == 0)))
        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1 = (2 * precision * recall) / max(precision + recall, 1e-9)
        return {"acc": acc, "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    # --- KPI 바 (가로 카드) ---
    @output(id="mon_kpi_bar")
    @render.ui
    def _mon_kpi_bar():
        m = metrics()
        def kcard(title, value, sub=""):
            return ui.div(
                ui.div(
                    ui.div(title, class_="title"),
                    ui.div(value, class_="value"),
                    ui.div(sub, class_="sub"),
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

    # --- 혼동 행렬 (Plotly) ---
    @output(id="mon_confusion_plot")
    @render.ui
    def _mon_confusion_plot():
        df = view_df()
        y_t = df["y_true"].to_numpy()
        y_p = df["y_pred(τ)"].to_numpy()
        cm = confusion_matrix(y_t, y_p, labels=[0,1])  # [[TN, FP],[FN, TP]]
        z = cm
        text = np.array([["TN","FP"],["FN","TP"]]) + "<br>" + z.astype(str)
        fig = go.Figure(data=go.Heatmap(
            z=z, x=["예측:정상(0)","예측:불량(1)"], y=["실제:정상(0)","실제:불량(1)"],
            colorscale="Blues", text=text, texttemplate="%{text}", hoverongaps=False
        ))
        fig.update_layout(
            template="plotly_white", height=360,
            margin=dict(l=50,r=20,t=10,b=40),
            xaxis_title="", yaxis_title=""
        )
        return ui.HTML(fig.to_html(include_plotlyjs='cdn', div_id="mon_conf_mat"))

    # --- 샘플 테이블 (최근) ---
    @output(id="mon_sample_table")
    @render.table
    def _mon_sample_table():
        df = view_df().copy()
        df = df.sort_values("sample_id", ascending=False)
        df["flag"] = np.where((df["y_true"]==1) & (df["y_pred(τ)"]==0), "❗ FN",
                       np.where((df["y_true"]==0) & (df["y_pred(τ)"]==1), "⚠️ FP",
                       np.where((df["y_true"]==1) & (df["y_pred(τ)"]==1), "✅ TP", "✅ TN")))
        df = df[["sample_id","y_true","y_prob","y_pred(τ)","flag"]]
        return df

    # --- ROC 곡선 ---
    @output(id="mon_roc_plot")
    @render.ui
    def _mon_roc_plot():
        df = view_df()
        y_t = df["y_true"].to_numpy()
        y_prob = df["y_prob"].to_numpy()
        fpr, tpr, _ = roc_curve(y_t, y_prob)
        auc_score = auc(fpr, tpr)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"AUC={auc_score:.3f}"))
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
        fig.update_layout(
            template="plotly_white", height=360,
            margin=dict(l=50,r=20,t=10,b=40),
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate"
        )
        return ui.HTML(fig.to_html(include_plotlyjs='cdn', div_id="mon_roc_curve"))

    # --- Precision–Recall 곡선 ---
    @output(id="mon_pr_plot")
    @render.ui
    def _mon_pr_plot():
        df = view_df()
        y_t = df["y_true"].to_numpy()
        y_prob = df["y_prob"].to_numpy()
        precision, recall, _ = precision_recall_curve(y_t, y_prob)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='PR curve'))
        fig.update_layout(
            template="plotly_white", height=360,
            margin=dict(l=50,r=20,t=10,b=40),
            xaxis_title="Recall", yaxis_title="Precision"
        )
        return ui.HTML(fig.to_html(include_plotlyjs='cdn', div_id="mon_pr_curve"))
