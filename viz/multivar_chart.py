# viz/multivar_chart.py
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from shiny import ui
from scipy.stats import f
from shared import multivar_info  # ✅ 새로 추가된 기준정보

def render_multivar_plot(input, df_view, df_baseline, PROCESS_GROUPS):
    df = df_view()
    if df.empty:
        return ui.p("⚠️ 선택한 몰드코드에 데이터가 없습니다.",
                    style="text-align:center;color:#777;padding:2rem;")

    mold = int(input.mold())
    process = input.process_select()

    # ✅ 기준정보 조회
    info = multivar_info[
        (multivar_info["mold_code"] == mold)
        & (multivar_info["chart_group"].str.contains(process.split(')')[1].strip()))
    ]
    if info.empty:
        return ui.p("⚠️ 해당 몰드코드 기준정보 없음",
                    style="text-align:center;color:#777;padding:2rem;")

    row = info.iloc[0]
    mean_dict = row["mean_vector"]
    vars_used = [v.strip() for v in row["vars_used"].split(",")]

    X = df[vars_used].dropna().to_numpy()
    if X.shape[0] < len(vars_used) + 5:
        return ui.p("⚠️ 표본 수가 부족합니다.",
                    style="text-align:center;color:#777;padding:2rem;")

    # ✅ 기준정보 기반 평균·공분산·UCL 계산
    mu = np.array([mean_dict[v] for v in vars_used])
    S = np.cov(X, rowvar=False)
    invS = np.linalg.pinv(S)
    T2 = [(x - mu).T @ invS @ (x - mu) for x in X]

    n, p = int(row["n"]), int(row["p"])
    alpha = float(row["alpha_for_F"])
    Fcrit = f.ppf(1 - alpha, p, n - p)
    UCL = (p * (n - 1) * (n + 1) / (n * (n - p))) * Fcrit

    viol_idx = np.where(T2 > UCL)[0]
    colors = ["#ef4444" if i in viol_idx else "#3b82f6" for i in range(len(T2))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(1, len(T2) + 1),
        y=T2,
        mode="lines+markers",
        line=dict(color="#3b82f6", width=2),
        marker=dict(color=colors, size=5)
    ))
    fig.add_hline(y=UCL, line_dash="dash", line_color="#ef4444",
                  annotation_text=f"UCL ({UCL:.2f})", annotation_position="right")

    fig.update_layout(
        title=f"{process} 다변량 관리도 (몰드 {mold}, α={alpha:.3f})",
        xaxis_title="샘플 번호",
        yaxis_title="T² 값",
        template="plotly_white",
        height=380,
        hovermode="x unified"
    )

    return ui.HTML(fig.to_html(include_plotlyjs="cdn", div_id="t2_chart"))

def render_multivar_table(input, df_view, df_baseline, PROCESS_GROUPS):
    df = df_view()
    if df.empty:
        return pd.DataFrame({"상태": ["⚠️ 데이터 없음"]})

    mold = int(input.mold())
    process = input.process_select()

    info = multivar_info[
        (multivar_info["mold_code"] == mold)
        & (multivar_info["chart_group"].str.contains(process.split(')')[1].strip()))
    ]
    if info.empty:
        return pd.DataFrame({"상태": ["⚠️ 기준정보 없음"]})

    row = info.iloc[0]
    mean_dict = row["mean_vector"]
    vars_used = [v.strip() for v in row["vars_used"].split(",")]

    X = df[vars_used].dropna().to_numpy()
    if X.shape[0] < len(vars_used) + 5:
        return pd.DataFrame({"상태": ["⚠️ 표본 부족"]})

    mu = np.array([mean_dict[v] for v in vars_used])
    S = np.cov(X, rowvar=False)
    invS = np.linalg.pinv(S)
    T2 = np.array([(x - mu).T @ invS @ (x - mu) for x in X])

    n, p, alpha = int(row["n"]), int(row["p"]), float(row["alpha_for_F"])
    Fcrit = f.ppf(1 - alpha, p, n - p)
    UCL = (p * (n - 1) * (n + 1) / (n * (n - p))) * Fcrit

    viol = np.where(T2 > UCL)[0]
    if len(viol) == 0:
        return pd.DataFrame({"상태": ["✅ 관리 상태 양호"]})

    return pd.DataFrame({
        "샘플": viol + 1,
        "T²": T2[viol].round(3),
        "UCL": np.round(UCL, 3),
        "유형": ["T² 초과"] * len(viol)
    })
