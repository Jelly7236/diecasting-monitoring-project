import numpy as np
import pandas as pd
import plotly.graph_objs as go
from shiny import ui
from utils.control_utils import calculate_hotelling_t2, phaseII_ucl_t2

def render_multivar_plot(input, df_view, df_baseline, PROCESS_GROUPS):
    process = input.process_select()
    var_list = PROCESS_GROUPS[process]
    df = df_view()
    base = df_baseline()

    X = df[var_list].dropna().to_numpy()
    p = len(var_list)
    if X.shape[0] < max(30, p + 5):
        return ui.p("표본이 부족합니다.", style="color:#6b7280;text-align:center;padding:2rem;")

    base_df = base[var_list].dropna() if base is not None else df[var_list].dropna()
    mu = base_df.mean().to_numpy()
    cov = np.cov(base_df.T)
    try:
        inv_cov = np.linalg.inv(cov)
    except:
        inv_cov = np.linalg.pinv(cov)

    t2 = calculate_hotelling_t2(X, mu, inv_cov)
    ucl = phaseII_ucl_t2(X.shape[0], p)
    viol_idx = np.where(t2 > ucl)[0]

    colors = ["red" if i in viol_idx else "#3b82f6" for i in range(len(t2))]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(t2) + 1)),
            y=t2,
            mode="lines+markers",
            line=dict(color="#3b82f6", width=2),
            marker=dict(color=colors, size=5),
        )
    )
    fig.add_hline(y=ucl, line_dash="dash", line_color="#ef4444",
                  annotation_text=f"UCL ({ucl:.2f})", annotation_position="right")
    fig.update_layout(template="plotly_white", height=350,
                      hovermode="x unified", title=f"Hotelling T² (n={X.shape[0]})")

    return ui.HTML(fig.to_html(include_plotlyjs="cdn", div_id="t2_chart"))

def render_multivar_table(input, df_view, df_baseline, PROCESS_GROUPS):
    process = input.process_select()
    var_list = PROCESS_GROUPS[process]
    df = df_view()
    base = df_baseline()

    X = df[var_list].dropna().to_numpy()
    p = len(var_list)
    if X.shape[0] < max(30, p + 5):
        return pd.DataFrame({"상태": ["표본 부족"]})

    base_df = base[var_list].dropna() if base is not None else df[var_list].dropna()
    mu = base_df.mean().to_numpy()
    cov = np.cov(base_df.T)
    try:
        inv_cov = np.linalg.inv(cov)
    except:
        inv_cov = np.linalg.pinv(cov)

    t2 = calculate_hotelling_t2(X, mu, inv_cov)
    ucl = phaseII_ucl_t2(X.shape[0], p)
    viol = np.where(t2 > ucl)[0]
    if len(viol) == 0:
        return pd.DataFrame({"상태": ["✅ 관리 상태 양호"]})
    return pd.DataFrame({
        "샘플": viol + 1,
        "T²": t2[viol].round(3),
        "UCL": np.round(ucl, 3),
        "유형": ["T² 초과"] * len(viol)
    })
