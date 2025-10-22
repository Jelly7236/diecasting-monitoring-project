# ============================================================
# viz/multivar_chart.py
# ============================================================
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from shiny import ui
from scipy.stats import f
from shared import multivar_info, cov_matrices  # ✅ 공분산 행렬도 함께 import

# ============================================================
# (NEW) 다변량 관리도 파라미터 테이블 정의
# ============================================================
MULTIVAR_PARAMS = pd.DataFrame([
    (8412, "1.용탕준비·가열", 2, 16092, 0.048),
    (8412, "3.사출·금형충전", 5, 16092, 0.048),
    (8412, "4.응고", 5, 16092, 0.048),

    (8413, "1.용탕준비·가열", 2, 60, 0.048),
    (8413, "3.사출·금형충전", 5, 60, 0.048),
    (8413, "4.응고", 5, 60, 0.048),

    (8576, "1.용탕준비·가열", 2, 56, 0.060),
    (8576, "3.사출·금형충전", 5, 56, 0.060),
    (8576, "4.응고", 5, 56, 0.060),

    (8722, "1.용탕준비·가열", 2, 18336, 0.132),
    (8722, "3.사출·금형충전", 5, 18336, 0.132),
    (8722, "4.응고", 5, 18336, 0.06),

    (8917, "1.용탕준비·가열", 2, 21575, 0.048),
    (8917, "3.사출·금형충전", 5, 21575, 0.048),
    (8917, "4.응고", 5, 21575, 0.004),
], columns=["mold_code", "chart_group", "p_vars", "n_rows", "alpha_for_F"])

PROC_NUM_TO_NAME = {
    1: "1.용탕준비·가열",
    3: "3.사출·금형충전",
    4: "4.응고",
}

def lookup_params(mold_code: int, process_label: str):
    """몰드코드·공정명에 맞는 p, n, alpha 반환"""
    try:
        proc_num = int(process_label.split(')')[0])
        chart_group_name = PROC_NUM_TO_NAME.get(proc_num, process_label.split(')')[1].strip())
    except Exception:
        chart_group_name = process_label.strip()

    row = MULTIVAR_PARAMS[
        (MULTIVAR_PARAMS["mold_code"] == mold_code) &
        (MULTIVAR_PARAMS["chart_group"] == chart_group_name)
    ]
    if row.empty:
        row = MULTIVAR_PARAMS[
            (MULTIVAR_PARAMS["mold_code"] == mold_code) &
            (MULTIVAR_PARAMS["chart_group"].str.contains(chart_group_name, na=False))
        ]

    if row.empty:
        return None

    r = row.iloc[0]
    return {
        "p": int(r["p_vars"]),
        "n": int(r["n_rows"]),
        "alpha": float(r["alpha_for_F"]),
        "chart_group": r["chart_group"],
    }

# ============================================================
# 다변량 관리도 (Hotelling T²) - Plot
# ============================================================
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

    # ✅ 관측 데이터 준비
    X = df[vars_used].dropna().to_numpy()
    if X.shape[0] < len(vars_used) + 5:
        return ui.p("⚠️ 표본 수가 부족합니다.",
                    style="text-align:center;color:#777;padding:2rem;")

    mu = np.array([mean_dict[v] for v in vars_used])

    # ✅ 기준 공분산 행렬 적용
    mold_int = int(mold)
    proc_num = int(process.split(')')[0])
    if mold_int in cov_matrices and proc_num in cov_matrices[mold_int]:
        S = cov_matrices[mold_int][proc_num]
        print(f"📦 기준 공분산 행렬 사용 (mold={mold_int}, process={proc_num})")
    else:
        S = np.cov(X, rowvar=False)
        print(f"⚠️ 기준 공분산 행렬 없음 → 데이터 기반 공분산 사용")

    # ✅ T² 통계량 계산
    invS = np.linalg.pinv(S)
    T2 = np.array([(x - mu).T @ invS @ (x - mu) for x in X])

    # ============================================================
    # ✅ UCL/CL 계산 (파라미터 테이블 기반)
    # ============================================================
    params = lookup_params(mold, process)
    if params is None:
        return ui.p("⚠️ 파라미터 테이블에 해당 몰드/공정 정보가 없습니다.",
                    style="text-align:center;color:#777;padding:2rem;")

    p = params["p"]
    n = params["n"]
    alpha = params["alpha"]

    if n <= p + 1:
        return ui.p(f"⚠️ n(={n})이 p(={p})에 비해 너무 작습니다. UCL 계산 불가.",
                    style="text-align:center;color:#777;padding:2rem;")

    Fcrit = f.ppf(1 - alpha, p, n - p)
    UCL = Fcrit * (p * (n - 1)) / (n - p)
    CL = p

    # ============================================================
    # ✅ 색상 분류
    # ============================================================
    colors = []
    for val in T2:
        if val > UCL:
            colors.append("#ef4444")  # 경고
        elif val > CL:
            colors.append("#f59e0b")  # 주의
        else:
            colors.append("#3b82f6")  # 정상

    # ============================================================
    # ✅ Plotly 시각화
    # ============================================================
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(1, len(T2) + 1),
        y=T2,
        mode="lines+markers",
        line=dict(color="#3b82f6", width=2),
        marker=dict(color=colors, size=6)
    ))

    fig.add_hline(
        y=UCL, line_dash="dash", line_color="#ef4444",
        annotation_text=f"UCL ({UCL:.2f})", annotation_position="right"
    )
    fig.add_hline(
        y=CL, line_dash="dot", line_color="#f59e0b",
        annotation_text=f"CL ({CL:.2f})", annotation_position="right"
    )

    fig.update_layout(
        title=f"{process} 다변량 관리도 (몰드 {mold}, α={alpha:.3f})",
        xaxis_title="샘플 번호",
        yaxis_title="T² 값",
        template="plotly_white",
        height=380,
        hovermode="x unified"
    )

    return ui.HTML(fig.to_html(include_plotlyjs="cdn", div_id="t2_chart"))

# ============================================================
# 다변량 관리도 - 표 요약
# ============================================================
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

    mold_int = int(mold)
    proc_num = int(process.split(')')[0])
    if mold_int in cov_matrices and proc_num in cov_matrices[mold_int]:
        S = cov_matrices[mold_int][proc_num]
    else:
        S = np.cov(X, rowvar=False)

    invS = np.linalg.pinv(S)
    T2 = np.array([(x - mu).T @ invS @ (x - mu) for x in X])

    # ✅ 동일한 파라미터 테이블 기반 UCL 계산
    params = lookup_params(mold, process)
    if params is None:
        return pd.DataFrame({"상태": ["⚠️ 파라미터 테이블에 (몰드/공정) 정보 없음"]})

    p = params["p"]
    n = params["n"]
    alpha = params["alpha"]

    if n <= p + 1:
        return pd.DataFrame({"상태": [f"⚠️ n(={n})이 p(={p})에 비해 너무 작아 UCL 계산 불가"]})

    Fcrit = f.ppf(1 - alpha, p, n - p)
    UCL = Fcrit * (p * (n - 1)) / (n - p)

    viol = np.where(T2 > UCL)[0]
    if len(viol) == 0:
        return pd.DataFrame({"상태": ["✅ 관리 상태 양호"]})

    return pd.DataFrame({
        "샘플": viol + 1,
        "T²": np.round(T2[viol], 3),
        "UCL": np.round(UCL, 3),
        "유형": ["T² 초과"] * len(viol)
    })
