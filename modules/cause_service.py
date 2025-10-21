# cause_service.py
import numpy as np
import pandas as pd
from io import StringIO
from math import sqrt

# ---- 스냅샷 필터 ----
def snapshot_filter(
    df_all: pd.DataFrame,
    mold: str,
    end_date: pd.Timestamp,
    start_date: pd.Timestamp | None = None,
):
    """
    선택한 몰드에 대해 [start_date ~ end_date] 구간 필터.
    start_date가 None이면 해당 몰드의 '최초 일자'를 자동 사용.
    """
    if df_all.empty or mold is None or pd.isna(end_date):
        return pd.DataFrame(columns=df_all.columns)

    end = pd.to_datetime(end_date).normalize()
    sub = df_all.copy()
    sub = sub[sub["mold_code"] == mold] if mold is not None else sub

    if start_date is None:
        smin = sub["date"].dropna().min()
        if pd.isna(smin):
            smin = df_all["date"].dropna().min()
        start = pd.to_datetime(smin).normalize() if pd.notna(smin) else end
    else:
        start = pd.to_datetime(start_date).normalize()

    ms = sub["date"].isna() | ((sub["date"] >= start) & (sub["date"] <= end))
    return sub.loc[ms].copy()


# ---- 로그 테이블(일자 집계용) ----
def build_log_table(dff: pd.DataFrame) -> pd.DataFrame:
    if dff.empty:
        return pd.DataFrame({"메시지": ["데이터/기간 없음"]})

    dt = (
        dff.dropna(subset=["date"])
           .groupby("date", as_index=False)
           .agg(d=("d", "sum"), n=("n", "sum"))
    )
    dt["p"] = np.where(dt["n"] > 0, dt["d"] / dt["n"], 0.0)
    if dt.empty:
        return pd.DataFrame({"메시지": ["일자 정보 없음"]})

    pbar = float(dt["p"].mean()); nbar = float(dt["n"].mean() or 1)
    sigma = np.sqrt(max(pbar * (1 - pbar) / nbar, 1e-12))
    UCL = pbar + 3 * sigma; LCL = max(0.0, pbar - 3 * sigma)

    roll = dt["p"].rolling(10, min_periods=6)
    z = (dt["p"] - roll.mean()) / roll.std().replace(0, np.nan)
    z = z.fillna(0.0)
    z_anom = (z.abs() > 3).fillna(False)

    shap_cols = [c for c in dff.columns if c.lower().startswith("shap_")]
    rows = []
    seq = 1
    for day, grp in dff.groupby(dff["date"].dt.floor("D"), dropna=False):
        pi = float(dt.loc[dt["date"] == day, "p"].values[0]) if (dt["date"] == day).any() else 0.0
        oc_state = "UCL 초과" if pi > UCL else ("LCL 미만" if pi < LCL else "정상")
        anom_state = bool(z_anom.loc[dt["date"] == day].values[0]) if (dt["date"] == day).any() else False
        anom_score = float(abs(z.loc[dt["date"] == day].values[0])) if (dt["date"] == day).any() else 0.0

        for _, r in grp.iterrows():
            shap1 = shap2 = ""; top_var = ""
            if shap_cols:
                absvals = r[shap_cols].abs()
                order = absvals.sort_values(ascending=False).index.tolist()
                if len(order) >= 1:
                    shap1 = f"{order[0].replace('shap_', '')}: {r[order[0]]:.4f}"
                    top_var = order[0].replace("shap_", "")
                if len(order) >= 2:
                    shap2 = f"{order[1].replace('shap_', '')}: {r[order[1]]:.4f}"

            rows.append({
                "일시": (r["date"] if pd.notna(r["date"]) else day),
                "몰드": r.get("mold_code", ""),
                "순번": seq,
                "예측불량확률": (round(float(r["prob"]), 4) if "prob" in dff.columns and pd.notna(r["prob"]) else ""),
                "shap1": shap1,
                "shap2": shap2,
                "변수상태": "",
                "관리도 상태": oc_state,
                "이탈변수": top_var,
                "이상탐지": ("✅" if anom_state else ""),
                "Anomaly Score": round(anom_score, 3),
                "임계값 이탈변수": "",
                "이탈유형": ("관리도 이탈" if oc_state != "정상" else ("이상탐지" if anom_state else "")),
            })
            seq += 1

    cols = ["일시","몰드","순번","예측불량확률","shap1","shap2","변수상태",
            "관리도 상태","이탈변수","이상탐지","Anomaly Score","임계값 이탈변수","이탈유형"]
    log = pd.DataFrame(rows)
    return log[cols] if not log.empty else pd.DataFrame({"메시지": ["로그 없음"]})


# ---- 다운로드용 CSV 바이트 ----
def report_csv_bytes(df_log: pd.DataFrame) -> bytes:
    buf = StringIO()
    df_log.to_csv(buf, index=False, encoding="utf-8-sig")
    return buf.getvalue().encode("utf-8-sig")


# ===== 롤링 60샷 p-관리도 계산 (A-샷만) =====
def _agresti_coull(x: int, n: int) -> float:
    return (x + 2) / (n + 4) if n > 0 else float("nan")

def build_rolling_p_series(dff: pd.DataFrame, n_window: int = 60):
    """
    입력: 선택 기간·몰드로 필터된 df (date, tryshot_signal, passorfail 포함)
    출력: (series_df, limits_dict)
      - series_df: date, p_hat, risk_level
      - limits: {"pbar":..., "UCL2":..., "UCL3":..., "LCL3":..., "n_window":...}
    """
    if dff.empty or "date" not in dff.columns:
        return pd.DataFrame(columns=["date","p_hat","risk_level"]), {"pbar":np.nan,"UCL2":np.nan,"UCL3":np.nan,"LCL3":np.nan,"n_window":n_window}

    df = dff.dropna(subset=["date"]).copy()
    df = df.sort_values("date", kind="mergesort")

    # A-샷만
    if "tryshot_signal" in df.columns:
        isA = df["tryshot_signal"].astype(str).str.strip().str.upper().eq("A")
        dfA = df[isA].copy()
    else:
        dfA = df.copy()

    if dfA.empty:
        return pd.DataFrame(columns=["date","p_hat","risk_level"]), {"pbar":np.nan,"UCL2":np.nan,"UCL3":np.nan,"LCL3":np.nan,"n_window":n_window}

    # 불량여부 시퀀스
    if "passorfail" in dfA.columns:
        defect = pd.to_numeric(dfA["passorfail"], errors="coerce").fillna(0).astype(int).clip(0,1)
    else:
        defect = pd.to_numeric(dfA.get("d", 0), errors="coerce").fillna(0).astype(int).clip(0,1)

    # pbar (Agresti–Coull) — 선택 구간의 A-샷 기반
    x = int(defect.sum()); nA = int(defect.shape[0])
    pbar = _agresti_coull(x, nA) if nA > 0 else np.nan
    if not pd.notna(pbar):
        return pd.DataFrame(columns=["date","p_hat","risk_level"]), {"pbar":np.nan,"UCL2":np.nan,"UCL3":np.nan,"LCL3":np.nan,"n_window":n_window}

    sigma = sqrt(max(pbar * (1 - pbar) / n_window, 1e-16))
    UCL2 = pbar + 2*sigma
    UCL3 = pbar + 3*sigma
    LCL3 = max(0.0, pbar - 3*sigma)

    # 롤링 60샷 불량합 / 60
    roll_def = defect.rolling(window=n_window, min_periods=n_window).sum()
    p_hat = (roll_def / n_window).rename("p_hat")

    # 유효 구간만 (창이 찬 A-행들)
    valid = p_hat.dropna()
    series = pd.DataFrame({
        "date": dfA.loc[valid.index, "date"].values,
        "p_hat": valid.values
    })

    # 위험도 라벨
    series["risk_level"] = "NORMAL"
    series.loc[series["p_hat"] >= UCL3, "risk_level"] = "CRITICAL"
    series.loc[(series["p_hat"] < UCL3) & (series["p_hat"] >= UCL2), "risk_level"] = "CAUTION"

    limits = {"pbar":pbar, "UCL2":UCL2, "UCL3":UCL3, "LCL3":LCL3, "n_window":n_window}
    return series, limits
