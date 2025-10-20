import numpy as np
import pandas as pd
from datetime import timedelta
from io import StringIO

def snapshot_filter(df_all: pd.DataFrame, mold: str, end_date: pd.Timestamp, window_days: int = 20):
    if df_all.empty or mold is None or pd.isna(end_date):
        return pd.DataFrame(columns=df_all.columns)

    end = pd.to_datetime(end_date).normalize()
    start = end - timedelta(days=window_days)
    ms = (df_all["mold_code"] == mold) & (
        df_all["date"].isna() | ((df_all["date"] >= start) & (df_all["date"] <= end))
    )
    return df_all.loc[ms].copy()

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
        pi = float(dt.loc[dt["date"] == day, "p"].values[0])
        oc_state = "UCL 초과" if pi > UCL else ("LCL 미만" if pi < LCL else "정상")
        anom_state = bool(z_anom.loc[dt["date"] == day].values[0])
        anom_score = float(abs(z.loc[dt["date"] == day].values[0]))

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

def report_csv_bytes(df_log: pd.DataFrame) -> bytes:
    buf = StringIO()
    df_log.to_csv(buf, index=False, encoding="utf-8-sig")
    return buf.getvalue().encode("utf-8-sig")
