import pandas as pd
import numpy as np

# ------------------------ 데이터 로더 ------------------------
def load_quality_from_file() -> pd.DataFrame:
    candidates = [
        "./data/train2.csv", "/mnt/data/train2.csv",
        "/mnt/data/test2.xlsx", "/mnt/data/test2.csv",
    ]
    df = None
    for p in candidates:
        try:
            if p.endswith(".xlsx"):
                tmp = pd.read_excel(p)
            elif p.endswith(".csv"):
                tmp = pd.read_csv(p)
            else:
                continue
            if not tmp.empty:
                df = tmp
                break
        except Exception:
            continue

    if df is None:
        return pd.DataFrame(columns=["date", "mold_code", "n", "d"])

    rename_map = {
        "Date": "date", "DATE": "date",
        "MOLD": "mold_code", "Mold": "mold_code", "mold": "mold_code",
        "N": "n", "D": "d", "defect": "d", "Defect": "d",
        "PassOrFail": "passorfail", "PASSORFAIL": "passorfail",
        "probability": "prob", "Probability": "prob",
    }
    df = df.rename(columns=rename_map)

    # date 파싱(없어도 동작)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.NaT

    # passorfail -> n/d 생성
    if "passorfail" in df.columns:
        df["passorfail"] = pd.to_numeric(df["passorfail"], errors="coerce").fillna(0).astype(int)
        df["n"] = 1
        df["d"] = df["passorfail"].clip(0, 1).astype(int)

    # 숫자형 보정
    for c in ["n", "d"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        else:
            df[c] = 0

    if "prob" in df.columns:
        df["prob"] = pd.to_numeric(df["prob"], errors="coerce")

    df["mold_code"] = df.get("mold_code", "UNKNOWN").astype(str)
    df["p"] = np.where(df["n"] > 0, df["d"] / df["n"], 0.0)
    return df
