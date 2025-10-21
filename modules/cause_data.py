# cause_data.py
import pandas as pd
import numpy as np
import os

# ------------------------ 메인 품질 데이터 로더 ------------------------
def load_quality_from_file() -> pd.DataFrame:
    """
    Ctest.csv를 우선 읽고, 없을 경우 기존 후보(train2/test2)를 시도.
    표준 컬럼: date, mold_code, n, d, prob, p, tryshot_signal, passorfail
    (개별 샷: n=1, d=passorfail(1=불량))
    """
    candidates = [
        "./data/Ctest.csv", "/mnt/data/Ctest.csv",
        "./Ctest.csv", "Ctest.csv",
        "./data/train2.csv", "/mnt/data/train2.csv",
        "/mnt/data/test2.xlsx", "/mnt/data/test2.csv",
    ]
    df = None; used = None
    for p in candidates:
        try:
            if not os.path.exists(p):
                continue
            if p.lower().endswith(".xlsx"):
                tmp = pd.read_excel(p)
            elif p.lower().endswith(".csv"):
                tmp = pd.read_csv(p)
            else:
                continue
            if tmp is not None and not tmp.empty:
                df = tmp; used = p; break
        except Exception as e:
            print(f"[LOAD] failed {p}: {e}")
            continue

    if df is None:
        print("[LOAD] no file found; returning empty frame")
        return pd.DataFrame(columns=["date","mold_code","n","d","prob","p","tryshot_signal","passorfail"])

    print(f"[LOAD] using: {used} shape={df.shape}")

    rename_map = {
        "Date":"date","DATE":"date","date":"date",
        "MOLD":"mold_code","Mold":"mold_code","mold":"mold_code","mold_code":"mold_code",
        "N":"n","D":"d","defect":"d","Defect":"d",
        "PassOrFail":"passorfail","PASSORFAIL":"passorfail","passorfail":"passorfail",
        "probability":"prob","Probability":"prob","prob":"prob",
        "time":"time","Time":"time",
        "tryshot_signal":"tryshot_signal","trysignal_shot":"tryshot_signal",
    }
    df = df.rename(columns=rename_map)

    # date/time 결합
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.NaT

    if "time" in df.columns:
        tm = pd.to_datetime(df["time"], errors="coerce").dt.time
        df["date"] = np.where(
            df["date"].notna() & pd.Series(tm).notna(),
            pd.to_datetime(df["date"].dt.date.astype(str) + " " + pd.Series(tm).astype(str)),
            df["date"]
        )

    # passorfail → n/d
    if "passorfail" in df.columns:
        s = pd.to_numeric(df["passorfail"], errors="coerce").fillna(0).astype(int)
        df["n"] = 1
        df["d"] = s.clip(0, 1)
    else:
        df["n"] = pd.to_numeric(df.get("n", 0), errors="coerce").fillna(0).astype(int)
        df["d"] = pd.to_numeric(df.get("d", 0), errors="coerce").fillna(0).astype(int)

    # 안전 보정
    df["mold_code"] = df.get("mold_code", "UNKNOWN").astype(str).str.strip()
    df["prob"] = pd.to_numeric(df.get("prob", np.nan), errors="coerce")
    df["tryshot_signal"] = df.get("tryshot_signal", np.nan)
    df["passorfail"] = pd.to_numeric(df.get("passorfail", np.nan), errors="coerce")

    # 파생 p(관측 불량률)
    df["p"] = np.where(df["n"] > 0, df["d"] / df["n"], 0.0)

    if df["date"].notna().any():
        df = df.sort_values("date").reset_index(drop=True)

    molds = sorted(df["mold_code"].dropna().astype(str).unique().tolist())
    print(f"[LOAD] molds detected: {molds} (n={len(molds)})")

    # ✅ 롤링 p̂ 계산용 컬럼 포함해서 반환
    return df[["date","mold_code","n","d","prob","p","tryshot_signal","passorfail"]]


# ------------------------ "불량 샘플 로그 전용" CSV 로더 ------------------------
def load_fault_samples() -> pd.DataFrame:
    """
    사용자 제공 CSV만 사용하여 '실제 불량 샘플 로그'를 표시한다.
    우선순위: /mnt/data/fault_analysis_dataframe_filtered.csv
    컬럼 자동 매핑:
      - date:  ['date','Date','DATE','datetime','timestamp','time']
      - mold_code: ['mold_code','MOLD','Mold','mold']
      - passorfail: ['passorfail','PassOrFail','PASSORFAIL','defect','Defect','DEFECT']
      - defect_name: ['defect_name','DefectName','defectType','defect_type','불량명','결함명']
      - defect_code: ['defect_code','DefectCode','불량코드','결함코드']
      - reason/note/prob 등은 있으면 사용
    """
    candidates = [
        "/mnt/data/fault_analysis_dataframe_filtered.csv",
        "./data/fault_analysis_dataframe_filtered.csv",
        "fault_analysis_dataframe_filtered.csv",
    ]

    df = None; used = None
    for p in candidates:
        try:
            if os.path.exists(p):
                tmp = pd.read_csv(p)
                if tmp is not None and not tmp.empty:
                    df = tmp; used = p; break
        except Exception as e:
            print(f"[FAULT-LOAD] failed {p}: {e}")
            continue

    if df is None:
        print("[FAULT-LOAD] no CSV found; returning empty frame")
        return pd.DataFrame(columns=["date","mold_code"])

    print(f"[FAULT-LOAD] using: {used} shape={df.shape}")

    # ---- 컬럼 표준화(느슨한 매칭) ----
    def _find(cols, cand):
        cl = {c.lower(): c for c in cols}
        for k in cand:
            if k.lower() in cl:
                return cl[k.lower()]
        return None

    cols = list(df.columns)

    # 표준 키 후보
    k_date = _find(cols, ["date","Date","DATE","datetime","timestamp","time"])
    k_mold = _find(cols, ["mold_code","MOLD","Mold","mold"])
    k_pfail= _find(cols, ["passorfail","PassOrFail","PASSORFAIL","defect","Defect","DEFECT"])
    k_dname= _find(cols, ["defect_name","DefectName","defectType","defect_type","불량명","결함명"])
    k_dcode= _find(cols, ["defect_code","DefectCode","불량코드","결함코드"])
    k_reason=_find(cols, ["reason","Reason","원인","사유"])
    k_note  =_find(cols, ["note","Note","비고","메모"])
    k_prob  =_find(cols, ["prob","Prob","Probability","probability","예측불량확률"])

    # 타입/정규화
    if k_date is not None:
        df["date"] = pd.to_datetime(df[k_date], errors="coerce")
    else:
        df["date"] = pd.NaT

    if k_mold is not None:
        df["mold_code"] = df[k_mold].astype(str).str.strip()
    else:
        df["mold_code"] = "UNKNOWN"

    if k_pfail is not None:
        df["passorfail"] = pd.to_numeric(df[k_pfail], errors="coerce")

    if k_dname is not None:
        df["defect_name"] = df[k_dname]
    if k_dcode is not None:
        df["defect_code"] = df[k_dcode]
    if k_reason is not None:
        df["reason"] = df[k_reason]
    if k_note is not None:
        df["note"] = df[k_note]
    if k_prob is not None:
        df["prob"] = pd.to_numeric(df[k_prob], errors="coerce")

    # 정렬
    if df["date"].notna().any():
        df = df.sort_values("date").reset_index(drop=True)

    # 표준 컬럼을 앞으로 정렬(있을 때만)
    prefer = ["date","mold_code","defect_code","defect_name","reason","note","passorfail","prob"]
    front = [c for c in prefer if c in df.columns]
    tail  = [c for c in df.columns if c not in front]
    df = df[front + tail]

    print(f"[FAULT-LOAD] columns(final): {list(df.columns)}")
    return df

# --- ADD: 변수명→한글명 매핑 로더 ---
def load_var_labels() -> dict:
    """
    data/var_labels.csv 를 읽어 변수명 -> 한글라벨 매핑 dict 반환.
    컬럼명은 유연하게 처리:
      - (var, ko)  또는
      - (variable/name/feat, label_ko/ko/korean)
    파일이 없거나 비어있으면 빈 dict 반환.
    """
    import os
    import pandas as pd

    candidates = [
        "/mnt/data/var_labels.csv",
        "./data/var_labels.csv",
        "data/var_labels.csv",
        "var_labels.csv",
    ]

    df = None
    for p in candidates:
        try:
            if os.path.exists(p):
                tmp = pd.read_csv(p)
                if tmp is not None and not tmp.empty:
                    df = tmp
                    break
        except Exception as e:
            print(f"[VAR-LABEL] read fail {p}: {e}")

    if df is None or df.empty:
        print("[VAR-LABEL] no mapping file, using empty map")
        return {}

    # 느슨한 컬럼 추출
    cols = {c.lower(): c for c in df.columns}
    def pick(keys):
        for k in keys:
            if k in cols: return cols[k]
        return None

    k_var = pick(["var","variable","name","feature","feat"])
    k_ko  = pick(["ko","label_ko","korean","label","라벨","한글"])

    if k_var is None or k_ko is None:
        # 컬럼 2개뿐이면 첫 번째를 var, 두 번째를 ko로 가정
        if df.shape[1] >= 2:
            k_var = df.columns[0]
            k_ko  = df.columns[1]
        else:
            print("[VAR-LABEL] cannot detect columns")
            return {}

    df = df[[k_var, k_ko]].dropna().astype(str)
    mapping = dict(zip(df[k_var].str.strip(), df[k_ko].str.strip()))
    print(f"[VAR-LABEL] loaded {len(mapping)} mappings")
    return mapping
