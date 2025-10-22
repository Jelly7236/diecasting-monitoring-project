# -*- coding: utf-8 -*-
"""
금형코드별 × 변수별 ARIMA(p,d,q) 적합 + 모델 저장 + (잔차) Ljung-Box & Shapiro-Wilk 검정 + XLSX 저장
- 시간 대신 생산순번(count)을 시계열 인덱스로 사용
- 행 순서가 곧 시계열 순서
- 저장 파일명: a{금형번호}_{변수}.pkl  (예: d8412_4.xlsx -> a8412_molten_temp.pkl)

필요 패키지:
  pip install pandas numpy statsmodels scipy openpyxl
"""

import os
import re
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import shapiro

warnings.filterwarnings("ignore")

# =========================================================
# 0) 설정
# =========================================================
DATA_FILES: List[str] = [
    "d8412_4.xlsx",
    "d8413_4.xlsx",
    "d8576_4.xlsx",
    "d8722_4.xlsx",
    "d8917_4.xlsx",
]

# count 컬럼을 시계열 인덱스로 사용
INDEX_COL: str = "count"

TARGET_VARS: List[str] = [
    "molten_volume",
    "molten_temp",
    "upper_mold_temp1",
    "upper_mold_temp2",
    "lower_mold_temp1",
    "lower_mold_temp2",
    "Coolant_temperature",
    "cast_pressure",
    "high_section_speed",
    "biscuit_thickness",
    "sleeve_temperature",
]

# === 최신 (p,d,q) 반영 ===
VAR_PDQS: Dict[str, Tuple[int, int, int]] = {
    "molten_volume":        (0, 1, 2),
    "molten_temp":          (1, 1, 3),
    "upper_mold_temp1":     (1, 1, 1),  # 변경
    "upper_mold_temp2":     (1, 1, 5),
    "lower_mold_temp1":     (1, 1, 5),
    "lower_mold_temp2":     (1, 1, 2),  # 변경
    "Coolant_temperature":  (1, 1, 2),
    "cast_pressure":        (1, 1, 1),  # 변경
    "high_section_speed":   (1, 1, 1),
    "biscuit_thickness":    (1, 0, 1),
    "sleeve_temperature":   (1, 0, 1),
}

MODEL_DIR: str = "./saved_models"
RESULT_XLSX: str = "./residual_tests_ARIMA.xlsx"

LB_MAX_LAG: int = 24
ALPHA: float = 0.05
MIN_SAMPLES_FOR_TESTS: int = 20

# =========================================================
# 1) 유틸
# =========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))

def extract_mold_digits(mold_code_base: str) -> str:
    """
    파일 베이스명에서 연속된 숫자 블록을 추출 (가장 첫 번째 숫자 블록)
    예) 'd8412_4' -> '8412', 'M12345' -> '12345'
    """
    m = re.search(r"(\d+)", mold_code_base)
    return m.group(1) if m else ""

def filename_for_model(mold_code_base: str, var: str) -> str:
    """
    요구형식: a{숫자}_{변수}.pkl
    숫자가 없으면 a{원본베이스}_{변수}.pkl 로 대체
    """
    digits = extract_mold_digits(mold_code_base)
    if digits:
        return f"a{digits}_{safe_name(var)}.pkl"
    else:
        return f"a{safe_name(mold_code_base)}_{safe_name(var)}.pkl"

def fit_arima_endog(y: pd.Series, order: Tuple[int,int,int]):
    y = pd.to_numeric(y, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if y.size < max(10, MIN_SAMPLES_FOR_TESTS):
        raise ValueError(f"Series too short for ARIMA: n={y.size}")
    model = sm.tsa.ARIMA(y, order=order)
    res = model.fit()
    return res

def ljung_box_summary(resid: pd.Series, max_lag: int) -> dict:
    resid = pd.Series(resid).dropna()
    if resid.size < MIN_SAMPLES_FOR_TESTS:
        return {"lb_pass": None, "lb_lags": None, "lb_pvalues": None}
    use_lag = int(min(max_lag, max(1, resid.size // 5)))
    lb = acorr_ljungbox(resid, lags=use_lag, return_df=True)
    pvals = lb["lb_pvalue"].tolist()
    lb_pass = bool((lb["lb_pvalue"] > ALPHA).all())
    return {"lb_pass": lb_pass, "lb_lags": use_lag, "lb_pvalues": pvals}

def shapiro_summary(resid: pd.Series) -> dict:
    resid = pd.Series(resid).dropna()
    n = resid.size
    if n < 3:
        return {"sw_pass": None, "sw_stat": None, "sw_pvalue": None, "sw_n": n}
    if n > 5000:
        resid = resid.sample(5000, random_state=42)
    stat, p = shapiro(resid.values)
    return {"sw_pass": bool(p > ALPHA), "sw_stat": float(stat), "sw_pvalue": float(p), "sw_n": int(n)}

# =========================================================
# 2) 메인
# =========================================================
def run():
    ensure_dir(MODEL_DIR)

    summary_rows = []
    failed_rows = []

    for file_path in DATA_FILES:
        mold_code_base = os.path.splitext(os.path.basename(file_path))[0]  # 예: d8412_4

        try:
            df = pd.read_excel(file_path)
        except Exception as e:
            failed_rows.append({"mold_code": mold_code_base, "var": None, "stage": "read_excel", "error": str(e)})
            continue

        if INDEX_COL not in df.columns:
            failed_rows.append({"mold_code": mold_code_base, "var": None, "stage": "index_check", "error": f"'{INDEX_COL}' 컬럼 없음"})
            continue

        # count 기준 정렬 및 인덱스 설정 (행 순서 = 시계열 순서)
        df = df.sort_values(INDEX_COL).set_index(INDEX_COL)

        for var in TARGET_VARS:
            if var not in df.columns:
                failed_rows.append({"mold_code": mold_code_base, "var": var, "stage": "var_check", "error": "변수 없음"})
                continue

            order = VAR_PDQS.get(var)
            if order is None:
                failed_rows.append({"mold_code": mold_code_base, "var": var, "stage": "pdq_check", "error": "pdq 미지정"})
                continue

            try:
                y = df[var]
                res = fit_arima_endog(y, order)

                # 모델 저장 경로 및 파일명 (a{숫자}_{변수}.pkl)
                mold_dir = os.path.join(MODEL_DIR, safe_name(mold_code_base))
                ensure_dir(mold_dir)
                model_filename = filename_for_model(mold_code_base, var)
                model_path = os.path.join(mold_dir, model_filename)
                res.save(model_path)

                # 잔차검정
                resid = res.resid
                lb = ljung_box_summary(resid, LB_MAX_LAG)
                sw = shapiro_summary(resid)

                summary_rows.append({
                    "mold_code_base": mold_code_base,
                    "model_filename": model_filename,
                    "variable": var,
                    "order_p": order[0], "order_d": order[1], "order_q": order[2],
                    "n_obs": int(res.nobs),
                    "aic": float(res.aic) if res.aic is not None else np.nan,
                    "bic": float(res.bic) if res.bic is not None else np.nan,
                    "model_path": model_path,
                    # Ljung–Box
                    "lb_lags": lb["lb_lags"],
                    f"lb_pass_all_lags(alpha={ALPHA:.2f})": lb["lb_pass"],
                    "lb_pvalues": lb["lb_pvalues"],
                    # Shapiro–Wilk
                    "sw_n": sw["sw_n"],
                    "sw_stat": sw["sw_stat"],
                    "sw_pvalue": sw["sw_pvalue"],
                    f"sw_pass(alpha={ALPHA:.2f})": sw["sw_pass"],
                })

            except Exception as e:
                failed_rows.append({"mold_code": mold_code_base, "var": var, "stage": "fit_or_tests", "error": str(e)})

    # 결과 저장
    with pd.ExcelWriter(RESULT_XLSX, engine="openpyxl") as w:
        pd.DataFrame(summary_rows).sort_values(
            ["mold_code_base", "variable"], na_position="last"
        ).to_excel(w, sheet_name="residual_tests", index=False)
        if failed_rows:
            pd.DataFrame(failed_rows).to_excel(w, sheet_name="failed_cases", index=False)

    print(f"[완료] 모델 저장 폴더: {MODEL_DIR}")
    print(f"[완료] 잔차검정 결과 엑셀: {RESULT_XLSX}")

if __name__ == "__main__":
    run()
