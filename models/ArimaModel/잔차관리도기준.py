# -*- coding: utf-8 -*-
"""
저장된 ARIMA 모델의 '훈련 잔차'로 잔차 관리도 통계 산출:
- 각 모델의 resid(결측 제거) 기준으로
  CL = resid.mean()
  sigma = resid.std(ddof=1)
  UCL = CL + 3*sigma
  LCL = CL - 3*sigma
- 결과를 엑셀 파일로 저장
- 선택적으로 단일 (금형, 변수) 잔차 관리도 그리는 함수 제공

전제:
- 학습 시 저장한 모델이 ./saved_models/<mold_code_base>/a{digits}_{variable}.pkl 형태로 존재
- 예: saved_models/d8412_4/a8412_molten_temp.pkl
"""

import os
import re
import warnings
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

MODEL_DIR = "./saved_models"
OUT_XLSX  = "./residual_control_limits.xlsx"

# (참고) 변수 목록: 필요 시 필터링에 활용
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

# ---------- 유틸 ----------
def parse_var_from_filename(fname: str) -> Optional[str]:
    """
    파일명에서 변수명 추출: a{digits}_{var}.pkl
    예) a8412_molten_temp.pkl -> molten_temp
    """
    m = re.match(r"^a\d+_(.+)\.pkl$", fname)
    if m:
        return m.group(1)
    return None

def list_model_paths(model_root: str) -> List[Tuple[str, str, str]]:
    """
    saved_models 하위 모든 금형 폴더를 순회하며 (mold_code_base, variable, model_path) 목록 반환
    """
    rows = []
    if not os.path.isdir(model_root):
        return rows
    for mold in sorted(os.listdir(model_root)):
        mold_dir = os.path.join(model_root, mold)
        if not os.path.isdir(mold_dir):
            continue
        for fn in sorted(os.listdir(mold_dir)):
            if not fn.endswith(".pkl"):
                continue
            var = parse_var_from_filename(fn)
            if var is None:
                continue
            rows.append((mold, var, os.path.join(mold_dir, fn)))
    return rows

def compute_limits_from_resid(resid: pd.Series) -> Dict[str, float]:
    """
    resid(Series)에서 CL, sigma, UCL/LCL(3σ) 반환
    """
    r = pd.Series(resid).dropna()
    n = int(r.size)
    if n < 3:
        return {
            "n_resid": n, "CL": np.nan, "sigma": np.nan, "UCL_3sigma": np.nan, "LCL_3sigma": np.nan
        }
    CL = float(r.mean())
    sigma = float(r.std(ddof=1))
    return {
        "n_resid": n,
        "CL": CL,
        "sigma": sigma,
        "UCL_3sigma": CL + 3.0 * sigma,
        "LCL_3sigma": CL - 3.0 * sigma,
    }

# ---------- 메인 처리 ----------
def build_residual_limits_table():
    records = []
    failures = []

    triplets = list_model_paths(MODEL_DIR)
    if not triplets:
        print(f"[경고] '{MODEL_DIR}'에서 모델을 찾지 못했습니다.")
        return pd.DataFrame(), pd.DataFrame()

    for mold_code_base, variable, model_path in triplets:
        # (선택) 변수 필터링을 원하면 아래 주석 해제
        # if variable not in TARGET_VARS:
        #     continue
        try:
            res = sm.load(model_path)   # statsmodels ARIMAResults 로드
            resid = pd.Series(res.resid).dropna()
            lim = compute_limits_from_resid(resid)
            records.append({
                "mold_code_base": mold_code_base,
                "variable": variable,
                "model_path": model_path,
                "n_obs_model": int(getattr(res, "nobs", np.nan)),
                "aic": float(res.aic) if res.aic is not None else np.nan,
                "bic": float(res.bic) if res.bic is not None else np.nan,
                **lim
            })
        except Exception as e:
            failures.append({
                "mold_code_base": mold_code_base,
                "variable": variable,
                "model_path": model_path,
                "error": str(e)
            })

    df_ok = pd.DataFrame(records).sort_values(["mold_code_base", "variable"])
    df_fail = pd.DataFrame(failures)
    return df_ok, df_fail

def save_limits_to_excel(df_ok: pd.DataFrame, df_fail: pd.DataFrame, out_path: str = OUT_XLSX):
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        if not df_ok.empty:
            df_ok.to_excel(w, sheet_name="control_limits", index=False)
        if not df_fail.empty:
            df_fail.to_excel(w, sheet_name="failed_models", index=False)
    print(f"[완료] 잔차 관리도 한계치 엑셀 저장: {out_path}")

# ---------- 시각화(옵션) ----------
def plot_residual_control_chart(mold_code_base: str, variable: str, save_path: Optional[str] = None):
    """
    특정 (금형, 변수) 모델의 훈련 잔차 관리도를 그려 파일로 저장(또는 화면 표시)
    """
    mold_dir = os.path.join(MODEL_DIR, mold_code_base)
    if not os.path.isdir(mold_dir):
        raise FileNotFoundError(f"폴더 없음: {mold_dir}")

    # 해당 변수의 모델 파일 찾기
    target = None
    for fn in os.listdir(mold_dir):
        if fn.endswith(".pkl") and parse_var_from_filename(fn) == variable:
            target = os.path.join(mold_dir, fn)
            break
    if target is None:
        raise FileNotFoundError(f"{mold_code_base} 에서 변수 '{variable}' 모델 파일을 찾지 못함.")

    res = sm.load(target)
    resid = pd.Series(res.resid).dropna()
    lim = compute_limits_from_resid(resid)

    # 플롯
    plt.figure(figsize=(10, 4))
    plt.plot(resid.index, resid.values, marker="o", linewidth=1)
    plt.axhline(lim["CL"], linestyle="--", label=f"CL={lim['CL']:.4f}")
    plt.axhline(lim["UCL_3sigma"], linestyle=":", label=f"UCL(3σ)={lim['UCL_3sigma']:.4f}")
    plt.axhline(lim["LCL_3sigma"], linestyle=":", label=f"LCL(3σ)={lim['LCL_3sigma']:.4f}")
    plt.title(f"Residual Control Chart | {mold_code_base} - {variable}")
    plt.xlabel("index (after dropna)")
    plt.ylabel("residual")
    plt.legend(loc="best")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[저장] {save_path}")
    else:
        plt.show()

# ---------- 실행 ----------
if __name__ == "__main__":
    df_limits, df_fail = build_residual_limits_table()
    if not df_limits.empty or not df_fail.empty:
        save_limits_to_excel(df_limits, df_fail, OUT_XLSX)

    # (예시) 하나 그려보기:
    # plot_residual_control_chart("d8412_4", "molten_temp", save_path="d8412_4_molten_temp_resid_chart.png")
