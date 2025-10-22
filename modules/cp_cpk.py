from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd

# ========= 사용자 설정 =========
DATA_PATH = Path("./data2/train_final.xlsx")
SHEET_NAME = 0
TIME_COL   = "time"  # 있으면 within-σ 추정을 위해 정렬에 사용 (여기선 전체σ 기반 옵티마)

# 단위 변환: 원자료 × factor = 스펙 단위
UNIT_CONV: Dict[str, float] = {
    "cast_pressure": 0.1,                  # bar -> MPa (가정)
    "low_section_speed": 0.01,             # cm/s -> m/s (가정; mm/s라면 0.001)
    "high_section_speed": 0.01,            # cm/s -> m/s
    "facility_operation_cycleTime": 1.0,   # s (가정)
    "production_cycletime": 1.0,           # s (가정)
}

# Cp 허용 범위 (원하는 대로 조정 가능)
CP_MIN = 0.5
CP_MAX = 1.8

# (중요) 변수별 물리적 하한/상한 — 현장 지식으로 조정해도 됨
PHYSICAL_FLOOR: Dict[str, float] = {
    "cast_pressure": 0.0,                 # MPa
    "low_section_speed": 0.0,             # m/s
    "high_section_speed": 0.0,            # m/s
    "facility_operation_cycleTime": 0.0,  # s
    "production_cycletime": 0.0,          # s
    "biscuit_thickness": 0.0,             # mm
    "molten_temp": 500.0,                 # °C (용탕: 현실적 하한)
    "upper_mold_temp1": 80.0,             # °C
    "upper_mold_temp2": 80.0,             # °C
    "upper_mold_temp3": 80.0,             # °C
    "lower_mold_temp1": 80.0,             # °C
    "lower_mold_temp2": 80.0,             # °C
    "lower_mold_temp3": 80.0,             # °C
    "sleeve_temperature": 50.0,           # °C
    "Coolant_temperature": 0.0,           # °C (영하 비허용 가정)
}

PHYSICAL_CEILING: Dict[str, float] = {
    "cast_pressure": 200.0,               # MPa (장비 한계 가정)
    "low_section_speed": 5.0,             # m/s
    "high_section_speed": 10.0,           # m/s
    "facility_operation_cycleTime": 600.0,# s (10분)
    "production_cycletime": 600.0,        # s
    "biscuit_thickness": 200.0,           # mm
    "molten_temp": 900.0,                 # °C
    "upper_mold_temp1": 400.0,            # °C
    "upper_mold_temp2": 400.0,            # °C
    "upper_mold_temp3": 400.0,            # °C
    "lower_mold_temp1": 400.0,            # °C
    "lower_mold_temp2": 400.0,            # °C
    "lower_mold_temp3": 400.0,            # °C
    "sleeve_temperature": 800.0,          # °C
    "Coolant_temperature": 60.0,          # °C
}

# 분석 대상 변수(파일에 없으면 자동 스킵)
CANDIDATES: List[str] = [
    "cast_pressure","high_section_speed","low_section_speed",
    "facility_operation_cycleTime","production_cycletime",
    "molten_temp","Coolant_temperature","sleeve_temperature",
    "biscuit_thickness","upper_mold_temp1","upper_mold_temp2",
    "lower_mold_temp1","lower_mold_temp2","upper_mold_temp3","lower_mold_temp3",
]

# ========= 유틸 =========
def to_numeric(s: pd.Series) -> pd.Series:
    if s.dtype.kind in "biufc":
        return s.astype(float)
    st = s.astype(str).str.strip()
    num = st.str.extract(r'([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', expand=False)
    return pd.to_numeric(num.str.replace(",", "", regex=False), errors="coerce")

def apply_unit_conversions(df: pd.DataFrame, conv: Dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    for col, factor in conv.items():
        if col in out.columns and factor not in (None, 1, 1.0):
            out[col] = to_numeric(out[col]) * float(factor)
    return out

def estimate_sigma_within_mr(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size < 2: return np.nan
    mr = np.abs(np.diff(x))
    if mr.size == 0: return np.nan
    d2 = 1.128
    return float(np.mean(mr) / d2)

def cp_cpk_from(mu: float, sigma: float, L: float, U: float) -> Tuple[float, float]:
    if any(np.isnan(v) for v in [mu, sigma, L, U]) or sigma <= 0 or U <= L:
        return np.nan, np.nan
    Cp  = (U - L) / (6.0 * sigma)
    Cpu = (U - mu) / (3.0 * sigma)
    Cpl = (mu - L) / (3.0 * sigma)
    return float(Cp), float(min(Cpu, Cpl))

# ========= 핵심: Cp 구간 내 최적 LSL/USL 탐색 (Cpk 최대) =========
def best_specs_in_cp_range(mu: float, sigma: float,
                           cp_min: float, cp_max: float,
                           floor: Optional[float]=None,
                           ceiling: Optional[float]=None) -> Tuple[float, float, float, float, float]:
    """
    Cp ∈ [cp_min, cp_max]에서 Cpk 최대가 되도록 L/U 선택.
    - 원칙: Cpk는 'μ에서 가까운 규격선까지의 거리'로 결정 → μ를 중앙에 두고 폭을 가능한 크게.
    - 물리한계가 있으면 그 안에서 가능한 최대 폭을 쓰고, 불가하면 Cp를 줄여 적합.
    반환: (LSL, USL, Cp, Cpk, chosen_cp)
    """
    if np.isnan(mu) or np.isnan(sigma) or sigma <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    w_for_cp = lambda cp: 6.0 * sigma * cp
    # 한계 없음: cp_max로 대칭 배치(최적)
    if floor is None and ceiling is None:
        cp_chosen = cp_max
        w = w_for_cp(cp_chosen)
        L, U = mu - w/2.0, mu + w/2.0
        Cp, Cpk = cp_cpk_from(mu, sigma, L, U)
        return L, U, Cp, Cpk, cp_chosen

    # 한계 있을 때
    min_L = -np.inf if floor is None else float(floor)
    max_U =  np.inf if ceiling is None else float(ceiling)
    if max_U <= min_L:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    max_centerable_width = 2.0 * min(mu - min_L, max_U - mu)  # μ를 중앙에 둘 수 있는 최대 폭
    feasible_cp_max = max_centerable_width / (6.0 * sigma)

    if feasible_cp_max >= cp_max:
        cp_chosen = cp_max
        w = w_for_cp(cp_chosen)
        L, U = mu - w/2.0, mu + w/2.0
        Cp, Cpk = cp_cpk_from(mu, sigma, L, U)
        return L, U, Cp, Cpk, cp_chosen

    # cp_max가 불가 → 가능한 최대 cp로 낮춤 (최소 cp_min은 보장하려 시도)
    cp_candidate = max(feasible_cp_max, cp_min)
    cp_chosen = cp_candidate
    w = w_for_cp(cp_chosen)
    L, U = mu - w/2.0, mu + w/2.0

    # 경계 벗어나면 한쪽으로 평행이동, 그래도 불가하면 폭 축소(=cp 더 낮춤)
    if L < min_L:
        shift = min_L - L
        L += shift; U += shift
        if U > max_U:  # 그래도 불가 → 최대 들어갈 폭으로 축소
            w = max_U - min_L
            cp_chosen = w / (6.0 * sigma)
            L, U = min_L, max_U
    elif U > max_U:
        shift = U - max_U
        L -= shift; U -= shift
        if L < min_L:
            w = max_U - min_L
            cp_chosen = w / (6.0 * sigma)
            L, U = min_L, max_U

    Cp, Cpk = cp_cpk_from(mu, sigma, L, U)
    return L, U, Cp, Cpk, cp_chosen

# # ========= 실행 =========
# def main():
#     if not DATA_PATH.exists():
#         raise FileNotFoundError(f"데이터 파일이 없어요: {DATA_PATH.resolve()}")

#     raw = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)

#     # 단위 변환
#     df = apply_unit_conversions(raw, UNIT_CONV)

#     rows = []
#     for col in CANDIDATES:
#         if col not in df.columns:
#             continue
#         x = to_numeric(df[col]).dropna().to_numpy(float)
#         if x.size < 2:
#             rows.append({
#                 "Variable": col, "N_valid": x.size,
#                 "Mean": np.nan, "Std_overall": np.nan, "Std_within(MR)": np.nan,
#                 "Best_LSL": np.nan, "Best_USL": np.nan,
#                 "Chosen_Cp": np.nan, "Cp_overall": np.nan, "Cpk_overall": np.nan,
#                 "Cp_within": np.nan, "Cpk_within": np.nan
#             })
#             continue

#         mu = float(np.mean(x))
#         sigma_overall = float(np.std(x, ddof=1))
#         sigma_within  = estimate_sigma_within_mr(x)

#         floor = PHYSICAL_FLOOR.get(col, None)
#         ceil  = PHYSICAL_CEILING.get(col, None)

#         # 전체σ 기준 최적 스펙 (Cpk 최대)
#         L, U, Cp_o, Cpk_o, cp_chosen = best_specs_in_cp_range(
#             mu, sigma_overall, CP_MIN, CP_MAX, floor=floor, ceiling=ceil
#         )
#         # 같은 L/U로 within-σ 기준도 참고 계산
#         Cp_w, Cpk_w = cp_cpk_from(mu, sigma_within, L, U)

#         rows.append({
#             "Variable": col, "N_valid": x.size,
#             "Mean": mu, "Std_overall": sigma_overall, "Std_within(MR)": sigma_within,
#             "Best_LSL": L, "Best_USL": U,
#             "Chosen_Cp": cp_chosen,
#             "Cp_overall": Cp_o, "Cpk_overall": Cpk_o,
#             "Cp_within": Cp_w, "Cpk_within": Cpk_w
#         })

#     res = pd.DataFrame(rows).sort_values("Cpk_overall", na_position="last").reset_index(drop=True)
#     show = ["Variable","N_valid","Best_LSL","Best_USL","Chosen_Cp",
#             "Mean","Std_overall","Cp_overall","Cpk_overall",
#             "Std_within(MR)","Cp_within","Cpk_within"]
#     print("\n===== Best Specs under Cp ∈ [%.2f, %.2f] (maximize Cpk) =====" % (CP_MIN, CP_MAX))
#     if not res.empty:
#         print(res[show].to_string(index=False))
#     else:
#         print("대상 컬럼이 없습니다.")

# if __name__ == "__main__":
#     main()
