# shared.py
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import shap
from matplotlib import font_manager as fm
import os
import json
import sys
from shiny import reactive
import numpy as np



from models.FinalModel.smote_sampler import MajorityVoteSMOTENC

# app.py가 있는 위치를 기준으로 절대 경로 관리
app_dir = Path(__file__).parent
# 데이터 경로
data_dir = app_dir / "data"
models_dir = app_dir / "models"
fonts_dir = app_dir / "www" / "fonts"

def setup_korean_font():
    """로컬 + 리눅스 서버에서 모두 한글 깨짐 방지"""
    font_candidates = []

    # 1. 프로젝트 폰트 (권장)
    font_path = Path(__file__).parent / "www" / "fonts" / "NotoSansKR-Regular.ttf"
    if font_path.exists():
        font_prop = fm.FontProperties(fname=str(font_path))
        plt.rcParams['font.family'] = font_prop.get_name()
        fm.fontManager.addfont(str(font_path))
        print(f"✅ 내부 폰트 적용: {font_prop.get_name()}")
        return

    # 2. 로컬 OS별 기본 폰트
    font_candidates = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans KR"]

    for font in font_candidates:
        if font in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
            plt.rcParams['font.family'] = font
            print(f"✅ 시스템 폰트 적용: {font}")
            return

    # 3. fallback
    plt.rcParams['font.family'] = "DejaVu Sans"
    print("⚠️ 한글 폰트를 찾지 못해 DejaVu Sans로 대체합니다.")

    # 마이너스 기호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False
setup_korean_font()


# plt.rcParams['axes.unicode_minus'] = False

# Data Load
# streaming_df = pd.read_csv(data_dir / "train2.csv")
streaming_df = pd.read_excel(data_dir / "test2.xlsx")
prediction_state = reactive.Value(pd.DataFrame()) ### 여기추가
current_state = reactive.Value(pd.DataFrame())  # ✅ 실시간 공정 데이터 (센서 기반)

# 시간 칼럼 통합 (옵션)
# date 컬럼이 datetime이면 str로 변환, 아니면 그대로
if np.issubdtype(streaming_df["date"].dtype, np.datetime64):
    date_str = streaming_df["date"].dt.strftime("%Y-%m-%d")
else:
    date_str = streaming_df["date"].astype(str)

streaming_df["datetime"] = pd.to_datetime(date_str + " " + streaming_df["time"].astype(str))

streaming_df = streaming_df.sort_values("datetime").reset_index(drop=True)


# 이상치 제거 데이터
df2 = pd.read_csv(data_dir / "outlier_remove_data2.csv")


rf_models = {
    "8412": joblib.load(models_dir / "RandomForest" /"rf_mold_8412.pkl"),
    "8573": joblib.load(models_dir / "RandomForest" /"rf_mold_8573.pkl"),
    "8600": joblib.load(models_dir / "RandomForest" /"rf_mold_8600.pkl"),
    "8722": joblib.load(models_dir / "RandomForest" /"rf_mold_8722.pkl"),
    "8917": joblib.load(models_dir / "RandomForest" /"rf_mold_8917.pkl"),
    
}

rf_explainers = {
    "8412": shap.TreeExplainer(rf_models["8412"].named_steps["model"]),
    "8573": shap.TreeExplainer(rf_models["8573"].named_steps["model"]),
    "8600": shap.TreeExplainer(rf_models["8600"].named_steps["model"]),
    "8722": shap.TreeExplainer(rf_models["8722"].named_steps["model"]),
    "8917": shap.TreeExplainer(rf_models["8917"].named_steps["model"]),
    
}

# shared.py에 추가
rf_preprocessors = {
    "8412": rf_models["8412"].named_steps["preprocess"],
    "8573": rf_models["8573"].named_steps["preprocess"],
    "8600": rf_models["8600"].named_steps["preprocess"],
    "8722": rf_models["8722"].named_steps["preprocess"],
    "8917": rf_models["8917"].named_steps["preprocess"],
}

# 전처리된 컬럼명 → 원래 변수명
feature_name_map = {
    "num__molten_temp": "molten_temp",
    "num__molten_volume": "molten_volume",
    "num__sleeve_temperature": "sleeve_temperature",
    "num__EMS_operation_time": "EMS_operation_time",
    "num__cast_pressure": "cast_pressure",
    "num__biscuit_thickness": "biscuit_thickness",
    "num__low_section_speed": "low_section_speed",
    "num__high_section_speed": "high_section_speed",
    "num__physical_strength": "physical_strength",
    "num__upper_mold_temp1": "upper_mold_temp1",
    "num__upper_mold_temp2": "upper_mold_temp2",
    "num__lower_mold_temp1": "lower_mold_temp1",
    "num__lower_mold_temp2": "lower_mold_temp2",
    "num__Coolant_temperature": "Coolant_temperature",
    "num__facility_operation_cycleTime": "facility_operation_cycleTime",
    "num__production_cycletime": "production_cycletime",
    "num__count": "count",
    "cat__working_가동": "working=가동",
    "cat__working_정지": "working=정지",
    "cat__tryshot_signal_A": "tryshot_signal=A",
    "cat__tryshot_signal_D": "tryshot_signal=D",
}

feature_name_map_kor = {
    "num__molten_temp": "용탕 온도(℃)",
    "num__molten_volume": "용탕 부피",
    "num__sleeve_temperature": "슬리브 온도(℃)",
    "num__EMS_operation_time": "EMS 작동시간(s)",
    "num__cast_pressure": "주조 압력(bar)",
    "num__biscuit_thickness": "비스킷 두께(mm)",
    "num__low_section_speed": "저속 구간 속도",
    "num__high_section_speed": "고속 구간 속도",
    "num__physical_strength": "형체력",
    "num__upper_mold_temp1": "상형 온도1(℃)",
    "num__upper_mold_temp2": "상형 온도2(℃)",
    "num__lower_mold_temp1": "하형 온도1(℃)",
    "num__lower_mold_temp2": "하형 온도2(℃)",
    "num__coolant_temp": "냉각수 온도(℃)",
    "num__facility_operation_cycleTime": "설비 가동 사이클타임",
    "num__production_cycletime": "생산 사이클타임",
    "num__count": "생산 횟수",
    "cat__working_가동": "작업 여부=가동",
    "cat__working_정지": "작업 여부=정지",
    "cat__tryshot_signal_A": "트라이샷 신호=A",
    "cat__tryshot_signal_D": "트라이샷 신호=D",
}

name_map_kor = {
    # 메타/식별자
    "id": "행 ID",
    "line": "작업 라인",
    "name": "제품명",
    "mold_name": "금형명",
    "time": "수집 시간",
    "date": "수집 일자",
    "registration_time": "등록 일시",

    # 생산 관련
    "count": "생산 횟수",
    "working": "작업 여부",
    "emergency_stop": "비상정지 여부",
    "passorfail": "양/불 판정 결과",
    "tryshot_signal": "트라이샷 여부",
    "mold_code": "금형 코드",
    "heating_furnace": "가열로 구분",

    # 공정 변수
    "molten_temp": "용탕 온도",
    "molten_volume": "용탕 부피",
    "sleeve_temperature": "슬리브 온도",
    "EMS_operation_time": "EMS 작동시간",
    "cast_pressure": "주조 압력(bar)",
    "biscuit_thickness": "비스킷 두께(mm)",
    "low_section_speed": "저속 구간 속도",
    "high_section_speed": "고속 구간 속도",
    "physical_strength": "형체력",

    # 금형 온도
    "upper_mold_temp1": "상형 온도1",
    "upper_mold_temp2": "상형 온도2",
    "upper_mold_temp3": "상형 온도3",
    "lower_mold_temp1": "하형 온도1",
    "lower_mold_temp2": "하형 온도2",
    "lower_mold_temp3": "하형 온도3",

    # 냉각 관련
    "Coolant_temperature": "냉각수 온도",

    # 사이클 관련
    "facility_operation_cycleTime": "설비 가동 사이클타임",
    "production_cycletime": "생산 사이클타임",

    # 파생 변수
    "day": "일",
    "month": "월",
    "weekday": "요일"
}

mold_list = ["8412", "8573", "8600", "8722", "8917"]

# Isolation Forest 모델 로드
iso_dir = models_dir / "isolation forest" / "saved_models(IF)"
iso_meta_path = models_dir / "isolation forest" /"saved_models(IF)" / "isoforest_20251015_150622_meta.json"

iso_models = {}

for mold in mold_list:
    model_path = iso_dir / f"isolation_forest_{mold}.pkl"
    try:
        iso_models[mold] = joblib.load(model_path)
        print(f"✅ IsolationForest 모델 로드 완료: {model_path.name}")
    except Exception as e:
        print(f"⚠️ 모델 로드 실패 ({mold}): {e}")

try:
    with open(iso_meta_path, "r", encoding="utf-8") as f:
        iso_meta = json.load(f)
    iso_features = iso_meta.get("features", [])
    print(f"✅ IsolationForest 메타 정보 로드 완료: {iso_meta_path.name}")
except Exception as e:
    iso_meta, iso_features = None, []
    print(f"⚠️ IsolationForest 메타 로드 실패: {e}")

print(f"\n총 로드된 모델 수: {len(iso_models)}개")
print(f"사용 가능한 피처 수: {len(iso_features)}개 → {iso_features if iso_features else '메타 없음'}")

# RandomForestTimeSeries 모델 로드
rft_dir = models_dir / "RandomForestTimeSeries"

rft_models = {}

# ✅ scikit-learn 내부 클래스 호환 패치
import sklearn.compose._column_transformer as ct
sys.modules["__main__"].MajorityVoteSMOTENC = MajorityVoteSMOTENC

if not hasattr(ct, "_RemainderColsList"):
    class _RemainderColsList(list):
        """Dummy placeholder for sklearn 1.6.x compatibility"""
        pass
    ct._RemainderColsList = _RemainderColsList
    print("✅ sklearn.compose._column_transformer._RemainderColsList 더미 클래스 등록 완료")
    
for mold in mold_list:
    model_path = rft_dir / f"model_{mold}.pkl"
    try:
        rft_models[mold] = joblib.load(model_path)
        print(f"✅ RandomForestTimeSeries 모델 로드 완료: {model_path.name}")
    except Exception as e:
        print(f"⚠️ 모델 로드 실패 ({mold}): {e}")

print(f"\n총 로드된 모델 수: {len(rft_models)}개")

### 다변량 관리도 정보 로드
def load_multivar_chart_info():
    df = pd.read_csv("./data/multivar_chart_info.csv")
    df["mean_vector"] = df["mean_vector"].apply(json.loads)
    return df
multivar_info = load_multivar_chart_info()

### 단변량 관리도 ARIMA 모델 로드
BASE_DIR = Path(__file__).parent
ARIMA_INFO_PATH = BASE_DIR / "data" / "arima_model_info_updated.csv"

arima_models = {}

try:
    arima_info = pd.read_csv(ARIMA_INFO_PATH)
    print(f"📄 ARIMA 메타데이터 로드 완료: {len(arima_info)}행")
except FileNotFoundError:
    print(f"⚠️ 파일을 찾을 수 없습니다 → {ARIMA_INFO_PATH}")
    arima_info = pd.DataFrame()

for _, row in arima_info.iterrows():
    mold = str(row.get("mold_code", "")).strip()
    var = str(row.get("variable", "")).strip()
    model_path = Path(row.get("model_path", "")).as_posix()
    key = f"{mold}_{var}"

    try:
        model_file = BASE_DIR / Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"모델 파일이 존재하지 않음: {model_file}")

        with open(model_file, "rb") as f:
            model = joblib.load(f)

        arima_models[key] = {
            "model": model,
            "mold": mold,
            "variable": var,
            "p": int(row.get("p", 0)),
            "d": int(row.get("d", 0)),
            "q": int(row.get("q", 0)),
            "n_obs": int(row.get("n_obs_model", 0)),
            "aic": float(row.get("aic", 0)),
            "bic": float(row.get("bic", 0)),
            "cl": float(row.get("CL", 0)),
            "sigma": float(row.get("sigma", 0)),
            "ucl": float(row.get("UCL_3sigma", 0)),
            "lcl": float(row.get("LCL_3sigma", 0)),
        }

    except Exception as e:
        print(f"⚠️ [{key}] 로드 실패 → {e}")

print(f"✅ 총 {len(arima_models)}개 ARIMA 모델 로드 완료")


# ====================================================
# 3️⃣ 접근 유틸리티
# ====================================================
def get_arima_model(mold_code: str, variable: str):
    """
    mold_code와 variable로 ARIMA 모델 검색.
    예:
        m = get_arima_model("8412", "Coolant_temperature")
        m["model"].forecast(steps=5)
    """
    key = f"{mold_code}_{variable}"
    return arima_models.get(key)


def list_arima_models():
    """현재 로드된 ARIMA 모델 전체 목록 반환"""
    return pd.DataFrame([
        {
            "key": k,
            "mold": v["mold"],
            "variable": v["variable"],
            "p": v["p"],
            "d": v["d"],
            "q": v["q"]
        }
        for k, v in arima_models.items()
    ])

# 다변량 기준 공분산 행렬 등록

cov_matrices = {
    8412: {
        3: np.array([
            [127.876960, 134.655342, 22.687752, -8.754930, -53.034360],
            [134.655342, 150.045114, 25.964681, -11.089454, -57.921923],
            [22.687752, 25.964681, 73.645743, -7.729699, -41.922842],
            [-8.754930, -11.089454, -7.729699, 11.045913, 14.949799],
            [-53.034360, -57.921923, -41.922842, 14.949799, 195.273963]
        ]),
        4: np.array([
            [1591.384130, 171.737395, -68.344648, 694.267212, 10.983294],
            [171.737395, 313.833970, 122.461660, 194.277955, 12.568786],
            [-68.344648, 122.461660, 1056.410256, -76.971875, 5.942511],
            [694.267212, 194.277955, -76.971875, 1124.600061, 20.073014],
            [10.983294, 12.568786, 5.942511, 20.073014, 4.745007]
        ])
    },
    8413: {
        1: np.array([
            [2.181921, 5.210734],
            [5.210734, 231.542090]
        ]),
        3: np.array([
            [0, 0, 0, 0, 0],
            [0, 0.281356, -0.001695, -0.166102, 0.059322],
            [0, -0.001695, 0.048305, -0.003390, 0.072034],
            [0, -0.166102, -0.003390, 6.820339, -1.881356],
            [0, 0.059322, 0.072034, -1.881356, 6.586158]
        ]),
        4: np.array([
            [11.616949, 21.393220, 37.664407, 26.400000, 3.489831],
            [21.393220, 56.298305, 85.518644, 54.657627, 8.037288],
            [37.664407, 85.518644, 142.541243, 96.341243, 13.339548],
            [26.400000, 54.657627, 96.341243, 69.097175, 8.990395],
            [3.489831, 8.037288, 13.339548, 8.990395, 1.406497]
        ])
    },
    8576: {
        1: np.array([[2.519156, 0], [0, 0]]),
        3: np.array([
            [0.097403, -0.022078, -0.016883, 0.161688, 0.311039],
            [-0.022078, 0.362338, 0.042857, -0.082468, 0.246104],
            [-0.016883, 0.042857, 0.222078, -0.060390, 1.060390],
            [0.161688, -0.082468, -0.060390, 3.872403, 0.145779],
            [0.311039, 0.246104, 1.060390, 0.145779, 120.054221]
        ]),
        4: np.array([
            [894.244156, 94.612987, 108.516883, 15.051948, -6.353247],
            [94.612987, 68.970130, 97.344805, 89.366883, 6.246104],
            [108.516883, 97.344805, 154.119156, 142.822403, 10.485390],
            [15.051948, 89.366883, 142.822403, 177.160714, 12.100325],
            [-6.353247, 6.246104, 10.485390, 12.100325, 1.137338]
        ])
    },
    8722: {
        1: np.array([[223.366186, -83.535979], [-83.535979, 685.859302]]),
        3: np.array([
            [25.368105, 13.603162, 6.913507, 0.884923, 18.318729],
            [13.603162, 58.279450, 1.330811, -3.100607, 6.466901],
            [6.913507, 1.330811, 15.496278, -3.820112, 44.425305],
            [0.884923, -3.100607, -3.820112, 14.772464, -15.712934],
            [18.318729, 6.466901, 44.425305, -15.712934, 303.934901]
        ]),
        4: np.array([
            [958.654604, 347.361164, 348.391187, -78.622508, -15.440608],
            [347.361164, 733.723937, -45.736142, -210.819360, -33.941177],
            [348.391187, -45.736142, 1944.734352, -962.277183, 44.056452],
            [-78.622508, -210.819360, -962.277183, 1825.466947, -1.335291],
            [-15.440608, -33.941177, 44.056452, -1.335291, 8.105254]
        ])
    },
    8917: {
        1: np.array([[119.587302, 100.379136], [100.379136, 1304.503413]]),
        3: np.array([
            [10.208602, 5.215599, 1.226577, -0.387230, 0.525340],
            [5.215599, 10.401384, 0.556699, -1.010207, -1.981685],
            [1.226577, 0.556699, 3.658302, 0.979020, 1.160507],
            [-0.387230, -1.010207, 0.979020, 14.234561, 7.710724],
            [0.525340, -1.981685, 1.160507, 7.710724, 59.029980]
        ]),
        4: np.array([
            [2184.691133, 59.490486, 215.500601, 969.129771, -24.260635],
            [59.490486, 229.163234, 322.152171, 173.589358, 0.917755],
            [215.500601, 322.152171, 1163.697249, -425.509129, -5.583567],
            [969.129771, 173.589358, -425.509129, 1900.700261, -16.166537],
            [-24.260635, 0.917755, -5.583567, -16.166537, 4.328931]
        ])
    }
}

    
###  X–R 관리도 기준선 로드 (ARIMA 없는 변수용)

def _load_xr_control_info(path: str = "./data/xr_control_info.csv") -> dict:
    """X–R 관리도용 통계치(X̄̄, R̄, UCL/LCL) 계산"""
    try:
        xr_df = pd.read_csv(path)
    except Exception as e:
        print(f"⚠️ X–R 관리도 파일 로드 실패: {e}")
        return {}

    # X–R 관리도 상수 테이블 (서브그룹 크기 n별)
    XR_CONSTANTS = {
        2: {"A2": 1.880, "D3": 0.000, "D4": 3.267},
        3: {"A2": 1.023, "D3": 0.000, "D4": 2.574},
        4: {"A2": 0.729, "D3": 0.000, "D4": 2.282},
        5: {"A2": 0.577, "D3": 0.000, "D4": 2.114},
        6: {"A2": 0.483, "D3": 0.000, "D4": 2.004},
        7: {"A2": 0.419, "D3": 0.076, "D4": 1.924},
        8: {"A2": 0.373, "D3": 0.136, "D4": 1.864},
        9: {"A2": 0.337, "D3": 0.184, "D4": 1.816},
        10: {"A2": 0.308, "D3": 0.223, "D4": 1.777},
    }

    xr_limits = {}

    for _, row in xr_df.iterrows():
        n = int(row["n"])
        const = XR_CONSTANTS.get(n, XR_CONSTANTS[5])  # 기본값 n=5
        A2, D3, D4 = const["A2"], const["D3"], const["D4"]

        cl_x = row["X_barbar"]
        cl_r = row["R_bar"]

        key = f"{row['mold_code']}_{row['variable']}"

        xr_limits[key] = {
            "mold": row["mold_code"],
            "variable": row["variable"],
            "n": n,
            "k_subgroups": row["k_subgroups"],
            "CL_X": cl_x,
            "UCL_X": cl_x + A2 * cl_r,
            "LCL_X": cl_x - A2 * cl_r,
            "CL_R": cl_r,
            "UCL_R": D4 * cl_r,
            "LCL_R": D3 * cl_r,
        }

    print(f"✅ X–R 관리도 기준선 로드 완료: {len(xr_limits)}개")
    return xr_limits


# 전역 변수로 보관
xr_limits = _load_xr_control_info()