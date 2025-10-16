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
streaming_df = pd.read_csv(data_dir / "train2.csv")

# 시간 칼럼 통합 (옵션)
streaming_df["datetime"] = pd.to_datetime(streaming_df["date"] + " " + streaming_df["time"])
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