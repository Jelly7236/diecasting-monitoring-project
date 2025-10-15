import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score

# 1) 데이터 불러오기
train = pd.read_csv('./data2/train_split.csv')
valid = pd.read_csv('./data2/valid_split.csv')
test  = pd.read_csv('./data2/test_split.csv')

# 2) 특징(수치형 센서)
features = [
    'molten_volume','molten_temp','facility_operation_cycleTime','production_cycletime',
    'low_section_speed','high_section_speed','cast_pressure','biscuit_thickness',
    'upper_mold_temp1','upper_mold_temp2','lower_mold_temp1','lower_mold_temp2',
    'sleeve_temperature','physical_strength','Coolant_temperature'
]

X_train = train[features]
X_valid = valid[features]
X_test  = test[features]

# 3) 라벨: IF 규칙에 맞춰 변환  (1=정상, -1=이상)
y_valid = valid['passorfail'].replace({0:1, 1:-1}).astype(int)
y_test  = test['passorfail'].replace({0:1, 1:-1}).astype(int)

# 4) 파이프라인: RobustScaler → PCA(95%) → IsolationForest
pipe = Pipeline([
    ("scaler", RobustScaler()),
    ("pca", PCA(n_components=0.95, random_state=42)),
    ("iforest", IsolationForest(random_state=42, contamination=0.05))
])

pipe.fit(X_train)

# 5) 예측
pred_val  = pipe.predict(X_valid)   # 1=정상, -1=이상
pred_test = pipe.predict(X_test)

# 6) F2 스코어 헬퍼: -1(이상)을 양성(1)으로 간주
def f2_score_anomaly(y_true, y_pred):
    y_true_bin = (y_true == -1).astype(int)
    y_pred_bin = (y_pred == -1).astype(int)
    return fbeta_score(y_true_bin, y_pred_bin, beta=2, zero_division=0)

# 7) 전체 평가
print("=== VALID 전체 평가 ===")
print(confusion_matrix(y_valid, pred_val))
print(classification_report(y_valid, pred_val, digits=3, zero_division=0))
print("F2-score (이상=-1 기준):", round(f2_score_anomaly(y_valid, pred_val), 3))

print("\n=== TEST 전체 평가 ===")
print(confusion_matrix(y_test, pred_test))
print(classification_report(y_test, pred_test, digits=3, zero_division=0))
print("F2-score (이상=-1 기준):", round(f2_score_anomaly(y_test, pred_test), 3))

# 8) 몰드 코드별 리포트 & F2 (있을 때만)
mold_col = 'mold_code' if 'mold_code' in valid.columns else None

if mold_col:
    print("\n==================== Mold Code별 VALID 결과 ====================")
    for mold in sorted(valid[mold_col].dropna().unique()):
        idx = valid[mold_col] == mold
        y_true_m = y_valid[idx]
        y_pred_m = pred_val[idx]
        if len(y_true_m) == 0: 
            continue
        print(f"\n[MOLD CODE: {mold}] (VALID, n={len(y_true_m)})")
        print(classification_report(y_true_m, y_pred_m, digits=3, zero_division=0))
        print(f"F2-score (이상=-1 기준): {round(f2_score_anomaly(y_true_m, y_pred_m), 3)}")

    print("\n==================== Mold Code별 TEST 결과 ====================")
    for mold in sorted(test[mold_col].dropna().unique()):
        idx = test[mold_col] == mold
        y_true_m = y_test[idx]
        y_pred_m = pred_test[idx]
        if len(y_true_m) == 0:
            continue
        print(f"\n[MOLD CODE: {mold}] (TEST, n={len(y_true_m)})")
        print(classification_report(y_true_m, y_pred_m, digits=3, zero_division=0))
        print(f"F2-score (이상=-1 기준): {round(f2_score_anomaly(y_true_m, y_pred_m), 3)}")


# 이상치 탐지모델을 전체 다들어가게 그리고, 다변량관리도는 공정단계별로
# 이상치 알림이 울리면, 실제 불량비율이 얼마인지 
# 공정과정에서의 관리도 반영
# 다변량관리도, 각변수별 관리도 돌아가고 이상탐지
