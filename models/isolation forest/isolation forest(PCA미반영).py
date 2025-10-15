import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import fbeta_score, classification_report, confusion_matrix
import joblib, json

# 1. 데이터 불러오기
train = pd.read_csv('./data2/train_split.csv')
valid = pd.read_csv('./data2/valid_split.csv')
test = pd.read_csv('./data2/test_split.csv')

# 2. 특징 선택(수치형 센서 변수만)
train.dtypes
features = ['molten_volume', 'molten_temp','facility_operation_cycleTime','production_cycletime',
'low_section_speed','high_section_speed','cast_pressure','biscuit_thickness','upper_mold_temp1',
'upper_mold_temp2','lower_mold_temp1','lower_mold_temp2','sleeve_temperature',
'physical_strength','Coolant_temperature']

# 3. 데이터 분리 
X_train = train[features]
X_valid = valid[features]
X_test = test[features]

y_valid = valid['passorfail'] #검증용 실제 정답
y_test = test['passorfail'] #테스트용 실제 정답

# 4. Isolation Forest 모델 학습
clf = IsolationForest(random_state=42, contamination=0.05) #contamination은 전체데이터 중 이상치가 차지하는 비율(추정값),
#제조/품질 데이터에서 불량률은 보통 1~3%사이라 0.02로 잡음 
clf.fit(X_train)

# 5. 예측(IF결과: 1=정상, -1=이상 -> 0/1로 변환)
pred_val = clf.predict(X_valid)
pred_test = clf.predict(X_test)

# y값을 IsolationForest 기준에 맞춰 변환 
y_valid = y_valid.replace({0:1, 1:-1})
y_test = y_test.replace({0:1, 1:-1})

# 6. 평가
print("=== VALID ===")
print(confusion_matrix(y_valid, pred_val))
print(classification_report(y_valid, pred_val, digits=3, zero_division=0))

print("=== TEST ===")
print(confusion_matrix(y_test, pred_test))
print(classification_report(y_test, pred_test, digits=3, zero_division=0))

# 7. 모델 저장(joblib)
import os, joblib, json
from datetime import datetime

SAVE_DIR = "./saved_models(IF)"
os.makedirs(SAVE_DIR, exist_ok=True)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = os.path.join(SAVE_DIR, f"isoforest_{ts}.pkl")
meta_path  = os.path.join(SAVE_DIR, f"isoforest_{ts}_meta.json")

# 파이프라인이 아니라 IF 단독이므로 그대로 저장
joblib.dump(clf, model_path)
print(f"[SAVED] IsolationForest 모델 → {model_path}")

# (선택) 메타정보 저장: 재현을 위해 features/contamination 등 기록 권장
meta = {
    "saved_at": ts,
    "features": features,
    "model_type": "IsolationForest",
    "params": {
        "random_state": 42,
        "contamination": 0.05,
        "n_estimators": getattr(clf, "n_estimators", None),
        "max_samples": getattr(clf, "max_samples", None)
    }
}
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print(f"[SAVED] 메타정보 → {meta_path}")

# ====== 8. (참고) 로드 & 재사용 예시 ======
# pipe_loaded = joblib.load(model_path)
# pred_again  = pipe_loaded.predict(X_test)   # 1=정상, -1=이상
