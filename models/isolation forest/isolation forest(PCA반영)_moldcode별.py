# Isolation Forest PCA/Robust Scaler/mold code별 반영

import pandas as pd
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, recall_score, f1_score
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# ===== 1️⃣ 몰드별 데이터 불러오기 =====
train_8412 = pd.read_csv('./data_group/train_group_8412.csv')
test_8412  = pd.read_csv('./data_group/test_group_8412.csv')

train_8573 = pd.read_csv('./data_group/train_group_8573.csv')
test_8573  = pd.read_csv('./data_group/test_group_8573.csv')

train_8600 = pd.read_csv('./data_group/train_group_8600.csv')
test_8600  = pd.read_csv('./data_group/test_group_8600.csv')

train_8722 = pd.read_csv('./data_group/train_group_8722.csv')
test_8722  = pd.read_csv('./data_group/test_group_8722.csv')

train_8917 = pd.read_csv('./data_group/train_group_8917.csv')
test_8917  = pd.read_csv('./data_group/test_group_8917.csv')

# ===== 2️⃣ 공통 설정 =====
TIME_COL = "time"
FEATURES = [
    'molten_volume','molten_temp','facility_operation_cycleTime','production_cycletime',
    'low_section_speed','high_section_speed','cast_pressure','biscuit_thickness',
    'upper_mold_temp1','upper_mold_temp2','lower_mold_temp1','lower_mold_temp2',
    'sleeve_temperature','physical_strength','Coolant_temperature'
]
CANDIDATES = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]  # contamination 후보값

# 저장 경로 설정 
SAVE_DIR = './saved_models(IF)'

def to_if_labels(s):
    """0→1(정상), 1→-1(이상) 변환"""
    return s.replace({0:1, 1:-1}).astype(int)

def recall_anomaly(y_true_if, y_pred_if):
    """-1(이상)을 양성으로 본 recall 계산"""
    y_true_bin = (y_true_if == -1).astype(int)
    y_pred_bin = (y_pred_if == -1).astype(int)
    return recall_score(y_true_bin, y_pred_bin, zero_division=0)

def f1_anomaly(y_true_if, y_pred_if):
    """-1(이상)을 양성으로 본 F1 계산"""
    y_true_bin = (y_true_if == -1).astype(int)
    y_pred_bin = (y_pred_if == -1).astype(int)
    return f1_score(y_true_bin, y_pred_bin, zero_division=0)

def make_pipeline(contamination):
    return Pipeline([
        ("scaler", RobustScaler()),
        ("pca", PCA(n_components=0.95, random_state=42)),
        ("iforest", IsolationForest(random_state=42, contamination=contamination))
    ])


# ===== 3. Isolation Forest 모델 학습 =====
def run_isolation_forest(mold_code, train_df, test_df, save_name=None):
    print(f"\n==================== MOLD {mold_code} ====================")

    # (1) 데이터 준비  ※ 시계열 split 파일이라 추가 정렬/변환 불필요
    X_train = train_df[FEATURES]
    y_train_if = to_if_labels(train_df['passorfail'])
    X_test  = test_df[FEATURES]
    y_test_if = to_if_labels(test_df['passorfail'])

    # (2) train의 80% / 20% 시계열 분할 (검증용) — 시간 순서 유지
    split_idx = int(len(X_train) * 0.8)
    X_tr, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
    y_val = y_train_if.iloc[split_idx:]

    # (3) contamination 튜닝 (validation F1 기준, 동률이면 recall 큰 값)
    best_c, best_f1, best_r = None, -1.0, -1.0
    for c in CANDIDATES:
        pipe = make_pipeline(c)
        pipe.fit(X_tr)                       # 스케일러/PCA는 오직 X_tr로만 학습 → 누수 방지
        pred_val = pipe.predict(X_val)       # 1=정상, -1=이상

        f1 = f1_anomaly(y_val, pred_val)
        r  = recall_anomaly(y_val, pred_val)
        if (f1 > best_f1) or (f1 == best_f1 and r > best_r):
            best_f1, best_r, best_c = f1, r, c

    print(f"Best contamination = {best_c} (VAL F1={best_f1:.3f}, Recall={best_r:.3f})")

    # (4) 최종 모델 학습 및 테스트 예측  ※ 전처리 포함 파이프라인을 전체 train에 재학습
    pipe_final = make_pipeline(best_c)
    pipe_final.fit(X_train)
    pred_test = pipe_final.predict(X_test)

    print(confusion_matrix(y_test_if, pred_test))
    print(classification_report(y_test_if, pred_test, digits=3, zero_division=0))

    # ✅ (5) 모델 저장
    model_path = os.path.join(SAVE_DIR, f'isolation_forest_{mold_code}.pkl')
    joblib.dump(pipe_final, model_path)
    print(f"💾 모델 저장 완료 → {model_path}")   


# ===== 3️⃣ 몰드별 실행 =====
run_isolation_forest(8412, train_8412, test_8412)
run_isolation_forest(8573, train_8573, test_8573)
run_isolation_forest(8600, train_8600, test_8600)
run_isolation_forest(8722, train_8722, test_8722)
run_isolation_forest(8917, train_8917, test_8917)
