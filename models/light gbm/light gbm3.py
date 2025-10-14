import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, recall_score, make_scorer
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import optuna

# ==============================================================
# 1. 데이터 불러오기
# ==============================================================
df = pd.read_csv("C:\\Users\\USER\\Desktop\\diecasting-project\\data\\data1.csv")
X = df.drop(columns="passorfail")
y = df["passorfail"]

# ==============================================================
# 2. 컬럼 타입 분리
# ==============================================================
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
print("Categorical:", cat_cols)
print("Numerical:", num_cols)

# ==============================================================
# 3. 전처리기 정의
# ==============================================================
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols)
    ]
)

# ==============================================================
# 4. Optuna 목적 함수 정의
# ==============================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial):
    params = {
        "objective": "binary",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "max_depth": trial.suggest_int("max_depth", -1, 20),
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 10),
        "random_state": 42,
        "n_jobs": -1,
        "class_weight": "balanced"   # 불균형 추가 대응
    }

    lgb_model = LGBMClassifier(**params)
    clf = ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(sampling_strategy=0.2, random_state=42)),
        ("model", lgb_model)
    ])

    score = cross_val_score(
        clf, X, y,
        scoring=make_scorer(f1_score, pos_label=1),
        cv=cv,
        n_jobs=-1
    ).mean()
    return score

# ==============================================================
# 5. Optuna 실행
# ==============================================================
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, n_jobs=-1)

print("\n✅ [Optuna] Best Params:", study.best_params)
print("✅ [Optuna] Best CV F1 Score:", study.best_value)

# ==============================================================
# 6. 최적 파라미터로 최종 모델 학습
# ==============================================================
best_params = study.best_params
best_params["class_weight"] = "balanced"

lgb_best = LGBMClassifier(**best_params)

final_clf = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", SMOTE(sampling_strategy=0.2, random_state=42)),
    ("model", lgb_best)
])

train_x, test_x, train_y, test_y = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=0
)

final_clf.fit(train_x, train_y)

# ==============================================================
# 7. 최종 평가
# ==============================================================
y_pred = final_clf.predict(test_x)
y_proba = final_clf.predict_proba(test_x)[:, 1]

print("\n📊 [Optuna + SMOTE(20%)] Test Results")
print("Accuracy :", accuracy_score(test_y, y_pred))
print("F1 Score :", f1_score(test_y, y_pred, pos_label=1))
print("AUC      :", roc_auc_score(test_y, y_proba))
print("Recall(Sensitivity, 불량=1):", recall_score(test_y, y_pred, pos_label=1))
print("\n✅ [Optuna] Best Params:", study.best_params)