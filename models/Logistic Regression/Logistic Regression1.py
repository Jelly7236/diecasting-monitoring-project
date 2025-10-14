import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, recall_score, make_scorer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import optuna

# -------------------------------
# 1. 데이터 불러오기
# -------------------------------
df = pd.read_csv("C:\\Users\\USER\\Desktop\\diecasting-project\\data\\data1.csv")
X = df.drop(columns="passorfail")
y = df["passorfail"]

# -------------------------------
# 2. 컬럼 타입 분리
# -------------------------------
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

print("Categorical:", cat_cols)
print("Numerical:", num_cols)

# -------------------------------
# 3. 전처리기 정의
# -------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols)
    ]
)

# -------------------------------
# 4. Optuna 목적 함수 정의
# -------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial):
    # 하이퍼파라미터 탐색 범위 정의
    C = trial.suggest_float("C", 1e-3, 10.0, log=True)   # 규제 강도
    penalty = trial.suggest_categorical("penalty", ["l1", "l2"])  # 규제 방식
    solver = "liblinear" if penalty == "l1" else "lbfgs"

    log_reg = LogisticRegression(
        C=C,
        penalty=penalty,
        solver=solver,
        class_weight="balanced",  # 불균형 대응
        max_iter=1000,
        random_state=42
    )

    # SMOTE + 전처리 + 모델 파이프라인
    clf = ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(sampling_strategy=0.2, random_state=42)),  # 불량=양품의 20%
        ("model", log_reg)
    ])

    score = cross_val_score(
        clf, X, y,
        scoring=make_scorer(f1_score, pos_label=1),
        cv=cv,
        n_jobs=-1
    ).mean()

    return score

# -------------------------------
# 5. Optuna 실행
# -------------------------------
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, n_jobs=-1)

print("\n✅ [Optuna] Best Params:", study.best_params)
print("✅ [Optuna] Best CV F1 Score:", study.best_value)

# -------------------------------
# 6. 최적 파라미터로 최종 모델 학습
# -------------------------------
best_params = study.best_params
solver = "liblinear" if best_params["penalty"] == "l1" else "lbfgs"

log_reg_best = LogisticRegression(
    C=best_params["C"],
    penalty=best_params["penalty"],
    solver=solver,
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

final_clf = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", SMOTE(sampling_strategy=0.2, random_state=42)),
    ("model", log_reg_best)
])

train_x, test_x, train_y, test_y = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=0
)

final_clf.fit(train_x, train_y)

# -------------------------------
# 7. 평가
# -------------------------------
y_pred = final_clf.predict(test_x)
y_proba = final_clf.predict_proba(test_x)[:, 1]

print("\n📊 [Final Logistic Regression + SMOTE] Test Results")
print("Accuracy :", accuracy_score(test_y, y_pred))
print("F1 Score :", f1_score(test_y, y_pred, pos_label=1))
print("AUC      :", roc_auc_score(test_y, y_proba))
print("Recall(Sensitivity, 불량=1):", recall_score(test_y, y_pred, pos_label=1))
print("\n✅ [Optuna] Best Params:", study.best_params)
