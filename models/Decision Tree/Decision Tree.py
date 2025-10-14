import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, make_scorer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import optuna

# 1. 데이터 불러오기
df = pd.read_csv("C:\\Users\\USER\\Desktop\\diecasting-project\\data\\data1.csv")
X = df.drop(columns="passorfail")
y = df["passorfail"]

# 2. 컬럼 타입 분리
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# 3. 전처리기
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols)
    ]
)

# 4. Stratified K-Fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 5. Optuna Objective 함수 정의
def objective(trial):
    # 하이퍼파라미터 탐색 공간 정의
    params = {
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        "max_depth": trial.suggest_int("max_depth", 2, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "random_state": 42
    }

    # 모델 정의
    dt_model = DecisionTreeClassifier(**params)

    # 파이프라인 (전처리 + SMOTE + 모델)
    clf = ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(sampling_strategy=0.2, random_state=42)),
        ("model", dt_model)
    ])

    # 교차검증 (불량=1 기준 F1 점수)
    score = cross_val_score(
        clf, X, y,
        scoring=make_scorer(f1_score, pos_label=1),
        cv=cv,
        n_jobs=-1
    ).mean()

    return score

# 6. Optuna 실행
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)  # 시도 횟수 조정 가능

print("\n✅ Best Params:", study.best_params)
print("✅ Best CV F1 Score:", study.best_value)

# 7. 최적 모델 학습 및 평가
best_params = study.best_params
best_model = DecisionTreeClassifier(**best_params)

clf_best = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", SMOTE(sampling_strategy=0.2, random_state=42)),
    ("model", best_model)
])

# 데이터 분할
train_x, test_x, train_y, test_y = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=0
)

clf_best.fit(train_x, train_y)

# 예측
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, classification_report

y_pred = clf_best.predict(test_x)
y_proba = clf_best.predict_proba(test_x)[:, 1]

print("\n📊 Test Results with Decision Tree + SMOTE(20%) + Bayesian Optimization")
print("Accuracy :", accuracy_score(test_y, y_pred))
print("F1 Score :", f1_score(test_y, y_pred, pos_label=1))
print("Recall   :", recall_score(test_y, y_pred, pos_label=1))  # 민감도
print("AUC      :", roc_auc_score(test_y, y_proba))
print("\nClassification Report:\n", classification_report(test_y, y_pred, target_names=["양품(0)", "불량(1)"]))
