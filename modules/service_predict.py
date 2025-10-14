# service_predict.py
import pandas as pd
from shiny import reactive
import numpy as np

shap_values_state = reactive.Value(None)
X_input_state = reactive.Value(None)
X_input_raw = reactive.Value(None)

def do_predict(input, shap_values_state, X_input_state, X_input_raw, models, explainers):
    """
    버튼 클릭 시 실행되는 예측 함수
    - mold_code 모델이 있으면 해당 모델 사용
    - 없으면 전체 모델 soft voting
    """
    features = {
        "molten_temp": input.molten_temp(),
        "molten_volume": input.molten_volume(),
        "sleeve_temperature": input.sleeve_temperature(),
        "EMS_operation_time": input.EMS_operation_time(),
        "cast_pressure": input.cast_pressure(),
        "biscuit_thickness": input.biscuit_thickness(),
        "low_section_speed": input.low_section_speed(),
        "high_section_speed": input.high_section_speed(),
        "physical_strength": input.physical_strength(),
        "upper_mold_temp1": input.upper_mold_temp1(),
        "upper_mold_temp2": input.upper_mold_temp2(),
        "lower_mold_temp1": input.lower_mold_temp1(),
        "lower_mold_temp2": input.lower_mold_temp2(),
        "Coolant_temperature": input.coolant_temp(),
        "facility_operation_cycleTime": input.facility_operation_cycleTime(),
        "production_cycletime": input.production_cycletime(),
        "count": input.count(),
        "working": input.working(),
        "tryshot_signal": "D" if input.tryshot_check() else "A"
    }

    X = pd.DataFrame([features])
    mold_code = input.mold_code()

    model = models.get(mold_code)
    explainer = explainers.get(mold_code)

    # ---------------------------
    # Case 1: 해당 mold_code 모델 있음
    # ---------------------------
    if model is not None and explainer is not None:
        try:
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0][1]
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            return -1, None

        # 전처리 + shap
        try:
            X_transformed = model.named_steps["preprocess"].transform(X)
            feature_names = model.named_steps["preprocess"].get_feature_names_out()
            X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)
        except Exception as e:
            print(f"[ERROR] Preprocessing failed: {e}")
            return pred, proba

        try:
            shap_values = explainer(X_transformed_df)
        except Exception as e:
            print(f"[ERROR] SHAP calculation failed: {e}")
            shap_values = None

        shap_values_state.set(shap_values)
        X_input_state.set(X_transformed_df)
        X_input_raw.set(X.copy())

        return pred, proba

    # ---------------------------
    # Case 2: mold_code 모델 없음 → soft voting
    # ---------------------------
    else:
        all_probas = []
        for mc, mdl in models.items():
            try:
                p = mdl.predict_proba(X)[0][1]
                all_probas.append(p)
            except Exception as e:
                print(f"[WARN] 모델 {mc} 예측 실패: {e}")

        if not all_probas:
            return -1, None

        avg_proba = np.mean(all_probas)
        pred = 1 if avg_proba >= 0.5 else 0

        # soft voting일 때는 shap 계산이 애매 → None으로
        shap_values_state.set(None)
        X_input_state.set(None)
        X_input_raw.set(X.copy())

        return pred, avg_proba
