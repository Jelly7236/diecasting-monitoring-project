# utils/realtime_predictor.py
import pandas as pd
from shared import rft_models, streaming_df

def get_latest_input(mold_code: str):
    """해당 금형의 최신 데이터를 가져와 feature 구성"""
    df = streaming_df[streaming_df["mold_code"] == mold_code]
    if df.empty:
        return None
    latest = df.sort_values("datetime").iloc[-1:]
    exclude_cols = ["id", "passorfail", "date", "time", "mold_name", "line"]
    features = [c for c in latest.columns if c not in exclude_cols]
    return latest[features]

def predict_quality(mold_code: str):
    """RandomForestTimeSeries 모델을 활용해 실시간 양/불 예측"""
    model = rft_models.get(mold_code)
    if model is None:
        return None, f"모델 없음 ({mold_code})"
    X_input = get_latest_input(mold_code)
    if X_input is None:
        return None, "데이터 없음"

    try:
        y_pred = model.predict(X_input)[0]
        y_prob = model.predict_proba(X_input)[0][1]
        return {"mold": mold_code, "pred": int(y_pred), "prob": float(y_prob)}, None
    except Exception as e:
        return None, str(e)
