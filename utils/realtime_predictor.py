# utils/realtime_predictor.py
import pandas as pd
from shared import rft_models

def get_latest_input(df: pd.DataFrame, mold_code: str):
    """실시간 데이터프레임에서 해당 몰드의 최신 샘플 1건을 추출"""
    # ✅ mold_code 타입 통일 (문자열)
    df["mold_code"] = df["mold_code"].astype(str)
    mold_code = str(mold_code)

    df_mold = df[df["mold_code"] == mold_code]
    if df_mold.empty:
        return None

    if "datetime" not in df_mold.columns:
        df_mold = df_mold.assign(datetime=pd.RangeIndex(len(df_mold)))

    latest = df_mold.sort_values("datetime").iloc[-1:]
    exclude_cols = ["id", "passorfail", "date", "time", "mold_name", "line"]
    features = [c for c in latest.columns if c not in exclude_cols]
    return latest[features]

def predict_quality(df: pd.DataFrame, mold_code: str):
    """현재 실시간 데이터 기반으로 RandomForestTimeSeries 예측 수행"""
    model = rft_models.get(str(mold_code))
    if model is None:
        return None, f"모델 없음 ({mold_code})"

    X_input = get_latest_input(df, mold_code)
    if X_input is None or X_input.empty:
        # ✅ 금형별 데이터가 아직 안 들어왔을 때는 스킵
        return None, f"{mold_code} 데이터 없음"

    try:
        y_pred = model.predict(X_input)[0]
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_input)[0][1]
        else:
            y_prob = float(y_pred)
        return {"mold": mold_code, "pred": int(y_pred), "prob": float(y_prob)}, None
    except Exception as e:
        return None, str(e)
