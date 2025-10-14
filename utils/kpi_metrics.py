# utils/kpi_metrics.py
import numpy as np
import pandas as pd


def calculate_realtime_metrics(df: pd.DataFrame, mold_codes: list[str]):
    """실시간 KPI, OEE, 몰드별 통계 계산"""
    if df.empty:
        molds_init = {m: {"good": 0, "defect": 0, "rate": 0.0} for m in mold_codes}
        return {
            "abnormal": 0, "good_rate": 0.0, "prod_count": 0,
            "cycle_time": 0.0, "oee": 0.0,
            "availability": 0, "performance": 0, "quality": 0,
            "molds": molds_init
        }

    n = len(df)
    abnormal = np.sum((df["tryshot_signal"] == "D") | (df["molten_temp"] > 800))
    good = np.sum(df["passorfail"] == 0)
    defect = np.sum(df["passorfail"] == 1)
    good_rate = (good / n) * 100 if n > 0 else 0
    prod_count = n
    cycle_time = df["production_cycletime"].mean()

    running = np.sum(df["working"] == "가동")
    availability = running / n
    std_cycle = df["facility_operation_cycleTime"].mean()
    actual_cycle = df["production_cycletime"].mean()
    performance = min(1.0, std_cycle / actual_cycle) if actual_cycle > 0 else 0
    quality = good_rate / 100
    oee_value = availability * performance * quality

    mold_data = {}
    mold_group = df.groupby("mold_code")["passorfail"].value_counts().unstack(fill_value=0)
    for mold in mold_codes:
        if mold in mold_group.index:
            g = mold_group.loc[mold].get(0.0, 0)
            d = mold_group.loc[mold].get(1.0, 0)
            total = g + d
            rate = (g / total * 100) if total > 0 else 0.0
            mold_data[mold] = {"good": int(g), "defect": int(d), "rate": rate}
        else:
            mold_data[mold] = {"good": 0, "defect": 0, "rate": 0.0}

    return {
        "abnormal": int(abnormal),
        "good_rate": good_rate,
        "prod_count": prod_count,
        "cycle_time": cycle_time,
        "oee": oee_value,
        "availability": availability,
        "performance": performance,
        "quality": quality,
        "molds": mold_data,
    }
