# utils/control_stats.py
import numpy as np
import pandas as pd
from scipy import stats

# -------------------------------
# 넬슨 룰
# -------------------------------
def check_nelson_rules(data, mean, ucl, lcl, sigma):
    violations = []
    n = len(data)
    for i in range(n):
        # Rule 1
        if data[i] > ucl:
            violations.append((i+1, "Rule 1", "UCL 초과", data[i]))
        elif data[i] < lcl:
            violations.append((i+1, "Rule 1", "LCL 미만", data[i]))
        # Rule 2
        if i >= 8:
            if all(data[i-j] > mean for j in range(9)):
                violations.append((i+1, "Rule 2", "연속 9개 점 중심선 위", data[i]))
            elif all(data[i-j] < mean for j in range(9)):
                violations.append((i+1, "Rule 2", "연속 9개 점 중심선 아래", data[i]))
        # Rule 3
        if i >= 5:
            inc = all(data[i-j] < data[i-j+1] for j in range(5,0,-1))
            dec = all(data[i-j] > data[i-j+1] for j in range(5,0,-1))
            if inc: violations.append((i+1, "Rule 3", "연속 6개 점 증가", data[i]))
            if dec: violations.append((i+1, "Rule 3", "연속 6개 점 감소", data[i]))
        # Rule 5
        if i >= 2:
            zone2u, zone2l = mean + 2*sigma, mean - 2*sigma
            count = sum(1 for j in range(3) if data[i-j] > zone2u or data[i-j] < zone2l)
            if count >= 2:
                violations.append((i+1, "Rule 5", "3개 중 2개가 2σ 밖", data[i]))
    return violations


# -------------------------------
# Hotelling T²
# -------------------------------
def calculate_hotelling_t2(X, mu, inv_cov):
    t2 = []
    for row in X:
        diff = row - mu
        t2.append(diff @ inv_cov @ diff.T)
    return np.array(t2)

def phaseII_ucl_t2(n, p, alpha=0.01):
    return (p * (n-1) * (n+1) / (n * (n-p))) * stats.f.ppf(1-alpha, p, n-p)


# -------------------------------
# Cp / Cpk
# -------------------------------
def calculate_cp_cpk(x, usl, lsl):
    mean = np.mean(x)
    std = np.std(x, ddof=1)
    cp = (usl - lsl) / (6 * std)
    cpu = (usl - mean) / (3 * std)
    cpl = (mean - lsl) / (3 * std)
    cpk = min(cpu, cpl)
    return cp, cpk, cpu, cpl, mean, std


# -------------------------------
# 안전한 datetime 변환
# -------------------------------
def to_datetime_safe(df):
    if "date" in df.columns and "time" in df.columns:
        return pd.to_datetime(df["date"] + " " + df["time"], errors="coerce")
    elif "timestamp" in df.columns:
        return pd.to_datetime(df["timestamp"], errors="coerce")
    return None
