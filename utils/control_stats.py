# utils/control_stats.py
import numpy as np
import pandas as pd
from scipy import stats

# ---------- 유틸/통계 ----------
def check_nelson_rules(data, mean, ucl, lcl, sigma):
    violations = []
    n = len(data)
    for i in range(n):
        # Rule 1
        if data[i] > ucl: violations.append((i+1, "Rule 1", "UCL 초과", data[i]))
        elif data[i] < lcl: violations.append((i+1, "Rule 1", "LCL 미만", data[i]))
        # Rule 2
        if i >= 8:
            if all(data[i-j] > mean for j in range(9)):
                violations.append((i+1, "Rule 2", "연속 9개 중심선 위", data[i]))
            elif all(data[i-j] < mean for j in range(9)):
                violations.append((i+1, "Rule 2", "연속 9개 중심선 아래", data[i]))
        # Rule 3
        if i >= 5:
            inc = all(data[i-j] < data[i-j+1] for j in range(5, 0, -1))
            dec = all(data[i-j] > data[i-j+1] for j in range(5, 0, -1))
            if inc: violations.append((i+1, "Rule 3", "연속 6개 증가 추세", data[i]))
            elif dec: violations.append((i+1, "Rule 3", "연속 6개 감소 추세", data[i]))
        # Rule 5
        if i >= 2:
            z2u, z2l = mean + 2*sigma, mean - 2*sigma
            cnt = sum(1 for j in range(3) if data[i-j] > z2u or data[i-j] < z2l)
            if cnt >= 2: violations.append((i+1, "Rule 5", "3개 중 2개 2σ 영역 밖", data[i]))
    return violations

def calculate_hotelling_t2(X, mu, inv_cov):
    diff = X - mu
    return np.einsum("ij,jk,ik->i", diff, inv_cov, diff)

def phaseII_ucl_t2(n, p, alpha=0.01):
    return (p*(n-1)*(n+1)/(n*(n-p))) * stats.f.ppf(1-alpha, p, n-p)

def calculate_cp_cpk(data, usl, lsl):
    data = np.asarray(data)
    if len(data) < 5 or np.allclose(np.std(data, ddof=1), 0):
        return (0, 0, 0, 0, np.nan, np.nan)
    mean = np.mean(data)
    std  = np.std(data, ddof=1)
    cp   = (usl - lsl) / (6 * std) if std > 0 else 0
    cpu  = (usl - mean) / (3 * std) if std > 0 else 0
    cpl  = (mean - lsl) / (3 * std) if std > 0 else 0
    cpk  = min(cpu, cpl)
    return cp, cpk, cpu, cpl, mean, std

def to_datetime_safe(df):
    if "date" in df and "time" in df:
        try:
            return pd.to_datetime(df["date"] + " " + df["time"], errors="coerce")
        except Exception:
            pass
    return None
