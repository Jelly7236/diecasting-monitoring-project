import pandas as pd
import numpy as np
from scipy import stats

# ==================== 통계 함수 ====================

def check_nelson_rules(data, mean, ucl, lcl, sigma):
    violations = []
    n = len(data)

    for i in range(n):
        if data[i] > ucl:
            violations.append((i+1, "Rule 1", "UCL 초과", data[i]))
        elif data[i] < lcl:
            violations.append((i+1, "Rule 1", "LCL 미만", data[i]))

        if i >= 8:
            if all(data[i-j] > mean for j in range(9)):
                violations.append((i+1, "Rule 2", "연속 9개 점이 중심선 위", data[i]))
            elif all(data[i-j] < mean for j in range(9)):
                violations.append((i+1, "Rule 2", "연속 9개 점이 중심선 아래", data[i]))

        if i >= 5:
            increasing = all(data[i-j] < data[i-j+1] for j in range(5, 0, -1))
            decreasing = all(data[i-j] > data[i-j+1] for j in range(5, 0, -1))
            if increasing:
                violations.append((i+1, "Rule 3", "연속 6개 점 증가 추세", data[i]))
            elif decreasing:
                violations.append((i+1, "Rule 3", "연속 6개 점 감소 추세", data[i]))

        if i >= 2:
            zone2_upper = mean + 2*sigma
            zone2_lower = mean - 2*sigma
            count = sum(1 for j in range(3)
                        if data[i-j] > zone2_upper or data[i-j] < zone2_lower)
            if count >= 2:
                violations.append((i+1, "Rule 5", "3개 중 2개가 2σ 영역 밖", data[i]))

    return violations


def calculate_hotelling_t2(data_matrix, mean_vector, inv_cov):
    t2_values = []
    for i in range(len(data_matrix)):
        diff = data_matrix[i] - mean_vector
        t2 = diff @ inv_cov @ diff.T
        t2_values.append(t2)
    return np.array(t2_values)


def phaseII_ucl_t2(n, p, alpha=0.01):
    return (p * (n-1) * (n+1) / (n * (n-p))) * stats.f.ppf(1-alpha, p, n-p)


def to_datetime_safe(df):
    if "tryshot_time" in df.columns:
        return pd.to_datetime(df["tryshot_time"], errors='coerce')
    return None
