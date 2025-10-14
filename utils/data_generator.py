import pandas as pd
import numpy as np
from datetime import datetime

def generate_sensor_data():
    """온도·압력 등 센서의 더미 데이터 생성"""
    now = datetime.now()
    times = pd.date_range(now, periods=30, freq="s")
    temp = 200 + np.random.randn(30).cumsum() * 0.5
    press = 50 + np.random.randn(30).cumsum() * 0.2
    return pd.DataFrame({"time": times, "temperature": temp, "pressure": press})
