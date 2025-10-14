# utils/data_state.py
from shiny import reactive
import pandas as pd

# 전역 센서 데이터 상태 (비어 있는 DataFrame)
sensor_data_state = reactive.Value(
    pd.DataFrame(columns=["time", "temperature", "pressure"])
)
