# utils/data_updater.py
from shiny import reactive
import pandas as pd
from utils.data_state import sensor_data_state
from utils.data_generator import generate_sensor_row


def update_sensor_data(interval_ms: int = 1000, max_rows: int = 200):
    """interval_ms 간격으로 sensor_data_state를 업데이트"""
    @reactive.effect
    def _update():
        reactive.invalidate_later(interval_ms)
        old_df = sensor_data_state.get()

        # ✅ 비어있는 DataFrame인 경우 바로 새 행으로 대체
        if old_df is None or old_df.empty:
            sensor_data_state.set(generate_sensor_row())
            return

        new_row = generate_sensor_row()

        # ✅ 안전한 concat (빈 DF 무시)
        updated = (
            pd.concat([old_df, new_row], ignore_index=True)
            .tail(max_rows)
            .reset_index(drop=True)
        )

        sensor_data_state.set(updated)

    return _update
