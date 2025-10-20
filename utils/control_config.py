# ==================== 공정별 변수 정의 ====================
PROCESS_GROUPS = {
    "1) 용탕 준비 및 가열": ["molten_temp", "molten_volume"],
    "2) 반고체 슬러리 제조": ["sleeve_temperature", "EMS_operation_time"],
    "3) 사출 & 금형 충전": [
        "cast_pressure", "low_section_speed", "high_section_speed",
        "physical_strength", "biscuit_thickness"
    ],
    "4) 응고": [
        "upper_mold_temp1", "upper_mold_temp2",
        "lower_mold_temp1", "lower_mold_temp2",
        "Coolant_temperature"
    ]
}

FEATURES_ALL = [
    "molten_temp", "molten_volume", "sleeve_temperature", "EMS_operation_time",
    "cast_pressure", "low_section_speed", "high_section_speed", "physical_strength",
    "biscuit_thickness", "upper_mold_temp1", "upper_mold_temp2",
    "lower_mold_temp1", "lower_mold_temp2", "Coolant_temperature"
]

# 규격 한계
SPEC_LIMITS = {
    "molten_temp": {"usl": 750, "lsl": 650},
    "cast_pressure": {"usl": 370, "lsl": 250},
    "upper_mold_temp1": {"usl": 250, "lsl": 150},
    "sleeve_temperature": {"usl": 500, "lsl": 400},
    "Coolant_temperature": {"usl": 45, "lsl": 35},
    "physical_strength": {"usl": 750, "lsl": 600}
}
