# utils/control_config.py

# ---- 공정/변수 설정 ----
SPEC_LIMITS = {
    "molten_temp":        {"usl": 750, "lsl": 650},
    "cast_pressure":      {"usl": 400, "lsl": 250},
    "upper_mold_temp1":   {"usl": 250, "lsl": 150},
    "sleeve_temperature": {"usl": 300, "lsl": 200},
    "Coolant_temperature":{"usl": 40,  "lsl": 15},
}

PROCESS_GROUPS = {
    "1️⃣ 용탕 준비/가열": ["molten_temp", "sleeve_temperature"],
    "2️⃣ 몰드 준비":      ["upper_mold_temp1", "Coolant_temperature"],
    "3️⃣ 사출/충전":      ["molten_temp", "cast_pressure"],
    "4️⃣ 응고":           ["upper_mold_temp1", "sleeve_temperature"],
}

FEATURES_ALL = [
    "molten_temp",
    "cast_pressure",
    "upper_mold_temp1",
    "sleeve_temperature",
    "Coolant_temperature",
]
