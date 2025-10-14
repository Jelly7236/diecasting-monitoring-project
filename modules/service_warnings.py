# modules/service_warnings.py
from shiny import ui
import numpy as np
from shared import feature_name_map, feature_name_map_kor

# -----------------------------------
# 1) Cut-off 기준 정의
# -----------------------------------
CUTOFFS = {
    "num__low_section_speed": {"low": 100, "high": 114},
    "num__high_section_speed": {"low": 100},
    "num__coolant_temp": {"low": 20},
    "num__biscuit_thickness": {"low": 42, "high": 56},
    "num__sleeve_temperature": {"low": 128},
    "num__cast_pressure": {"low": 314},
    "num__upper_mold_temp1": {"low": 103},
    "num__upper_mold_temp2": {"low": 80},
    "num__lower_mold_temp1": {"low": 92},
    "num__lower_mold_temp2": {"low": 71},
}

# -----------------------------------
# 2) 실제 데이터 범위 (CSV에서 추출)
# -----------------------------------
DATA_RANGES = {
    "num__production_cycletime": {"min": 60, "max": 500},
    "num__facility_operation_cycleTime": {"min": 60, "max": 500},
    "num__molten_volume": {"min": -1, "max": 600},
    "num__molten_temp": {"min": 70, "max": 750},
    "num__sleeve_temperature": {"min": 20, "max": 1000},
    "num__cast_pressure": {"min": 40, "max": 370},
    "num__biscuit_thickness": {"min": 0, "max": 450},
    "num__low_section_speed": {"min": 0, "max": 200},
    "num__high_section_speed": {"min": 0, "max": 400},
    "num__physical_strength": {"min": 0, "max": 750},
    "num__upper_mold_temp1": {"min": 10, "max": 400},
    "num__upper_mold_temp2": {"min": 10, "max": 250},
    "num__lower_mold_temp1": {"min": 10, "max": 400},
    "num__lower_mold_temp2": {"min": 10, "max": 550},
    "num__coolant_temp": {"min": 10, "max": 50},  # Coolant_temperature -> coolant_temp
}

# 공정별 변수 매핑
PROCESS_VARS = {
    "molten": ["num__molten_temp", "num__molten_volume"],
    "slurry": ["num__sleeve_temperature", "num__EMS_operation_time"],
    "injection": ["num__cast_pressure", "num__low_section_speed",
                  "num__high_section_speed", "num__biscuit_thickness"],
    "solidify": ["num__upper_mold_temp1", "num__upper_mold_temp2",
                 "num__lower_mold_temp1", "num__lower_mold_temp2", "num__coolant_temp"],
    "overall": [
        "num__facility_operation_cycleTime",
        "num__production_cycletime",
        "num__count",
        "cat__working_가동",
        "cat__working_정지",
        "cat__tryshot_signal_A",
        "cat__tryshot_signal_D",
    ],
}

# -----------------------------------
# 2) 개선된 정규화 함수들
# -----------------------------------
def normalize_shap_contribution(shap_value, prediction_prob):
    """SHAP 기여도를 예측 확률 대비 비율로 정규화 (0~1)"""
    if prediction_prob <= 0 or prediction_prob > 1:
        return 0.0
    
    positive_contrib = max(0, shap_value)
    normalized = positive_contrib / prediction_prob
    
    return min(normalized, 1.0)

def normalize_rule_severity(var, value):
    """규칙 위반도를 데이터 범위 대비 비율로 정규화 (0~1)"""
    if var not in CUTOFFS or var not in DATA_RANGES:
        return 0.0
    
    cut = CUTOFFS[var]
    data_range = DATA_RANGES[var]
    severity = 0.0
    
    if "low" in cut and value < cut["low"]:
        denominator = cut["low"] - data_range["min"]
        if denominator > 0:
            severity += (cut["low"] - value) / denominator
    
    if "high" in cut and value > cut["high"]:
        denominator = data_range["max"] - cut["high"]
        if denominator > 0:
            severity += (value - cut["high"]) / denominator
    
    return min(severity, 1.0)

# -----------------------------------
# 3) 경고 메시지 생성 함수
# -----------------------------------
def shap_based_warning(process: str,
                       shap_values_state,
                       X_input_state,
                       X_input_raw,
                       feature_name_map,
                       pred_state=None):
    """SHAP + Cut-off 융합 기반 공정별 경고 메시지"""
    shap_values = shap_values_state.get()
    X = X_input_state.get()
    X_raw = X_input_raw.get()
    # print("-------------------------------------1")
    # print(X)
    # print("-------------------------------------2")
    # print(X_raw)
    pred = pred_state.get() if pred_state is not None else None

    # 값 없는 경우
    if shap_values is None or X is None:
        return {
            "header": ui.div(
                ui.p("⚠️ SHAP 계산 불가"),
                class_="text-center text-white",
                style="background-color:#6c757d; border-radius:6px; font-weight:600; padding:0.8rem;"
            ),
            "details": ui.div(ui.p("데이터가 없습니다.", style="font-size:1.5rem;"))
        }

    # SHAP 기여도 추출
    if hasattr(shap_values, "values"):
        vals = shap_values.values[0]
        if vals.ndim == 2 and vals.shape[1] == 2:
            vals = vals[:, 1]
        contrib = dict(zip(X.columns, vals))
    else:
        return {
            "header": ui.div(
                ui.p("⚠️ SHAP 형식 오류"),
                class_="text-center text-white",
                style="background-color:#6c757d; border-radius:6px; font-weight:600; padding:0.8rem;"
            ),
            "details": ui.div(ui.p("SHAP 값을 읽을 수 없습니다.", style="font-size:1.5rem;"))
        }

    key_vars = PROCESS_VARS.get(process, [])
    prediction_prob = float(pred) if isinstance(pred, (int, float)) and pred > 0 else 0.8

    # A. SHAP 신호 처리
    shap_normalized = {}
    shap_raw_values = {}
    for v in key_vars:
        if v in contrib:
            shap_raw = contrib[v]
            shap_raw_values[v] = shap_raw
            shap_norm = normalize_shap_contribution(shap_raw, prediction_prob)
            shap_normalized[v] = shap_norm
        else:
            shap_raw_values[v] = 0.0
            shap_normalized[v] = 0.0
    
    shap_values_list = list(shap_normalized.values())
    shap_max = max(shap_values_list) if shap_values_list else 0.0
    shap_avg = sum(shap_values_list) / max(len(key_vars), 1)
    shap_score = 0.7 * shap_max + 0.3 * shap_avg

    # # B. Rule 신호 처리
    # rule_normalized = {}
    # current_values = {}
    # for v in key_vars:
    #     if v in X.columns:  # 여기 수정해야됨
    #         current_value = float(X.iloc[0][v])  # 여기 수정해야됨
    #         current_values[v] = current_value
    #         rule_norm = normalize_rule_severity(v, current_value)
    #         rule_normalized[v] = rule_norm
    #     else:
    #         current_values[v] = None
    #         rule_normalized[v] = 0.0
    
    # B. Rule 신호 처리 (원본 데이터 사용)
    rule_normalized = {}
    current_values = {}
    for v in key_vars:
        # 전처리 컬럼명(num__xxx, cat__xxx)을 원본 컬럼명으로 변환
        raw_col_name = v.replace("num__", "").replace("cat__", "")
        
        if raw_col_name in X_raw.columns:
            current_value = float(X_raw.iloc[0][raw_col_name])
            current_values[v] = current_value
            rule_norm = normalize_rule_severity(v, current_value)
            rule_normalized[v] = rule_norm
        else:
            current_values[v] = None
            rule_normalized[v] = 0.0
    
    rule_values_list = list(rule_normalized.values())
    rule_max = max(rule_values_list) if rule_values_list else 0.0
    rule_avg = sum(rule_values_list) / max(len(key_vars), 1)
    rule_score = 0.7 * rule_max + 0.3 * rule_avg
    
    
    
    
    
    
    
    

    # C. 통합 스코어
    w_shap, w_rule = 0.5, 0.5
    proc_score = w_shap * shap_score + w_rule * rule_score

    # D. 의사결정
    HIGH_THRESHOLD = 0.15
    
    if shap_score > HIGH_THRESHOLD and rule_score > HIGH_THRESHOLD:
        color, header = "#ff5733", "⚡ 강한 원인 후보"
    elif shap_score > HIGH_THRESHOLD:
        color, header = "#ffc107", "⚠️ 모델 신호 경고 (관찰 필요)"
    elif rule_score > HIGH_THRESHOLD:
        color, header = "#fd7e14", "⚠️ 기준치 초과 (관찰 필요)"
    else:
        color, header = "#198754", "✅ 이상 없음"

    # E. 변수별 통합 분석
    var_combined = {}
    total_shap = sum(shap_normalized.values()) + 1e-6
    total_rule = sum(rule_normalized.values()) + 1e-6
    
    for v in key_vars:
        shap_relative = shap_normalized[v] / total_shap
        rule_relative = rule_normalized[v] / total_rule
        combined_relative = w_shap * shap_relative + w_rule * rule_relative
        
        var_combined[v] = {
            'combined_relative': combined_relative,
            'shap_normalized': shap_normalized[v],
            'rule_normalized': rule_normalized[v],
            'shap_raw': shap_raw_values.get(v, 0.0),
            'shap_relative': shap_relative,
            'rule_relative': rule_relative,
            'current_value': current_values.get(v)
        }
    
    top_vars = sorted(var_combined.items(), 
                     key=lambda x: x[1]['combined_relative'], 
                     reverse=True)[:3]

    # ====================================
    # F. 상세 분석 UI 생성
    # ====================================
    detail_sections = []

    for v, scores in top_vars:
        if scores['combined_relative'] > 0.05:
            pretty = feature_name_map.get(v, v)
            current_value = scores['current_value']
            shap_abs = scores['shap_normalized']
            rule_abs = scores['rule_normalized']
            
            # 신호 타입 결정
            if shap_abs > 0.1 and rule_abs > 0.1:
                signal_badge = "🔴 두 신호 모두 감지"
                badge_color = "#dc3545"
            elif shap_abs > 0.1:
                signal_badge = "🟡 모델 신호"
                badge_color = "#ffc107"
            elif rule_abs > 0.1:
                signal_badge = "🟠 기준치 신호"
                badge_color = "#fd7e14"
            else:
                signal_badge = "⚪ 약한 신호"
                badge_color = "#6c757d"
            
            # ✅ SHAP 기여도 해석 (개선됨)
            shap_raw = scores['shap_raw']
            
            # 변수 방향성 판단
            if current_value is not None and v in DATA_RANGES:
                data_range = DATA_RANGES[v]
                median_value = (data_range["min"] + data_range["max"]) / 2
                direction = "높아서" if current_value > median_value else "낮아서"
            else:
                direction = "변화로 인해"
            
            # 불량 확률 기여도 계산
            prob_contribution = shap_raw * 100
            
            # 기여도 해석
            if shap_abs > 0.3:
                shap_impact = f"매우 높음 - 해당 변수가 {direction} 불량 확률을 약 {abs(prob_contribution):.1f}%p 상승시켰습니다."
                impact_color = "#dc3545"
            elif shap_abs > 0.15:
                shap_impact = f"높음 - 해당 변수가 {direction} 불량 확률을 약 {abs(prob_contribution):.1f}%p 상승시켰습니다."
                impact_color = "#fd7e14"
            elif shap_abs > 0.05:
                shap_impact = f"보통 - 해당 변수가 {direction} 불량 확률을 약 {abs(prob_contribution):.1f}%p 상승시켰습니다."
                impact_color = "#ffc107"
            elif shap_abs > 0.01:
                shap_impact = f"낮음 - 해당 변수가 {direction} 약간의 영향을 주었으나 미미합니다 ({abs(prob_contribution):.2f}%p)."
                impact_color = "#28a745"
            else:
                shap_impact = "해당 변수는 불량에 거의 영향을 미치지 않습니다."
                impact_color = "#6c757d"
            
            # 기준치 이탈 분석
            cutoff_details = []
            if v in CUTOFFS and current_value is not None:
                cut = CUTOFFS[v]
                
                if "low" in cut:
                    if current_value < cut["low"]:
                        diff = cut["low"] - current_value
                        pct = (diff / cut["low"]) * 100
                        cutoff_details.append(
                            f"• 하한 기준({cut['low']:.1f})보다 {diff:.1f} 낮음 ({pct:.1f}% 이탈) ❌"
                        )
                    else:
                        cutoff_details.append(f"• 하한 기준({cut['low']:.1f}) 이상 ✅")
                
                if "high" in cut:
                    if current_value > cut["high"]:
                        diff = current_value - cut["high"]
                        pct = (diff / cut["high"]) * 100
                        cutoff_details.append(
                            f"• 상한 기준({cut['high']:.1f})보다 {diff:.1f} 높음 ({pct:.1f}% 이탈) ❌"
                        )
                    else:
                        cutoff_details.append(f"• 상한 기준({cut['high']:.1f}) 이하 ✅")
            
            if not cutoff_details:
                cutoff_details.append("• 기준치 설정 없음")
            
            # 개별 변수 카드 생성
            var_card = ui.div(
                # 변수명 헤더
                ui.div(
                    ui.h5(pretty, style="margin:0; font-weight:700; color:#212529;"),
                    ui.span(signal_badge, 
                        style=f"background-color:{badge_color}; color:white; padding:4px 12px; border-radius:12px; font-size:0.85rem; font-weight:600;"),
                    style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem; padding-bottom:0.5rem; border-bottom:2px solid #dee2e6;"
                ),
                
                # 점수 정보
                ui.div(
                    ui.div(
                        ui.strong("📊 점수 정보"),
                        style="font-size:1.1rem; margin-bottom:0.5rem; color:#495057;"
                    ),
                    ui.p(f"• SHAP 점수 (정규화): {shap_abs:.4f} ({scores['shap_relative']*100:.1f}%)", 
                        style="margin:0.3rem 0; font-size:1rem;"),
                    ui.p(f"• SHAP 원본값: {shap_raw:.4f}", 
                        style="margin:0.3rem 0; font-size:0.9rem; color:#6c757d;"),
                    ui.p(f"• Rule 점수 (정규화): {rule_abs:.4f} ({scores['rule_relative']*100:.1f}%)", 
                        style="margin:0.3rem 0; font-size:1rem;"),
                    ui.p(f"• 현재 입력값: {current_value:.2f}" if current_value is not None else "• 현재 값: N/A", 
                        style="margin:0.3rem 0; font-size:1rem; font-weight:600; color:#0d6efd;"),
                    style="margin-bottom:1rem; padding:0.8rem; background-color:#f8f9fa; border-radius:8px;"
                ),
                
                # SHAP 기여도 해석
                ui.div(
                    ui.div(
                        ui.strong("불량 기여도 분석 (SHAP)"),
                        style="font-size:1.1rem; margin-bottom:0.5rem; color:#495057;"
                    ),
                    ui.p(shap_impact, 
                        style=f"margin:0; font-size:1rem; color:{impact_color}; font-weight:600;"),
                    # ui.p(f"이 변수는 전체 예측에서 상대적으로 {scores['shap_relative']*100:.1f}%의 영향력을 가집니다.", 
                    #     style="margin-top:0.5rem; font-size:0.9rem; color:#6c757d;"),
                    style="margin-bottom:1rem; padding:0.8rem; background-color:#fff3cd; border-radius:8px; border-left:4px solid #ffc107;"
                ),
                
                # 기준치 이탈 분석
                ui.div(
                    ui.div(
                        ui.strong("기준치 대비 상태 (Rule)"),
                        style="font-size:1.1rem; margin-bottom:0.5rem; color:#495057;"
                    ),
                    *[ui.p(detail, style="margin:0.3rem 0; font-size:1rem;") for detail in cutoff_details],
                    style="margin-bottom:1rem; padding:0.8rem; background-color:#e7f3ff; border-radius:8px; border-left:4px solid #0d6efd;"
                ),
                
                style="padding:1.5rem; margin-bottom:1.5rem; background-color:white; border-radius:12px; box-shadow:0 2px 8px rgba(0,0,0,0.1);"
            )
            
            detail_sections.append(var_card)

    if not detail_sections:
        detail_sections.append(
            ui.div(
                ui.p("✅ 모든 변수가 정상 범위 내에 있습니다.", 
                    style="font-size:1.3rem; text-align:center; color:#198754; font-weight:600;"),
                style="padding:2rem;"
            )
        )
    
    # ====================================
    # G. UI 반환
    # ====================================
    header_ui = ui.div(
        ui.h6(f"{header}", ui.br(), f"(Score={proc_score:.2f})", class_="mb-2"),
        class_="text-white text-center",
        style=f"background-color:{color}; border-radius:6px; font-weight:600; padding:0.8rem;"
    )

    details_ui = ui.div(
        # 전체 요약
        ui.div(
            ui.h4("종합 분석", style="margin-bottom:1rem; color:#212529;"),
            ui.div(
                ui.p(f"• SHAP 신호: {shap_score:.3f}", style="margin:0.3rem 0; font-size:1.1rem;"),
                ui.p(f"• Rule 신호: {rule_score:.3f}", style="margin:0.3rem 0; font-size:1.1rem;"),
                style="padding:1rem;"
            ),
            style="padding:1rem; background-color:#f8f9fa; border-radius:8px; margin-bottom:1.5rem;"
        ),
        
        # 개별 변수 상세
        ui.h4("주요 변수 상세 분석", style="margin-bottom:1rem; color:#212529;"),
        *detail_sections,
        
        style="max-height:70vh; overflow-y:auto;"
    )

    return {"header": header_ui, "details": details_ui}

# -----------------------------------
# 4) 데이터 범위 업데이트 함수
# -----------------------------------
def update_data_ranges(new_ranges):
    """외부에서 실제 데이터 범위를 업데이트하는 함수"""
    global DATA_RANGES
    DATA_RANGES.update(new_ranges)
    print(f"데이터 범위 업데이트 완료: {len(new_ranges)}개 변수")