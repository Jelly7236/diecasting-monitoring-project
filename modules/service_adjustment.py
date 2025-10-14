# modules/service_adjustment.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


# ============================================
# 1) 설정값
# ============================================
ADJUSTMENT_STEP = {
    "molten_temp": 2.1,
    "molten_volume": 5.6,
    "sleeve_temperature": 14.5,
    "EMS_operation_time": 1.0,  # ✅ 추가
    "cast_pressure": 5.4,
    "biscuit_thickness": 3.8,
    "low_section_speed": 1.9,
    "high_section_speed": 2.3,
    "physical_strength": 5.4,
    "upper_mold_temp1": 7.3,
    "upper_mold_temp2": 4.1,
    "lower_mold_temp1": 8.3,
    "lower_mold_temp2": 6.7,
    "coolant_temp": 0.4,
    "facility_operation_cycleTime": 1.8,
    "production_cycletime": 2.1,
    "count": 10.0,  # ✅ 추가
}

GOOD_SAMPLE_MEANS = {
    "molten_temp": 720.2,
    "molten_volume": 88.9,
    "sleeve_temperature": 446.5,
    "EMS_operation_time": 15.0,  # ✅ 추가 (적절한 값으로 설정)
    "cast_pressure": 328.5,
    "biscuit_thickness": 49.9,
    "low_section_speed": 110.0,
    "high_section_speed": 112.7,
    "physical_strength": 701.9,
    "upper_mold_temp1": 184.9,
    "upper_mold_temp2": 163.3,
    "lower_mold_temp1": 202.5,
    "lower_mold_temp2": 196.3,
    "coolant_temp": 32.5,
    "facility_operation_cycleTime": 121.3,
    "production_cycletime": 122.7,
    "count": 100.0,  # ✅ 추가
}

DATA_RANGES = {
    "molten_temp": {"min": 70, "max": 750},
    "molten_volume": {"min": -1, "max": 600},
    "sleeve_temperature": {"min": 20, "max": 1000},
    "EMS_operation_time": {"min": 3, "max": 25},  # ✅ 추가
    "cast_pressure": {"min": 40, "max": 370},
    "biscuit_thickness": {"min": 0, "max": 450},
    "low_section_speed": {"min": 0, "max": 200},
    "high_section_speed": {"min": 0, "max": 400},
    "physical_strength": {"min": 0, "max": 750},
    "upper_mold_temp1": {"min": 10, "max": 400},
    "upper_mold_temp2": {"min": 10, "max": 250},
    "lower_mold_temp1": {"min": 10, "max": 400},
    "lower_mold_temp2": {"min": 10, "max": 550},
    "coolant_temp": {"min": 0, "max": 50},
    "facility_operation_cycleTime": {"min": 60, "max": 500},
    "production_cycletime": {"min": 60, "max": 500},
    "count": {"min": 1, "max": 1000},  # ✅ 추가
}

# 규칙 컷오프 (원본 변수명 기준)
CUTOFFS = {
    "low_section_speed": {"low": 100, "high": 114},
    "high_section_speed": {"low": 100},
    "coolant_temp": {"low": 20},
    "biscuit_thickness": {"low": 42, "high": 56},
    "sleeve_temperature": {"low": 128},
    "cast_pressure": {"low": 314},
    "upper_mold_temp1": {"low": 103},
    "upper_mold_temp2": {"low": 80},
    "lower_mold_temp1": {"low": 92},
    "lower_mold_temp2": {"low": 71},
}

# SHAP 변수명 → 원본 변수명 매핑 (Force Plot 기준 추가)
SHAP_TO_RAW_MAP = {
    "num__molten_temp": "molten_temp",
    "num__molten_volume": "molten_volume", 
    "num__sleeve_temperature": "sleeve_temperature",
    "num__EMS_operation_time": "EMS_operation_time",
    "num__cast_pressure": "cast_pressure",
    "num__biscuit_thickness": "biscuit_thickness",
    "num__low_section_speed": "low_section_speed",
    "num__high_section_speed": "high_section_speed",
    "num__physical_strength": "physical_strength",
    "num__upper_mold_temp1": "upper_mold_temp1",
    "num__upper_mold_temp2": "upper_mold_temp2",
    "num__lower_mold_temp1": "lower_mold_temp1",
    "num__lower_mold_temp2": "lower_mold_temp2",
    "num__coolant_temp": "coolant_temp",
    "num__facility_operation_cycleTime": "facility_operation_cycleTime",
    "num__production_cycletime": "production_cycletime",
    "num__count": "count",
    
    # Force Plot 한글명 매핑 추가
    "고속 구간 속도": "high_section_speed",
    "용탕 부피": "molten_volume",
    "상형 온도2(℃)": "upper_mold_temp2",
    "하형 온도1(℃)": "lower_mold_temp1",
    "상형 온도1(℃)": "upper_mold_temp1",
    "EMS 작동시간(s)": "EMS_operation_time",
}


# ============================================
# 2) 예측 함수 (원본 데이터 → 전처리 → 예측)
# ============================================
def predict_with_raw_data(raw_sample, preprocessor, model) -> float:
    """원본 데이터를 전처리 후 모델 예측"""
    if isinstance(raw_sample, pd.Series):
        raw_df = raw_sample.to_frame().T
    elif isinstance(raw_sample, pd.DataFrame):
        raw_df = raw_sample
    else:
        raise ValueError("raw_sample must be a pandas Series or DataFrame")

    # 전처리 수행
    transformed = preprocessor.transform(raw_df)
    prob = model.predict_proba(transformed)[0, 1]
    return float(prob)


# ============================================
# 3) Rule 기반 보정 (원본 값 기준)
# ============================================
def fix_rule_violations(raw_sample: pd.Series) -> Tuple[pd.Series, List[str]]:
    """원본 변수명 기준으로 Rule 기반 보정"""
    adjusted = raw_sample.copy()
    logs = []

    for var, cut in CUTOFFS.items():
        if var not in raw_sample:
            continue
        # ✅ FutureWarning 수정
        orig_val = float(raw_sample[var].iloc[0]) if hasattr(raw_sample[var], 'iloc') else float(raw_sample[var])
        rng = DATA_RANGES.get(var, {"min": -np.inf, "max": np.inf})

        if "low" in cut and orig_val < cut["low"]:
            new_val = max(cut["low"], rng["min"])
            adjusted[var] = new_val
            logs.append(f"{var}: {orig_val:.1f} → {new_val:.1f} (상향 조정)")
        if "high" in cut and orig_val > cut["high"]:
            new_val = min(cut["high"], rng["max"])
            adjusted[var] = new_val
            logs.append(f"{var}: {orig_val:.1f} → {new_val:.1f} (하향 조정)")

    return adjusted, logs


# ============================================
# 4) SHAP 기반 우선순위 (SHAP 변수명 → 원본 변수명 변환)
# ============================================
def calculate_priority(shap_values: Dict) -> List[Tuple[str, float, str]]:
    """SHAP 변수명을 원본 변수명으로 변환하여 우선순위 계산"""
    priorities = []
    
    print("🔍 SHAP 값 분석 (전체):")
    print(f"   총 {len(shap_values)}개 변수 수신")
    
    # 변수별 처리 (정렬하지 않고 모든 변수 확인)
    for shap_var, val in shap_values.items():
        # SHAP 변수명을 원본 변수명으로 변환
        raw_var = SHAP_TO_RAW_MAP.get(shap_var, shap_var)
        
        print(f"   {shap_var} → {raw_var}: {val:.4f} (절댓값: {abs(val):.4f})")
        
        # 매핑되지 않은 변수는 건너뛰기
        if raw_var not in ADJUSTMENT_STEP:
            print(f"      ⚠️ 스킵: (설정값 없음)")
            continue
        
        if abs(val) < 1e-6:  # 거의 0인 값은 제외
            print(f"      ⚠️ 스킵: 영향도 너무 낮음")
            continue
        
        if val > 0:
            # 양수: 불량률 증가 요인 → 변수값 감소 필요
            direction = "↓"
            priorities.append((raw_var, abs(val), direction))  # 절댓값으로 우선순위
            print(f"      ✅ 추가: {direction} (불량률↑ → 값 감소)")
        elif val < 0:
            # 음수: 불량률 감소 요인 → 변수값 증가 필요
            direction = "↑"
            priorities.append((raw_var, abs(val), direction))  # 절댓값으로 우선순위
            print(f"      ✅ 추가: {direction} (불량률↓ → 값 증가)")
    
    # 절댓값이 큰 순서대로 정렬 (불량 예측 기여도가 높은 순)
    sorted_priorities = sorted(priorities, key=lambda x: x[1], reverse=True)
    
    print(f"\n📊 최종 조정 우선순위 (절댓값 기준):")
    for i, (var, importance, dir) in enumerate(sorted_priorities[:10], 1):
        print(f"   {i}. {var}: {importance:.4f} ({dir})")
        
    return sorted_priorities


# ============================================
# 5) 메인 알고리즘 (R-SG) - 수정된 버전
# ============================================
def adjust_variables_to_target(
    raw_sample: pd.Series,
    shap_values: Dict,
    preprocessor,
    model,
    target_prob: float = 0.3,
    max_iterations: int = 10
) -> Dict:
    """
    R-SG 알고리즘: Rule 기반 + SHAP Greedy
    
    Args:
        raw_sample: 원본 입력 데이터 (사용자 입력값)
        shap_values: SHAP 값들 (전처리된 변수명 기준)
        preprocessor: 전처리기
        model: 모델
        target_prob: 목표 불량률
        max_iterations: 최대 반복 횟수
    """
    
    # ✅ SHAP Explanation 객체 자동 변환
    if not isinstance(shap_values, dict):
        if hasattr(shap_values, "values") and hasattr(shap_values, "feature_names"):
            vals = shap_values.values[0]
            if vals.ndim == 2 and vals.shape[1] == 2:  # 이진 분류 shap
                vals = vals[:, 1]
            shap_values = dict(zip(shap_values.feature_names, vals))
        else:
            raise ValueError("shap_values must be dict or shap.Explanation")

    # 초기 예측
    initial_prob = predict_with_raw_data(raw_sample, preprocessor, model)

    result = {
        "initial_prob": initial_prob,
        "target_prob": target_prob,
        "final_prob": initial_prob,
        "rule_adjustments": [],
        "shap_adjustments": [],
        "initial_sample": raw_sample.to_dict(),
        "final_sample": raw_sample.to_dict(),
        "success": False,
    }

    # ✅ Step 1: Rule 기반 보정 (원본 값 기준)
    print("🔧 Step 1: Rule 기반 보정 시작...")
    adjusted_raw, rule_logs = fix_rule_violations(raw_sample)
    prob_after_rule = predict_with_raw_data(adjusted_raw, preprocessor, model)
    
    result["rule_adjustments"] = rule_logs
    result["final_sample"] = adjusted_raw.to_dict()
    result["final_prob"] = prob_after_rule
    
    print(f"   Rule 보정 후: {prob_after_rule:.3f}")

    if prob_after_rule <= target_prob:
        print("✅ Rule 보정만으로 목표 달성!")
        result["success"] = True
        return result

    # ✅ Step 2: SHAP 기반 Greedy (SHAP → 원본 변수명 변환)
    print("🎯 Step 2: SHAP 기반 최적화 시작...")
    current_raw = adjusted_raw.copy()
    best_prob = prob_after_rule
    
    # SHAP 변수명을 원본 변수명으로 변환하여 우선순위 계산
    priority_list = calculate_priority(shap_values)
    print(f"   우선순위: {[f'{var}({dir})' for var, _, dir in priority_list[:5]]}")

    for var, importance, direction in priority_list:
        if var not in ADJUSTMENT_STEP or var not in DATA_RANGES:
            continue
            
        # ✅ Series 문제 해결: 명시적으로 스칼라 값 추출
        val = float(current_raw[var].iloc[0]) if hasattr(current_raw[var], 'iloc') else float(current_raw[var])
        step = ADJUSTMENT_STEP[var]
        mean_val = GOOD_SAMPLE_MEANS.get(var)
        rng = DATA_RANGES[var]

        best_val, val_now = val, val
        
        for iteration in range(max_iterations):
            if direction == "↑":
                new_val = val_now + step
                # 범위 제한
                new_val = min(new_val, rng["max"])
                # ✅ 양품 평균값 제한 제거 (혼란 방지)
                # if mean_val: new_val = min(new_val, mean_val)
            else:  # direction == "↓"
                new_val = val_now - step
                # 범위 제한
                new_val = max(new_val, rng["min"])
                # ✅ 양품 평균값 제한 제거 (혼란 방지)
                # if mean_val: new_val = max(new_val, mean_val)

            # 임시로 변수 변경하여 예측
            temp_raw = current_raw.copy()
            temp_raw[var] = new_val
            new_prob = predict_with_raw_data(temp_raw, preprocessor, model)

            if new_prob < best_prob:
                best_prob, best_val = new_prob, new_val
                val_now = new_val
                print(f"   {var}: {val:.1f} → {new_val:.1f} (확률: {new_prob:.3f}) {direction}")
            else:
                break

        # 실제 적용
        if best_val != val:
            current_raw[var] = best_val
            # ✅ 실제 변화 방향 확인
            actual_direction = "↑" if best_val > val else "↓"
            expected_direction = direction
            direction_match = "✅" if actual_direction == expected_direction else "❌"
            
            result["shap_adjustments"].append(
                f"{var}: {val:.1f} → {best_val:.1f} ({actual_direction}) {direction_match}"
            )
            print(f"   📝 최종: {var} {val:.1f} → {best_val:.1f} (예상:{expected_direction}, 실제:{actual_direction}) {direction_match}")

        result["final_prob"] = best_prob

        if best_prob <= target_prob:
            print(f"✅ 목표 달성! 최종 확률: {best_prob:.3f}")
            break

    result["final_sample"] = current_raw.to_dict()
    result["success"] = result["final_prob"] <= target_prob
    
    return result


# ============================================
# 6) 출력 요약
# ============================================
def print_adjustment_summary(result: Dict, feature_name_map: Dict = None):
    print("="*60)
    print("📊 불량률 개선 조정 가이드")
    print(f"초기: {result['initial_prob']:.1%} → 최종: {result['final_prob']:.1%} (목표 {result['target_prob']:.1%})")
    print("✅ 목표 달성!" if result["success"] else "⚠️ 목표 미달성")

    if result["rule_adjustments"]:
        print("\n🔧 Rule 기반 조정")
        for adj in result["rule_adjustments"]:
            var, change = adj.split(": ", 1)
            name = feature_name_map.get(var, var) if feature_name_map else var
            print(f"  • {name}: {change}")

    if result["shap_adjustments"]:
        print("\n🎯 SHAP 기반 조정")
        for adj in result["shap_adjustments"]:
            var, change = adj.split(": ", 1)
            name = feature_name_map.get(var, var) if feature_name_map else var
            print(f"  • {name}: {change}")
    print("="*60)


# ============================================
# 7) 사용 예시
# ============================================
def example_usage(preprocessor, model):
    """사용 예시"""
    # 원본 입력 데이터 (사용자 입력값)
    raw_sample = pd.Series({
        "cast_pressure": 350,
        "coolant_temp": 25,
        "low_section_speed": 105,
        "biscuit_thickness": 55,
    })

    # SHAP 값 (전처리된 변수명 기준)
    shap_values = {
        "num__cast_pressure": 0.25,
        "num__coolant_temp": -0.15,
        "num__low_section_speed": -0.05,
        "num__biscuit_thickness": 0.08,
    }

    result = adjust_variables_to_target(
        raw_sample, shap_values, preprocessor, model, target_prob=0.3
    )
    print_adjustment_summary(result)