# modules/page_control.py
from shiny import ui, render, reactive
import pandas as pd
import numpy as np
from shared import current_state
from utils.control_utils import (
    check_nelson_rules,
    calculate_hotelling_t2,
    phaseII_ucl_t2,
    to_datetime_safe,
)
from viz.multivar_chart import render_multivar_plot, render_multivar_table
from viz.univar_chart import make_univar_cards, make_univar_modal


# ==================== 공정별 변수 정의 ====================
PROCESS_GROUPS = {
    "1) 용탕 준비 및 가열": ["molten_temp", "molten_volume"],
    "2) 반고체 슬러리 제조": ["sleeve_temperature", "EMS_operation_time"],
    "3) 사출 & 금형 충전": [
        "cast_pressure",
        "low_section_speed",
        "high_section_speed",
        "physical_strength",
        "biscuit_thickness",
    ],
    "4) 응고": [
        "upper_mold_temp1",
        "upper_mold_temp2",
        "lower_mold_temp1",
        "lower_mold_temp2",
        "Coolant_temperature",
    ],
}

FEATURES_ALL = sum(PROCESS_GROUPS.values(), [])

# ✅ 몰드코드 고정 리스트 (전체 제거, 기본값 8412)
MOLD_CHOICES = ["8412", "8413", "8576", "8722", "8917"]


# ==================== UI ====================
def ui_control():
    return ui.page_fluid(
        ui.head_content(ui.tags.link(rel="stylesheet", href="/css/control.css")),

        ui.div(
            ui.h3("📊 공정 관리 상태 분석", class_="text-center mb-3"),

            # ⚙️ 분석 설정
            ui.card(
                ui.card_header("⚙️ 분석 설정"),
                ui.layout_columns(
                    ui.input_select(
                        "process_select",
                        "공정 선택",
                        choices={k: k for k in PROCESS_GROUPS.keys()},
                        selected=list(PROCESS_GROUPS.keys())[0],
                    ),
                    ui.input_select("mold", "몰드 선택", MOLD_CHOICES, selected="8412"),
                    ui.input_numeric(
                        "win", "윈도우(샘플 수)", 200, min=50, max=5000, step=50
                    ),
                    col_widths=[4, 4, 4],
                ),
            ),

            # 🔬 다변량 관리도
            ui.card(
                ui.card_header(ui.output_text("multivar_title")),
                ui.layout_columns(
                    ui.output_ui("t2_plot"),
                    ui.div(
                        ui.h5("📄 T² 이탈 로그", class_="mb-2"),
                        ui.div(ui.output_table("t2_table"), class_="scroll-table"),
                    ),
                    col_widths=[7, 5],
                ),
            ),

            # 📈 단변량 관리도
            ui.card(
                ui.card_header("📈 단변량 관리도 (클릭하여 상세 차트 보기)"),
                ui.output_ui("variable_cards"),
            ),

            # 🕒 전체 이탈 로그
            ui.card(
                ui.card_header("🕒 전체 이탈 로그 (단변량 + 다변량 통합)"),
                ui.div(
                    ui.output_table("timeline_table"),
                    class_="scroll-table",
                    style="max-height: 400px;",
                ),
            ),
            style="max-width: 1600px; margin: 0 auto; padding: 0 0.75rem;",
        ),
    )


# ==================== SERVER ====================
def server_control(input, output, session):
    # ==================== 데이터 뷰 ====================
    @reactive.calc
    def df_view():
        df = current_state().copy()
        if df is None or df.empty:
            return pd.DataFrame()

        if "id" in df:
            df = df.sort_values("id")
        df = df.tail(int(input.win()))

        mold_selected = input.mold()
        if "mold_code" in df:
            df = df[df["mold_code"].astype(str) == str(mold_selected)]

        if df.empty:
            return pd.DataFrame()

        dt = to_datetime_safe(df)
        df["__dt__"] = dt if dt is not None else pd.RangeIndex(len(df)).astype(float)
        return df.reset_index(drop=True)

    # ==================== 기준선 ====================
    @reactive.calc
    def df_baseline():
        df = current_state().copy()
        if df is None or df.empty:
            return None

        mold_selected = input.mold()
        if "mold_code" in df:
            df = df[df["mold_code"].astype(str) == str(mold_selected)]

        if df.empty:
            return None

        mask = (df["passorfail"] == 0) if "passorfail" in df else np.ones(len(df), bool)
        base = df.loc[mask, FEATURES_ALL].dropna()
        return None if len(base) < 50 else base

    # ==================== 다변량 관리도 ====================
    @output
    @render.text
    def multivar_title():
        # ✅ 데이터 유무와 상관없이 제목은 항상 유지
        process = input.process_select()
        var_list = PROCESS_GROUPS[process]
        mold = input.mold()
        return f"🔬 다변량 관리도 (Hotelling T²) - {process} [몰드 {mold}] [{', '.join(var_list)}]"

    @output
    @render.ui
    def t2_plot():
        df = df_view()
        if df.empty:
            # ✅ 데이터 없음 시 메시지만 표시
            return ui.p(
                "⚠️ 선택한 몰드코드에 해당하는 데이터가 없습니다.",
                style="color:#6b7280; text-align:center; padding:2rem;",
            )
        return render_multivar_plot(input, df_view, df_baseline, PROCESS_GROUPS)

    @output
    @render.table
    def t2_table():
        df = df_view()
        if df.empty:
            return pd.DataFrame({"상태": ["⚠️ 데이터 없음"]})
        return render_multivar_table(input, df_view, df_baseline, PROCESS_GROUPS)

    # ==================== 단변량 관리도 ====================
    @output
    @render.ui
    def variable_cards():
        df = df_view()
        if df.empty:
            return ui.p(
                "⚠️ 선택한 몰드코드에 데이터가 없습니다.",
                style="color:#6b7280; text-align:center; padding:2rem;",
            )
        return make_univar_cards(input, df_view, df_baseline, PROCESS_GROUPS)

    @reactive.effect
    @reactive.event(input.card_click)
    def _():
        df = df_view()
        if df.empty:
            ui.notification_show("데이터가 없습니다.", type="warning")
            return
        make_univar_modal(input, df_view, df_baseline)

    # ==================== 타임라인 ====================
    @output
    @render.table
    def timeline_table():
        df = df_view()
        if df.empty:
            return pd.DataFrame({"상태": ["⚠️ 선택한 몰드코드에 데이터가 없습니다."]})

        base = df_baseline()
        out_rows = []
        dtcol = "__dt__" if "__dt__" in df.columns else None

        # 단변량 로그
        for var in FEATURES_ALL:
            s = df[var].dropna()
            if len(s) < 10:
                continue
            if base is None or var not in base.columns or len(base) < 5:
                mu0, sd0 = s.mean(), s.std(ddof=1)
            else:
                mu0, sd0 = base[var].mean(), base[var].std(ddof=1)
            vio = check_nelson_rules(s.to_numpy(), mu0, mu0 + 3 * sd0, mu0 - 3 * sd0, sd0)
            for (idx, r, desc, val) in vio[-20:]:
                ts = df.iloc[s.index.min() + idx - 1][dtcol] if dtcol else np.nan
                out_rows.append(
                    {"시각": ts, "유형": "단변량", "변수": var, "룰": r, "설명": desc, "값": round(val, 3)}
                )

        if not out_rows:
            return pd.DataFrame({"상태": ["최근 이상 없음"]})
        timeline = pd.DataFrame(out_rows)
        if "시각" in timeline.columns and timeline["시각"].notna().any():
            timeline = timeline.sort_values("시각", ascending=False)
        return timeline.head(200)
