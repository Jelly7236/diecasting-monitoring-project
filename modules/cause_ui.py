from shiny import ui
# ---- 공용 스타일 ----
MAXW = "max-width:1400px; margin:0 auto;"
ROW  = "display:flex; gap:16px; justify-content:space-between; flex-wrap:wrap; " + MAXW
SCARD = "flex:1 1 260px; min-width:240px;"
MCARD = "flex:1; min-width:560px;"
RIGHT = "flex:0.9; min-width:420px;"

def section(title: str, sub: str = ""):
    return ui.div(
        ui.h4(title, style="margin:0;"),
        (ui.p(sub, style="margin:4px 0 0 0; color:#6b7280;") if sub else ui.div()),
        style=f"{MAXW} padding:2px 4px;"
    )

def sticky_toolbar():
    return ui.div(
        ui.card(
            ui.div(
                ui.input_date("p_date", "기준일", value=None),
                ui.input_select("p_mold", "몰드", choices=[], multiple=False),
                ui.input_action_button("btn_update_date", "일자 업데이트"),
                ui.input_action_button("btn_apply", "적용", class_="btn-primary"),
                ui.download_button("btn_report", "리포트 다운로드"),
                style="display:flex; gap:12px; align-items:end; flex-wrap:wrap;"
            ),
            style="padding:12px;"
        ),
        style=f"position:sticky; top:0; z-index:8; background:white; {MAXW}"
    )

# ============================== 페이지 UI ===============================
def page_ui():
    return ui.page_fluid(
        ui.div(
            ui.h3("🎯 불량 원인 분석"),
            ui.p("상단: 몰드별 누적 카드 → 분석 설정 → p-관리도 & SHAP → 실제 불량 로그 → 변수/원인 분석",
                 style="color:#6b7280; margin-top:4px;"),
            style=MAXW
        ),

        ui.hr(),

        # 1) 몰드별 누적 카드
        section("몰드별 누적 현황", "각 카드: 누적 불량률 · 누적 이상 · 누적 관리도 이탈"),
        ui.div(
            ui.output_ui("mold_cards"),     # 서버에서 렌더
            style=ROW,
        ),

        ui.hr(),

        # 2) 분석 설정(스티키)
        section("분석 설정", "기준일과 몰드를 선택 후 [적용]을 눌러 갱신"),
        sticky_toolbar(),

        ui.hr(style=MAXW),

        # 3) 분석: p-관리도 + SHAP
        section("분석", "좌: p-관리도 / 우: SHAP 중요변수 기여도"),
        ui.div(
            ui.card(ui.card_header("📊 p-관리도"), ui.output_ui("p_chart"), style=MCARD),
            ui.card(ui.card_header("🔥 SHAP 중요변수 기여도"), ui.output_ui("shap_plot"), style=RIGHT),
            style=ROW,
        ),

        ui.hr(),

        # 4) 실제 불량 샘플 로그
        section("실제 불량 샘플 로그", "일시 | 몰드 | 순번 | 예측불량확률 | shap1 | shap2 | 변수상태 | 관리도 상태 | 이탈변수 | 이상탐지 | Anomaly Score | 임계값 이탈변수 | 이탈유형"),
        ui.card(ui.output_table("detect_log"), style=MAXW),

        ui.hr(),
    )
