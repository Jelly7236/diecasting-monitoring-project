# modules/cause_ui.py
from shiny import ui

# ---- 공용 스타일 ----
MAXW  = "max-width:1400px; margin:0 auto;"
ROW   = "display:flex; gap:16px; justify-content:space-between; flex-wrap:wrap; " + MAXW
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
    # p_mold는 서버에서 렌더(ui.output_ui("p_mold_ui")) – “전체” 포함
    return ui.div(
        ui.card(
            ui.div(
                # 날짜: 시작/끝 + 최신일자 버튼
                ui.input_date("p_start", "시작일", value=None, min="2019-03-01", max="2019-03-12"),
                ui.input_date("p_end",   "종료일", value=None, min="2019-03-01", max="2019-03-12"),
                # 몰드 셀렉트(서버에서 렌더)
                ui.output_ui("p_mold_ui"),
                # 동작 버튼
                ui.input_action_button("btn_update_date", "최신 일자"),
                ui.input_action_button("btn_apply", "적용", class_="btn-primary"),
                ui.input_action_button("btn_pdf", "보고서(PDF) 다운로드"),
                style="display:flex; gap:12px; align-items:end; flex-wrap:wrap;"
            ),
            ui.div(
                ui.output_text("sel_summary"),
                style="margin-top:6px; color:#6b7280;"
            ),
            style="padding:12px;"
        ),
        style=f"position:sticky; top:0; z-index:8; background:white; {MAXW}"
    )

# ============================== 페이지 UI ===============================
def page_ui():
    pdf_support = [
        ui.tags.style("""
#btn_update_date, #btn_apply, #btn_pdf {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 4px;
    padding: 8px 16px;
    border-radius: 6px;
    border: 1px solid transparent;
    font-weight: 600;
    font-size: 14px;
    line-height: 1.2;
    transition: all 0.15s ease-in-out;
}
#btn_update_date { background-color: #4B5563; color: #fff; }
#btn_apply { background-color: #2563EB; color: #fff; }
#btn_pdf { background-color: #047857; color: #fff; }
#btn_update_date:hover, #btn_apply:hover, #btn_pdf:hover {
    filter: brightness(0.92);
}
#btn_update_date:disabled, #btn_apply:disabled, #btn_pdf:disabled {
    opacity: 0.7;
    cursor: not-allowed;
}
        """),
        ui.tags.script(src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"),
        ui.tags.script(src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"),
        ui.tags.script("""
(function(){
    function bindPdfButton(){
        var btn = document.getElementById('btn_pdf');
        if(!btn || btn.dataset.pdfBound === '1'){
            return;
        }
        btn.dataset.pdfBound = '1';
        btn.addEventListener('click', function(){
            var target = document.getElementById('cause-tab-root') || document.body;
            if(!window.html2canvas || !window.jspdf){
                alert('PDF 도구 로드 중입니다. 잠시 후 다시 시도하세요.');
                return;
            }
            var originalText = btn.innerText;
            btn.disabled = true;
            btn.innerText = 'PDF 생성 중...';
            html2canvas(target, {background:'#ffffff', scale:2, scrollY:-window.scrollY}).then(function(canvas){
                var imgData = canvas.toDataURL('image/png');
                var pdf = new window.jspdf.jsPDF('p','pt','a4');
                var pageWidth = pdf.internal.pageSize.getWidth();
                var pageHeight = pdf.internal.pageSize.getHeight();
                var imgWidth = pageWidth;
                var imgHeight = canvas.height * imgWidth / canvas.width;
                var heightLeft = imgHeight - pageHeight;
                pdf.addImage(imgData, 'PNG', 0, 0, imgWidth, imgHeight);
                while(heightLeft > 0){
                    pdf.addPage();
                    pdf.addImage(imgData, 'PNG', 0, -heightLeft, imgWidth, imgHeight);
                    heightLeft -= pageHeight;
                }
                pdf.save('불량원인_보고서.pdf');
            }).catch(function(err){
                console.error(err);
                alert('PDF 생성 중 오류가 발생했습니다.');
            }).finally(function(){
                btn.disabled = false;
                btn.innerText = originalText;
            });
        });
    }
    if(document.readyState !== 'loading'){
        bindPdfButton();
    } else {
        document.addEventListener('DOMContentLoaded', bindPdfButton);
    }
    var observer = new MutationObserver(bindPdfButton);
    observer.observe(document.body, {childList:true, subtree:true});
})();
        """)
    ]

    content = [
        # 타이틀
        ui.div(
            ui.h3("불량 원인 분석"),
            ui.p(
                "상단: 몰드별 누적 현황 카드 → 분석 설정 → p-관리도 & SHAP → 변수별 관계분석 → 실제 불량 샘플 로그",
                style="color:#6b7280; margin-top:4px;",
            ),
            style=MAXW,
        ),

        ui.hr(),

        # 1) 몰드별 누적 카드 (기간 반영)
        section("몰드별 누적 현황", "선택한 기간 기준 누적 불량률 / 누적 불량 건수"),
        ui.div(
            ui.output_ui("mold_cards"),   # 서버(server_cause)에서 기간 반영 카드 생성
            style=ROW,
        ),

        ui.hr(),

        # 2) 분석 설정(스티키)
        section("분석 설정", "시작일 종료일과 몰드를 선택 후 [적용]을 눌러 갱신 / 최신 일자를 눌러 현재 시간까지 갱신"),
        sticky_toolbar(),

        ui.hr(style=MAXW),

        # 3) 분석: p-관리도 + SHAP
        section("분석", "좌: p-관리도 / 우: SHAP 중요변수 기여도"),
        ui.div(
            ui.card(ui.card_header("p-관리도"), ui.output_ui("p_chart"), style=MCARD),
            ui.card(ui.card_header("SHAP 중요변수 기여도"), ui.output_ui("shap_plot"), style=RIGHT),
            style=ROW,
        ),

        ui.hr(),

        # 4) 변수별 관계분석 (표 + 그래프)
        section(
            "변수별 관계분석",
            "이탈변수/SHAP 기준 ‘변수+상태’ 사건횟수 Top 5 — (원인 분석 횟수 = SHAP횟수 + HIGH횟수 + LOW횟수)"
        ),
        ui.div(
            ui.card(ui.card_header("순위표 · Top 5"), ui.output_ui("var_rel_table"), style=MCARD),
            ui.card(ui.card_header("막대 그래프 · Top 5"), ui.output_ui("var_rel_bar"), style=RIGHT),
            style=ROW,
        ),

        ui.hr(),

        # 5) 실제 불량 샘플 로그 + CSV 다운로드
        section("실제 불량 샘플 로그", "기간·몰드 필터된 사용자 CSV 기준으로 표시"),
        ui.card(
            ui.output_ui("detect_log"),
            ui.div(
                ui.download_button("btn_report", "CSV 다운로드"),
                style="padding:6px 0 0 6px; text-align:left;"
            ),
            style=MAXW
        ),

        ui.hr(),
    ]

    return ui.page_fluid(
        *pdf_support,
        ui.div(*content, id="cause-tab-root")
    )
