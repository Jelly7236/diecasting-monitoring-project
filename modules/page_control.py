# modules/page_control.py
from shiny import ui, render, reactive
import pandas as pd
import numpy as np
import io
import os
import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

from shared import current_state
from utils.control_utils import (
    check_nelson_rules,
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
MOLD_CHOICES = ["8412", "8413", "8576", "8722", "8917"]


# ==================== UI ====================
def ui_control():
    return ui.page_fluid(
        ui.head_content(ui.tags.link(rel="stylesheet", href="/css/control.css"),
                        ui.tags.link(rel="stylesheet", href="/css/control_enhanced.css")),
        
        # 헤더
        ui.div(
            ui.h2("공정 관리 상태 분석", class_="title"),
            # ui.div(
            #     ui.h4("전자교반 3라인 2호기 TM Carrier RH", class_="machine"),
            #     ui.div(ui.output_ui("working_badge"), ui.output_ui("tryshot_badge"), class_="badge-row"),
            #     class_="machine-row",
            # ),
            class_="header",
        ),
        
        ui.div(
            # ui.h3("📊 공정 관리 상태 분석", class_="text-center mb-3"),

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

            # 🔬 다변량 관리도 + 보고서 다운로드 버튼
            ui.card(
                ui.card_header(
                    ui.div(
                        {
                            "style": "display:flex; justify-content:space-between; align-items:center;"
                        },
                        ui.h4("🔬 다변량 관리도 (Hotelling T²)", style="margin:0;"),
                        ui.download_button(
                            "download_report_btn",
                            "📘 보고서 PDF 받기",
                            class_="btn btn-primary",
                        ),
                    )
                ),
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
    @render.ui
    def t2_plot():
        df = df_view()
        if df.empty:
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

    # ==================== 📘 보고서 PDF 받기 ====================
    @output
    @render.download(filename="Final_Report.pdf")
    def download_report_btn():
        file_path = "www/files/final_report.pdf"
        if not os.path.exists(file_path):
            # 파일이 없을 경우 간단한 PDF 자동 생성
            buf = io.BytesIO()
            c = canvas.Canvas(buf, pagesize=A4)
            c.drawString(100, 750, "⚠️ 보고서 파일을 찾을 수 없습니다.")
            c.save()
            buf.seek(0)
            yield from buf
        else:
            with open(file_path, "rb") as f:
                yield from f

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



    def _compute_t2_violations(df: pd.DataFrame, base: pd.DataFrame, features: list[str], alpha: float = 0.99):
        """
        baseline(base)으로 평균/공분산을 잡고, df에 대해 Hotelling T² 계산 후
        경험적 한계(CL=baseline T²의 alpha 백분위수)를 넘는 이탈들 반환.
        반환: list[dict] (타임라인에 바로 append 가능한 딕셔너리들)
        """
        out = []
        if base is None or df is None or df.empty:
            return out

        cols = [c for c in features if c in df.columns]
        if len(cols) < 2:
            # 다변량이 의미 있으려면 최소 2변수 이상
            return out

        # 기준/대상 데이터 정리(결측 제거)
        B = base[cols].dropna()
        if len(B) < max(30, len(cols) + 5):
            # 기준 데이터가 충분치 않으면 skip
            return out

        X = df[cols].dropna()
        if X.empty:
            return out

        # 평균/공분산/역행렬
        mu = B.mean().values
        S = np.cov(B.values, rowvar=False)
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        # T² 계산: T²_i = (x_i - mu)^T S^{-1} (x_i - mu)
        diff_base = B.values - mu
        T2_base = np.einsum("ij,jk,ik->i", diff_base, S_inv, diff_base)

        diff = X.values - mu
        T2 = np.einsum("ij,jk,ik->i", diff, S_inv, diff)

        # 경험적 한계(CL)
        CL = float(np.percentile(T2_base, alpha * 100.0))

        # df 인덱스에 매핑
        for idx_raw, t2_val in zip(X.index.tolist(), T2.tolist()):
            if t2_val > CL:
                out.append({
                    "__idx__": idx_raw,       # 나중에 시간/몰드코드 매핑용
                    "T2": float(t2_val),
                    "CL": CL,
                })
        return out
    
    @output
    @render.table
    def timeline_table():
        df = df_view()
        if df.empty:
            return pd.DataFrame({"상태": ["⚠️ 선택한 몰드코드에 데이터가 없습니다."]})

        base = df_baseline()
        out_rows = []
        dtcol = "__dt__" if "__dt__" in df.columns else None

        # 현재 선택된 공정 그룹(다변량 대상 특징)
        proc_name = input.process_select()
        features_mv = PROCESS_GROUPS.get(proc_name, [])
        # 단변량 대상은 전체 FEATURES_ALL을 유지
        features_uv = FEATURES_ALL

        # ---- 단변량: Nelson rule 위반 수집 (+ 몰드 코드 포함)
        for var in features_uv:
            if var not in df.columns:
                continue
            s = df[var].dropna()
            if len(s) < 10:
                continue

            # 기준선 통계
            if base is None or var not in (base.columns if hasattr(base, "columns") else [] ) or len(base) < 5:
                mu0, sd0 = s.mean(), s.std(ddof=1)
            else:
                mu0, sd0 = base[var].mean(), base[var].std(ddof=1)

            vio = check_nelson_rules(
                s.to_numpy(), mu0, mu0 + 3 * sd0, mu0 - 3 * sd0, sd0
            )

            # 최근 위반만 (너무 많으면 200개 제한 전에도 과다)
            for (idx, r, desc, val) in vio[-200:]:
                # s는 dropna 후이므로 원본 df 인덱스로 변환
                src_idx = s.index.min() + idx - 1 if len(s.index) else None
                if src_idx is None or src_idx not in df.index:
                    continue
                ts = df.loc[src_idx, dtcol] if (dtcol and src_idx in df.index) else np.nan
                mold_code = df.loc[src_idx, "mold_code"] if "mold_code" in df.columns else input.mold()
                out_rows.append({
                    "시각": ts,
                    "유형": "단변량",
                    "몰드": str(mold_code),
                    "변수": var,
                    "룰": r,
                    "설명": desc,
                    "값": round(float(val), 3),
                })

        # ---- 다변량: Hotelling T² 이탈 수집 (+ 몰드 코드 포함)
        t2_viol = _compute_t2_violations(df, base, features_mv, alpha=0.99)
        for v in t2_viol:
            src_idx = v["__idx__"]
            ts = df.loc[src_idx, dtcol] if (dtcol and src_idx in df.index) else np.nan
            mold_code = df.loc[src_idx, "mold_code"] if "mold_code" in df.columns else input.mold()
            out_rows.append({
                "시각": ts,
                "유형": "다변량",
                "몰드": str(mold_code),
                "변수": "T²",
                "룰": "T²>CL",
                "설명": f"T²={v['T2']:.2f} > CL={v['CL']:.2f}",
                "값": round(float(v["T2"]), 3),
            })

        if not out_rows:
            return pd.DataFrame({"상태": ["최근 이상 없음"]})

        timeline = pd.DataFrame(out_rows)

        # 시각 정렬(가능한 경우)
        if "시각" in timeline.columns and timeline["시각"].notna().any():
            timeline = timeline.sort_values("시각", ascending=False)

        # 최종 컬럼 순서 정리
        cols = ["시각", "유형", "몰드", "변수", "룰", "설명", "값"]
        show_cols = [c for c in cols if c in timeline.columns]
        return timeline[show_cols].head(200)
