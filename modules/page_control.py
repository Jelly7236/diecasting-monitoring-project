# modules/page_control.py
from shiny import ui, render, reactive
import pandas as pd
import numpy as np

from shared import streaming_df  # 실시간 DF
from utils.control_config import SPEC_LIMITS, PROCESS_GROUPS, FEATURES_ALL
from utils.control_stats import (
    check_nelson_rules, calculate_hotelling_t2, phaseII_ucl_t2,
    calculate_cp_cpk, to_datetime_safe
)
from viz.control_plots import build_univar_figure, build_t2_figure, build_cap_hist

# ==================== UI ====================
def ui_control():
    return ui.page_fluid(
        # 외부 CSS
        ui.tags.link(rel="stylesheet", href="css/control.css"),

        # 컨트롤 바
        ui.div(
            ui.div(
                ui.card(
                    ui.card_header("⚙️ 컨트롤"),
                    ui.div(
                        ui.layout_columns(
                            ui.input_select(
                                "var_uni","단변량 변수",
                                choices={
                                    "molten_temp":"용탕온도",
                                    "cast_pressure":"주조압력",
                                    "upper_mold_temp1":"상형온도1",
                                    "sleeve_temperature":"슬리브온도",
                                    "Coolant_temperature":"냉각수온도",
                                }, selected="molten_temp"
                            ),
                            ui.output_ui("mold_select"),
                            ui.input_numeric("win","윈도우(n)",200,min=50,max=5000,step=50),
                            ui.input_switch("phase_guard","Phase I(정상만) 기준선",True),
                            col_widths=[4,4,2,2]
                        ),
                        class_="p-2"
                    ),
                ),
                class_="section"
            ),
            class_="container stickybar"
        ),

        # KPI
        ui.div(
            ui.card(
                ui.card_header("📌 KPI (선택 변수 한눈에)"),
                ui.output_ui("kpi_bar")
            ),
            class_="container section"
        ),

        # 섹션 1: 단변량
        ui.div(
            ui.card(
                ui.card_header("📈 단변량 관리도 + 넬슨 룰"),
                ui.layout_columns(
                    ui.div(
                        ui.output_ui("univar_plot"),
                        ui.div(ui.output_ui("nelson_badges"), class_="pt-2"),
                        class_="p-3"
                    ),
                    ui.div(
                        ui.h5("🚨 이상 패턴 로그", class_="mb-2"),
                        ui.div(ui.output_table("nelson_table"), class_="scroll-table"),
                        class_="p-3"
                    ),
                    col_widths=[8,4]
                )
            ),
            class_="container section"
        ),

        # 섹션 2: 다변량
        ui.div(
            ui.card(
                ui.card_header("🔬 다변량 관리도 (Hotelling T²)"),
                ui.layout_columns(
                    ui.div(
                        ui.input_select("t2_group","변수 그룹",
                            choices={k:k for k in PROCESS_GROUPS.keys()},
                            selected=list(PROCESS_GROUPS.keys())[0]
                        ),
                        ui.output_ui("t2_plot"),
                        class_="p-3"
                    ),
                    ui.div(
                        ui.h5("📄 T² 초과 로그", class_="mb-2"),
                        ui.div(ui.output_table("t2_table"), class_="scroll-table"),
                        class_="p-3"
                    ),
                    col_widths=[8,4]
                )
            ),
            class_="container section"
        ),

        # 섹션 3: Cp/Cpk
        ui.div(
            ui.card(
                ui.card_header("📐 공정능력 (Cp / Cpk)"),
                ui.layout_columns(
                    ui.div(
                        ui.input_select(
                            "cap_var","분석 변수",
                            choices={
                                "molten_temp":"용탕온도",
                                "cast_pressure":"주조압력",
                                "upper_mold_temp1":"상형온도1",
                                "sleeve_temperature":"슬리브온도",
                                "Coolant_temperature":"냉각수온도",
                            }, selected="cast_pressure"
                        ),
                        ui.output_ui("cap_plot"),
                        class_="p-3"
                    ),
                    ui.div(
                        ui.h5("📄 Cp/Cpk 표", class_="mb-2"),
                        ui.output_table("cap_table"),
                        class_="p-3"
                    ),
                    col_widths=[8,4]
                )
            ),
            class_="container section"
        ),

        # 섹션 4: 타임라인
        ui.div(
            ui.card(
                ui.card_header("🕒 최근 이상 타임라인 (단변량 룰/다변량 T² 합본)"),
                ui.div(ui.output_table("timeline_table"), class_="scroll-table", style="max-height:340px")
            ),
            class_="container section"
        ),
    )

# ==================== SERVER ====================
def server_control(input, output, session):
    # 동적 mold 선택
    @output
    @render.ui
    def mold_select():
        df = streaming_df; choices = ["(전체)"]
        if "mold_code" in df:
            choices += [str(m) for m in sorted(df["mold_code"].dropna().unique())]
        return ui.input_select("mold","몰드",choices=choices,selected="(전체)")

    # 공통 뷰
    @reactive.Calc
    def df_view():
        df = streaming_df.copy()
        if "id" in df: df = df.sort_values("id")
        df = df.tail(int(input.win()))
        if "mold_code" in df and input.mold() not in (None,"","(전체)"):
            try:
                sel = int(input.mold()); df = df[df["mold_code"] == sel]
            except Exception:
                df = df[df["mold_code"].astype(str) == str(input.mold())]
        dt = to_datetime_safe(df)
        df["__dt__"] = dt if dt is not None else pd.RangeIndex(len(df)).astype(float)
        return df.reset_index(drop=True)

    # 기준선 (Phase I: passorfail==0)
    @reactive.Calc
    def df_baseline():
        df = streaming_df.copy()
        if "id" in df: df = df.sort_values("id")
        if "mold_code" in df and input.mold() not in (None,"","(전체)"):
            try:
                sel = int(input.mold()); df = df[df["mold_code"] == sel]
            except Exception:
                df = df[df["mold_code"].astype(str) == str(input.mold())]
        mask = (df["passorfail"] == 0) if "passorfail" in df else np.ones(len(df), dtype=bool)
        base = df.loc[mask, FEATURES_ALL].dropna()
        if len(base) < 50: return None
        return base

    # KPI
    @output
    @render.ui
    def kpi_bar():
        df = df_view()
        base = df_baseline() if input.phase_guard() else df_view()[FEATURES_ALL].dropna()
        var = input.var_uni()
        series = df[var].dropna()
        if len(series) < 5:
            return ui.div(ui.p("표본이 부족합니다.", class_="muted"))

        if base is None or len(base) < 5:
            mu0, sd0 = series.mean(), series.std(ddof=1)
        else:
            mu0, sd0 = base[var].mean(), base[var].std(ddof=1)
        ucl, lcl = mu0 + 3*sd0, mu0 - 3*sd0

        if var in SPEC_LIMITS:
            cp, cpk, *_ = calculate_cp_cpk(series, SPEC_LIMITS[var]["usl"], SPEC_LIMITS[var]["lsl"])
            cp_text = f"{cp:.2f} / {cpk:.2f}"
        else:
            cp_text = "—"

        def kcard(title, value, sub=""):
            return ui.div(
                ui.div(
                    ui.div(title, class_="title"),
                    ui.div(value, class_="value"),
                    ui.div(sub, class_="sub"),
                    class_="p-3"
                ),
                class_="kcard"
            )

        return ui.div(
            kcard("변수", f"{var}"),
            kcard("평균(μ)", f"{series.mean():.2f}", f"기준선 μ={mu0:.2f}"),
            kcard("표준편차(σ)", f"{series.std(ddof=1):.2f}", f"기준선 σ={sd0:.2f}"),
            kcard("UCL/LCL(±3σ)", f"{ucl:.2f} / {lcl:.2f}"),
            kcard("Cp / Cpk", cp_text),
            class_="kpi-row"
        )

    # 단변량
    @output
    @render.ui
    def univar_plot():
        df = df_view()
        base = df_baseline() if input.phase_guard() else df_view()[FEATURES_ALL].dropna()
        var = input.var_uni()
        x = df[var].dropna().to_numpy()
        if len(x) < 10:
            return ui.p("표본이 부족합니다.", class_="muted")
        mu = (base[var].mean() if base is not None and len(base)>5 else np.mean(x))
        sd = (base[var].std(ddof=1) if base is not None and len(base)>5 else np.std(x, ddof=1))
        vio = check_nelson_rules(x, mu, mu+3*sd, mu-3*sd, sd)
        fig = build_univar_figure(x, mu, sd, vio, title=f"{var} 관리도 (n={len(x)})")
        return ui.HTML(fig.to_html(include_plotlyjs='cdn', div_id=f"uni_{var}"))

    @output
    @render.ui
    def nelson_badges():
        df = df_view()
        base = df_baseline() if input.phase_guard() else df_view()[FEATURES_ALL].dropna()
        var = input.var_uni()
        x = df[var].dropna().to_numpy()
        if len(x) < 10: return ui.div()
        mu = (base[var].mean() if base is not None and len(base)>5 else np.mean(x))
        sd = (base[var].std(ddof=1) if base is not None and len(base)>5 else np.std(x, ddof=1))
        vio = check_nelson_rules(x, mu, mu+3*sd, mu-3*sd, sd)
        counts = {"Rule 1":0,"Rule 2":0,"Rule 3":0,"Rule 5":0}
        for _, r, _, _ in vio:
            if r in counts: counts[r]+=1
        return ui.div(
            ui.span(f"Rule1 {counts['Rule 1']}", class_="badge b-red",   style="margin-right:.5rem"),
            ui.span(f"Rule2 {counts['Rule 2']}", class_="badge b-amber", style="margin-right:.5rem"),
            ui.span(f"Rule3 {counts['Rule 3']}", class_="badge b-blue",  style="margin-right:.5rem"),
            ui.span(f"Rule5 {counts['Rule 5']}", class_="badge b-gray"),
        )

    @output
    @render.table
    def nelson_table():
        df = df_view()
        base = df_baseline() if input.phase_guard() else df_view()[FEATURES_ALL].dropna()
        var = input.var_uni()
        x = df[var].dropna().to_numpy()
        if len(x) < 10: return pd.DataFrame({"상태":["표본 부족"]})
        mu = (base[var].mean() if base is not None and len(base)>5 else np.mean(x))
        sd = (base[var].std(ddof=1) if base is not None and len(base)>5 else np.std(x, ddof=1))
        vio = check_nelson_rules(x, mu, mu+3*sd, mu-3*sd, sd)
        if not vio: return pd.DataFrame({"상태":["✅ 이상 패턴 없음"]})
        out = pd.DataFrame(vio, columns=["샘플","룰","설명","값"])
        out["값"] = out["값"].round(3)
        return out.tail(200)

    # 다변량
    @output
    @render.ui
    def t2_plot():
        df = df_view()
        base = df_baseline() if input.phase_guard() else df_view()[FEATURES_ALL].dropna()
        group_key = input.t2_group()
        var_list = PROCESS_GROUPS[group_key]
        X = df[var_list].dropna().to_numpy()
        p = len(var_list)
        if X.shape[0] < max(30, p+5): return ui.p("표본이 부족합니다.", class_="muted")

        base_df = base[var_list].dropna() if (base is not None and set(var_list).issubset(base.columns)) else df[var_list].dropna()
        mu = base_df.mean().to_numpy()
        cov = np.cov(base_df.to_numpy().T)
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov)

        t2 = calculate_hotelling_t2(X, mu, inv_cov)
        n = X.shape[0]
        ucl = phaseII_ucl_t2(n, p, alpha=0.01)
        viol_idx = np.where(t2 > ucl)[0]
        fig = build_t2_figure(t2, ucl, title=f"{group_key} · 변수: {', '.join(var_list)}", viol_idx=viol_idx)
        return ui.HTML(fig.to_html(include_plotlyjs='cdn', div_id=f"t2_{group_key}"))

    @output
    @render.table
    def t2_table():
        df = df_view()
        base = df_baseline() if input.phase_guard() else df_view()[FEATURES_ALL].dropna()
        group_key = input.t2_group()
        var_list = PROCESS_GROUPS[group_key]
        X = df[var_list].dropna().to_numpy()
        p = len(var_list)
        if X.shape[0] < max(30, p+5): return pd.DataFrame({"상태":["표본 부족"]})

        base_df = base[var_list].dropna() if (base is not None and set(var_list).issubset(base.columns)) else df[var_list].dropna()
        mu = base_df.mean().to_numpy()
        cov = np.cov(base_df.to_numpy().T)
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov)

        t2 = calculate_hotelling_t2(X, mu, inv_cov)
        n = X.shape[0]
        ucl = phaseII_ucl_t2(n, p, alpha=0.01)
        viol = np.where(t2 > ucl)[0]

        if len(viol) == 0:
            return pd.DataFrame({"상태":["✅ 관리 상태 양호"]})
        log = pd.DataFrame({
            "샘플": viol+1,
            "T²": t2[viol].round(3),
            "UCL": np.round(ucl,3),
            "변수": [", ".join(var_list)]*len(viol),
            "유형": ["T² 초과"]*len(viol),
        })
        return log.tail(200)

    # Cp/Cpk
    @output
    @render.ui
    def cap_plot():
        df = df_view()
        var = input.cap_var()
        x = df[var].dropna().to_numpy()
        if len(x) < 20 or var not in SPEC_LIMITS:
            return ui.p("표본이 부족거나 규격 한계 미정의.", class_="muted")
        usl, lsl = SPEC_LIMITS[var]["usl"], SPEC_LIMITS[var]["lsl"]
        cp, cpk, cpu, cpl, mean_s, std_s = calculate_cp_cpk(x, usl, lsl)
        fig = build_cap_hist(x, usl, lsl, mean_s, cp, cpk, title=f"{var} Cp/Cpk")
        return ui.HTML(fig.to_html(include_plotlyjs='cdn', div_id=f"cap_{var}"))

    @output
    @render.table
    def cap_table():
        df = df_view()
        var = input.cap_var()
        x = df[var].dropna().to_numpy()
        if len(x) < 20 or var not in SPEC_LIMITS:
            return pd.DataFrame({"상태":["표본 부족 또는 규격 한계 미정의"]})
        usl, lsl = SPEC_LIMITS[var]["usl"], SPEC_LIMITS[var]["lsl"]
        cp, cpk, cpu, cpl, mean_s, std_s = calculate_cp_cpk(x, usl, lsl)
        status = "✅ 우수(≥1.33)" if cpk >= 1.33 else ("⚠️ 양호(≥1.00)" if cpk >= 1.00 else "❌ 개선 필요")
        return pd.DataFrame({
            "지표":["USL","LSL","평균(μ)","표준편차(σ)","Cp","Cpu","Cpl","Cpk","평가"],
            "값":[usl, lsl, round(mean_s,3), round(std_s,3),
                 round(cp,3), round(cpu,3), round(cpl,3), round(cpk,3), status]
        })

    # 타임라인
    @output
    @render.table
    def timeline_table():
        df = df_view()
        base = df_baseline() if input.phase_guard() else df_view()[FEATURES_ALL].dropna()
        out_rows = []
        dtcol = "__dt__" if "__dt__" in df.columns else None

        # 단변량
        for var in FEATURES_ALL:
            s = df[var].dropna()
            if len(s) < 10: continue
            if base is None or var not in base.columns or len(base) < 5:
                mu0, sd0 = s.mean(), s.std(ddof=1)
            else:
                mu0, sd0 = base[var].mean(), base[var].std(ddof=1)
            vio = check_nelson_rules(s.to_numpy(), mu0, mu0+3*sd0, mu0-3*sd0, sd0)
            for (idx, r, desc, val) in vio[-50:]:
                ts = df.iloc[s.index.min() + idx - 1][dtcol] if dtcol else np.nan
                out_rows.append({"시각": ts, "유형":"단변량", "세부": r, "설명": f"{var}: {desc}", "값": round(val,3)})

        # 다변량
        for key, vars_ in PROCESS_GROUPS.items():
            sub = df[vars_].dropna()
            p = len(vars_)
            if sub.shape[0] < max(30, p+5): continue
            base_df = base[vars_].dropna() if (base is not None and set(vars_).issubset(base.columns)) else sub
            mu = base_df.mean().to_numpy()
            cov = np.cov(base_df.to_numpy().T)
            try:
                inv_cov = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                inv_cov = np.linalg.pinv(cov)
            t2 = calculate_hotelling_t2(sub.to_numpy(), mu, inv_cov)
            ucl = phaseII_ucl_t2(len(sub), p, 0.01)
            viol_idx = np.where(t2 > ucl)[0][-50:]
            for idx in viol_idx:
                orig_idx = sub.index[idx]
                ts = df.loc[orig_idx, dtcol] if dtcol else np.nan
                out_rows.append({"시각": ts, "유형":"다변량", "세부":"T²", "설명": f"{key} 초과", "값": round(t2[idx],3)})

        if not out_rows:
            return pd.DataFrame({"상태":["최근 이상 없음"]})
        timeline = pd.DataFrame(out_rows)
        if "시각" in timeline.columns and timeline["시각"].notna().any():
            timeline = timeline.sort_values("시각")
        return timeline.tail(300)
