# modules/page_quality.py
from shiny import ui, render
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path
import glob

# =========================================
# 0) 데이터 파일 우선순위 설정
# =========================================
CSV_PATHS = [
    "/mnt/data/test1.csv",            # 업로드 파일(우선)
    "./data/quality/*.csv",           # 프로젝트 폴더 내 데이터
    "./data/quality/*.svc",
]

# =========================================
# 1) 데이터 로더
#    필수: date, value, mold_code
#    선택: n(검사수), d(불량수) → 있으면 p-관리도에 직접 사용
# =========================================
def load_quality_source() -> pd.DataFrame:
    paths: List[str] = []
    for p in CSV_PATHS:
        if any(ch in p for ch in ["*", "?"]):
            paths += glob.glob(p)
        else:
            if Path(p).exists():
                paths.append(p)

    dfs: List[pd.DataFrame] = []

    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        cols_lower = {c.lower(): c for c in df.columns}

        def pick(*names):
            for n in names:
                if n in cols_lower:
                    return cols_lower[n]
            return None

        c_date  = pick("date", "datetime", "ts", "timestamp")
        c_val   = pick("value", "val", "measurement", "y")
        c_mold  = pick("mold_code", "mold", "moldid", "tool", "code")
        c_n     = pick("n", "sample", "count", "inspected", "inspections")
        c_d     = pick("d", "defect", "fail", "ng", "rejects")

        need = []
        if c_date is None: need.append("date")
        if c_val  is None: need.append("value")
        if c_mold is None: need.append("mold_code")
        if need:
            raise ValueError(f"필수 컬럼 누락: {need} / 원본: {list(df.columns)}")

        out = pd.DataFrame({
            "date": pd.to_datetime(df[c_date], errors="coerce"),
            "value": pd.to_numeric(df[c_val], errors="coerce"),
            "mold_code": df[c_mold].astype(str),
        })
        if c_n is not None:
            out["n"] = pd.to_numeric(df[c_n], errors="coerce").astype("Int64")
        if c_d is not None:
            out["d"] = pd.to_numeric(df[c_d], errors="coerce").astype("Int64")
        return out.dropna(subset=["date", "value", "mold_code"])

    for fp in paths:
        try:
            df = pd.read_csv(fp)
            dfs.append(_normalize_columns(df))
        except Exception:
            try:
                df = pd.read_csv(fp, sep=";")
                dfs.append(_normalize_columns(df))
            except Exception:
                print(f"[load_quality_source] 파일 로딩 실패: {fp}")

    if dfs:
        return pd.concat(dfs, ignore_index=True).sort_values("date")

    # ---- 파일이 없으면 더미 생성 ----
    rng = pd.date_range("2020-01-01", periods=200, freq="D")
    molds = ["8412", "8573", "8600", "8722", "8917"]
    out = []
    rng_seed = np.random.default_rng(42)
    for m in molds:
        vals = rng_seed.normal(250, 50, len(rng))
        vals[rng_seed.integers(0, len(rng), 5)] += rng_seed.integers(300, 700, 5)
        n = rng_seed.integers(80, 140, len(rng))
        base_p = 0.02 + np.clip((vals - 250) / 2000.0, 0, 0.2)
        d = np.array([rng_seed.binomial(int(n[i]), min(max(float(base_p[i]),0.001),0.4)) for i in range(len(n))])
        out.append(pd.DataFrame({"date": rng, "value": vals, "mold_code": m, "n": n, "d": d}))
    return pd.concat(out, ignore_index=True)


# ——— 간단 예측/설명 훅(데모) ———
def model_predict_passfail(df: pd.DataFrame) -> pd.Series:
    thr = df["value"].quantile(0.98)
    return (df["value"] >= thr).astype(int)

def explain_shap_for_week_row(row: pd.Series) -> Tuple[List[str], List[float]]:
    """주간 포인트에 대한 대체 SHAP(특징 중요도) — 데모용."""
    feats = ["molten_temp", "cast_pressure", "low_section_speed", "high_section_speed", "time_since_maint"]
    vals = np.abs(np.random.default_rng(int(pd.Timestamp.now().timestamp()) % 10_000).normal(0, 1, len(feats)))
    vals = (vals / vals.sum() * 100).tolist()
    return feats, vals


# =========================================
# 2) 통계/탐지 유틸 & p-관리도 유틸
# =========================================
def calc_control_limits(series: pd.Series):
    mu = series.mean()
    sigma = series.std(ddof=1)
    return mu, mu + 3 * sigma, mu - 3 * sigma, sigma

def flag_violations_control(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().sort_values("date")
    cl, ucl, lcl, sigma = calc_control_limits(d["value"])
    d["CL"], d["UCL"], d["LCL"], d["sigma"] = cl, ucl, lcl, sigma
    d["viol_control"] = (d["value"] > ucl) | (d["value"] < lcl)
    return d

def detect_outliers_zscore(df: pd.DataFrame, z: float = 3.0) -> pd.Series:
    mu, sigma = df["value"].mean(), df["value"].std(ddof=1)
    if sigma == 0 or np.isnan(sigma):
        return pd.Series(False, index=df.index)
    return (np.abs((df["value"] - mu) / sigma) >= z)

def detect_by_rules(df: pd.DataFrame, use_rule_a: bool = True, use_rule_b: bool = True) -> pd.Series:
    d = flag_violations_control(df)
    n = len(d); hit = pd.Series(False, index=d.index)
    if use_rule_a:
        above = d["value"] > d["CL"]; run = 0
        for i in range(n):
            run = run + 1 if above.iloc[i] else 0
            if run >= 8: hit.iloc[i-run+1:i+1] = True
    if use_rule_b and n >= 6:
        inc = d["value"].diff() > 0; run = 0
        for i in range(n):
            run = run + 1 if (inc.iloc[i] if i>0 else False) else 0
            if run >= 6: hit.iloc[i-run:i+1] = True
    return hit

def _daily_p_from_value(v: float, cl: float, sigma: float) -> float:
    if sigma <= 0 or np.isnan(sigma): return 0.02
    raw = 0.02 + max(0.0, (v - cl) / (3 * sigma)) * 0.25
    return float(np.clip(raw, 0.005, 0.35))

def build_p_weekly(df_all: pd.DataFrame, mold: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    d0 = df_all[(df_all["mold_code"] == mold)].copy()
    d0 = d0[(d0["date"] >= start) & (d0["date"] <= end)].copy()
    if d0.empty:
        return pd.DataFrame(columns=["week_start","n","d","p","UCL","LCL","pbar"])

    # n,d 있으면 그대로 사용. 없으면 value→임시 p 추정
    if not {"n","d"}.issubset(d0.columns):
        cl, ucl, lcl, sigma = calc_control_limits(d0["value"])
        d0["n"] = 100
        d0["p_day"] = d0["value"].apply(lambda x: _daily_p_from_value(float(x), cl, sigma))
        d0["d"] = np.round(d0["n"] * d0["p_day"]).astype(int)

    w = (
        d0.set_index("date")
          .groupby(pd.Grouper(freq="W-MON"))
          .agg(n=("n","sum"), d=("d","sum"))
          .reset_index()
          .rename(columns={"date":"week_start"})
    )
    w["p"] = w["d"] / w["n"]
    pbar = (w["d"].sum() / w["n"].sum()) if w["n"].sum() > 0 else 0.0
    w["UCL"] = pbar + 3.0 * np.sqrt(np.maximum(pbar * (1 - pbar) / w["n"], 0))
    w["LCL"] = (pbar - 3.0 * np.sqrt(np.maximum(pbar * (1 - pbar) / w["n"], 0))).clip(lower=0.0)
    w["pbar"] = pbar
    return w

def monthly_summary(df: pd.DataFrame, col_flag: str):
    tmp = df.copy()
    tmp["ym"] = df["date"].dt.to_period("M").astype(str)
    return tmp.groupby("ym", as_index=False).agg(out_cnt=(col_flag,"sum"), avg_val=("value","mean"))


# =========================================
# 3) UI
# =========================================
def ui_quality():
    df_all = load_quality_source()
    molds = [str(m) for m in sorted(df_all["mold_code"].unique())]
    tabs = [ui.nav_panel(f"Mold {m}", _mold_content_ui(m)) for m in molds]

    return ui.page_fluid(
        ui.tags.style("""
        *{font-family:'Noto Sans KR',-apple-system,blinkmacsystemfont,'Segoe UI',roboto,'Helvetica Neue',arial,'Apple Color Emoji','Segoe UI Emoji';}
        .kpi-strip{display:flex;gap:14px;align-items:stretch;width:100%;padding:6px 4px;}
        .kpi-tile{flex:1;min-width:140px;background:#fafafa;border:1px solid #eaeaea;border-radius:14px;padding:14px 12px;text-align:center;box-shadow:0 1px 2px rgba(0,0,0,.04);}
        .kpi-lbl{color:#6b7280;font-size:13px;line-height:16px;}
        .kpi-val{font-size:30px;font-weight:800;color:#111827;margin-top:2px;}
        .kpi-yellow{color:#f59e0b}.kpi-red{color:#ef4444}.kpi-blue{color:#2563eb}
        """),
        ui.h3("품질관리팀 탭 (Quality Control)"),
        ui.navset_tab(*tabs),
    )


def _mold_content_ui(mold_code: str):
    sid = lambda base: f"{base}_{mold_code}"

    # 상단: 관리도 + KPI
    top_panel = ui.card(
        ui.card_header(f"관리도 (Mold {mold_code})"),
        ui.output_ui(sid("control_chart")),
        ui.div(ui.output_ui(sid("kpi_strip")),
               style="padding:10px 6px 2px 6px; border-top:1px solid #eee;"),
        full_screen=True,
    )

    # 중단: 탐지 설정/결과 + (좌)결과표 / (우)다운로드 탭
    detect_controls = ui.card(
        ui.card_header("불량 탐지 설정"),
        ui.input_checkbox_group(
            sid("detectors"), "",
            choices={
                "zscore": "이상치 탐지 (z-score)",
                "control": "관리도 기준 (±3σ)",
                "rules": "우리 룰 기준",
                "predict": "양/불량 예측",
            },
            selected=["zscore","control","rules","predict"],
        ),
        ui.row(
            ui.column(6, ui.input_numeric(sid("z_th"), "z-score 임계값", 3.0, min=1.0, max=6.0, step=0.5)),
            ui.column(6, ui.input_switch(sid("rule_a"), "룰A: 연속 8포인트 CL 위", True)),
        ),
        ui.row(
            ui.column(6, ui.input_switch(sid("rule_b"), "룰B: 연속 6포인트 증가", True)),
            ui.column(6, ui.input_action_button(sid("btn_detect"), "🔎 불량 탐지 실행", class_="btn-primary w-100")),
        ),
    )

    detect_left = ui.card(
        ui.card_header("탐지 결과"),
        ui.output_text(sid("detect_summary")),
        ui.div(  # 폭을 100% 사용
            ui.output_data_frame(sid("detect_table")),
            style="width:100%;"
        ),
    )

    detect_right = ui.card(
        ui.card_header("탐지 결과 파일"),
        ui.navset_tab(
            ui.nav_panel(
                "다운로드",
                ui.p("탐지 결과를 CSV 혹은 Excel 파일로 저장할 수 있습니다."),
                ui.div(
                    ui.download_button(sid("dl_csv"), "CSV로 다운로드", class_="btn-secondary"),
                    ui.download_button(sid("dl_xlsx"), "Excel로 다운로드", class_="btn-secondary ms-2"),
                ),
                ui.br(),
                ui.output_text(sid("dl_info")),
            ),
            ui.nav_panel(
                "미리보기",
                ui.output_data_frame(sid("detect_preview")),
            ),
        ),
    )

    detect_panel = ui.card(
        ui.card_header("불량 탐지 설정 · 결과"),
        detect_controls,
        ui.hr(),
        ui.layout_columns(
            detect_left,
            detect_right,
            col_widths=[7, 3],   # ← 왼쪽을 더 넓게
        ),
    )

    # 하단: p-관리도 + SHAP (몰드/기간 동적)
    cause_panel = ui.card(
        ui.card_header("불량 원인 분석 (p-관리도 + SHAP)"),
        ui.layout_columns(
            ui.card(
                ui.card_header("p-관리도 (주간)"),
                ui.div(
                    ui.row(
                        ui.column(4, ui.output_ui(sid("p_mold_ctrl"))),   # 서버 렌더
                        ui.column(8, ui.output_ui(sid("p_date_ctrl"))),  # 동적 DateRange
                    ),
                    style="padding:6px 8px;"
                ),
                ui.output_ui(sid("p_chart")),  # p-관리도 출력
            ),
            ui.card(
                ui.card_header("SHAP 원인 기여도"),
                ui.output_ui(sid("shap_bar")),
                ui.output_text(sid("shap_notice")),
            ),
            col_widths=[7,5],
        ),
    )

    monthly_panel = ui.card(
        ui.card_header("월별 한계초과 개수 (Mold 관리도 기준)"),
        ui.output_ui(sid("viol_bar")),
    )

    return ui.page_fluid(
        top_panel, ui.br(),
        detect_panel, ui.br(),
        cause_panel, ui.br(),
        monthly_panel,
    )


# =========================================
# 4) SERVER
# =========================================
@dataclass
class DetectResult:
    date: pd.Timestamp
    value: float
    method: str
    idx: int

def server_quality(input, output, session):
    df_all = load_quality_source()
    df_all["date"] = pd.to_datetime(df_all["date"])
    molds = [str(m) for m in sorted(df_all["mold_code"].unique())]
    for mold in molds:
        _bind_mold_outputs(mold, df_all.copy(), output, input)

def _bind_mold_outputs(mold: str, df_all: pd.DataFrame, output, input):
    sid = lambda base: f"{base}_{mold}"

    df_mold = df_all[df_all["mold_code"] == mold].copy()
    df_control = flag_violations_control(df_mold)
    mbar = monthly_summary(df_control, col_flag="viol_control")

    # ===== 상단 관리도 =====
    @output(id=sid("control_chart"))
    @render.ui
    def _control_chart():
        d = df_control
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d["date"], y=d["value"], mode="lines+markers",
                                 line=dict(width=1), marker=dict(size=5), name=f"Mold {mold}"))
        fig.add_hline(y=float(d["CL"].iloc[0]),  line_color="#fbbf24", annotation_text="CL")
        fig.add_hline(y=float(d["UCL"].iloc[0]), line_color="#ef4444", annotation_text="UCL")
        fig.add_hline(y=float(d["LCL"].iloc[0]), line_color="#3b82f6", annotation_text="LCL")
        outs = d[d["viol_control"]]
        if not outs.empty:
            fig.add_trace(go.Scatter(x=outs["date"], y=outs["value"], mode="markers",
                                     marker=dict(size=10, color="#ef4444"), name="Out of Control"))
        fig.update_layout(template="plotly_white", height=420, margin=dict(l=20,r=20,t=40,b=20),
                          title=f"관리도 (Individuals, CL±3σ)")
        return ui.HTML(fig.to_html(include_plotlyjs=False, full_html=False))

    @output(id=sid("kpi_strip"))
    @render.ui
    def _kpi_strip():
        d = df_control
        html = f"""
        <div class="kpi-strip">
          <div class="kpi-tile"><div class="kpi-lbl">Groups</div><div class="kpi-val">{len(d):,}</div></div>
          <div class="kpi-tile"><div class="kpi-lbl">Stddev</div><div class="kpi-val">{d['sigma'].iloc[0]:,.3f}</div></div>
          <div class="kpi-tile"><div class="kpi-lbl">CL</div><div class="kpi-val kpi-yellow">{d['CL'].iloc[0]:,.0f}</div></div>
          <div class="kpi-tile"><div class="kpi-lbl">UCL</div><div class="kpi-val kpi-red">{d['UCL'].iloc[0]:,.0f}</div></div>
          <div class="kpi-tile"><div class="kpi-lbl">LCL</div><div class="kpi-val kpi-blue">{d['LCL'].iloc[0]:,.0f}</div></div>
          <div class="kpi-tile"><div class="kpi-lbl">관리도 한계밖</div><div class="kpi-val">{int(d['viol_control'].sum()):,}</div></div>
        </div>
        """
        return ui.HTML(html)

    @output(id=sid("viol_bar"))
    @render.ui
    def _viol_bar():
        mm = mbar
        fig = go.Figure(go.Bar(x=mm["ym"], y=mm["out_cnt"], name="Out-of-control count"))
        fig.update_layout(height=260, template="plotly_white",
                          margin=dict(l=10, r=10, t=30, b=40),
                          title=f"월별 한계초과 개수 (Mold {mold})")
        return ui.HTML(fig.to_html(include_plotlyjs=False, full_html=False))

    # ===== 탐지 공통 함수 =====
    def run_all_detectors() -> List[DetectResult]:
        chosen = set(getattr(input, sid("detectors"))())
        results: List[DetectResult] = []
        if "zscore" in chosen:
            z = float(getattr(input, sid("z_th"))()); zflag = detect_outliers_zscore(df_mold, z)
            for i in df_mold.index[zflag]:
                results.append(DetectResult(df_mold.at[i,"date"], float(df_mold.at[i,"value"]), "zscore", i))
        if "control" in chosen:
            for i in df_control.index[df_control["viol_control"]]:
                results.append(DetectResult(df_control.at[i,"date"], float(df_control.at[i,"value"]), "control", i))
        if "rules" in chosen:
            rule_a = bool(getattr(input, sid("rule_a"))()); rule_b = bool(getattr(input, sid("rule_b"))())
            rflag = detect_by_rules(df_mold, rule_a, rule_b)
            for i in df_mold.index[rflag]:
                results.append(DetectResult(df_mold.at[i,"date"], float(df_mold.at[i,"value"]), "rules", i))
        if "predict" in chosen:
            pred = model_predict_passfail(df_mold)
            for i in df_mold.index[pred == 1]:
                results.append(DetectResult(df_mold.at[i,"date"], float(df_mold.at[i,"value"]), "predict", i))
        results.sort(key=lambda r: (r.date, r.method))
        return results

    def get_detect_df() -> pd.DataFrame:
        res = run_all_detectors()
        if not res:
            return pd.DataFrame(columns=["date","value","method","idx"])
        df_res = pd.DataFrame([r.__dict__ for r in res]).sort_values(["date","method"]).reset_index(drop=True)
        df_res["date"] = pd.to_datetime(df_res["date"]).dt.strftime("%Y-%m-%d")
        return df_res[["date","value","method","idx"]]

    # ===== 탐지: 좌측 표/요약 =====
    @output(id=sid("detect_summary"))
    @render.text
    def _detect_summary():
        if not getattr(input, sid("btn_detect"))(): return "탐지 방법을 선택한 뒤, ‘불량 탐지 실행’을 클릭하세요."
        df_res = get_detect_df()
        if df_res.empty: return "탐지 결과: 이상 없음 ✅"
        cnts = df_res.groupby("method")["idx"].nunique().to_dict()
        return "탐지 결과: " + " · ".join([f"{k}: {v}건" for k,v in sorted(cnts.items())])

    @output(id=sid("detect_table"))
    @render.data_frame
    def _detect_table():
        grid_height = 600
        if not getattr(input, sid("btn_detect"))():
            return render.DataGrid(
                pd.DataFrame(columns=["date","value","method","idx"]),
                row_selection_mode="single",
                filters=True,
                height=grid_height,
                width="100%"
            )
        df_res = get_detect_df()
        return render.DataGrid(
            df_res,
            row_selection_mode="single",
            filters=True,
            height=grid_height,
            width="100%"
        )

    # ===== 탐지: 우측 다운로드 탭 =====
    @output(id=sid("detect_preview"))
    @render.data_frame
    def _detect_preview():
        prev = get_detect_df().head(20).copy()
        return render.DataGrid(prev, row_selection_mode="none", filters=True, height=360, width="100%")

    @output(id=sid("dl_csv"))
    @render.download(filename=lambda: f"detect_results_mold_{mold}.csv", media_type="text/csv")
    def _dl_csv():
        import io
        df_res = get_detect_df()
        with io.StringIO() as s:
            df_res.to_csv(s, index=False)
            return s.getvalue()

    @output(id=sid("dl_xlsx"))
    @render.download(filename=lambda: f"detect_results_mold_{mold}.xlsx", media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    def _dl_xlsx():
        import io
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
            get_detect_df().to_excel(writer, index=False, sheet_name="detect")
        return bio.getvalue()

    @output(id=sid("dl_info"))
    @render.text
    def _dl_info():
        df_res = get_detect_df()
        return f"행 {len(df_res):,}건 · 열 {df_res.shape[1]}개 · Mold {mold}"

    # ===== 하단: p-관리도 + SHAP =====
    @output(id=sid("p_mold_ctrl"))
    @render.ui
    def _p_mold_ctrl():
        molds_all = [str(m) for m in sorted(df_all["mold_code"].unique())]
        default_sel = mold if mold in molds_all else (molds_all[0] if molds_all else "")
        return ui.input_select(
            sid("p_mold"),
            "몰드 선택",
            choices={m: m for m in molds_all},
            selected=default_sel
        )

    @output(id=sid("p_date_ctrl"))
    @render.ui
    def _p_date_ctrl():
        try:
            cur_mold = getattr(input, sid("p_mold"))() or mold
        except Exception:
            cur_mold = mold

        dsel = df_all[df_all["mold_code"] == cur_mold].copy()
        if dsel.empty:
            d_min = pd.to_datetime(df_all["date"]).min()
            d_max = pd.to_datetime(df_all["date"]).max()
        else:
            d_min = pd.to_datetime(dsel["date"]).min()
            d_max = pd.to_datetime(dsel["date"]).max()

        try:
            dr = getattr(input, sid("p_date"))()
            start_cur = pd.to_datetime(dr[0]) if dr and dr[0] else d_min
            end_cur   = pd.to_datetime(dr[1]) if dr and dr[1] else d_max
        except Exception:
            start_cur, end_cur = d_min, d_max

        start_cur = max(d_min, min(start_cur, d_max))
        end_cur   = max(d_min, min(end_cur,   d_max))
        if start_cur > end_cur:
            start_cur, end_cur = d_min, d_max

        return ui.input_date_range(
            sid("p_date"),
            "기간",
            start=start_cur.date(),
            end=end_cur.date(),
            min=d_min.date(),
            max=d_max.date(),
        )

    @output(id=sid("p_chart"))
    @render.ui
    def _p_chart():
        try:
            cur_mold = getattr(input, sid("p_mold"))() or mold
        except Exception:
            cur_mold = mold

        dsel = df_all[df_all["mold_code"] == cur_mold].copy()
        if dsel.empty:
            d_min = pd.to_datetime(df_all["date"]).min()
            d_max = pd.to_datetime(df_all["date"]).max()
        else:
            d_min = pd.to_datetime(dsel["date"]).min()
            d_max = pd.to_datetime(dsel["date"]).max()

        try:
            dr = getattr(input, sid("p_date"))()
            start = pd.to_datetime(dr[0]) if dr and dr[0] else d_min
            end   = pd.to_datetime(dr[1]) if dr and dr[1] else d_max
        except Exception:
            start, end = d_min, d_max

        start = max(d_min, min(start, d_max))
        end   = max(d_min, min(end,   d_max))
        if start > end:
            start, end = d_min, d_max

        w = build_p_weekly(df_all, cur_mold, start, end)
        if w.empty:
            return ui.HTML("<div>선택한 조건에 해당하는 데이터가 없습니다.</div>")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=w["week_start"], y=w["p"], mode="lines+markers", name="p (불량비율)", marker=dict(size=7)))
        fig.add_trace(go.Scatter(x=w["week_start"], y=w["UCL"], mode="lines", name="UCL", line=dict(color="#ef4444")))
        fig.add_trace(go.Scatter(x=w["week_start"], y=w["LCL"], mode="lines", name="LCL", line=dict(color="#3b82f6")))
        fig.add_hline(y=float(w["pbar"].iloc[0]), line_color="#fbbf24", annotation_text="CL (p̄)")
        outs = (w["p"] > w["UCL"]) | (w["p"] < w["LCL"])
        if outs.any():
            fig.add_trace(go.Scatter(x=w.loc[outs,"week_start"], y=w.loc[outs,"p"],
                                     mode="markers", marker=dict(size=12, color="#dc2626", symbol="x", line=dict(width=2)),
                                     name="Out of Control"))
        fig.update_layout(template="plotly_white", height=320, margin=dict(l=20,r=20,t=40,b=20),
                          title=f"p-관리도 (주간) · Mold {cur_mold}", yaxis_tickformat=".1%")
        return ui.HTML(fig.to_html(include_plotlyjs=False, full_html=False))

    @output(id=sid("shap_bar"))
    @render.ui
    def _shap_bar():
        try:
            cur_mold = getattr(input, sid("p_mold"))() or mold
        except Exception:
            cur_mold = mold

        dsel = df_all[df_all["mold_code"] == cur_mold].copy()
        if dsel.empty:
            d_min = pd.to_datetime(df_all["date"]).min()
            d_max = pd.to_datetime(df_all["date"]).max()
        else:
            d_min = pd.to_datetime(dsel["date"]).min()
            d_max = pd.to_datetime(dsel["date"]).max()

        try:
            dr = getattr(input, sid("p_date"))()
            start = pd.to_datetime(dr[0]) if dr and dr[0] else d_min
            end   = pd.to_datetime(dr[1]) if dr and dr[1] else d_max
        except Exception:
            start, end = d_min, d_max

        start = max(d_min, min(start, d_max))
        end   = max(d_min, min(end,   d_max))
        if start > end:
            start, end = d_min, d_max

        w = build_p_weekly(df_all, cur_mold, start, end)
        if w.empty:
            return ui.HTML("원인 기여도를 계산할 데이터가 없습니다.")

        row = w.iloc[[-1]]
        feats, vals = explain_shap_for_week_row(row.iloc[0])
        fig = go.Figure(go.Bar(x=vals, y=feats, orientation="h",
                               marker=dict(color=vals, colorscale="Blues", showscale=False),
                               text=[f"{v:.1f}%" for v in vals], textposition="outside"))
        fig.update_layout(template="plotly_white", height=280, margin=dict(l=60,r=20,t=30,b=20),
                          title="불량 원인 기여도(주간 p-관리도 기준)")
        return ui.HTML(fig.to_html(include_plotlyjs=False, full_html=False))

    @output(id=sid("shap_notice"))
    @render.text
    def _shap_notice():
        return "※ CSV의 n,d가 있으면 그대로 p-관리도에 사용하고, 없으면 value로 임시 추정합니다. DateRange는 몰드의 실제 데이터 범위로 제한됩니다."
