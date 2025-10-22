# modules/page_monitoring.py
from shiny import ui, render, reactive
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
# 🔹 추가: 실시간 예측 결과 공유용
from shared import prediction_state

# ===== 모니터링 설정 =====
MON_BATCH_SIZE = 200          # 배치 크기
MON_LOOKBACK_BATCHES = 5      # 최근 몇 배치를 볼지(총 행수 = 배치 크기 * 개수)
INCLUDE_PARTIAL_BATCH = False  # 마지막 부분 배치 포함 여부
X_ALIGN = "end"               # "end" or "center" (배치의 끝/중앙 시간)
SHOW_BATCH_LINES = True       # 배치 경계선 표시

# ==================== UI ====================
def ui_monitoring():
    return ui.page_fluid(
        ui.tags.style("""
            * {font-family: 'Noto Sans KR', sans-serif}
            body {background: #f6f7fb}
            .container {max-width: 1300px; margin: 0 auto; padding: 0 12px}
            .card {border: 1px solid #e5e7eb; border-radius: 14px; box-shadow: 0 2px 6px rgba(0,0,0,.05); background: #fff}
            .card-header {background: #fafbfc; border-bottom: 1px solid #eef0f3; padding: .9rem 1.1rem; font-weight: 800; color: #111827}
            .section {margin-bottom: 18px}
            .kpi-row {display: grid; grid-auto-flow: column; grid-auto-columns: minmax(180px, 1fr); gap: 12px; overflow-x: auto; padding: 12px; align-items: stretch}
            .kcard {border: 1px solid #e5e7eb; border-radius: 12px; background: #fff; padding: 1rem}
            .kcard .title {color: #6b7280; font-size: .85rem; font-weight: 700; margin-bottom: .5rem}
            .kcard .value {font-size: 1.35rem; font-weight: 900; color: #111827}
            .muted {color: #6b7280}
            .scroll-table {max-height: 340px; overflow: auto; border-radius: 8px; background: #fff}
            .scroll-table table {width: 100%; border-collapse: collapse}
            .scroll-table thead th {position: sticky; top: 0; background: #fafbfc; z-index: 1; padding: .75rem; border-bottom: 2px solid #e5e7eb; text-align: left}
            .scroll-table tbody td {padding: .75rem; border-bottom: 1px solid #f3f4f6}
        """),

        # 헤더
        ui.div(
            ui.h3("모델 모니터링 및 성능 분석"),
            ui.p("실시간 예측 결과와 실제 결과를 비교하여 모델의 성능을 평가합니다.", class_="muted"),
            class_="container section"
        ),

        # 모델 설명 아코디언
        ui.div(
            ui.card(
                ui.card_header("모델 설명"),
                ui.accordion(
                    ui.accordion_panel(
                        "개요 · 프로세스",
                        ui.markdown(
                            """
### 1) 개요
- **목적**: 시계열 생산 로그로 *다음 샷 불량(1)* 사전 예측 → 조기 대응
- **대상**: 몰드코드별 **독립 모델링** (8412, 8573, 8600, 8722, 8917)
- **튜닝 지표(목표)**: **F2-score** *(재현율 가중)*  
  참고: Precision · F1 · ROC AUC · AP(PR AUC)

---

### 2) 데이터 구성
1. **몰드코드 분리**
2. **정렬/클린**: `datetime` 기준 이상치/결측 제거 → 오름차순 정렬
3. **시계열 분할**
   - Train **80%** / Validation **20%** (과거→미래 고정)
   - Train 내부 검증: **TimeSeriesSplit** *(미래 누수 방지)*

---

### 3) 특징공학 & 라벨 보조
- **보조 라벨**
  - `realfail`: `(tryshot_signal == 'A' | count ≥ 7 ) & passorfail == 1)` → 1
  - `check_passorfail`: `passorfail + realfail → {0, 1, 2}
- **전처리**
  - 범주형: **One-Hot Encoding**
  - 수치형: **RobustScaler**
  - **sanitize 단계**로 `NaN/Inf`, 타입 캐스팅 안전 처리
                            """
                        ),
                        value="p_overview"
                    ),
                    ui.accordion_panel(
                        "오버샘플링",
                        ui.markdown(
                            """
### 오버샘플링 전략
**커스텀 MajorityVoteSMOTENC**

- **대상**: `y == 1` **AND** `check_passorfail == 2` *(진짜 불량)* 만 합성
- **합성 개수**: `n_new = ⌊ 1.5 × #가짜불량 (y==1 & cp==1) ⌋`
- **생성 방식**
  - 수치형: 선형 보간
  - 범주형: k-이웃 **다수결**(동률 랜덤)
- **파이프라인 위치**: **전처리(OHE/Scaling) 이전** 단계에서 동작

> **누수 방지**  
> 교차검증(TimeSeriesSplit) 각 fold의 **train 폴드에만** 오버샘플링 적용
                            """
                        ),
                        value="p_sampling"
                    ),
                    ui.accordion_panel(
                        "모델 · 튜닝 · 평가",
                        ui.markdown(
                            """
### 모델 & 튜닝
- **모델**: `RandomForestClassifier(class_weight="balanced")`
- **파이프라인**: `sanitize → sampler → preprocess → model`
- **교차검증**: `TimeSeriesSplit(n_splits=5)` *(expanding-window)*
- **튜너**: `BayesSearchCV(n_iter=30)` / **목표 스코어**: **F2**
- **탐색 공간**  
  `n_estimators, max_depth, min_samples_* , max_features, bootstrap, ccp_alpha ...`

---

### 테스트 평가 (20% 홀드아웃, 단 1회)
- **임계값(τ)**: 기본 **0.50**  
- **보고 지표**: Precision · Recall · F1 · **F2** · ROC AUC · AP · Confusion Matrix
                            """
                        ),
                        value="p_eval"
                    ),
                    id="acc_model_doc_v2",
                    multiple=True,
                    open=[]
                ),
            ),
            class_="container section"
        ),

        # KPI 한 줄 (몰드코드 선택 추가)
        ui.div(
            ui.card(
                ui.div(
                    ui.div(
                        ui.span("실시간 성능 지표", style="font-weight: 800; font-size: 1rem;"),
                        ui.div(
                            ui.input_select(
                                "mon_mold_code",
                                None,
                                choices=["전체", "8412", "8413", "8576", "8722", "8917"],
                                selected="전체",
                                width="180px"
                            ),
                            style="display: inline-block; margin-left: 1rem;"
                        ),
                        style="display: flex; align-items: center; background: #fafbfc; border-bottom: 1px solid #eef0f3; padding: .9rem 1.1rem;"
                    ),
                ),
                ui.output_ui("mon_kpi_bar")
            ),
            class_="container section"
        ),

        # ───────── 시계열 그래프 ─────────
        ui.div(
            ui.card(
                ui.card_header("실시간 예측 추이"),
                ui.output_ui("mon_timeseries_plot")
            ),
            class_="container section"
        ),
        
        # ───────── 최근 샘플 10건 + 오류 샘플(FP/FN) (2열) ─────────
        ui.div(
            ui.layout_columns(
                ui.card(
                    ui.card_header("최근 샘플 10건"),
                    ui.div(ui.output_table("mon_sample_table"), class_="scroll-table")
                ),
                ui.card(
                    ui.card_header("오류 샘플 (FP/FN)"),
                    ui.div(ui.output_table("mon_error_table"), class_="scroll-table")
                ),
                col_widths=[6, 6]
            ),
            class_="container section"
        ),
    )

MOLD_COL_CANDIDATES = ["mold_code", "moldcode", "mold", "MOLD_CODE"]

def _find_mold_col(df: pd.DataFrame):
    for c in MOLD_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None

# ▼ 추가: 타임스탬프 컬럼 후보와 탐지 유틸
TS_COL_CANDIDATES = ["timestamp", "time", "datetime", "ts"]

def _find_ts_col(df: pd.DataFrame):
    for c in TS_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None

# ==================== SERVER ====================
def server_monitoring(input, output, session):

    # ▼ 기본값(컨트롤 제거한 대신 상수로 둠)
    DEFAULT_TAU = 0.5               # 임계값 τ
    DEFAULT_NSHOW = MON_BATCH_SIZE * MON_LOOKBACK_BATCHES  # 최근 N개(배치기반)

    # --- 관측 프레임 (필터 + 정렬 + 윈도우 + 파생) ---
    @reactive.calc
    def view_df() -> pd.DataFrame:
        df = prediction_state()
        if df is None or df.empty or not {"pred", "prob", "actual"}.issubset(df.columns):
            return pd.DataFrame()

        mold_col = _find_mold_col(df)
        ts_col   = _find_ts_col(df)

        # 몰드 필터
        sel = input.mon_mold_code() if hasattr(input, "mon_mold_code") else "전체"
        if sel and sel != "전체" and mold_col:
            df = df[df[mold_col].astype(str) == str(sel)]
        if df.empty:
            return pd.DataFrame()

        # 시간 정렬
        if ts_col:
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
            df = df.dropna(subset=[ts_col]).sort_values(ts_col)

        # 최근 N개(최근 k배치만 유지)
        df = df.tail(DEFAULT_NSHOW).copy()

        # 파생
        thr = DEFAULT_TAU
        df["y_true"]    = pd.to_numeric(df["actual"], errors="coerce").fillna(0).astype(int)
        df["y_prob"]    = pd.to_numeric(df["prob"],   errors="coerce")
        df["y_pred(τ)"] = (df["y_prob"] >= thr).astype(int)
        df["sample_id"] = np.arange(1, len(df) + 1)

        # ── 'time' 보장: datetime/timestamp/ts → time(HH:MM:SS) ──
        ts_for_time = next((c for c in ["time", "timestamp", "datetime", "ts"] if c in df.columns), None)
        if ts_for_time:
            if ts_for_time != "time" or df["time"].dtype != "object":
                ts = pd.to_datetime(df[ts_for_time], errors="coerce")
                df["time"] = ts.dt.strftime("%H:%M:%S")
        else:
            df["time"] = np.nan

        # 반환 컬럼 구성
        cols = ["sample_id", "y_true", "y_prob", "y_pred(τ)"]
        if mold_col: cols = [mold_col] + cols
        if ts_col:   cols = [ts_col]   + cols
        if "time" in df.columns and "time" not in cols:
            cols = ["time"] + cols

        return df[cols]

    # --- 성능 지표 계산 ---
    @reactive.calc
    def metrics():
        df = view_df()
        if df.empty:
            return {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                    "tp": 0, "fp": 0, "fn": 0, "n": 0}
        y_t = df["y_true"].to_numpy()
        y_p = df["y_pred(τ)"].to_numpy()
        n = len(y_t)
        acc = float((y_t == y_p).mean()) if n > 0 else 0.0
        tp = int(np.sum((y_t == 1) & (y_p == 1)))
        fp = int(np.sum((y_t == 0) & (y_p == 1)))
        fn = int(np.sum((y_t == 1) & (y_p == 0)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (2 * precision * recall) / max(precision + recall, 1e-9)
        return {"acc": acc, "precision": precision, "recall": recall, "f1": f1,
                "tp": tp, "fp": fp, "fn": fn, "n": n}

    # --- KPI 바 ---
    @output(id="mon_kpi_bar")
    @render.ui
    def _mon_kpi_bar():
        m = metrics()

        def kcard(title, value, subtitle=None):
            return ui.div(
                ui.div(
                    ui.div(title, class_="title"),
                    ui.div(value, class_="value"),
                    ui.div(subtitle or "", class_="muted") if subtitle else None,
                    class_="p-3"
                ),
                class_="kcard"
            )

        # 보조 부제: 최근 N개 기준
        sel = input.mon_mold_code() if hasattr(input, "mon_mold_code") else "전체"
        subtitle = f"{sel} · 최근 {m['n']}건"

        return ui.div(
            kcard("정확도", f"{m['acc']:.3f}", subtitle),
            kcard("정밀도", f"{m['precision']:.3f}"),
            kcard("재현율", f"{m['recall']:.3f}"),
            kcard("F1-score", f"{m['f1']:.3f}"),
            class_="kpi-row"
        )

    # --- 배치 기준 시계열 그래프 ---
    @output(id="mon_timeseries_plot")
    @render.ui
    def _mon_timeseries_plot():
        df = view_df()
        if df.empty:
            return ui.p("데이터 없음", class_="text-muted")

        # ✅ time 없으면 datetime 등에서 즉석 생성
        if "time" not in df.columns:
            ts_col = next((c for c in ["timestamp", "datetime", "ts"] if c in df.columns), None)
            if ts_col:
                tmp_ts = pd.to_datetime(df[ts_col], errors="coerce")
                df = df.assign(time=tmp_ts.dt.strftime("%H:%M:%S"))
            else:
                return ui.p("'time' 칼럼이 없습니다. time(시:분[:초]) 칼럼을 추가해 주세요.", class_="text-muted")

        # time → 공통 기준일(2000-01-01)로 파싱
        t_parsed = pd.to_datetime(df["time"], errors="coerce")
        needs_rescan = t_parsed.isna() & df["time"].notna()
        if needs_rescan.any():
            t_parsed.loc[needs_rescan] = pd.to_datetime("2000-01-01 " + df.loc[needs_rescan, "time"].astype(str),
                                                        errors="coerce")
        _tod = pd.to_datetime(t_parsed.dt.strftime("2000-01-01 %H:%M:%S"), errors="coerce")

        dfd = df.assign(_tod=_tod).dropna(subset=["_tod"]).sort_values("_tod").reset_index(drop=True)
        if dfd.empty:
            return ui.p("유효한 time 값이 없습니다.", class_="text-muted")

        n = len(dfd)
        if n < (MON_BATCH_SIZE if not INCLUDE_PARTIAL_BATCH else 1):
            return ui.p(f"배치 계산을 위해 최소 {MON_BATCH_SIZE}개 이상이 권장됩니다.", class_="text-muted")

        # 배치 인덱스
        dfd["_batch"] = (np.arange(n) // MON_BATCH_SIZE).astype(int)

        # 완전 배치만 사용할 경우 필터
        if not INCLUDE_PARTIAL_BATCH:
            full_batches = (dfd["_batch"].value_counts().sort_index() >= MON_BATCH_SIZE)
            keep_batches = set(full_batches[full_batches].index.tolist())
            dfd = dfd[dfd["_batch"].isin(keep_batches)]
            if dfd.empty:
                return ui.p(f"{MON_BATCH_SIZE}개 단위의 완전 배치가 아직 없습니다.", class_="text-muted")

        # 배치별 Precision/Recall/F1 계산
        def prf1_for_group(g: pd.DataFrame):
            yt = g["y_true"].astype(int).to_numpy()
            yp = g["y_pred(τ)"].astype(int).to_numpy()
            tp = ((yt == 1) & (yp == 1)).sum()
            fp = ((yt == 0) & (yp == 1)).sum()
            fn = ((yt == 1) & (yp == 0)).sum()
            p  = tp / max(tp + fp, 1)
            r  = tp / max(tp + fn, 1)
            f1 = (2 * p * r) / max(p + r, 1e-9)
            # x축 대표 시간
            if X_ALIGN == "center":
                mid_idx = len(g) // 2
                x_time = g["_tod"].iloc[mid_idx]
            else:  # "end"
                x_time = g["_tod"].iloc[-1]
            return pd.Series({"x_time": x_time, "precision": p, "recall": r, "f1": f1, "count": len(g)})

        agg = dfd.groupby("_batch", sort=True, as_index=False).apply(prf1_for_group)
        agg = agg.sort_values("x_time")

        # Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=agg["x_time"], y=agg["precision"], mode="lines+markers", name=f"Precision[{MON_BATCH_SIZE}]"))
        fig.add_trace(go.Scatter(x=agg["x_time"], y=agg["recall"],    mode="lines+markers", name=f"Recall[{MON_BATCH_SIZE}]"))
        fig.add_trace(go.Scatter(x=agg["x_time"], y=agg["f1"],        mode="lines+markers", name=f"F1[{MON_BATCH_SIZE}]"))

        # (선택) 배치 경계선: 각 배치 끝 시간에 세로선
        if SHOW_BATCH_LINES:
            end_times = dfd.groupby("_batch")["_tod"].max().sort_index().tolist()
            for xt in end_times:
                fig.add_vline(x=xt, line_width=1, line_dash="dot", opacity=0.2)

        fig.update_xaxes(type="date", tickformat="%H:%M", title_text="시간")
        fig.update_layout(
            template="plotly_white",
            height=380,
            margin=dict(l=50, r=20, t=40, b=40),
            yaxis=dict(title="Score", range=[0.0, 1.0]),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            title=f"{MON_BATCH_SIZE}개 단위 Batch Precision/Recall/F1"
        )
        fig.update_traces(hovertemplate="시간=%{x|%H:%M:%S}<br>Score=%{y:.3f}")

        return ui.HTML(fig.to_html(include_plotlyjs='cdn', div_id="mon_timeseries"))

    # --- 샘플 테이블 (최근 10개) ---
    @output(id="mon_sample_table")
    @render.table
    def _mon_sample_table():
        df = view_df().copy()
        if df.empty:
            return pd.DataFrame({"상태": ["예측 결과 없음"]})
    
        mold_col = _find_mold_col(df)
    
        # 최신이 위로
        df = df.sort_values("sample_id", ascending=False).reset_index(drop=True)
    
        # 판정 플래그
        df["flag"] = np.where(
            (df["y_true"] == 1) & (df["y_pred(τ)"] == 0), "❗ FN",
            np.where(
                (df["y_true"] == 0) & (df["y_pred(τ)"] == 1), "⚠️ FP",
                np.where((df["y_true"] == 1) & (df["y_pred(τ)"] == 1), "✅ TP", "✅ TN")
            )
        )
    
        # 보기 컬럼
        cols = ["sample_id", "y_true", "y_prob", "y_pred(τ)", "flag"]
        if mold_col and mold_col in df.columns:
            cols.insert(1, mold_col)
        df = df[cols].copy()
    
        # 표기 정리
        df["y_prob"] = df["y_prob"].astype(float).round(3)
        rename_dict = {
            "sample_id": "샘플ID",
            "y_true": "실제",
            "y_prob": "불량확률",
            "y_pred(τ)": "예측(τ)",
            "flag": "판정"
        }
        if mold_col and mold_col in df.columns:
            rename_dict[mold_col] = "몰드코드"
        df.rename(columns=rename_dict, inplace=True)
    
        # 최근 10건만
        return df.head(10)
    
    # --- 오류 샘플 테이블 (FP/FN만) ---
    @output(id="mon_error_table")
    @render.table
    def _mon_error_table():
        df = view_df().copy()
        if df.empty:
            return pd.DataFrame({"상태": ["예측 결과 없음"]})
    
        mold_col = _find_mold_col(df)
    
        # 판정 플래그
        df["flag"] = np.where(
            (df["y_true"] == 1) & (df["y_pred(τ)"] == 0), "❗ FN",
            np.where(
                (df["y_true"] == 0) & (df["y_pred(τ)"] == 1), "⚠️ FP",
                "OK"  # TP/TN
            )
        )
    
        # FP/FN만 추출, 최신순
        err = df[df["flag"].isin(["❗ FN", "⚠️ FP"])].copy()
        if err.empty:
            return pd.DataFrame({"상태": ["FP/FN 오류 없음(최근 창 기준)"]})
    
        err = err.sort_values("sample_id", ascending=False).reset_index(drop=True)
    
        # 보기 컬럼
        cols = ["sample_id", "y_true", "y_prob", "y_pred(τ)", "flag"]
        if mold_col and mold_col in err.columns:
            cols.insert(1, mold_col)
        err = err[cols].copy()
    
        # 표기 정리
        err["y_prob"] = err["y_prob"].astype(float).round(3)
        rename_dict = {
            "sample_id": "샘플ID",
            "y_true": "실제",
            "y_prob": "불량확률",
            "y_pred(τ)": "예측",
            "flag": "판정"
        }
        if mold_col and mold_col in err.columns:
            rename_dict[mold_col] = "몰드코드"
        err.rename(columns=rename_dict, inplace=True)
    
        # 많을 수 있으니 최근 30건만 노출
        return err.head(30)
