# modules/page_monitoring.py
from shiny import ui, render, reactive
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from scipy.stats import ks_2samp
import os

# 🔹 실시간 예측 결과 및 현재 상태 공유
from shared import prediction_state, current_state

# ===== 모니터링 설정 =====
MON_BATCH_SIZE = 30          # 배치 크기
MON_LOOKBACK_BATCHES = 5      # 최근 몇 배치를 볼지(총 행수 = 배치 크기 * 개수)
INCLUDE_PARTIAL_BATCH = False  # 마지막 부분 배치 포함 여부
X_ALIGN = "end"               # "end" or "center" (배치의 끝/중앙 시간)
SHOW_BATCH_LINES = True       # 배치 경계선 표시

# 선입력 데이터 파일 경로
PRELOADED_DATA_PATH = "./data/train2.csv"

# 수치형 변수 리스트
NUMERIC_VARS = [
    'molten_volume', 'molten_temp', 'facility_operation_cycleTime',
    'production_cycletime', 'low_section_speed', 'high_section_speed',
    'cast_pressure', 'biscuit_thickness', 'upper_mold_temp1',
    'upper_mold_temp2', 'lower_mold_temp1', 'lower_mold_temp2',
    'sleeve_temperature', 'physical_strength', 'Coolant_temperature'
]

# K-S 테스트 유의수준
KS_ALPHA = 0.05

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
            .btn-primary {background: #3b82f6; color: white; border: none; padding: .6rem 1.2rem; border-radius: 8px; font-weight: 600; cursor: pointer}
            .btn-primary:hover {background: #2563eb}
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

        # KPI 한 줄 (몰드코드 선택 + 지표 선택 추가)
        ui.div(
            ui.card(
                ui.div(
                    ui.div(
                        ui.span("실시간 성능 지표", style="font-weight: 800; font-size: 1rem;"),
                        ui.div(
                            ui.input_select(
                                "mon_mold_code",
                                "몰드코드",
                                choices=["전체", "8412", "8413", "8576", "8722", "8917"],
                                selected="전체",
                                width="140px"
                            ),
                            ui.input_select(
                                "mon_metric_select",
                                "표시 지표",
                                choices=["Accuracy", "Precision", "Recall", "F1-Score"],
                                selected="F1-Score",
                                width="140px"
                            ),
                            style="display: inline-flex; gap: 1rem; margin-left: 1rem;"
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
        
        # ───────── 분포 비교 버튼 ─────────
        ui.div(
            ui.card(
                ui.div(
                    ui.h5("변수별 분포 비교 (K-S 테스트)", style="margin: 0; font-weight: 700;"),
                    ui.p("선입력 데이터와 실시간 데이터의 수치형 변수 분포를 비교합니다.", 
                         class_="muted", style="margin-top: 0.5rem; margin-bottom: 1rem;"),
                    ui.input_action_button(
                        "btn_ks_test",
                        "결과 확인",
                        class_="btn-primary"
                    ),
                    style="padding: 1rem;"
                )
            ),
            class_="container section"
        ),
        
        # ───────── 변수별 분포 비교 + 오류 샘플(FP/FN) (2열) ─────────
        ui.div(
            ui.layout_columns(
                ui.card(
                    ui.card_header("변수별 분포 비교 결과 (K-S Test)"),
                    ui.div(ui.output_table("ks_test_table"), class_="scroll-table")
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
    
    # ▼ 기본값
    DEFAULT_TAU = 0.5
    DEFAULT_NSHOW = MON_BATCH_SIZE * MON_LOOKBACK_BATCHES
    
    # --- 실시간 데이터 뷰 ---
    @reactive.calc
    def view_df() -> pd.DataFrame:
        """prediction_state()에서 실시간 데이터 가져오기"""
        df = prediction_state()
        
        if df is None or df.empty:
            return pd.DataFrame()
        
        # 필수 컬럼 체크
        required_cols = {"pred", "prob", "actual"}
        if not required_cols.issubset(df.columns):
            return pd.DataFrame()
        
        # 컬럼 정규화
        df = df.copy()
        if "y_true" not in df.columns:
            df["y_true"] = df["actual"]
        if "y_pred(τ)" not in df.columns:
            df["y_pred(τ)"] = df["pred"]
        if "y_prob" not in df.columns:
            df["y_prob"] = df["prob"]
        
        # 몰드코드 필터
        sel_mold = input.mon_mold_code() if hasattr(input, "mon_mold_code") else "전체"
        mold_col = _find_mold_col(df)
        
        if sel_mold != "전체" and mold_col and mold_col in df.columns:
            df[mold_col] = df[mold_col].astype(str)
            df = df[df[mold_col] == sel_mold].copy()
        
        return df.reset_index(drop=True)
    
    # --- Macro 평균 지표 계산 ---
    @reactive.calc
    def metrics():
        """현재 실시간 데이터의 macro avg 지표"""
        df = view_df()
        
        if df.empty:
            return {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "n": 0}
        
        if not {"y_true", "y_pred(τ)"}.issubset(df.columns):
            return {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "n": 0}
        
        try:
            y_true = df["y_true"].astype(int).to_numpy()
            y_pred = df["y_pred(τ)"].astype(int).to_numpy()
        except Exception:
            return {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "n": 0}
        
        TP = ((y_true == 1) & (y_pred == 1)).sum()
        FP = ((y_true == 0) & (y_pred == 1)).sum()
        TN = ((y_true == 0) & (y_pred == 0)).sum()
        FN = ((y_true == 1) & (y_pred == 0)).sum()
        
        n_total = TP + FP + TN + FN
        
        if n_total == 0:
            return {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "n": 0}
        
        # Accuracy
        acc = (TP + TN) / n_total
        
        # 실제로 존재하는 클래스만 Macro 평균에 포함
        precisions = []
        recalls = []
        f1_scores = []
        
        # Class 0 (정상) - TN 또는 FP가 있으면 계산
        if (TN + FP) > 0:
            prec_0 = TN / (TN + FN) if (TN + FN) > 0 else 0
            rec_0 = TN / (TN + FP) if (TN + FP) > 0 else 0
            f1_0 = 2 * prec_0 * rec_0 / (prec_0 + rec_0) if (prec_0 + rec_0) > 0 else 0
            precisions.append(prec_0)
            recalls.append(rec_0)
            f1_scores.append(f1_0)
        
        # Class 1 (불량) - TP 또는 FN이 있으면 계산
        if (TP + FN) > 0:
            prec_1 = TP / (TP + FP) if (TP + FP) > 0 else 0
            rec_1 = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1_1 = 2 * prec_1 * rec_1 / (prec_1 + rec_1) if (prec_1 + rec_1) > 0 else 0
            precisions.append(prec_1)
            recalls.append(rec_1)
            f1_scores.append(f1_1)
        
        # Macro 평균 (존재하는 클래스만 평균)
        macro_prec = sum(precisions) / len(precisions) if precisions else 0.0
        macro_rec = sum(recalls) / len(recalls) if recalls else 0.0
        macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        
        return {
            "acc": acc,
            "precision": macro_prec,
            "recall": macro_rec,
            "f1": macro_f1,
            "n": n_total
        }
    
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

        sel = input.mon_mold_code() if hasattr(input, "mon_mold_code") else "전체"
        subtitle = f"{sel} · 실시간 {m['n']}건 (macro avg)"

        return ui.div(
            kcard("정확도", f"{m['acc']:.3f}", subtitle),
            kcard("정밀도", f"{m['precision']:.3f}"),
            kcard("재현율", f"{m['recall']:.3f}"),
            kcard("F1-score", f"{m['f1']:.3f}"),
            class_="kpi-row"
        )
    
    # --- 시계열 그래프 ---
    @output(id="mon_timeseries_plot")
    @render.ui
    def _mon_timeseries_plot():
        df = view_df()
        
        if df.empty:
            return ui.p("실시간 데이터가 없습니다.", class_="text-muted")
        
        if not {"y_true", "y_pred(τ)"}.issubset(df.columns):
            return ui.p("필요한 컬럼이 없습니다.", class_="text-muted")
        
        # time 컬럼 확인 및 생성
        if "time" not in df.columns:
            ts_col = _find_ts_col(df)
            if ts_col:
                tmp_ts = pd.to_datetime(df[ts_col], errors="coerce")
                df = df.assign(time=tmp_ts.dt.strftime("%H:%M:%S"))
            else:
                # 시간 정보가 없으면 임시로 생성
                df = df.assign(time=pd.date_range("2000-01-01", periods=len(df), freq="1min").strftime("%H:%M:%S"))
        
        # time → datetime 파싱
        t_parsed = pd.to_datetime(df["time"], errors="coerce")
        needs_rescan = t_parsed.isna() & df["time"].notna()
        if needs_rescan.any():
            t_parsed.loc[needs_rescan] = pd.to_datetime("2000-01-01 " + df.loc[needs_rescan, "time"].astype(str), errors="coerce")
        _tod = pd.to_datetime(t_parsed.dt.strftime("2000-01-01 %H:%M:%S"), errors="coerce")
        
        df = df.assign(_tod=_tod).dropna(subset=["_tod"]).sort_values("_tod").reset_index(drop=True)
        
        if df.empty:
            return ui.p("유효한 time 값이 없습니다.", class_="text-muted")
        
        n = len(df)
        
        # 최소 30개 이상 필요
        if n < MON_BATCH_SIZE:
            return ui.p(f"시계열 그래프 표시를 위해 최소 {MON_BATCH_SIZE}개 이상의 데이터가 필요합니다. (현재: {n}개)", class_="text-muted")
        
        # 배치 인덱스 생성
        df["_batch"] = (np.arange(n) // MON_BATCH_SIZE).astype(int)
        
        # 완전 배치만 사용 (부분 배치 제거)
        full_batches = (df["_batch"].value_counts().sort_index() >= MON_BATCH_SIZE)
        keep_batches = set(full_batches[full_batches].index.tolist())
        df = df[df["_batch"].isin(keep_batches)]
        
        if df.empty:
            return ui.p(f"{MON_BATCH_SIZE}개 단위의 완전 배치가 아직 없습니다.", class_="text-muted")
        
        # 누적 지표 계산: 각 배치까지의 누적 데이터로 계산
        cumulative_results = []
        
        for batch_id in sorted(df["_batch"].unique()):
            # 현재 배치까지의 모든 데이터 (누적)
            cumulative_data = df[df["_batch"] <= batch_id].copy()
            
            yt = cumulative_data["y_true"].astype(int).to_numpy()
            yp = cumulative_data["y_pred(τ)"].astype(int).to_numpy()
            
            # 누적 혼동행렬
            TP = ((yt == 1) & (yp == 1)).sum()
            FP = ((yt == 0) & (yp == 1)).sum()
            TN = ((yt == 0) & (yp == 0)).sum()
            FN = ((yt == 1) & (yp == 0)).sum()
            
            n_total = TP + FP + TN + FN
            
            if n_total == 0:
                continue
            
            # 실제로 존재하는 클래스만 Macro 평균에 포함
            acc = (TP + TN) / n_total
            
            precisions = []
            recalls = []
            f1_scores = []
            
            # Class 0 (정상)
            if (TN + FP) > 0:
                prec_0 = TN / (TN + FN) if (TN + FN) > 0 else 0
                rec_0 = TN / (TN + FP) if (TN + FP) > 0 else 0
                f1_0 = 2 * prec_0 * rec_0 / (prec_0 + rec_0) if (prec_0 + rec_0) > 0 else 0
                precisions.append(prec_0)
                recalls.append(rec_0)
                f1_scores.append(f1_0)
            
            # Class 1 (불량)
            if (TP + FN) > 0:
                prec_1 = TP / (TP + FP) if (TP + FP) > 0 else 0
                rec_1 = TP / (TP + FN) if (TP + FN) > 0 else 0
                f1_1 = 2 * prec_1 * rec_1 / (prec_1 + rec_1) if (prec_1 + rec_1) > 0 else 0
                precisions.append(prec_1)
                recalls.append(rec_1)
                f1_scores.append(f1_1)
            
            # Macro 평균
            macro_prec = sum(precisions) / len(precisions) if precisions else 0.0
            macro_rec = sum(recalls) / len(recalls) if recalls else 0.0
            macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
            
            # x축 대표 시간: 현재 배치의 마지막 시간
            batch_data = df[df["_batch"] == batch_id]
            if X_ALIGN == "center":
                mid_idx = len(batch_data) // 2
                x_time = batch_data["_tod"].iloc[mid_idx]
            else:  # "end"
                x_time = batch_data["_tod"].iloc[-1]
            
            cumulative_results.append({
                "batch_id": batch_id,
                "x_time": x_time,
                "accuracy": acc,
                "precision": macro_prec,
                "recall": macro_rec,
                "f1": macro_f1,
                "n_cumulative": n_total
            })
        
        if not cumulative_results:
            return ui.p("누적 지표 계산 결과가 없습니다.", class_="text-muted")
        
        agg = pd.DataFrame(cumulative_results)
        
        # 선택된 지표
        selected_metric = input.mon_metric_select() if hasattr(input, "mon_metric_select") else "F1-Score"
        metric_key = _get_metric_key(selected_metric)
        
        # Plotly 그래프
        fig = go.Figure()
        
        # 메인 라인
        fig.add_trace(go.Scatter(
            x=agg["x_time"],
            y=agg[metric_key],
            mode="lines+markers",
            name=f"{selected_metric} (누적)",
            line=dict(width=3, color='#3b82f6'),
            marker=dict(size=8, color='#3b82f6')
        ))
        
        # 임계선 추가 (y=0.85)
        fig.add_hline(
            y=0.85,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text="임계값 (0.85)",
            annotation_position="right"
        )
        
        # 배치 경계선
        if SHOW_BATCH_LINES and len(df["_batch"].unique()) > 1:
            end_times = df.groupby("_batch")["_tod"].max().sort_index().tolist()
            for xt in end_times:
                fig.add_vline(x=xt, line_width=1, line_dash="dot", opacity=0.2, line_color="gray")
        
        fig.update_xaxes(type="date", tickformat="%H:%M", title_text="시간")
        fig.update_layout(
            template="plotly_white",
            height=380,
            margin=dict(l=50, r=20, t=40, b=40),
            yaxis=dict(title=selected_metric, range=[0.0, 1.0]),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            title=f"누적 {selected_metric} 추이 (30개 단위 업데이트, Macro Avg)"
        )
        fig.update_traces(
            hovertemplate="시간=%{x|%H:%M:%S}<br>Score=%{y:.3f}<br>누적 샘플=%{customdata}개",
            customdata=agg["n_cumulative"],
            selector=dict(mode='lines+markers')
        )
        
        return ui.HTML(fig.to_html(include_plotlyjs='cdn', div_id="mon_timeseries"))
    
    # --- K-S 테스트 결과 테이블 (버튼 클릭 시 실행) ---
    @output(id="ks_test_table")
    @render.table
    @reactive.event(input.btn_ks_test)  # 버튼 클릭 시에만 실행
    def _ks_test_table():
        # 선입력 데이터 로드
        if not os.path.exists(PRELOADED_DATA_PATH):
            return pd.DataFrame({"상태": [f"선입력 데이터 파일을 찾을 수 없습니다: {PRELOADED_DATA_PATH}"]})
        
        df_preloaded = pd.read_csv(PRELOADED_DATA_PATH)
        
        # 실시간 데이터
        df_current = current_state()
        if df_current is None or df_current.empty:
            return pd.DataFrame({"상태": ["실시간 데이터가 없습니다."]})
        
        # 몰드코드 필터링
        sel_mold = input.mon_mold_code() if hasattr(input, "mon_mold_code") else "전체"
        mold_col = _find_mold_col(df_preloaded)
        
        if mold_col is None:
            return pd.DataFrame({"상태": ["몰드코드 컬럼을 찾을 수 없습니다."]})
        
        # 선택한 몰드코드만 필터링
        if sel_mold == "전체":
            return pd.DataFrame({"상태": ["특정 몰드코드를 선택해주세요."]})
        
        # 🔧 데이터 타입 통일: 문자열로 변환하여 비교
        df_preloaded[mold_col] = df_preloaded[mold_col].astype(str)
        df_current[mold_col] = df_current[mold_col].astype(str)
        
        # 디버깅: 실제 값 확인
        print(f"📊 선택한 몰드코드: {sel_mold} (타입: {type(sel_mold)})")
        print(f"📊 선입력 데이터의 고유 몰드코드: {df_preloaded[mold_col].unique()}")
        print(f"📊 실시간 데이터의 고유 몰드코드: {df_current[mold_col].unique()}")
        
        df_pre_filtered = df_preloaded[df_preloaded[mold_col] == sel_mold].copy()
        df_cur_filtered = df_current[df_current[mold_col] == sel_mold].copy()
        
        if df_pre_filtered.empty:
            available_codes = df_preloaded[mold_col].unique().tolist()
            return pd.DataFrame({"상태": [f"선입력 데이터에 '{sel_mold}' 몰드코드가 없습니다. 사용 가능: {available_codes}"]})
        
        if df_cur_filtered.empty:
            available_codes = df_current[mold_col].unique().tolist()
            return pd.DataFrame({"상태": [f"실시간 데이터에 '{sel_mold}' 몰드코드가 없습니다. 사용 가능: {available_codes}"]})
        
        # K-S 테스트 수행
        results = []
        for var in NUMERIC_VARS:
            # 변수가 데이터프레임에 존재하는지 확인
            if var not in df_pre_filtered.columns or var not in df_cur_filtered.columns:
                results.append({
                    "몰드명": sel_mold,
                    "변수명": var,
                    "p-value": "N/A",
                    "결과": "변수 없음"
                })
                continue
            
            # 결측치 제거
            pre_data = df_pre_filtered[var].dropna()
            cur_data = df_cur_filtered[var].dropna()
            
            # 데이터가 충분한지 확인
            if len(pre_data) < 2 or len(cur_data) < 2:
                results.append({
                    "몰드명": sel_mold,
                    "변수명": var,
                    "p-value": "N/A",
                    "결과": "데이터 부족"
                })
                continue
            
            # K-S 테스트
            try:
                statistic, p_value = ks_2samp(pre_data, cur_data)
                result = "분포 달라짐" if p_value < KS_ALPHA else "분포 같음"
                
                results.append({
                    "몰드명": sel_mold,
                    "변수명": var,
                    "p-value": round(p_value, 4),
                    "결과": result
                })
            except Exception as e:
                results.append({
                    "몰드명": sel_mold,
                    "변수명": var,
                    "p-value": "오류",
                    "결과": str(e)[:20]
                })
        
        # 데이터프레임으로 변환
        result_df = pd.DataFrame(results)
        
        if result_df.empty:
            return pd.DataFrame({"상태": ["테스트 결과가 없습니다."]})
        
        return result_df
    
    # --- 오류 샘플 테이블 (FP/FN만) ---
    @output(id="mon_error_table")
    @render.table
    def _mon_error_table():
        df = view_df().copy()
        
        if df.empty:
            return pd.DataFrame({"상태": ["예측 결과 없음"]})
        
        mold_col = _find_mold_col(df)
        
        # 필수 컬럼 체크
        if not {"y_true", "y_pred(τ)"}.issubset(df.columns):
            return pd.DataFrame({"상태": ["필수 컬럼 누락"]})
        
        # 판정 플래그
        df["flag"] = np.where(
            (df["y_true"] == 1) & (df["y_pred(τ)"] == 0), "❗ FN",
            np.where(
                (df["y_true"] == 0) & (df["y_pred(τ)"] == 1), "⚠️ FP",
                "OK"  # TP/TN
            )
        )
        
        # FP/FN만 추출
        err = df[df["flag"].isin(["❗ FN", "⚠️ FP"])].copy()
        
        if err.empty:
            return pd.DataFrame({"상태": ["FP/FN 오류 없음 (모두 정답!)"]})
        
        # sample_id가 있으면 사용, 없으면 인덱스 사용
        if "sample_id" in err.columns:
            err = err.sort_values("sample_id", ascending=False).reset_index(drop=True)
        else:
            err = err.reset_index(drop=True)
            err.insert(0, "sample_id", range(len(err), 0, -1))
        
        # 보기 컬럼
        cols = ["sample_id", "y_true", "y_prob", "y_pred(τ)", "flag"]
        if mold_col and mold_col in err.columns:
            cols.insert(1, mold_col)
        
        # 컬럼 필터링 (존재하는 컬럼만)
        cols = [c for c in cols if c in err.columns]
        err = err[cols].copy()
        
        # 표기 정리
        if "y_prob" in err.columns:
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
        
        err.rename(columns={k: v for k, v in rename_dict.items() if k in err.columns}, inplace=True)
        
        return err.head(30)


# ==================== 헬퍼 함수 ====================
def _get_metric_key(metric_name):
    """UI 지표 이름 → 딕셔너리 키 매핑"""
    mapping = {
        "Accuracy": "accuracy",
        "Precision": "precision",
        "Recall": "recall",
        "F1-Score": "f1"
    }
    return mapping.get(metric_name, "f1")