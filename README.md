## 🏭 프로젝트 제목

> **본 프로젝트 2 – 다이캐스팅 공정 데이터 기반 운영·관리·분석·모니터링 대시보드**

---

## 📌 목차

1. [프로젝트 개요](#-프로젝트-개요)
2. [주요 데이터 컬럼 소개](#-주요-데이터-컬럼-소개)
3. [데이터 분석 및 전처리 과정](#-데이터-분석-및-전처리-과정)
4. [결론](#-결론)
5. [사용 기술](#-사용-기술)
6. [프로젝트 구조](#-프로젝트-구조)
7. [팀원 소개](#-팀원-소개)

---

## 📝 프로젝트 개요

- **목적**: 연속 공정(sensor → 사출/충전 → 응고)에서 발생하는 **실시간/준실시간 데이터**를 표준화·정제하여 **현장 운영자·품질관리(QC)·데이터/AI팀**에 **역할별 대시보드**로 제공하고, **이상 탐지·불량 예측·관리도 기반 통제**를 일원화.
- **핵심 기능(프로젝트2 가이드 반영)**
  - **현장 운영자**: 1초 단위 스파크라인 갱신, **임계치 + 모델 이중 알림**, OEE 및 라인 상태 보드, 이벤트 로그
  - **품질관리(QC)**: **P/NP, X̄–R/S 관리도**, **Cp/Cpk** 산출, 상관 히트맵, **Nelson Rules** 위반 알림, 주/월 리포트 다운로드
  - **데이터/AI**: 금형별 분류모델 학습/평가(시계열 검증), **Isolation Forest 이상 탐지**, **SHAP/PDP 해석**, 버전·성능 로그
- **배경/문제의식**
  - 리셋(`count`) 직후/저온 구간에서 불량 집중, 금형별(`mold_code`) 특성 상이, 센서 이상치·결측 다수 → **표준화·알림·XAI** 필요
  - 현업 니즈: “불량 사후확인”보다 **1초라도 빠른 이상 포착**과 **원인 가시화**가 중요
- **기대효과**
  - 임계+모델 알림으로 **불량률 저감** / **초기-후기 취약구간 선제 대응**
  - 금형별 최적 조건/레시피 재현 → **품질 안정화**
  - **설명 가능한 AI(XAI)**로 원인·조정 방향을 **즉시 의사결정**에 연결  
    :contentReference[oaicite:0]{index=0}

---

## 📂 주요 데이터 컬럼 소개

| 공정 단계             | 주요 컬럼                                                                                                 | Shiny UI 입력 요소                                                                 | 설명 / 활용 목적                            |
| --------------------- | --------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------- |
| **1) 용탕 준비·가열** | `molten_temp`, `molten_volume`                                                                            | `ui.input_slider("molten_temp")`<br>`ui.input_slider("molten_volume")`             | 저온/부족 용탕 시 불량↑                     |
| **2) 슬러리 제조**    | `sleeve_temperature`, `EMS_operation_time`                                                                | `ui.input_slider("sleeve_temperature")`<br>`ui.input_select("EMS_operation_time")` | 워밍업·초기 안정성 영향                     |
| **3) 사출·금형 충전** | `cast_pressure`, `low_section_speed`, `high_section_speed`, `physical_strength`, `biscuit_thickness`      | `ui.input_slider` 다수                                                             | 압력/속도/형체력 조건 → 기공/수축 주요 요인 |
| **4) 응고·냉각**      | `upper_mold_temp1/2`, `lower_mold_temp1/2`, `Coolant_temperature`                                         | `ui.input_slider` 다수                                                             | 금형/냉각수 온도 → 응고 균일성              |
| **기타(전체)**        | `mold_code`, `working`, `count`, `facility_operation_cycleTime`, `production_cycletime`, `tryshot_signal` | `ui.input_select`, `ui.input_numeric`, `ui.input_checkbox` 등                      | 금형/작업/사이클 메타 → 리셋·트라이샷 반영  |

> _비고_: 1차 프로젝트의 변수 제외/보정 규칙(상수·중복·결측·이상치)을 승계하되, 스트리밍 환경에 맞춰 **rolling fill + winsorization**을 추가.

---

## 🔍 데이터 분석 및 전처리 과정

### 0) 데이터 요약

- **기간**: 2019-01-02 ~ 2019-03-12 / **행 수**: 73,612
- **타깃**: `passorfail` (0=양품, 1=불량) — **불균형(~4–5%)**
- **특징**: 실제 주조 로그(온도/압력/속도/형체력/냉각/메타)로 **공정 단계-변수 연결**이 명확 → 설명 가능한 모델 구성 용이  
  :contentReference[oaicite:1]{index=1}

### 1) EDA 핵심

- **상관/분포**:
  - `molten_volume`↔`molten_temp` **음의 상관**, `low`↔`high_section_speed` **양의 상관**, 상형/하형 온도 **양의 상관**
  - **압력-두께**: `cast_pressure` < 150 → **불량 100%**, `cast_pressure` ≥ 300 & `biscuit_thickness` 40~60 → **양품 집중**
- **시간/Count 패턴**: **초기(1~6)** 불량률 높음, **후기(330~334)** 재상승 → **중간 구간 안정**  
  :contentReference[oaicite:2]{index=2}

### 2) 전처리(배치 + 스트리밍 공통 원칙)

- 센서 오류/비현실값 제거: `molten_temp<100`, 스파이크(예: 65535), 전구간 0 등
- 결측: 금형별 평균/보간, `tryshot_signal` 결측 → 정상값으로 대치
- 라벨/상태 정합: `working=정지`인데 양품 → 상태 수정
- 상수/결측 과다 컬럼 제거: `upper/lower_mold_temp3` 등
- 파생: `Count_stage`(초반/후반), `EMS_operation_time` 범주화
- **전처리 결과**: 핵심 변수 21개로 축약, 학습가능 형태로 정제  
  :contentReference[oaicite:3]{index=3}

### 3) 데이터 분할·검증 전략

- **금형별 분할** + **시계열 보존(80:20)**, Train의 20%로 **시계열 Valid** 생성
- 시계열 교차검증(**TimeSeriesSplit**, expanding window, n_splits=5)
- 목적: **누수 방지·일반화 성능 확보**  
  :contentReference[oaicite:4]{index=4}

### 4) 이상 탐지(Isolation Forest)

- **스케일·PCA 적용 비교** 결과: **비적용 모델이 탐지 성능 우수** → **비적용 채택**
- 금형별 최적 contamination 검색, 라벨(-1:이상, 1:정상) 매핑  
  :contentReference[oaicite:5]{index=5}

### 5) 분류모델(불량 예측)

- **모델군**: LightGBM, XGBoost, **RandomForest(최종)**
- **불균형 해소**: **MajorityVoteSMOTENC**(범주형 다수결 + 수치형 보간), 진성불량(`realfail`) 중심 증강
- 전처리 파이프라인: **OHE + RobustScaler**
- 하이퍼파라미터: **BayesSearchCV**
- **지표 우선순위**: *미검 최소화*를 위해 **F2** 최우선(Recall 가중)
- **모델 임계값**: 금형별 운영상 통일(0.5)로 설정  
  :contentReference[oaicite:6]{index=6}

### 6) 대시보드 설계(역할별 탭)

- **현장**: 실시간 센서 모니터링(1s), **이상치 탐지 로그**(시:분:초), **OEE 상태판**
- **QC**: 관리도(P/NP, X̄–R/S) + **Nelson Rules(1,2,3,5) 경고**, 단·다변량 관리도, 상관 히트맵, 리포트 다운로드
- **데이터/AI**: 모델 설명·버전 로그, 실시간 성능(정확도/정밀도/재현율/F1) 패널, 예측 실행, **SHAP Force Plot**  
  :contentReference[oaicite:7]{index=7}

---

## 🏛 결론

### 1) 핵심 발견

- **취약 구간**: **초기(1~6)**·**후기(330~334)**, **저온 영역**에서 불량 집중
- **공정 조건**: **압력 <150** 절대 위험 / **압력 ≥300 & 두께 40–60** 안정
- **금형 이질성**: 금형별 패턴 상이 → **금형별 모델·레시피가 효과적**  
  :contentReference[oaicite:8]{index=8}

### 2) 운영 가이드(대시보드 적용)

- **이중 알림**: 임계 위반 + 모델(이상/불량확률 상승) 동시 충족 시 **High Priority**
- **조정 순서**: Rule 보정 → **SHAP 상위 변수**(압력·온도·형체력·두께) 순으로 조정
- **목표**: 불량확률 **< 30%** 유지 / **Recall 우선(F2 최적화)**

---

## 🛠 사용 기술

**언어**

- Python 3.11

**라이브러리 & 프레임워크**

- 데이터 처리: `pandas`, `numpy`
- 분석/알고리즘: `scikit-learn`, `imbalanced-learn`, `xgboost`, `shap`, `scipy`, `statsmodels`
- 시각화: `matplotlib`, `seaborn`, `plotly`
- 웹/대시보드: **Shiny for Python(Core)** (`shiny`, `shinywidgets`)
- 기타 유틸: `joblib`, `pyyaml`, `tqdm`

**환경**

- Jupyter Notebook, VSCode
- Git/GitHub

---

## 📁 프로젝트 구조

```plaintext
📦 diecasting-project
 ┣ 📂 data              # 원본/전처리/스냅샷
 ┣ 📂 models            # 학습 모델·explainer(.pkl), 성능 로그
 ┣ 📂 modules           # Shiny 모듈
 ┃ ┣ ingest.py         # (옵션) 실시간 수집 mock/연동
 ┃ ┣ page_ops.py       # 현장 운영(실시간/알림/OEE/로그)
 ┃ ┣ page_qc.py        # 품질(QC: Cp/Cpk, 관리도, 히트맵, 리포트)
 ┃ ┣ page_ai.py        # 데이터/AI(학습·평가·XAI·버전)
 ┃ ┗ utils.py          # 전처리·임계·캐시·공통 위젯
 ┣ 📂 viz               # 스파크라인/관리도/SHAP/히트맵 등
 ┣ 📂 www               # CSS/폰트/아이콘
 ┣ 📜 app.py            # Shiny 메인(탭 라우팅/1s 갱신)
 ┣ 📜 shared.py         # 공통 로더/전역 상태
 ┣ 📜 README.md         # (본 문서)
 ┣ 📜 requirements.txt  # pip 환경
 ┗ 📜 environment.yml   # conda 환경

## 👥 팀원 소개

| 이름   | 주요 역할                          | 담당 업무                                                                                           |
| ------ | ---------------------------------- | --------------------------------------------------------------------------------------------------- |
| 김선준 | 데이터 전처리 / 보고서 작성 / 발표 | 결측치·이상치 처리 로직 설계, 변수 정제 및 임계치 설정 / 데이터 전처리 보고서 작성 / 최종 발표 주도    |
| 이우영 | 데이터 전처리 /발표              | 원본 데이터 클리닝, 이상치 검출 및 보정, 불량 구간 기준 도출 / 데이터 분석 보고서 작성 / 최종 발표 주도 |
| 임지원 | 프로젝트 매니저 (PM) / 모델링     | 프로젝트 일정 관리, 역할 분담 및 조율 / 발표 자료 제작 총괄 / RandomForest 최적화                  |
| 장승규 | 모델링                          | 머신러닝 모델 학습·튜닝, 성능 평가, 모델 비교 실험 /                                             |
| 황연주 | Tech Leader / 대시보드           | GitHub 형상 관리 / Shiny Core 구현, UI·UX 설계 및 시각화 통합                                 |
| 안형엽 | 대시보드                         | 불량 원인 분석 탭 개발, 시각화 요소 구현, 대시보드 기능 보완                                     |

> 모든 팀원이 **데이터 탐색** 과정에 참여했으며, 이후 전처리–모델링–대시보드–발표로 이어지는 전체 파이프라인을 협업하여 프로젝트를 완성했습니다.
```
