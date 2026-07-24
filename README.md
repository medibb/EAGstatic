# OpenBCI EAG Data Analyzer

OpenBCI BrainFlow RAW CSV 데이터를 분석하고 시각화하는 도구입니다.
EMG(근전도) 노이즈 제거에 최적화되어 있습니다.

## 기능

- 8채널 EEG 데이터 시각화 (개별/통합)
- EMG 노이즈 제거를 위한 Lowpass 필터 (기본 5Hz)
- 드리프트(기저선 변동) 보정
- 인터랙티브 데이터 선택 모드
- 결과 자동 저장 (result/ 폴더)

## 설치

```bash
# 가상환경 생성 (권장)
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 의존성 설치
pip install numpy pandas matplotlib scipy
```

## 저장소 구조

파이프라인은 **EAG 필터 → EAG+GRF 동기화 → edge/knee 추출 → 파라미터 추출 → 통계**의 흐름을 따릅니다. 2026-07 정리 시 현재 파이프라인(core)과 구코드(`old_code/`)·딥러닝 실험(`dl/`)을 분리했습니다.

```
EAGstatic/
├── eag_analyzer.py            # EAG 필터(LP 5Hz) 베이스
├── grf_viewer.py             # GRF/force plate 로딩 (sync_analyzer 하드 의존)
├── sync_analyzer.py           # EAG+GRF 동기화 엔진 (manual offset 반영)
├── offset_manager.py          # manual_offsets.json 관리
├── offset_review.py           # offset 검토 UI/worklist
├── edge_annotator.py          # EAG edge(knee) 검출
├── grf_triggered_annotator.py # GRF-anchored offset + knee 추출
├── edge_store.py              # manual_edges.json 관리
├── edge_editor.py             # edge 편집 CLI
├── edge_app.py                # edge 편집 GUI(로컬 브라우저)
├── parameter_extractor.py     # Phase 1/2/3 파라미터 추출
├── frequency_analyzer.py      # Phase 3 주파수 (parameter_extractor 의존)
├── stats_grf_eag.py           # dose-response 통계 (주 분석)
│
├── stats_analyzer.py          # 보조: session-type RM-ANOVA/APA/ICC/effect size
├── plot_fsi_verification.py   # 보조: FSI/APA 검증 시각화 (Phase 1 QC)
│
├── old_code/                  # 대체·폐기된 구코드 (파이프라인 비의존)
│   ├── adjust_offset.py            # → offset_review + edge_app로 대체
│   ├── plot_alignment_verification.py  # → offset_review 패널로 대체
│   ├── compare_raw_filtered.py     # 원시/필터링 비교(디버그)
│   ├── check_initial_events.py     # 디버그
│   ├── debug_sync_events*.py       # 디버그
│   └── validate_sync_*.py          # 구 sync 검증
│
├── dl/                        # 딥러닝 실험 (통계 파이프라인과 별도 트랙)
│   ├── dl_dataset.py · dl_models.py · dl_train.py
│   └── PLAN_dl_analysis.md
│
├── data/                      # 원본 데이터 (git 제외)
│   └── 피험자명/OpenBCISession_YYYY-MM-DD/BrainFlow-RAW_*.csv
└── result/                    # 분석 결과 PNG/JSON (git 제외)
    ├── manual_offsets.json · manual_edges.json
    └── 피험자명/…
```

> `old_code/`·`dl/`의 어떤 모듈도 core 파일이 import하지 않으므로, 이동으로 파이프라인이 깨지지 않습니다. `old_code/` 스크립트는 루트 기준 상대 import를 쓰므로 단독 실행 시 루트에서 실행해야 합니다.

## 사용법

### 기본 실행 (인터랙티브 모드)

```bash
python eag_analyzer.py
```

data 폴더가 있으면 자동으로 인터랙티브 모드로 전환되어 피험자/세션/파일을 선택할 수 있습니다.

### 인터랙티브 모드 명시적 실행

```bash
python eag_analyzer.py -i
python eag_analyzer.py --interactive
```

### 특정 파일 분석

```bash
python eag_analyzer.py --file ./data/피험자/세션/BrainFlow-RAW_2024-01-01.csv
```

### 특정 디렉토리 전체 분석

```bash
python eag_analyzer.py --dir ./data/김O진(F23)_25.12.23
```

### 필터 설정 변경

```bash
# 저역통과 필터 주파수 변경 (기본: 5Hz)
python eag_analyzer.py --lowpass 10

# 저역통과 필터 비활성화
python eag_analyzer.py --no-lowpass

# 드리프트 보정 OFF
python eag_analyzer.py --no-drift

# 드리프트 보정 방법 변경 (moving average)
python eag_analyzer.py --drift-method moving --drift-window 2.0
```

### 표시 설정 변경

```bash
# 시작 시간 변경 (초기 불안정 구간 제외)
python eag_analyzer.py --start-time 3

# Y축 눈금 간격 변경
python eag_analyzer.py --y-tick 1000
```

### 조합 예시

```bash
# 저역통과 15Hz + 이동평균 드리프트 보정 + 시작시간 3초
python eag_analyzer.py --lowpass 15 --drift-method moving --start-time 3
```

### 현재 설정 확인

```bash
python eag_analyzer.py --show-config
```

## 명령줄 옵션

| 옵션 | 단축 | 설명 | 기본값 |
|------|------|------|--------|
| `--file` | `-f` | 분석할 CSV 파일 경로 | - |
| `--dir` | `-d` | CSV 파일이 있는 디렉토리 | - |
| `--output` | `-o` | 결과 저장 디렉토리 | result/ |
| `--interactive` | `-i` | 인터랙티브 모드 | - |
| `--lowpass` | `-lp` | 저역통과 필터 차단 주파수 (Hz) | 5.0 |
| `--no-lowpass` | - | 저역통과 필터 비활성화 | - |
| `--no-drift` | - | 드리프트 보정 비활성화 | - |
| `--drift-method` | - | 드리프트 보정 방법 (detrend/moving/none) | detrend |
| `--drift-window` | - | 이동평균 윈도우 크기 (초) | 1.0 |
| `--start-time` | `-st` | 표시 시작 시간 (초) | 5.0 |
| `--y-tick` | - | Y축 눈금 간격 | 1500 |
| `--show-config` | - | 현재 필터 설정 출력 후 종료 | - |

## 출력 그래프

1. **개별 채널 그래프** (`*_individual_channels.png`)
   - 8개 채널을 각각 별도의 서브플롯으로 표시
   - 통일된 Y축 범위로 채널 간 비교 용이

2. **통합 오버레이 그래프** (`*_overlay.png`)
   - 8개 채널을 하나의 그래프에 겹쳐서 표시

## 원시 vs 필터링 비교 도구

별도의 `compare_raw_filtered.py` 스크립트를 사용하여 원시 데이터와 필터링된 데이터를 비교할 수 있습니다.

```bash
# 인터랙티브 모드 (구코드는 old_code/에 위치, 루트에서 실행)
python old_code/compare_raw_filtered.py

# 특정 파일, 특정 채널
python old_code/compare_raw_filtered.py --file ./data/피험자/세션/BrainFlow-RAW.csv --channel 3

# 필터 설정 변경
python old_code/compare_raw_filtered.py --lowpass 10 --no-drift
```

## BrainFlow RAW 데이터 형식

| 컬럼 | 내용 |
|------|------|
| 0 | Sample Index |
| 1-8 | EEG Channels (8채널, µV) |
| 9-11 | Accelerometer (X, Y, Z) |
| 12 | Package Counter |
| 13-20 | Digital/Analog Aux channels |
| 22 | Timestamp (Unix timestamp) |
| 23 | Other marker |

## 라이선스

MIT License
