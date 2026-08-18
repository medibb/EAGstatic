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
pip install -r requirements.txt
```

`requirements.txt`에 통계 패키지(`pingouin` 필수, `statsmodels` 권장)가 포함됩니다.
`statsmodels`가 없으면 `stats_grf_eag.py`는 혼합모형 대신 OLS 기울기 검정으로 자동 대체됩니다.

## 저장소 구조

파이프라인은 **EAG 필터 → EAG+GRF 동기화 → edge/knee 추출 → 파라미터 추출 → 통계**의 흐름을 따릅니다. 2026-07 정리 시 현재 파이프라인(core)과 구코드(`old_code/`)·딥러닝 실험(`dl/`)을 분리했습니다.

```
EAGstatic/
├── eag_analyzer.py            # EAG 필터(LP 5Hz) 베이스
├── grf_viewer.py             # GRF/force plate 로딩 (sync_analyzer 하드 의존)
├── sync_analyzer.py           # EAG+GRF 동기화 엔진 (manual offset 반영)
├── offset_manager.py          # manual_offsets.json 관리
├── offset_review.py           # offset 검토 UI/worklist (패널 PNG + CLI 확정)
├── offset_app.py              # offset 정렬 GUI(로컬 브라우저, 점 찍어 맞추기)
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

## 분석 파이프라인 (실행 순서)

전체 흐름은 **동기화 → offset 진단·검토 → edge 수동수정 → 파라미터 추출 → 통계**입니다.
각 단계의 중간 산출물(JSON)을 다음 단계가 자동 조회하며, "사람이 확정한 값이 최우선" 원칙으로 자동 결과를 오버라이드합니다.

| 단계 | 명령 | 산출물 |
|------|------|--------|
| 1. offset 진단 (GRF-anchored edge-align, 동시성 latency≈0) | `python3 grf_triggered_annotator.py --dir data --channels 1 --offset-report` | `result/offset_report.csv` |
| 2. offset 수동검토 (needs_review 세션) | `python3 offset_review.py --dir data` → 패널 확인 → GUI `python3 offset_app.py --host 0.0.0.0 --port 8766` (GRF·EAG 대응점 클릭 → Save) · CLI `offset_review.py --set --subject <S> --session-name <ss> --offset <v>` | `result/manual_offsets.json` |
| 3. EAG edge 수동수정 (부정확 세션/채널) | GUI `python3 edge_app.py --host 0.0.0.0 --port 8765` · CLI `edge_editor.py review/add/delete/move/reset --session <dir> --channel <n>` | `result/manual_edges.json` |
| 4. 파라미터 추출 (Phase 1/2/3, GRF 전이별 knee-pair) | `python3 parameter_extractor.py --batch [--phase3]` | `result/phase1_params/grf_triggered_params_*.csv` 등 |
| 5. 통계 (단계별 부하 dose-response) | `python3 stats_grf_eag.py [--exclude-review]` | `result/stats/` |

**핵심 설계**
- **GRF-anchored offset**: 깨끗한 GRF 전이 열을 기준으로 EAG edge 열을 정렬(동시성 가정). cycle-skip은 매칭수 최대 + 작은 이동량 우선으로 억제, 큰 교정(|offset|>2)은 자동적용 보류하고 `needs_review`로 넘김.
- **knee-pair 파라미터**: GRF 전이(rise=부하/fall=이탈)마다 EAG onset+offset 두 knee → 체중부하 cycle당 4 knee. `amplitude`(변화 크기), `grf_step`(부하 단계=dose), `latency`(≈0) 등 기록.
- **수동 확정 우선**: `manual_offsets.json`(offset)·`manual_edges.json`(edge). 설정 시 자동 검출을 무시하고 확정값 사용 → review 후 4단계를 재실행하면 반영됨.
- **edge 검출 기본값**: `edge_annotator` min_amp 25, slope_k 3.0, drift ON. 동일 검출기를 GRF signed imbalance에도 적용해 전이를 검출.

> ⚠️ manual review로 offset/edge를 확정한 뒤에는 4단계(`parameter_extractor --batch`)를 **다시 실행**해야 확정값과 `load_pct`·`accepted` 등 최신 컬럼이 반영됩니다.
> 실험 프로토콜(한발서기 4회 × 부하 시작/이탈 = 8 이벤트)과 판정 기준은 `ANNOTATION_PROTOCOL.md`, 상세 통계 설계는 `STATS_PLAN.md` 참조.

## 문서 안내

| 문서 | 역할 | 대상 |
|---|---|---|
| **`ANNOTATION_PROTOCOL.md`** | **판정 기준의 정본.** 조작적 정의, 허용 오차, 애매 상황 규칙, rater 자격, 신뢰도 설계 | Methods 작성·검증 |
| `ANNOTATION_GUIDE.md` | 접속·조작·저장 실무 안내 | 연구원 |
| **이 문서** | 코드 구조, 파이프라인, CLI | 개발·재현 |
| `STATS_PLAN.md` | 통계 설계 (dose-response) | 분석 |

> 판정 기준은 `ANNOTATION_PROTOCOL.md` **한 곳에만** 존재합니다. 다른 문서는 링크만 하고
> 기준을 다시 쓰지 않습니다. 기준을 고칠 때는 그 문서만 고치면 됩니다.
>
> 구 `ONBOARDING.md`·`REVIEW_WORKFLOW.md`는 위 세 문서로 재배치되어 폐지되었습니다
> (git 이력에 보존).

## annotation CLI 레퍼런스

GUI(`ANNOTATION_GUIDE.md`)로 하는 작업의 CLI 등가물입니다. 판정 기준은
`ANNOTATION_PROTOCOL.md`를 따릅니다.

**검토 대상 추리기**

```bash
cat result/offset_review/worklist.csv     # offset 검토 대상 (세션 단위)
python3 offset_review.py --dir data       # offset worklist + 패널 PNG 생성
python3 edge_review.py  --dir data        # edge worklist 생성 (offset 확정 후 재실행, 약 7분)
python3 edge_review.py  --list            # edge 검토 대상 요약
```

offset worklist의 `reason`: `저match`(정렬 약함) · `edge매칭부족`(GRF 전이 대비 EAG edge
매칭 적음) · `큰교정보류(재검토)` · `대안제시(res=…)`

edge worklist의 `priority`: `high`(측정 불가 cycle·cycle 수 이상, 사람이 봐야 함) ·
`low`(한쪽만 자동 채택, 값은 확보됨) · 빈칸(정상)

주요 컬럼: `priority · labels · ok · n_cycles · n_measured_cycles · n_matched · n_edges ·
n_single_sided · n_noise · load_pct · amp · asym · offset_source · offset_pending`

> `offset_pending=True` 행은 offset 미확정 세션입니다. **그 세션의 edge를 손으로 고치면
> 헛수고입니다** (edge는 `te_corr` 프레임에 저장되므로 offset이 바뀌면 어긋남).

**offset 확정**

```bash
python3 offset_app.py --host 0.0.0.0 --port 8766                    # GUI (권장)
python3 offset_review.py --set --subject "(02.02_17)김종문_1" --session-name s2 --offset -0.15
python3 offset_review.py --list
python3 offset_review.py --clear --subject "(02.02_17)김종문_1" --session-name s2
python3 offset_review.py --session "data/(02.02_17)김종문_1/OpenBCISession_...-s2"  # 패널 재생성
```

**edge 확정**

```bash
python3 edge_app.py --host 0.0.0.0 --port 8765                      # GUI (권장)
python3 edge_editor.py review --session "<경로>" --channel 1         # 번호라벨 PNG + edge 테이블
python3 edge_editor.py add    --session "<경로>" --channel 1 --onset 30.1 --offset 30.6 --snap
python3 edge_editor.py move   --session "<경로>" --channel 1 --id 2 --onset 19.2 --offset 19.7 --snap
python3 edge_editor.py delete --session "<경로>" --channel 1 --id 3
python3 edge_editor.py reset  --session "<경로>" --channel 1         # 자동검출로 복귀
python3 edge_editor.py list
```

매 명령이 `result/edge_edit/{피험자}_{세션}_ch{N}_edit.png`를 갱신합니다.
한글 폴더명은 **따옴표**로 감싸세요.

**제외 라벨**

```bash
python3 exclusion_store.py --list
python3 exclusion_store.py --set --subject "(02.02_10)주창민_1" --session s1 \
        --channel 3 --reason 노이즈 --note "드리프트 심함"        # channel 0 = 세션 전체
python3 exclusion_store.py --clear --subject "..." --session s1 --channel 3
```

파이프라인 반영: `parameter_extractor` → `excluded` · `exclude_reason` 컬럼 /
`stats_grf_eag` → `excluded=True` 행 자동 제외 (`accepted=False`도 함께) /
`edge_review` → 제외 채널은 worklist에 올리지 않음

**신뢰도 파일럿** (`ANNOTATION_PROTOCOL.md` §5.3 · §8)

```bash
python3 reliability_pilot.py sessions --n 10   # 층화 추출 후 동결 (시작 전 커밋)
python3 reliability_pilot.py baseline          # 자동 검출 스냅샷
python3 reliability_pilot.py report --a rater_A --b rater_B   # LoA · tolerance 후보
```

rater별 저장소 격리는 환경변수 `EAG_RESULT_DIR`로 한다 (`store_io.py`). 세 저장소
(`offset_manager` · `edge_store` · `exclusion_store`)와 `sync_analyzer`가 모두 따라온다.

```bash
EAG_RESULT_DIR=result/reliability/rater_A python3 offset_app.py --port 8768 --blank
EAG_RESULT_DIR=result/reliability/rater_B python3 edge_app.py   --port 8769 --blank
```

`--blank`는 알고리즘의 초기값(offset 보정·후보, EAG edge 자동검출)을 숨긴다. GRF anchor는
남긴다. 두 rater가 같은 자동값을 나란히 수용하면 일치도가 부풀려지기 때문이다.

**작업 루프**

```
[1회] offset_review.py --dir data     # offset worklist 생성
[1회] edge_review.py  --dir data      # edge worklist 생성

세션 하나 선택
  → offset 패널 확인 → (필요시) offset_app에서 대응점 클릭 → Save
  → edge_app에서 그 세션 Load → 노이즈 삭제 → 누락 anchor에 knee 추가 → Save
다음 세션 반복 (priority=high 부터)
  → 완료 후 edge_review --dir data 재실행 → parameter_extractor --batch → stats_grf_eag
```

**진행 현황은 문서에 적지 않고 `python3 run_pipeline.py status`로 확인합니다**
(손으로 유지하는 스냅샷은 반드시 낡습니다).

### 빠른 실행 (run_pipeline.py)

자동 단계(1·4·5)를 한 명령으로 묶어 실행하고, 남은 수동검토 물량을 요약한다.
`sys.executable`로 하위 스크립트를 호출하므로 Windows·Linux 공용이다.

```bash
python3 run_pipeline.py status     # 진행상태: offset_report / worklist 백로그 / override 현황
python3 run_pipeline.py all        # 진단→파라미터추출→통계 일괄 (diag+params+stats)
python3 run_pipeline.py analyze    # 수동검토 끝난 뒤 재분석 (params+stats, 기본 stage)
python3 run_pipeline.py params --phase3        # Phase3(주파수)까지 추출
python3 run_pipeline.py analyze --exclude-review   # needs_review 세션 제외
python3 run_pipeline.py all --dry-run          # 실행할 명령만 출력
```

**권장 흐름**: `status`로 백로그 확인 → 2·3단계 수동검토(우선순위: `큰offset` > `저match` > `edge매칭부족`) → `analyze`로 재분석.
기존 `grf_triggered_params_*.csv`가 `grf_step` 이전 구버전이면 `stats`가 실패하므로, 먼저 `params`(재추출)가 필요하다.

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
