# EAG-GRF Manual Offset Adjustment Guide

EAG(8ch electro-arthrography)와 GRF(force plate) 신호의 시간 동기화를 수동으로 조정하는 도구입니다.

## 배경

EAG와 GRF는 별도의 장비로 동시 측정되며, 녹화 시작 시각의 차이를 보정(offset)해야 합니다. 자동 동기화(`sync_analyzer.py`)는 두 가지 방법을 사용합니다:

1. **Event matching**: 초기 체중이동 이벤트의 시점을 EAG/GRF에서 각각 감지하여 매칭
2. **Cross-correlation fallback**: Event matching 실패 시 신호 상관분석으로 추정

자동 동기화가 부정확한 세션이 있어, 시각적으로 확인하고 수동으로 offset을 조정하는 워크플로우가 필요합니다.

## 파일 구조

```
EAGstatic/
├── adjust_offset.py           # CLI 도구 (이 문서의 주요 대상)
├── offset_manager.py          # manual offset JSON 관리 모듈
├── sync_analyzer.py           # 동기화 엔진 (manual offset 자동 반영)
├── parameter_extractor.py     # Phase 1/2/3 파라미터 추출
├── plot_alignment_verification.py   # 정렬 검증 4패널 PNG
├── plot_fsi_verification.py         # FSI/APA 검증 PNG
└── result/
    ├── manual_offsets.json     # 수동 offset 저장 파일 (자동 생성)
    └── alignment_check/       # 정렬 검증 PNG 저장 폴더
```

## 워크플로우

### Step 1. 현재 상태 확인 — `review`

```bash
cd /workspace/research/EAGstatic

# 전체 피험자 (s1 세션 기준)
python3 adjust_offset.py review

# 특정 피험자만
python3 adjust_offset.py review --subject 김은혜
```

출력 예시:
```
Subject                        Method    Auto Offset   Manual Status
--------------------------------------------------------------------------------
(02.02_10)주창민_1                event          -0.144        - ok
(02.02_15)김은혜_1                xcorr          -7.000   -0.350 ⚠ xcorr | ⚠ large | ✓ manual
```

- **Method**: `event`(정상) 또는 `xcorr`(fallback, 주의 필요)
- **Status**: `ok`(문제 없음), `⚠ xcorr`(xcorr fallback 사용), `⚠ large`(|offset| > 2초), `✓ manual`(수동 설정됨)

### Step 2. 정렬 시각화 확인

기존 alignment PNG를 확인하거나 새로 생성합니다:

```bash
# 전체 피험자 s1 alignment PNG 생성
python3 plot_alignment_verification.py --batch

# 특정 피험자만
python3 plot_alignment_verification.py --subject 김은혜 --session s1
```

결과: `result/alignment_check/{피험자}_{세션}_alignment.png`

4패널 구성:
- 좌상: EAG filtered 8ch
- 우상: GRF (Left/Right)
- 좌하: Timestamp 기반 정렬 overlay
- 우하: Event 기반 정렬 overlay

**EAG 변곡과 GRF 체중이동이 시간적으로 일치하는지 눈으로 확인합니다.**

### Step 3. Offset 탐색 — `explore`

문제가 있는 세션에 대해 여러 후보 offset을 한 PNG에 나열합니다:

```bash
# 기본: auto offset ±1초, 0.2초 간격 (11패널)
python3 adjust_offset.py explore --subject 김은혜 --session s1

# 범위/간격 지정
python3 adjust_offset.py explore --subject 김은혜 --session s1 --range -2.0 2.0 --step 0.4
```

결과: `result/alignment_check/{피험자}_{세션}_offset_explore.png`

- 각 패널에 후보 offset 값이 표시됩니다
- 초록색 배경 = 현재 auto offset
- EAG(회색 선)와 GRF(파란/빨간 선)의 이벤트가 가장 잘 일치하는 패널의 offset 값을 선택합니다

### Step 4. Offset 설정 — `set`

선택한 offset 값을 저장합니다:

```bash
python3 adjust_offset.py set --subject 김은혜 --session s1 --offset -0.35

# 메모 추가 (선택)
python3 adjust_offset.py set --subject 김은혜 --session s1 --offset -0.35 --note "visual inspection ok"
```

- `result/manual_offsets.json`에 저장됩니다
- 이후 모든 분석 도구가 자동으로 이 offset을 사용합니다
- 세션별로 개별 저장됩니다 (같은 피험자의 다른 세션에는 영향 없음)

### Step 5. 설정 검증

manual offset이 적용된 상태로 alignment PNG를 재생성합니다:

```bash
python3 plot_alignment_verification.py --subject 김은혜 --session s1
```

제목에 `Sync used: -0.350s`와 같이 manual offset이 표시됩니다.

### Step 6. Phase 1/2/3 재처리

manual offset이 설정된 세션만 골라서 재분석합니다:

```bash
# manual offset 세션만 재처리
python3 parameter_extractor.py --batch --phase3 --reprocess-manual

# 전체 재처리
python3 parameter_extractor.py --batch --phase3
```

### (선택) Offset 제거 — `clear`

manual offset을 제거하고 자동 정렬로 복귀합니다:

```bash
python3 adjust_offset.py clear --subject 김은혜 --session s1
```

## 동작 원리

### manual_offsets.json 구조

```json
{
  "(02.02_15)김은혜_1": {
    "s1": {
      "manual_offset": -0.35,
      "auto_offset": -7.0,
      "auto_method": "xcorr",
      "updated_at": "2026-04-28T14:30:00",
      "note": "visual inspection ok"
    }
  }
}
```

### 자동 반영

`SyncAnalyzer`가 초기화될 때 `manual_offsets.json`을 자동으로 조회합니다:

1. `manual_offset` 파라미터가 명시적으로 전달됨 → 해당 값 사용
2. 명시적 값 없음 → `manual_offsets.json`에서 해당 세션 조회
3. JSON에도 없음 → 기존 자동 정렬 (event matching → xcorr fallback)

따라서 `parameter_extractor.py`, `plot_fsi_verification.py`, `plot_alignment_verification.py` 등 기존 도구는 코드 변경 없이 자동으로 manual offset을 사용합니다.

## 권장 작업 순서

1. `python3 adjust_offset.py review` → 문제 세션 식별
2. `⚠ xcorr` 또는 `⚠ large` 세션부터 explore → set
3. alignment PNG로 나머지 세션도 눈으로 확인
4. 모든 조정 완료 후 `python3 parameter_extractor.py --batch --phase3` 전체 재처리

## 주의사항

- offset 값의 의미: `GRF_time + offset = EAG_time`. 양수이면 GRF가 EAG보다 먼저 시작, 음수이면 EAG가 먼저 시작.
- explore에서 xcorr fallback 세션은 자동으로 넓은 탐색 범위(±3초)를 사용합니다.
- `--subject` 옵션은 부분 매칭을 지원합니다. 예: `--subject 김` → 김씨 성을 가진 모든 피험자.
