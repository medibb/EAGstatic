# Manual Offset Adjustment 구현 계획

## 목표
EAG-GRF 동기화의 자동 offset이 부정확한 세션에 대해 수동으로 offset을 조정하고, 조정된 offset으로 Phase 1/2/3 분석을 재실행할 수 있도록 한다.

## 워크플로우 (사용자 관점)

```
Step 1: review — 현재 상태 확인
  python adjust_offset.py review
  → 18명 피험자별 auto offset, sync method, 문제 여부 테이블 출력

Step 2: explore — 후보 offset 시각화
  python adjust_offset.py explore --subject 김은혜 --session s1
  → auto offset 기준 ±1초를 0.2초 간격 11패널 PNG 생성
  → result/alignment_check/{subject}_{session}_offset_explore.png

Step 3: set — 최적 offset 저장
  python adjust_offset.py set --subject 김은혜 --session s1 --offset -0.35
  → result/manual_offsets.json에 기록

Step 4: 검증 — 설정한 offset으로 alignment PNG 재생성
  python plot_alignment_verification.py --subject 김은혜 --session s1
  → 자동으로 manual offset 사용

Step 5: Phase 1/2/3 재실행
  python parameter_extractor.py --batch --phase3 --reprocess-manual
  → manual offset이 설정된 세션만 재처리
```

## 구현 파일 목록

### 1. 신규: `offset_manager.py` (데이터 레이어)
- `load_manual_offsets(path)` → dict
- `save_manual_offsets(offsets, path)` → JSON 저장
- `get_manual_offset(subject, session)` → Optional[float]
- `set_manual_offset(subject, session, offset, auto_offset, auto_method, note)`
- `clear_manual_offset(subject, session)`
- 저장 경로: `result/manual_offsets.json`
- 구조: `{ "subject_name": { "session_name": { "manual_offset": float, "auto_offset": float, ... } } }`

### 2. 신규: `adjust_offset.py` (CLI 도구)
3개 서브커맨드:

**`review`**: 전체/특정 피험자의 offset 상태 테이블 출력
- auto method, auto offset, manual offset(있으면), 플래그(xcorr/큰 offset) 표시

**`explore`**: multi-panel offset 탐색 PNG 생성
- auto offset 기준 ±1초, 0.2초 간격 = 11패널
- 각 패널: EAG filtered 8ch(normalized) + GRF(twin axis) overlay, 첫 5초
- xcorr fallback 세션은 자동으로 넓은 범위 (±3초, 0.5초 간격)
- SyncAnalyzer 생성 없이 EAG/GRF 직접 로드하여 빠르게 처리

**`set`**: manual offset을 JSON에 저장
- 세션별 개별 저장 (--all-sessions 없음)
- 저장 전 overlap duration 검증 출력

**`clear`**: manual offset 제거 (auto로 복귀)

### 3. 수정: `sync_analyzer.py`
- `SyncAnalyzer.__init__()` 에 `manual_offset: Optional[float] = None` 파라미터 추가
- `manual_offset`이 None이면 → `manual_offsets.json`에서 자동 조회
- manual offset 존재 시 → auto alignment 건너뛰고 바로 unified time axis 계산
- `_align_time()` 리팩토링: time axis 계산 부분을 `_compute_unified_axes()`로 분리

변경 범위:
```python
def __init__(self, session_pair, config=None, utc_offset=9, manual_offset=None):
    ...
    # manual offset 조회 (명시적 > JSON > auto)
    if manual_offset is None:
        manual_offset = _lookup_manual_offset(subject, session)
    self._manual_offset = manual_offset
    self._align_time()

def _align_time(self):
    if self._manual_offset is not None:
        self.time_offset = self._manual_offset
        self.sync_method = "manual"
        self._compute_unified_axes()
        return
    # 기존 auto alignment 로직 유지
    ...
    self._compute_unified_axes()

def _compute_unified_axes(self):
    # 기존 _align_time()의 line 396~468 (time axis + filtering) 추출
```

### 4. 수정: `parameter_extractor.py`
- `--reprocess-manual` 플래그 추가
- 해당 플래그 시 manual_offsets.json에 있는 세션만 필터링하여 재처리

## 구현 순서
1. `offset_manager.py` — 독립 모듈, 의존성 없음
2. `sync_analyzer.py` 수정 — _align_time 리팩토링 + manual offset 지원
3. `adjust_offset.py` — CLI 도구 (review → explore → set)
4. `parameter_extractor.py` 수정 — --reprocess-manual 플래그
5. 테스트: 김은혜 s1 (xcorr 문제 세션)으로 전체 워크플로우 검증
