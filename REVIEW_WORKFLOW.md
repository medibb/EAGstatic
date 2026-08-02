# Offset / Edge 검토 워크플로우

GRF-triggered 파이프라인에서 **offset과 edge(knee)를 사람이 검토·확정**하는 절차.
자동 검출이 부정확한 세션을 육안으로 확인하고 수정하여, 파라미터·통계의 정확도를 확보한다.

## 대원칙

1. **offset 먼저, edge 나중.** edge는 offset이 보정된 시간축(`te_corr`)에 저장되므로, offset을 확정한 뒤 편집한다.
2. **사람이 확정한 값이 최우선.** `manual_offsets.json`(offset), `manual_edges.json`(edge)이 있으면 자동 검출을 무시하고 그 값을 사용한다.
3. **확정 후 재추출.** 확정값과 최신 컬럼(`load_pct`, `accepted` 등)은 `parameter_extractor.py --batch`를 다시 돌려야 반영된다.

---

## 연구 프로토콜 (모든 판정의 기준)

EAG(무릎 전위)가 GRF(족저 체중이동)에 반응하는지를 **점진적으로 부하를 늘린 한발서기 반복**으로
측정한다. 아래는 코드에 구현된 실제 분석 기준과 같은 내용이다.
원본 방법 문서는 `EAG 연구방법(26.02-03).hwp`.

### 측정 순서 (세션 1회)

```
초기 발구름(stomp)  →  한발서기 4회 (검사측 부하를 회차마다 증가: 설계 20-50-80-100%)  →  종료
```

1. **초기 발구름** — EAG-GRF 시간 동기화(offset) 세팅용. **분석 제외**
2. **한발서기 4회** — 검사측 다리에 싣는 체중을 회차마다 점진적으로 늘린다. **4회 모두 시행됨**
3. 프로토콜 **전후의 양발 서기(휴식)** — **분석 제외**

### 신호 구조

**GRF signed imbalance** `(L−R)/(L+R)` — 좌우 체중이동. 사각파 형태.

| 개념 | 뜻 |
|---|---|
| `rest_level` | 휴식 자세 plateau. 반대측 다리에 싣고 **검사측을 비운** 상태(`\|signed\| ≈ 0.9~1.0`).<br>부하 4회 **사이사이 반복해서 돌아오는** 레벨이라 plateau가 5개(부하 전/사이×3/후) |
| `load_level` | 각 부하 구간 plateau. 회차마다 1번씩, 크기 점증 |
| `load_step` | `load_level − rest_level` (부호 포함). 계단형으로 커진다 |
| `load_pct` | **검사측 체중부하율(%)** = 검사측 힘 / 전체 힘, 부하 구간 평균.<br>실측 중앙값 **18 / 46 / 73 / 92%** (설계 20/50/80/100%보다 일관되게 낮음) |

**EAG** — 각 부하 cycle에서 knee-pair 2개:
**부하가 실리면 하강(fall), 빠지면 상승(rise)** — 예외 없는 규칙.

### 분석 단위 (세션·채널당 8 이벤트)

```
1 cycle(한발서기 1회) = 부하 시작 anchor + 이탈 anchor = 2 이벤트
세션·채널당 4 cycle × 2 = 8 이벤트  →  EAG knee-pair 8개 (fall–rise 4회 교대)
```

- 코드 상수: `EXPECTED_CYCLES = 4`, `EXPECTED_EVENTS = 8`, `EXPECTED_DIR = {'on':'fall','off':'rise'}`
- 각 anchor마다 EAG **onset·offset 두 knee**로 변화 크기(`amplitude`), 전이시간, `latency`(≈0, 동시성) 기록
- **dose-response**: EAG `|amplitude|`가 부하(`load_pct`)에 비례해 커지는지가 핵심 분석.
  dose 축은 명목값(20/50/80/100)이 아니라 **실측 `load_pct`**를 쓴다
- 부하 구간 음영은 **복귀 램프가 끝나는 지점(`end_time`)까지** 그린다.
  이벤트 시각(`offset_time`)은 램프 **시작**이지만, 구간 자체는 램프를 포함해야 과소평가되지 않는다

### 코드에서의 정의 (ground truth: `grf_triggered_annotator.py`)

| 함수·자료구조 | 역할 |
|---|---|
| `LoadCycle` | 휴식→부하→복귀 1회. `load_step`(dose) · `load_pct`(실측 부하율) · `test_side` · `end_time` |
| `detect_load_cycles_expected()` | 휴식 레벨에서 벗어났다 돌아오는 구간 = 부하 cycle. **4회가 나올 때까지 문턱을 탐색**. 발구름·전후 양발서기 배제 |
| `cycles_to_transitions()` | cycle → 분석 anchor 8개(`role`=on/off). 시각은 `detect_grf_transitions`의 knee로 스냅해 정확도 유지 |
| `detect_eag_edges_protocol()` | 전역 검출 + **anchor별 국소 보강**(전역 문턱에 묻힌 약한 반응 회수, 방향 규칙 준수) |
| `match_edge()` | anchor ↔ edge 매칭. **방향 규칙을 만족하는 후보 중** 가장 가까운 것 |
| `validate_cycle_edges()` | cycle 수 · 이벤트 매칭 · 방향 · 노이즈 판정 |
| `label_single_sided()` | 파라미터 표에 '한쪽만 채택' 라벨 부여 |

> `detect_grf_transitions`는 사각파 **진폭** 문턱(`min_amp=0.3`)을 쓰기 때문에 가장 가벼운
> 1단계(예: signed 0.99→0.82, 진폭 0.17)를 놓친다. 그래서 cycle 검출은 진폭이 아니라
> **"휴식 레벨로부터의 이탈"**을 문턱으로 쓴다.

### 프로토콜이 지켜졌는지 확인하는 3가지 방법

| 방법 | 무엇을 보나 | 명령 |
|---|---|---|
| **GUI (시각, 권장)** | 부하 4 cycle 음영 + anchor 8개 + `cycle n/4 · 이벤트 m/8` 실시간 검증 | `python3 edge_app.py --host 0.0.0.0 --port 8765` |
| **배치 검증** | 전 세션·채널의 8/8 충족 여부 | `python3 edge_review.py --dir data` → `result/edge_review/all_channels.csv` |
| **파라미터** | 이벤트별 `load_pct` · `eag_direction` · `amplitude` · `latency` | `parameter_extractor.py --batch` 산출물 |

---

## Step 0. 검토 대상 확인

```bash
cat result/offset_review/worklist.csv     # offset 검토 대상 (세션 단위)
python3 edge_review.py --list             # edge 검토 대상 요약 (세션·채널 단위)
```

offset worklist의 `reason`:
- `저match` : 정렬 약함
- `edge매칭부족` : GRF 전이 대비 EAG edge 매칭 적음
- `큰교정보류(재검토)` : offset 큰 교정이 자동 보류됨 (cycle-skip 또는 기기 시작차)
- `대안제시(res=…)` : match-profile이 다른 offset 후보 제시

---

## Step 1. OFFSET 검토·확정 (세션별)

### ① 패널 보기
`result/offset_review/{피험자}_{세션}_review.png` (3단 구성)
- 상단 : match-rate 프로파일 (초록 점선 = best-match 후보)
- 중단 : **AUTO** 오버레이 (초록=GRF signed, 빨강=EAG @ auto offset)
- 하단 : **CORRECTED** 오버레이 (EAG @ corrected offset)

### ② 판단
- 하단(CORRECTED)에서 **GRF 사각파와 EAG가 잘 겹치면 → 그대로 OK** (아무 작업 안 함, corrected offset 자동 적용)
- 안 겹치면 → 겹치게 만드는 offset 값을 판단

### ③ 값 확정 (안 맞는 세션만)

#### 방법 A: GUI로 점 찍어 맞추기 (권장 — 눈금 읽을 필요 없음)
```bash
python3 offset_app.py --host 0.0.0.0 --port 8766
```
- 드롭다운에서 세션 선택 → 자동 Load. **전체 세션**이 나오며 검토대상 `▲`이 위로 정렬되고,
  확정된 세션은 `✅ … [확정 -0.35s]`로 표시된다(Save 즉시 갱신).
  옆 필터로 `전체 / 검토대상 / 확정 / 미확정` 전환
- **GRF 패널(위)에서 기준 변곡점 클릭 → EAG 패널(아래)에서 대응 변곡점 클릭** → 쌍 성립,
  두 점이 겹치도록 EAG가 즉시 이동 (최종 offset = `corrected + (EAG − GRF)`)
- 쌍을 여러 개 찍으면 **중앙값** 사용 → 한 쌍이 부정확해도 강건. 표에서 개별 쌍 삭제 가능
- **휠**=확대/축소, **드래그**=좌우 이동, 커서 위치의 시각(`t=…s`) 실시간 표시
- **클릭 흡착** — `가장 가까운 점`(기본, 클릭한 자리에서 거의 안 움직임) ·
  `변곡점`(근처 |2차미분| 최대점, 최대 ±0.3s 이동) · `자유`(클릭 좌표 그대로)
- 보조: match-rate 프로파일 클릭 → 그 residual로 점프 · `best-match 값 사용` · `nudge ±0.02/±0.1` · offset 직접 입력
- 잘 겹치면 **Save** → `result/manual_offsets.json` · **Clear manual** → 자동값 복귀

> 수동 offset이 있는 세션은 residual 재계산 없이(`recompute=False`) 불러오므로,
> **화면에서 본 정렬 = 파이프라인 정렬**이다 (`parameter_extractor`와 같은 규칙).

#### 방법 B: CLI
```bash
python3 offset_review.py --set --subject "(02.02_17)김종문_1" --session-name s2 --offset -0.15
```

### ④ 재확인 / 관리
```bash
python3 offset_review.py --session "data/(02.02_17)김종문_1/OpenBCISession_...-s2"  # 패널 재생성
python3 offset_review.py --list                                                     # 확정 목록
python3 offset_review.py --clear --subject "(02.02_17)김종문_1" --session-name s2    # 제거(auto 복귀)
```

---

## Step 2. EDGE 검토·수정 (offset 확정 후, 세션·채널별)

### 판정 규칙 (연구 설계 반영)

- **모든 세션은 4회를 시행했다.** 3회로 보이면 가장 가벼운 단계가 노이즈에 묻힌 것이므로,
  `detect_load_cycles_expected()`가 문턱을 낮춰가며 4회가 나오는 조합을 찾는다.
- **같은 부하의 rise와 fall은 크기가 거의 같다.** 따라서 한쪽이 노이즈로 못 쓰게 돼도
  다른 한쪽만 제대로 측정되면 그 부하에서의 EAG 크기를 알 수 있다
  → cycle은 **이벤트가 1개 이상**이면 측정 가능으로 본다 (**1단계도 분석에 포함**).
  `asymmetry`(두 값의 상대 차이)가 크면 한쪽이 오염됐다는 신호다.
- **edge는 부하의 시작과 끝에서만 발생한다.** 다만 노이즈 판정은 절대적이지 않다 —
  매칭되지 않았다는 이유만으로 지우면 경계에서 조금 벗어난 진짜 반응까지 사라진다.
  **anchor에서 `NOISE_MARGIN`(1.2s)를 넘게 떨어진 것만** 노이즈로 본다
  (= 부하/휴식 구간 '한가운데'). 경계 근처의 미매칭 edge는 표에 `후보`로 남는다.
  annotator의 `노이즈 삭제` 버튼은 노이즈로 판정된 것만 지운다.
- **방향 규칙은 절대적이다: 체중부하가 실릴 때 EAG는 하강(fall), 빠질 때 상승(rise).**
  매칭은 이 방향을 만족하는 후보 중 가장 가까운 것을 고른다(`match_edge`).
  단순히 가장 가까운 edge를 고르면, 노이즈가 anchor에 더 붙어 있을 때 반대 방향
  edge를 잘못 채택한다. anchor별 국소 보강(`detect_edge_near`)도 같은 부호의
  기울기 안에서만 knee를 찾는다.
- 방향이 맞는 후보가 한쪽에만 있으면 그쪽만 채택하고 `한쪽만 채택(c2:부하)` 라벨을
  남겨 **후순위 검토** 대상으로 둔다.

### 검토 우선순위

| priority | 뜻 | 대응 |
|---|---|---|
| `high` | 측정 불가 cycle · cycle 수 이상 → **사람이 봐야 함** | annotator에서 knee 추가/수정 |
| `low` | 한쪽만 자동 채택함 (값은 확보됨) | 여유 있을 때 확인 |
| (빈칸) | 양측 모두 정상 | 없음 |

`high` 사유: `측정 불가 cycle k개(cN …)` · `cycle N회(기대 4)` · `한 edge가 두 이벤트에 중복 매칭`

### 검토 대상 추리기

```bash
python3 edge_review.py --dir data                 # 전 세션 스캔 (약 7분)
python3 edge_review.py --session "<세션경로>"      # 단일 세션 채널별 진단표
python3 edge_review.py --list                      # 요약
```

산출물:
- `result/edge_review/worklist.csv` — priority가 있는 채널만 (**high → low 순 정렬**)
- `result/edge_review/all_channels.csv` — 전 채널

주요 컬럼: `priority · labels · ok · n_cycles · n_measured_cycles · n_matched · n_edges ·
n_single_sided · n_noise · load_pct · amp · asym · offset_source · offset_pending`
(`load_pct`/`amp`/`asym`는 cycle 4개 값을 `;`로 이어붙인 문자열)

> **`offset_pending=True`인 행은 offset이 아직 확정되지 않은 세션**의 결과다.
> 자동 주석 자체는 전 세션에 대해 수행되지만, 그 기준 시간축이 바뀔 수 있으므로
> **이 세션들의 edge를 손으로 고치는 것은 헛수고가 된다**(edge는 `te_corr` 프레임에
> 저장되므로 offset이 바뀌면 전부 어긋난다). 대원칙 "offset 먼저, edge 나중" 그대로다.
>
> 실제로 offset 미확정 세션은 결과가 눈에 띄게 나쁘다 — 통과 77.9% vs 88.1%,
> `high` 22.1% vs 11.9%. cycle 검출률은 차이가 없다(95.5% vs 96.0%).
> cycle은 GRF만으로 잡히고 offset과 무관한 반면, EAG-anchor 매칭은 offset 오차에
>직접 깨지기 때문이다. **offset을 확정하면 edge 결과가 상당수 저절로 좋아진다.**

### 방법 A: GUI (권장)

```bash
python3 edge_app.py --host 0.0.0.0 --port 8765
```

**세션 고르기**
- 상단 **세션 드롭다운**에서 선택 → 자동 Load (또는 경로 직접 입력 후 **Load**) · 채널은 `ch` 입력칸
- **전체 세션**이 나온다. 검토대상 `▲` 우선 정렬, `[프로토콜 6/8]` 배지,
  edge 확정 세션은 `✔ … [edge ch1,2]`, offset 확정 세션은 `[off✅]`
- 필터: `전체 / 검토대상 / edge 확정 / edge 미확정 / 프로토콜 미충족 ⚠️`

**화면에 프로토콜이 그려진다**
- **주황 음영** = 체중부하 cycle 4회 (`부하N (step …)`)
- **세로 점선** = anchor 8개 (빨강=부하 시작, 파랑=이탈)
- **굵은 빨강 `누락 cN…`** = 그 anchor에 knee가 없다 → 그 자리에 edge를 추가
- **회색 점선 edge** = 노이즈 후보 (anchor에서 1.2s 넘게 떨어진 것)

표의 `판정` 열은 세 가지다:
- `c2이탈` 등 — 그 anchor에 **매칭된** edge (분석에 쓰임)
- `후보` — 경계 근처인데 매칭되진 않은 edge. **지우지 않는다** (방향이 안 맞거나 더 가까운 것이 있었을 뿐)
- `노이즈` — 구간 한가운데. `노이즈 삭제` 대상

**상태 표시** (Save 즉시 재검증)
- `✅ 프로토콜 충족` / `🟡 후순위 검토(한쪽만 채택)` / `⚠️ 검토 필요`
- `cycle 4/4 · 측정가능 cycle 4/4 · 이벤트 8/8 · edge N개 · 노이즈 후보 k개`
- 라벨 예: `한쪽만 채택(c2:부하)` · `단측 측정(c1)`

**cycle 요약표** — `부하% · |amp| 부하 · |amp| 이탈 · 대표 amp · 비대칭 · 채택(양측/부하만/이탈만)`

**편집**
- knee 점 **드래그**로 이동 · **Add mode** 후 트레이스 2점 클릭으로 edge 추가 ·
  edge 선택 후 **Delete/Del키** · **snap** 체크 시 corner 자동정렬
- **노이즈 삭제** 버튼 = anchor에서 먼 edge 일괄 제거 (Save해야 확정)
- **확대/이동** — 확대한 상태에서 앞뒤로 훑어보며 검토하도록 만들어져 있다:

  | 조작 | 동작 |
  |---|---|
  | 휠 | 커서 위치 기준 가로축 확대/축소 |
  | **빈 공간 드래그** | 좌우 이동 (knee/edge 위가 아니면 잡고 끌면 된다) |
  | Shift+드래그 · 휠버튼 드래그 · Shift+휠 | 좌우 이동 |
  | **← →** | 보이는 폭의 25%씩 이동 |
  | **Shift+← →** | **이전/다음 이벤트(anchor)로 점프** — 배율은 유지된 채 그 anchor가 화면 중앙에 온다 |
  | Home / End | 세션 처음 / 끝 |
  | f · 전체보기 | 원래 배율로 |
  | ◀ ▶ ◀이벤트 이벤트▶ | 위 동작의 버튼 |

  화면은 데이터 범위 밖으로 벗어나지 않는다(자동 클램프).
  확대하면 y축도 그 구간에 맞춰 재조정되고 snap 범위도 좁아져 정밀해진다.
  **8개 anchor를 `Shift+→`로 차례로 넘기며 확인하는 것이 가장 빠르다.**
- **Save** → `manual_edges.json` · **Reset→auto** 복귀

### 방법 B: CLI
```bash
python3 edge_editor.py review --session "<세션경로>" --channel 1    # 번호라벨 PNG + edge 테이블
python3 edge_editor.py delete --session "<경로>" --channel 1 --id 3
python3 edge_editor.py add    --session "<경로>" --channel 1 --onset 30.1 --offset 30.6 --snap
python3 edge_editor.py move   --session "<경로>" --channel 1 --id 2 --onset 19.2 --offset 19.7 --snap
python3 edge_editor.py reset  --session "<경로>" --channel 1         # 자동검출로 복귀
python3 edge_editor.py list                                          # 확정 현황
```
매 명령이 `result/edge_edit/{피험자}_{세션}_ch{N}_edit.png`를 갱신 → 즉시 재검토.

> 주의: `review/add/delete/move/reset`은 다섯 서브커맨드 중 **하나**를 고른다.
> `<세션경로>`,`<채널>`은 실제 값으로 교체하고, 한글 폴더명은 **따옴표**로 감싼다.

---

## Step 3. 확정 후 재추출·통계

```bash
python3 parameter_extractor.py --batch        # manual offset/edge + 프로토콜 anchor 반영 재추출
python3 stats_grf_eag.py                      # dose-response 통계 (STATS_PLAN.md 참조)
```

### grf_triggered 출력에 추가된 컬럼

| 컬럼 | 뜻 |
|---|---|
| `cycle_id` | 부하 회차 (0~3) |
| `event_kind` | `on`=부하 시작 · `off`=이탈 |
| `load_pct` | **실측 체중부하율(%)** = 검사측 힘 / 전체 힘, 부하 구간 평균 |
| `test_side` | 검사측 다리 `L`/`R` (휴식 시 비어 있던 쪽) |
| `accepted` | 이 행을 파라미터로 채택했는지 (방향 오염으로 폐기되면 False) |
| `single_sided` | 그 cycle을 한쪽 이벤트만으로 측정했는지 |
| `review_priority` | `low`면 후순위 검토 대상 |
| `n_cycles` / `protocol_ok` | 검출된 cycle 수 / 4회 여부 |

> **실측 부하율은 설계값보다 일관되게 낮다** (중앙값 18 / 46 / 73 / 92% vs 설계 20 / 50 / 80 / 100%).
> dose 축은 명목값 대신 `load_pct`를 쓰는 것이 맞다.
> 통계 단계에서 `single_sided` 행만 따로 민감도 분석하거나 제외할 수 있다.

---

## "하나씩" 루프 요약

```
[1회] python3 offset_review.py --dir data     # offset worklist 생성
[1회] python3 edge_review.py  --dir data      # edge worklist 생성 (offset 확정 후 재실행)

세션 하나 선택
  → offset 패널 확인 → (필요시) offset_app에서 GRF·EAG 점 찍어 Save
  → edge_app에서 그 세션 Load → 노이즈 삭제 → 누락 anchor에 knee 추가 → Save
다음 세션 반복  (worklist의 priority=high 부터)
  → 다 끝나면 edge_review --dir data 재실행 → parameter_extractor --batch → stats_grf_eag
```

핵심 산출물: `result/manual_offsets.json`(확정 offset), `result/manual_edges.json`(확정 edge).
이 둘만 채워지면 재추출 시 전부 자동 반영된다.

---

## 현재 진행 상황 (2026-08-01 스캔 기준)

| 항목 | 값 |
|---|---|
| 스캔 대상 | 세션 425개 (로드 실패 11개 제외 → **416개 · 채널 3,328개**) |
| cycle 4회 검출 | **399 / 416 세션** |
| 프로토콜 통과 채널 | **2,861 / 3,328 (86%)** |
| 전 채널 통과 세션 | **257 / 416** |
| `high` (사람이 봐야 함) | **467 채널** |
| `low` (한쪽만 자동채택, 후순위) | 1,169 채널 |
| 라벨 없음 (양측 정상) | 1,692 채널 |
| 확정 offset / edge | 2 세션 / 1 채널 |

`high`는 대부분 `측정 불가 cycle`(방향이 맞는 반응이 창 안에 없음)과
`cycle 수 이상`(4회로 안 잡히는 17개 세션)이다. 후자를 먼저 보면 효율적이다.

> 방향 규칙을 절대 규칙으로 바꾸기 전에는 통과율이 93%였으나, 그 안에 **방향이 틀린
> 매칭**(이탈 anchor에 fall이 붙는 등)이 섞여 있었다. 규칙 적용 후 그런 것들이 `high`로
> 올라와 통과율은 86%로 내려갔지만, **라벨 없는 완전 정상 채널은 842 → 1,692로 늘었다.**
> 숫자보다 이쪽이 실질적인 품질 지표다.

---

## 참고

- **외부(DDNS) 접속**: 두 GUI 모두 code-server 내장 포트 프록시로 중계된다. 도커 포트 퍼블리시·공유기 포트포워딩 불필요.
  - offset GUI : `http://medibb.synology.me:18440/proxy/8766/`
  - edge GUI  : `http://medibb.synology.me:18440/proxy/8765/`
  - **끝 슬래시(`/`) 필수**, code-server에 로그인된 브라우저여야 한다(프록시가 인증 뒤에 있음).
  - 두 앱 모두 API를 문서 기준 상대경로로 호출하므로 프록시 prefix 아래에서 동작한다.
    새 GUI를 만들 때도 절대경로(`/api/...`) 대신 같은 방식을 쓸 것.
- **GUI 수정 시 검증**: canvas 코드는 문법 검사(`node --check`)로 미정의 참조를 잡지 못한다.
  실제 데이터를 넣어 `draw()`를 돌려보는 것이 안전하다 (과거 `vline` 미정의로 그래프가 통째로 안 그려진 적 있음).
- 한글 폰트: Linux/Docker는 `pip install koreanize-matplotlib`(또는 `apt-get install -y fonts-nanum`) 후 패널 재생성 시 한글 라벨 정상 표시. macOS는 자동.

### 관련 문서

| 문서 | 내용 |
|---|---|
| `README.md` | 전체 파일 구조·파이프라인 |
| `STATS_PLAN.md` | 통계 설계 (dose-response) |
| `EAG 연구방법(26.02-03).hwp` | 원본 연구방법 |
| **이 문서** | 실험 프로토콜 + offset/edge 검토·수정 절차 (프로토콜 단일 참조본) |
