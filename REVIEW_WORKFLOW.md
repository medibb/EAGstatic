# Offset / Edge 검토 워크플로우

GRF-triggered 파이프라인에서 **offset과 edge(knee)를 사람이 검토·확정**하는 절차.
자동 검출이 부정확한 세션을 육안으로 확인하고 수정하여, 파라미터·통계의 정확도를 확보한다.

## 대원칙

1. **offset 먼저, edge 나중.** edge는 offset이 보정된 시간축(`te_corr`)에 저장되므로, offset을 확정한 뒤 편집한다.
2. **사람이 확정한 값이 최우선.** `manual_offsets.json`(offset), `manual_edges.json`(edge)이 있으면 자동 검출을 무시하고 그 값을 사용한다.
3. **확정 후 재추출.** 확정값과 최신 컬럼(`grf_step` 등)은 `parameter_extractor.py --batch`를 다시 돌려야 반영된다.

---

## Step 0. 검토 대상 확인

```bash
# offset 검토가 필요한 세션 목록
cat result/offset_review/worklist.csv        # subject, session, reason
```

reason 의미:
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
- 안 겹치면 → 겹치게 만드는 offset 값을 판단 (제목의 `corrected=` 값 기준, profile의 best-match나 육안 참고)

### ③ 값 확정 (안 맞는 세션만)

#### 방법 A: GUI로 점 찍어 맞추기 (권장 — 눈금 읽을 필요 없음)
```bash
python3 offset_app.py --host 0.0.0.0 --port 8766
```
브라우저 `http://<서버IP>:8766` 접속 →
- 드롭다운에서 세션 선택 → 자동 Load. **전체 세션**이 나오며 검토대상 `▲`이 위로 정렬되고,
  확정된 세션은 `✅ … [확정 -0.35s]`로 표시된다 (Save 즉시 갱신). 옆 필터로 `전체 / 검토대상 / 확정 / 미확정` 전환
- **GRF 패널(위)에서 기준 변곡점 클릭 → EAG 패널(아래)에서 대응 변곡점 클릭** → 쌍 성립,
  두 점이 겹치도록 EAG가 즉시 이동 (최종 offset = `corrected + (EAG − GRF)`)
- 쌍을 여러 개 찍으면 **중앙값** 사용 → 한 쌍이 부정확해도 강건. 표에서 개별 쌍 삭제 가능
- **휠**=확대/축소, **드래그**=좌우 이동, 커서 위치의 시각(`t=…s`)이 실시간 표시됨
- **클릭 흡착** 드롭다운 — `가장 가까운 점`(기본, 모든 샘플 중 최근접 → 클릭한 자리에서 거의 안 움직임) ·
  `변곡점`(근처 |2차미분| 최대점으로 흡착, 최대 ±0.3s 이동) · `자유`(클릭 좌표 그대로)
- 보조: 맨 위 match-rate 프로파일 클릭 → 그 residual로 점프 · `best-match 값 사용` · `nudge ±0.02/±0.1` · offset 직접 입력
- 잘 겹치면 **Save** → `result/manual_offsets.json` 확정 · **Clear manual** → 자동값 복귀

> 수동 offset이 있는 세션은 residual 재계산 없이(`recompute=False`) 불러오므로,
> **화면에서 본 정렬 = 파이프라인 정렬**이다 (`parameter_extractor`와 같은 규칙).

#### 방법 B: CLI
```bash
python3 offset_review.py --set --subject "(02.02_17)김종문_1" --session-name s2 --offset -0.15
```
→ `result/manual_offsets.json` 기록 (SyncAnalyzer가 자동 조회)

### ④ 재확인 / 관리
```bash
python3 offset_review.py --session "data/(02.02_17)김종문_1/OpenBCISession_...-s2"  # 패널 재생성
python3 offset_review.py --list                                                     # 확정 목록
python3 offset_review.py --clear --subject "(02.02_17)김종문_1" --session-name s2    # 제거(auto 복귀)
```

---

## Step 2. EDGE 검토·수정 (offset 확정 후, 세션·채널별)

### 프로토콜 기준 — 세션당 8개 이벤트

연구 설계: **초기 발구름**(offset 세팅) → **한발서기 4회**(체중부하를 점진적으로 늘림).
분석 대상은 4회 각각의 **부하 시작 / 이탈**이므로, 세션·채널당

> **체중부하 cycle 4회 × 2 = 8개 이벤트**, EAG에서 **fall–rise가 4번 반복**되어 knee-pair 8개

가 나와야 한다. 발구름과 프로토콜 전후의 양발 서기는 분석에서 제외된다.

`detect_load_cycles()`가 "휴식 자세(한쪽 다리에 실은 상태)에서 벗어났다 돌아오는 구간"으로
4회를 잡고, `cycles_to_transitions()`가 anchor 8개를 만든다. anchor 시각은 `detect_grf_transitions`의
knee로 스냅해 정확도를 유지한다.

### 검토 대상 추리기
```bash
python3 edge_review.py --dir data                 # 전 세션 스캔 → worklist
python3 edge_review.py --session "<세션경로>"      # 단일 세션 채널별 진단표
python3 edge_review.py --list                      # 요약
```
→ `result/edge_review/worklist.csv`(8개가 안 나온 채널) · `all_channels.csv`(전체)

판정 실패 사유: `cycle N회(기대 4)` · `측정 불가 cycle k개(cN …)` · `부하/이탈 방향 같음(cN)` · `한 edge가 두 이벤트에 중복 매칭`

### 판정 규칙 (연구 설계 반영)

- **모든 세션은 4회를 시행했다.** 3회로 보이면 가장 가벼운 단계가 노이즈에 묻힌 것이므로,
  `detect_load_cycles_expected()`가 문턱을 낮춰가며 4회가 나오는 조합을 찾는다.
- **같은 부하의 rise와 fall은 크기가 거의 같다.** 따라서 한쪽이 노이즈로 못 쓰게 돼도
  다른 한쪽만 제대로 측정되면 그 부하에서의 EAG 크기를 알 수 있다
  → cycle은 **이벤트가 1개 이상**이면 측정 가능으로 본다 (1단계도 분석에 포함).
  `per_cycle.asymmetry`(두 값의 상대 차이)가 크면 한쪽이 오염됐다는 신호다.
- **edge는 부하의 시작과 끝에서만 발생한다.** 부하 구간 한가운데나 휴식 구간에서 검출된 edge는
  노이즈다 → anchor에 매칭되지 않은 edge는 **노이즈 후보**로 표시되고,
  annotator의 `노이즈 삭제` 버튼으로 일괄 제거할 수 있다.
- **부하/이탈 방향이 같으면 한쪽이 오염된 것**이므로 자동으로 한쪽만 채택한다.
  어느 쪽을 쓸지는 그 채널에서 방향이 정상인 cycle들의 **다수결(방향 규약)**로 정한다.
  값은 살리되 `한쪽만 채택(c2:부하)` 라벨이 남아 **후순위 검토** 대상이 된다.

### 검토 우선순위

| priority | 뜻 | 대응 |
|---|---|---|
| `high` | 측정 불가 cycle · cycle 수 이상 등 **사람이 봐야 함** | annotator에서 knee 추가/수정 |
| `low` | 한쪽만 자동 채택함 (값은 확보됨) | 여유 있을 때 확인 |
| (빈칸) | 양측 모두 정상 | 없음 |

worklist는 `high` → `low` 순으로 정렬되어 저장된다.
파라미터 출력에도 같은 정보가 `accepted · single_sided · review_priority` 컬럼으로 들어가므로,
통계 단계에서 `single_sided` 행만 따로 민감도 분석을 하거나 제외할 수 있다.

### 체중부하율 파라미터

설계는 20-50-80-100%지만 사람마다 다르므로, cycle별 **실측 부하율**을 파라미터로 남긴다.
`load_pct = 검사측 힘 / 전체 힘`의 부하 구간 평균 (검사측 = 휴식 시 비어 있던 다리).
`parameter_extractor`의 grf_triggered 출력에 `cycle_id · event_kind · load_pct · test_side` 컬럼으로 들어간다.

### 수정 원칙
각 anchor마다 EAG **knee-pair(onset+offset)**가 하나씩 붙어야 한다.
놓친 것은 **추가**, 가짜는 **삭제**, 어긋난 것은 **이동**.

### 방법 A: GUI (권장)
```bash
python3 edge_app.py --host 0.0.0.0 --port 8765
```
브라우저 `http://<서버IP>:8765` 접속 →
- 상단 **세션 드롭다운**에서 선택 → 자동 Load (또는 경로 직접 입력 후 **Load**) · 채널은 `ch` 입력칸
  - **전체 세션**이 나온다(worklist 세션만이 아님). 검토대상 `▲` 우선 정렬,
    edge가 확정된 세션은 `✅ … [edge ch1,2 · 24개]`, offset이 확정된 세션은 `[off✅]`로 표시
  - 옆 필터로 `전체 / 검토대상 / edge 확정 / edge 미확정` 전환 → 남은 작업량 파악에 사용
- **화면에 프로토콜이 그려진다**: 주황 음영 = 체중부하 cycle 4회(`부하N (step …)`),
  세로 점선 = anchor 8개(빨강=부하 시작, 파랑=이탈), **굵은 빨강 `누락 cN…`** = 그 anchor에
  knee가 없다는 뜻이니 그 자리에 edge를 추가하면 된다
- 상단 meta에 **`✅ 프로토콜 충족` / `⚠️ 검토 필요`** 와 `cycle n/4 · 이벤트 매칭 m/8`이 표시되고,
  **Save 즉시 재검증**된다 (8/8이 되면 ✅로 바뀜)
- knee 점 **드래그**로 이동 · **Add mode** 후 트레이스 2점 클릭으로 edge 추가 · edge 선택 후 **Delete/Del키** · **snap** 체크 시 corner 자동정렬
- **휠**=가로축 확대/축소(커서 기준) · **Shift+드래그**(또는 휠버튼 드래그)=좌우 이동 · **f키/전체보기**=원래대로.
  확대하면 y축도 그 구간에 맞춰 재조정되고 snap 탐색 범위도 함께 좁아져 정밀해진다
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

> 주의: `review/add/delete/move/reset`은 다섯 서브커맨드 중 **하나**를 고른다. `<세션경로>`,`<채널>`은 실제 값으로 교체하고, 한글 폴더명은 **따옴표**로 감싼다.

---

## Step 3. 확정 후 재추출·통계

```bash
python3 parameter_extractor.py --batch        # manual offset/edge + grf_step 반영 재추출
python3 stats_grf_eag.py                       # dose-response 통계 (STATS_PLAN.md 참조)
```

---

## "하나씩" 루프 요약

```
worklist 세션 하나 선택
  → offset 패널 확인 → (필요시) offset_app에서 GRF·EAG 점 찍어 Save
  → edge_app에서 그 세션 Load → knee 수정 → Save
다음 세션 반복
  → 다 끝나면 parameter_extractor --batch → stats_grf_eag
```

핵심 산출물: `result/manual_offsets.json`(확정 offset), `result/manual_edges.json`(확정 edge).
이 둘만 채워지면 재추출 시 전부 자동 반영된다.

## 참고

- **외부(DDNS) 접속**: 두 GUI 모두 code-server 내장 포트 프록시로 중계된다. 도커 포트 퍼블리시·공유기 포트포워딩 불필요.
  - offset GUI : `http://medibb.synology.me:18440/proxy/8766/`
  - edge GUI  : `http://medibb.synology.me:18440/proxy/8765/`
  - **끝 슬래시(`/`) 필수**, code-server에 로그인된 브라우저여야 한다(프록시가 인증 뒤에 있음).
  - 두 앱 모두 API를 문서 기준 상대경로로 호출하므로 프록시 prefix 아래에서 동작한다. 새 GUI를 만들 때도 절대경로(`/api/...`) 대신 같은 방식을 쓸 것.
- 한글 폰트: Linux/Docker는 `pip install koreanize-matplotlib`(또는 `apt-get install -y fonts-nanum`) 후 패널 재생성 시 한글 라벨 정상 표시. macOS는 자동.
- 상세 파일 구조·전체 파이프라인은 `README.md`, 통계 설계는 `STATS_PLAN.md` 참조.
