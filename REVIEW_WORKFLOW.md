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
- 드롭다운에서 세션 선택(검토대상 `▲` 우선 정렬) → 자동 Load
- **GRF 패널(위)에서 기준 변곡점 클릭 → EAG 패널(아래)에서 대응 변곡점 클릭** → 쌍 성립,
  두 점이 겹치도록 EAG가 즉시 이동 (최종 offset = `corrected + (EAG − GRF)`)
- 쌍을 여러 개 찍으면 **중앙값** 사용 → 한 쌍이 부정확해도 강건. 표에서 개별 쌍 삭제 가능
- **휠**=확대/축소, **드래그**=좌우 이동, **snap** 체크 시 클릭 지점 근처의 변곡점(|2차미분| 최대)으로 자동 흡착
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

각 GRF 전이(rise=부하 / fall=이탈)마다 EAG **knee-pair(onset+offset) 2개**가 plateau corner에 붙어야 한다 (체중부하 cycle당 4 knee). 놓친 것은 **추가**, 가짜는 **삭제**, 어긋난 것은 **이동**.

### 방법 A: GUI (권장)
```bash
python3 edge_app.py --host 0.0.0.0 --port 8765
```
브라우저 `http://<서버IP>:8765` 접속 →
- 상단 **worklist 드롭다운**에서 세션 선택 (또는 세션 경로 직접 입력) → 채널 입력 → **Load**
- knee 점 **드래그**로 이동 · **Add mode** 후 트레이스 2점 클릭으로 edge 추가 · edge 선택 후 **Delete/Del키** · **snap** 체크 시 corner 자동정렬
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
