# EAG-GRF Annotation Protocol

**이 문서가 판정 기준의 정본이다.** offset, edge(knee), 제외 라벨을 사람이 확정할 때
"무엇을 옳다고 보는가"는 전부 여기에만 정의한다. 다른 문서는 이 문서를 링크만 하고
기준을 다시 쓰지 않는다.

| 문서 | 역할 |
|---|---|
| **이 문서** | 조작적 정의, 허용 오차, 판정 규칙, rater 자격, 신뢰도 설계 (Methods 원본) |
| `ANNOTATION_GUIDE.md` | 연구원 실무 안내 (접속, 조작, 저장) |
| `README.md` | 코드 구조, CLI, 파이프라인 |
| `STATS_PLAN.md` | 통계 설계 |

---

## 1. 왜 이 문서가 필요한가

OBJ2는 선행 EAG 연구와 달리 **사람이 knee point를 확정하는 구조**를 도입했다.
그 순간부터 파라미터(`amplitude`, `slope`, `transition_time`, `latency`)가 사람의 판단에
의존하게 되므로, 판단 기준을 문서화하지 않으면 재현도 검증도 불가능하다.

특히 Ch.4는 test-retest **ICC(2,1) 0.463 (기울기) / 0.661 (진폭), MDC95 1.784**를 이미
보고했다. 이 낮은 재현성이 생물학적 변동인지 주석 잡음인지 구분하려면 주석 오차를
독립적으로 정량화해야 한다. 그것이 이 문서의 8장이다.

선행 EAG 문헌에는 이런 보고의 전례가 없다. 프로토콜상 정해진 시각에서 진폭을 읽는
방식에는 주석 오차라는 개념 자체가 없었기 때문이다. 따라서 기준은 EAG 분야가 아니라
**같은 종류의 문제를 오래 다뤄온 인접 분야**(보행 이벤트 라벨링, EMG onset 판정,
수면 EEG 이벤트 판독)의 관행을 따른다. 근거는 각 절에 표기했다.

---

## 2. 측정 프로토콜 (모든 판정의 기준)

### 2.1 세션 구조

```
초기 발구름(stomp)  →  한발서기 4회 (검사측 부하를 회차마다 증가)  →  종료
```

- **초기 발구름**: EAG-GRF 시간 동기화 세팅용. **분석 제외**
- **한발서기 4회**: 검사측 다리 부하를 점진적으로 증가. **모든 세션이 4회를 시행했다**
- 프로토콜 전후의 양발 서기(휴식): **분석 제외**

조건 3종은 검사측에 **동일한 graded PWB**를 적용하고, 나머지 부하를 어디로 보내는지만
다르다. s = 반대측 다리(좌우), f = 반대측 다리(전후), c = 반대측 상지 forearm crutch.

### 2.2 신호 정의

**GRF signed imbalance** `(L-R)/(L+R)`, 사각파 형태.

| 개념 | 정의 |
|---|---|
| `rest_level` | 휴식 plateau. 반대측에 싣고 검사측을 비운 상태 (`\|signed\| ≈ 0.9~1.0`). 부하 전/사이×3/후로 5개 |
| `load_level` | 각 부하 구간 plateau. 회차마다 1개, 크기 점증 |
| `load_step` | `load_level - rest_level` (부호 포함) |
| `load_pct` | **검사측 체중부하율(%)** = 검사측 힘 / 전체 힘, 부하 구간 평균. 실측 중앙값 **18 / 46 / 73 / 92%** (설계 20/50/80/100%보다 일관되게 낮음) |

**dose 축은 명목값이 아니라 실측 `load_pct`를 쓴다.**

### 2.3 분석 단위

```
1 cycle (한발서기 1회) = 부하 시작 anchor + 이탈 anchor = 2 이벤트
세션·채널당 4 cycle × 2 = 8 이벤트
```

코드 상수: `EXPECTED_CYCLES = 4`, `EXPECTED_EVENTS = 8`,
`EXPECTED_DIR = {'on':'fall', 'off':'rise'}` (`grf_triggered_annotator.py`)

### 2.4 방향 규칙 (절대 규칙)

> **부하가 실리면 EAG는 하강(fall), 빠지면 상승(rise).** 예외 없다.

방향이 맞지 않는 edge는 그 anchor의 반응이 아니다. 노이즈가 anchor에 더 가까이
붙어 있어도 방향이 틀리면 채택하지 않는다.

---

## 3. 전처리 동결

주석 작업은 아래 설정으로 생성된 트레이스 위에서 이루어진다. **작업 중 변경 금지.**

| 항목 | 값 | 위치 |
|---|---|---|
| 저역통과 | **5.0 Hz**, Butterworth order 5, `filtfilt` | `eag_analyzer.py:102-105` |
| 드리프트 | **`detrend`** (선형 추세 제거) | `eag_analyzer.py:108` |

**변경 금지 이유.** `grf_triggered_annotator.py:1014`는 수동 edge가 있으면 JSON에 저장된
`onset_amp` / `offset_amp`를 **그대로 읽는다**(트레이스에서 재계산하지 않는다).
따라서 전처리를 바꾸면 수동 확정 채널만 옛 진폭을 유지하고 자동 채널은 새 진폭을 갖는
혼재 상태가 된다.

> Ch.4 §5.5의 "취득 LP를 5 Hz에서 2 Hz로" 권고는 **후속 연구의 취득 설정**에 대한 것이며,
> 이 데이터를 재처리하라는 뜻이 아니다.

GUI가 그리는 트레이스(`edge_app.py:65`)와 자동 검출기가 쓰는 트레이스
(`grf_triggered_annotator.py:1018`)는 동일한 `detrend(sa.eag_filtered[:, ch-1])`이다.
따라서 **수동 진폭과 자동 진폭은 같은 척도 위에 있다.**

---

## 4. 주석 층위

| 층 | 단위 | 저장소 | 주 지표 |
|---|---|---|---|
| **offset** | 세션당 스칼라 1개 (초) | `result/manual_offsets.json` | Bland-Altman LoA |
| **edge (knee)** | 채널당 이벤트 목록 (시각 s, 진폭 µV) | `result/manual_edges.json` | 검출 F1 + 시점·진폭 LoA |
| **제외 라벨** | 세션 또는 채널 (범주형) | `result/exclusions.json` | Cohen's κ |

세 층은 척도가 다르므로 지표도 다르다. 특히 **범주형인 제외 라벨에 ICC를 쓰지 않는다.**

---

## 5. 조작적 정의

### 5.1 offset: "정렬이 맞다"

**정의.** CORRECTED 오버레이에서, GRF signed imbalance 사각파의 **전이 지점**과
EAG 트레이스의 대응 변곡점이 **허용 오차(§5.3) 이내**로 겹치는 상태.

판정은 **전이 지점 기준**이며 plateau 구간의 겹침이 아니다. plateau는 진폭이 서로 다른
신호라 겹칠 이유가 없다.

- 8개 anchor 중 **육안으로 확인 가능한 것 전부**를 대상으로 한다
- 일부만 맞고 일부가 어긋나면 정렬이 아니라 **cycle-skip**을 의심한다
  (5초 준주기 신호에서 한 주기 밀린 정렬은 국소적으로 잘 맞아 보인다)
- 확정 시 대응점 쌍을 **2개 이상** 찍고 중앙값을 쓴다. 단일 쌍은 그 쌍의 오차를
  그대로 물려받는다

### 5.2 edge: onset / offset knee

**정의.** 한 anchor에 대응하는 EAG 반응의 시작점(`onset`)과 종료점(`offset`).
각각 트레이스 기울기가 급변하는 **corner point**이며, 극값(peak)이 아니다.

- `onset`: 기울기가 baseline에서 반응 방향으로 꺾이는 지점
- `offset`: 반응이 끝나고 기울기가 다시 완만해지는 지점
- `amplitude` = `offset_amp - onset_amp`, `transition_time` = `offset_time - onset_time`
- `latency` = `onset_time - anchor_time`. **동시성 가정상 0에 가까워야 한다**

**onset과 offset은 성질이 다르다.** onset은 부하 인가에 따른 급격한 corner이고,
offset은 복귀 램프를 포함해 완만하다. 이 비대칭이 §5.3의 근거다.

### 5.3 허용 오차 (tolerance)

**사전에 임의로 정하지 않는다. 파일럿으로 측정한 뒤 그 값을 채택한다.**

이것은 보행 이벤트 라벨링의 표준 절차다. Wu et al. (2022)은 모션캡처 보행 이벤트의
inter-labeler LoA를 먼저 측정한 뒤 알고리즘을 그 기준으로 평가했고, 그 결과
**이벤트 종류에 따라 LoA가 4~5배 차이**났다: toe off 16 ms, heel strike 24 ms,
heel off 72 ms, flat foot 80 ms. 급격한 이벤트는 좁고, 완만한 이벤트는 넓다.
Vasseljen et al. (2006)의 육안 EMG onset 판정도 smallest detectable difference가
21~24 ms 수준이었다.

**절차.** `reliability_pilot.py`가 세 단계를 담당한다.

```bash
# 1. 대상 세션을 층화 추출해 동결 (worklist 5 + 비worklist 5). 시작 전 커밋할 것
python3 reliability_pilot.py sessions --n 10

# 1b. 채널 분모를 동결 (본 분석과 같은 PASS 규칙). 시작 전 커밋할 것
python3 reliability_pilot.py channels

# 2. 자동 검출 스냅샷 (나중에 "사람이 실제로 손댄 이벤트"를 가려내는 기준)
python3 reliability_pilot.py baseline

# 3. rater가 각자 격리된 저장소에 백지로 작업 (§8.3)
#    두 rater가 **같은 일을 각자** 해야 한다. 따라서 앱 4개 = 포트 4개.
#    EAG_RESULT_DIR은 절대경로로 준다 — store_io는 받은 값을 그대로 쓰므로,
#    상대경로면 앱을 띄운 위치에 따라 rater 자료가 딴 데 쌓인다.
R=$PWD/result/reliability
EAG_RESULT_DIR=$R/rater_main python3 offset_app.py --host 0.0.0.0 --port 8768 --blank
EAG_RESULT_DIR=$R/rater_main python3 edge_app.py   --host 0.0.0.0 --port 8769 --blank
EAG_RESULT_DIR=$R/rater_rel  python3 offset_app.py --host 0.0.0.0 --port 8770 --blank
EAG_RESULT_DIR=$R/rater_rel  python3 edge_app.py   --host 0.0.0.0 --port 8771 --blank
EAG_RESULT_DIR=$R/rater_ref  python3 offset_app.py --host 0.0.0.0 --port 8772 --blank
EAG_RESULT_DIR=$R/rater_ref  python3 edge_app.py   --host 0.0.0.0 --port 8773 --blank

# 4. 쌍별 LoA. 어느 쌍이 무엇을 재는지는 §8.3의 역할표를 볼 것
python3 reliability_pilot.py report --a rater_main --b rater_rel   # Methods 대표값
python3 reliability_pilot.py report --a rater_ref  --b rater_main  # §6 자격 판정
python3 reliability_pilot.py report --a rater_ref  --b rater_rel   # 참고
```

> **한 명에게 offset만, 다른 한 명에게 edge만 시키면 안 된다.** 비교할 공통 항목이
> 없어져 리포트가 통째로 빈 채로 나온다 — 에러 없이, 며칠치 주석 작업을 마친 뒤에야.
> `cmd_report`는 상대 rater에 항목이 없으면 조용히 건너뛴다.

리포트는 tolerance를 0.02~0.80 s로 스윕하며 매칭 수와 F1을 보여준다.
**F1이 평평해지기 시작하는 지점이 실질 상한**이다. 그보다 키우면 서로 다른 이벤트를
억지로 묶기 시작한다. 그 지점에서 onset·offset·진폭의 LoA를 각각 산출한다.

산출된 LoA 반폭을 반올림해 아래 표에 적고, 확정일과 리포트 경로를 함께 기록한 뒤
커밋한다. 확정된 tolerance는 이후 (a) rater 자격 판정(§6), (b) 신뢰도 매칭(§8),
(c) auto vs manual 비교에 **동일하게** 쓴다.

**표본을 사후에 고르지 않는다.** `sessions`가 만든 `pilot_sessions.csv`를 작업 시작 전에
커밋해 동결한다. 결과를 보고 세션을 바꾸면 표본 선택 편의가 된다.

| 항목 | 값 |
|---|---|
| onset tolerance | **[파일럿 후 기입]** |
| offset tolerance | **[파일럿 후 기입]** |
| offset(동기화) tolerance | **[파일럿 후 기입]** |
| 확정일 / 근거 | **[기입]** |

> **혼동 주의.** 자동 검출기의 매칭 창(`NOISE_MARGIN = 1.2 s`, `lat_lo`/`lat_hi`)은
> 여기서 정하는 주석 허용 오차와 **다른 파라미터**다. 전자는 알고리즘이 후보를 고르는
> 범위이고, 후자는 사람 둘의 일치를 판정하는 범위다. 후자가 훨씬 작다
> (`latency ≈ 0` 결과로 보아 0.05~0.15 s 수준이 예상된다).

### 5.4 애매·동점 상황

**강제로 판정시키지 않는다. 등급 라벨로 남기고 하류에서 처리한다.**

수면 EEG 이벤트 판독에서 확립된 방식이다. Zhao et al. (2017)은 spindle을
definite(가중치 1) / indefinite(0.5)로 나눠 기록했고, Lacourse et al. (2019)은 개별
전문가가 아니라 **전문가 집단의 합의**를 gold standard로 삼았다. Rahimi et al. (2026)은
전문가 8명이 주석한 결과 **경계가 완만한 전이일수록 일치도가 낮다**는 것을 보였다.
즉 애매 구간의 낮은 일치도는 실패가 아니라 신호의 성질이다.

**현행 라벨 체계** (이미 코드에 구현되어 있다):

| 라벨 | 뜻 | 처리 |
|---|---|---|
| 매칭됨 (`c2이탈` 등) | 그 anchor의 반응으로 채택 | 분석 사용 |
| `후보` | 경계 근처인데 매칭되지 않은 edge | **지우지 않는다.** 민감도 분석 대상 |
| `노이즈` | anchor에서 `NOISE_MARGIN`(1.2 s)을 넘게 떨어진 것 | 삭제 가능 |
| `한쪽만 채택` | 방향이 맞는 후보가 한쪽에만 있음 | 채택하되 후순위 검토 |

**결정 규칙.**

1. **미매칭이라는 이유만으로 지우지 않는다.** 경계에서 조금 벗어난 진짜 반응까지
   사라진다. 1.2 s 밖의 plateau 한가운데 것만 노이즈로 본다
2. **한쪽만 있으면 그쪽을 채택한다.** 같은 부하의 rise와 fall은 크기가 거의 같으므로,
   한쪽이 오염돼도 다른 쪽으로 그 부하의 EAG 크기를 알 수 있다.
   `asymmetry`가 크면 한쪽이 오염됐다는 신호다
3. **`후보` 라벨이 붙은 이벤트는 민감도 분석으로 처리한다.** 포함/제외 두 번 돌려
   결론이 바뀌는지만 보고한다. 제3자 판정은 신뢰도 하위연구(§8)에서만 쓴다
4. **아무리 봐도 못 맞추면 제외 라벨(§5.5)을 붙이고 넘어간다.** 붙잡고 있지 않는다

### 5.5 제외 라벨

**삭제가 아니라 라벨이다.** 데이터와 주석은 남고 통계에서만 빠지므로 되돌릴 수 있다.

- **범위**: 세션 전체 (동기화 불가, 프로토콜 파손) 또는 채널 단독
- **표준 사유**: `노이즈` · `동기화불가` · `프로토콜이상` · `기록오류` · `기타`
- **자유 메모 필수**: 나중에 왜 뺐는지 판단하는 근거가 된다
- **애매하면 제외하지 않는다.** `검토 필요`로 남겨 PI가 다시 본다

폴더명에 붙은 QC 접미사(`(분석안됨)` 등)는 **임시 표시**이며,
정본은 `result/exclusions.json`이다. 폴더명만 고치면 파이프라인에 반영되지 않는다.

---

## 6. rater 자격과 훈련

**기준 답이 있는 훈련 세트, 진입 임계값, 종료 시 드리프트 점검**의 3단 구조를 쓴다.
수면 판독(AASM)에서 판독자에게 정본 대비 일치도 기준을 요구하는 방식과 같다.
Zhao et al. (2017)은 비전문가 6~9명이 전문가 집단 기준을 대체할 수 있음을 보였는데,
비전문가에게 위임하는 본 연구 상황에 직접 대응된다.

| 단계 | 내용 |
|---|---|
| ① 기준 학습 | 이 문서 §2, §5를 읽는다 |
| ② 감독 하 연습 | **10세션**. PI 주석을 정본으로 대조하며 피드백 |
| ③ 자격 판정 | 별도 **5세션**을 독립 수행. **tolerance(§5.3) 내 일치 ≥ 80%** 면 본 작업 투입 |
| ④ 드리프트 점검 | 캠페인 종료 시 **초기 5세션을 재주석**. 초기 대비 계통 이동이 있으면 보고 |

④는 163세션을 여러 주에 걸쳐 수행하는 동안 기준이 서서히 이동할 수 있어서 넣는다.
비용이 거의 없고, Methods에서 "기준 안정성을 확인했다"고 쓸 수 있는 근거가 된다.

---

## 7. 작업 순서와 불변식

```
전처리 동결(§3)  →  offset 확정  →  edge 확정  →  재추출  →  통계
```

**불변식 4가지. 위반하면 앞 단계 작업이 무효가 된다.**

1. **offset 먼저, edge 나중.** edge는 보정된 시간축(`te_corr`)에 저장되므로, offset이
   바뀌면 그 세션의 edge가 전부 어긋난다
2. **전처리 상수를 바꾸지 않는다** (§3)
3. **사람이 확정한 값이 최우선.** `manual_offsets.json` / `manual_edges.json`이 있으면
   자동 검출을 무시한다
4. **확정 후 반드시 재추출.** `parameter_extractor.py --batch`를 다시 돌려야
   확정값과 `load_pct` · `accepted` 등 최신 컬럼이 반영된다

실측 근거: offset 미확정 세션은 edge 결과가 눈에 띄게 나쁘다 (프로토콜 통과 77.9% vs
88.1%, `high` 22.1% vs 11.9%). cycle 검출률은 차이가 없다 (95.5% vs 96.0%). cycle은
GRF만으로 잡혀 offset과 무관한 반면 EAG-anchor 매칭은 offset 오차에 직접 깨지기 때문이다.
**offset을 확정하면 edge 결과가 상당수 저절로 좋아진다.**

---

## 8. 신뢰도 하위연구

### 8.1 질문 3개, 지표 3개

**이벤트 시점에 ICC를 쓰지 않는다.** 이벤트가 50초 구간에 흩어져 있고 rater 불일치가
0.05초면 ICC는 0.999 이상이 나와 아무 의미가 없다. ICC는 대상 간 산포에 의존하는
상대 지표라 절대 시점 오차에는 부적합하다. Wu et al. (2022)과 Vasseljen et al. (2006)이
모두 LoA / SDD를 쓰는 이유다.

| 질문 | 지표 | 근거 |
|---|---|---|
| 두 사람이 **같은 자리에 이벤트를 찍었나** | precision / recall / **F1**, Cohen's κ | Lacourse et al. 2019 |
| 찍었다면 **시점·진폭이 얼마나 같나** | **Bland-Altman LoA** (bias ± 1.96 SD) | Wu et al. 2022; Vasseljen et al. 2006 |
| 최종 **파라미터**가 얼마나 같나 | **ICC(2,1)** + LoA | 오차 예산 (§8.5) |

이벤트 매칭은 §5.3의 tolerance로 판정한다. tolerance 안이면 "같은 이벤트를 이동",
밖이면 "삭제 + 추가"로 센다. **이 규칙을 사전에 고정하지 않으면 수치가 임의로 움직인다.**

### 8.2 표본

독립 단위는 이벤트가 아니라 **세션**이다. 한 세션 안 8채널 × 8이벤트는 같은 트레이스와
같은 판단 맥락을 공유하므로 독립이 아니다. 이벤트를 단순히 쌓아 계산하면 신뢰구간이
가짜로 좁아진다. 따라서 **채널을 늘리는 것보다 세션을 늘리는 것이 정밀도를 더 많이 산다.**

| 항목 | 규모 | 1인당 소요 |
|---|---|---|
| offset inter-rater | **40세션** (worklist 20 + 비worklist 20) | 약 1.5 시간 |
| edge inter-rater | **30세션 × 2채널** (내측 1 + 외측 1) = 60채널 | 약 4 시간 |
| intra-rater 재검토 | 위의 절반, **2주 이상 간격** | 약 3 시간 |

층화 이유: `worklist.csv`의 163세션은 이미 어려운 쪽으로 치우쳐 있다. 여기서만 뽑으면
최악 신뢰도, 전체에서만 뽑으면 낙관적 신뢰도가 나온다. 두 층에서 뽑아 **층별로도
보고**한다.

30세션에서 60세션으로 늘려도 신뢰구간은 약 1.4배만 좁아진다. 수확체감이 빠르다.

**본 분석에서 빠진 자료는 표본에서도 뺀다.** 분석에 쓰지 않을 세션의 일치도로 tolerance를
정하면, 그 tolerance가 적용될 자료와 다른 모집단에서 나온 값이 된다. 두 경로로 거른다.

| 경로 | 근거 파일 | 판정 |
|---|---|---|
| 방문 단위 | `result/cohort_manual.csv` → `data_flat/manifest.csv`의 `audit_ok` | 전수조사에서 방문째 제외 |
| 세션 단위 | `result/exclusions.json` | 노이즈 등으로 사람이 표시한 제외 |

`cohort_manual.csv`의 방문명에는 연구원이 붙인 QC 접미사가 남아 있고
(`(02.19_17)이다은_1(분석안됨)`) 미러·manifest는 그걸 뗀 이름을 쓴다. 정규화는
`build_flat_view.load_manual_cohort()`가 `strip_qc()`로 한 번만 수행하고, 하류는
`audit_ok`를 읽는다. 같은 규칙을 두 곳에 두면 한쪽만 고쳐진다.

**채널 분모도 같은 원리로 사전에 동결한다.** 노이즈로 오염된 채널은 본 분석
(`channel_quality == 'PASS'`)에서 이미 빠지므로, 그 불일치를 신뢰도에 넣으면 쓰지도 않을
자료로 tolerance를 정하게 된다. `reliability_pilot.py channels`가 `flag_noise()`
(SNR < 10 dB · 50 Hz · HF 비율 · 드리프트 · RMS)를 적용해 `pilot_channels.csv`를 만든다.

**빼는 주체가 rater여서는 안 된다.** rater가 작업 중 어려운 채널을 건너뛰면 분모가
사람마다 달라지고, 빠지는 쪽이 하필 불일치가 큰 자료라 LoA가 좁아진다
(informative missingness). 그래서 주석 **전에** 신호 지표만으로 정한다.
`compute_noise_metrics`는 원시 신호만 보므로 rater와 무관하다.

| 빼는 것 | 남기는 것 |
|---|---|
| **신호 품질** — SNR 낮음, 50 Hz, 드리프트 | **판정 난이도** — 신호는 깨끗한데 knee가 완만함 |
| 본 분석에서도 빠짐 | 본 분석에 들어감 |

두 번째를 같이 빼면 안 된다. §5.3에 인용한 Wu et al.의 toe off 16 ms vs flat foot 80 ms가
바로 그 경우다 — 노이즈가 아니라 이벤트가 본래 완만해서 넓다. 프로토콜이 감당해야 할
진짜 불확실성이고 tolerance가 포괄해야 할 대상이다. 구분 기준은 **rater의 체감이 아니라
신호 지표**다.

**동결 상태 (2026-08-19).** 제외 151세션을 뺀 1081에서 층화 추출한 10세션
(worklist 5 + 비worklist 5, 방문 중복 없음)과 그 80채널 중 PASS **73채널**
(LOW_SNR 7개 제외)을 `pilot_sessions.csv` · `pilot_channels.csv`로 동결하고,
같은 시점의 자동 검출 결과를 `auto_baseline.json`으로 함께 커밋했다(`fbfc325`).
표본을 다시 뽑으면 `baseline`도 반드시 다시 만든다 — §8.6이 "사람이 실제로 손댄
이벤트"를 가려내는 기준이 그 스냅샷이기 때문이다.

### 8.3 독립 관찰 확보

GUI는 자동 검출 결과를 초기값으로 그리고, 이미 확정된 세션은 `✅ [확정 ...s]`로 표시한다.
**두 rater가 같은 초기값에서 시작하면 둘 다 손을 안 댄 경우가 완전 일치로 잡혀,
재는 것이 사람의 일치도가 아니라 알고리즘의 결정론성이 된다.**

**구현 (2026-08-18).** 두 장치로 확보한다.

**(1) 저장 위치 격리** — 환경변수 `EAG_RESULT_DIR`가 세 저장소의 루트를 바꾼다
(`store_io.py`). rater별·라운드별 디렉터리를 쓰면 서로의 값을 보지 못하고, 본 분석
파이프라인도 파일럿 값을 보지 못한다. `sync_analyzer`도 같은 모듈을 통해 offset을
조회하므로 자동으로 따라온다.

```
result/reliability/
├── pilot_sessions.csv       # 대상 목록 (동결, git 추적)
├── auto_baseline.json       # 자동 검출 스냅샷
├── pilot_channels.csv       # 채널 분모 (동결, git 추적)
├── rater_main/{manual_offsets,manual_edges,exclusions}.json
├── rater_rel/…              # 신뢰도 전용 rater
├── rater_ref/…              # PI (정본)
├── round2_rater_main/…      # intra-rater 재검토 (2주 후)
└── report/                  # LoA 표, tolerance 스윕
```

rater ID와 라운드를 **파일 필드가 아니라 디렉터리로** 표현하므로 저장소 스키마를
바꿀 필요가 없다.

포트는 rater × 작업으로 나눈다. 본 캠페인(8765/8766)은 그대로 두고 별도 프로세스로 띄운다.
`EAG_RESULT_DIR`은 프로세스별 환경변수라 서로 간섭하지 않는다.

| 역할 | 누구 | offset | edge | 저장소 |
|---|---|---|---|---|
| `rater_main` | 주연구원 (본 캠페인도 이 사람) | **8768** | **8769** | `result/reliability/rater_main/` |
| `rater_rel` | 신뢰도 전용 rater (**가설 비공개**) | **8770** | **8771** | `result/reliability/rater_rel/` |
| `rater_ref` | PI (정본) | **8772** | **8773** | `result/reliability/rater_ref/` |
| (본 캠페인) | 주연구원 | 8766 | 8765 | `result/` |

세 사람이 **같은 동결 표본**을 각자 하면 쌍이 3개 나온다. 쌍마다 재는 것이 다르다.

| 쌍 | 산출 | 용도 |
|---|---|---|
| `rater_main` ↔ `rater_rel` | inter-rater LoA | **Methods 대표값** (둘 다 프로토콜 작성자가 아님) |
| `rater_ref` ↔ `rater_main` | criterion 일치 | §6 자격 판정 |
| `rater_ref` ↔ `rater_rel` | 참고 | 프로토콜 전달력 점검 |

**tolerance는 세 쌍 중 가장 넓은 것에서 채택한다.** 좁은 쌍(둘 다 숙련·한 명은 작성자)에서
뽑아 느슨한 쌍에 적용하면, 같은 이벤트를 가리킨 두 표시가 창 밖으로 밀려
`삭제 + 추가`로 계산된다 — 그 사람의 신뢰도가 아니라 창을 잘못 고른 결과가 보고된다.
가장 넓은 쌍에서 뽑으면 §6의 순환(정본 설정자가 포함된 쌍에서 나온 창으로 그 정본과의
일치를 판정하는 것)도 함께 풀린다.

`rater_rel`에게는 **가설을 알리지 않는다.** §2·§5만 훈련시킨다. 조건(s/f/c)은 세션명과
GRF 파형에 드러나 맹검이 불가능하므로, 이것이 확보 가능한 유일한 맹검 층이다.

**누가 어느 포트를 쓰는지 먼저 못박고 시작한다.** 두 rater가 같은 code-server 로그인을
공유하므로 주소만으로는 구분되지 않는다. 포트를 잘못 열면 상대 저장소에 쓰게 되고,
그 시점에서 독립 관찰이 깨진다. **3002는 금지**(api-server 전용).

**(2) 백지 모드** — 두 앱의 `--blank` 플래그.

| 앱 | `--blank`가 숨기는 것 | 남기는 것 |
|---|---|---|
| `offset_app.py` | 자동 보정(residual), best-match 후보, match 프로파일 | 원시 트레이스 |
| `edge_app.py` | EAG edge 자동검출 | **GRF anchor 8개** |

edge 쪽에서 anchor를 남기는 것이 중요하다. anchor는 GRF에서 나온 객관적 기준이고,
사람이 재는 것은 그 자리의 knee 위치다. anchor까지 숨기면 다른 과제를 재게 된다.

### 8.4 자동 수용 채널 층화

edge 작업의 상당 부분은 자동 결과를 그대로 받아들이는 것이다. 이를 섞어 계산하면
일치도가 부풀려진다. 따라서 **두 가지를 나눠 보고한다.**

- **전체(all-comers)**: 파이프라인 전체의 실제 신뢰도. 오차 예산(§8.5)에 쓴다
- **수정된 채널만**: 둘 중 한 명이라도 손댄 채널. 판단 작업 자체의 신뢰도

### 8.4b 커버리지와 제외 판단

리포트 §0은 **이행 점검**이다. 분모가 §8.2에서 동결돼 있으므로 편향 보정이 아니라
"양쪽이 분모를 다 했는가"를 확인하는 것이다. 그래도 반드시 찍는다 — 이 표가 없으면
누락을 알아낼 방법이 없고, 아래의 모든 LoA는 조용히 교집합에서만 계산된다.

리포트 §0b는 **제외 판단의 일치도**다. 한 명이 "못 쓴다"고 하고 다른 한 명이 knee를
찍었다면, 시점이 얼마나 다른가가 아니라 *잴 수 있는 자료인가*에 대한 이견이다.
edge 비교는 한쪽에 항목이 없다는 이유로 이걸 통째로 건너뛰므로 따로 센다.

**단순 일치율이 아니라 Cohen's kappa로 보고한다.** 제외 판정은 드물기 때문에 일치율은
자동으로 높게 나온다 — 검증 시나리오에서 일치율 95.7%인데 kappa는 0.553이었다.
일치율만 적으면 "판단이 일치했다"의 근거가 되지 못한다.

### 8.5 오차 예산 (이 하위연구의 목적)

Ch.4는 test-retest **MDC95 1.784** (평균 기울기 1.45)를 보고했다. 심사자는 이 낮은
재현성이 생물학적 변동인지 주석 잡음인지 반드시 묻는다.

**필요한 진술은 부등식 하나다: annotation LoA ≪ MDC95 1.784.**

- 격차가 크면 표본이 작아도 결론이 선다. LoA가 0.2~0.3 수준으로 나오면 n = 20에서도
  명확히 클리어된다
- 1.0 근처로 나오면 표본을 키워도 문제는 그대로다. 그때는 파이프라인 문제다

**단계적 설계.** 10세션 파일럿의 산포를 먼저 보고, 명확히 클리어되면 그 n으로 보고하고
멈춘다. 애매하면 §8.2 규모로 확장한다. **2단계 설계임을 사전에 명시**해야 사후 표본
조정이라는 지적을 피한다.

**집계 층위 주의.** Vasseljen et al. (2006)은 육안 onset 판정이 단일 시행에서는
수용 불가였으나 반복 시행을 평균하면 수용 가능했고, "onset 추정은 반복 시행의 평균값에
기반해야 한다"고 결론냈다. OBJ2의 분석 단위는 이미 방문×조건 집계값이므로,
**이벤트 수준 LoA는 최종 파라미터에 미치는 영향을 과대평가한다.**
신뢰도는 **분석이 실제로 쓰는 층위(집계 기울기)에서도** 보고한다.

### 8.6 auto vs manual (후속 논문 범위)

자동 검출기의 성능 평가(Bland-Altman bias·LoA, 수동 보정 비율의 추가/삭제/이동 분해)는
**본 연구가 아니라 montage 검증 후속 논문에서 다룬다.** Ch.4에서는 "수동 보정을
적용했다"는 사실과 §8.5의 오차 예산만 보고한다.

이유: 현재 GUI가 자동값을 초기값으로 보여주므로, 보정 비율은 자동의 정확도가 아니라
rater가 얼마나 손댈 의향이 있었는지를 재게 된다. 이 순환을 끊으려면 §8.3의 백지
주석을 전면 적용해야 하는데, 그 규모는 후속 논문에서 감당하는 것이 맞다.

---

## 9. 재현과 감사

### 9.1 저장소

| 파일 | 내용 |
|---|---|
| `result/manual_offsets.json` | 확정 offset (`manual_offset`, `auto_offset`, `auto_method`, `updated_at`, `note`) |
| `result/manual_edges.json` | 확정 edge (채널별 `edges[]`, `offset_used`, `updated_at`) |
| `result/exclusions.json` | 제외 라벨 (`reason`, `note`, `updated_at`) |

세 파일만 채워지면 재추출 시 전부 자동 반영된다.

세 저장소 모두 `store_io.save_json_atomic()`을 통해 **원자적으로** 기록한다
(임시 파일 → `os.replace`, 교체 전 `.bak` 1세대 보존). 기존 방식(`open(path,'w')` 직후
`json.dump`)은 쓰는 도중 끊기거나 두 명이 동시에 저장하면 파일이 통째로 유실됐다.
확정값 수백 건이 한 번에 날아갈 수 있어 신뢰도 연구와 무관하게 필요한 조치였다.

원자적 교체는 손상을 막을 뿐 **동시 저장의 경합 자체를 막지는 않는다.** 두 사람이 같은
세션을 저장하면 여전히 마지막 저장이 이긴다. 본 캠페인에서는 세션을 나눠 맡고,
신뢰도 하위연구에서는 `EAG_RESULT_DIR`로 저장소를 분리한다.

### 9.3 변경 이력

제외·재측정 결정은 `exclusions.json`에 등록해야 파이프라인에 반영된다.
대화나 폴더명으로만 남기면 재추출 때 그대로 살아 들어온다.

---

## 10. Methods 서술 템플릿

Ch.4 4.2.7에 들어갈 문장 구조다. 대괄호는 확정 후 채운다.

> Offsets and EAG knee points were confirmed by trained raters following a pre-specified
> protocol (§ Annotation Protocol). Raters completed [10] supervised training sessions and
> qualified by achieving ≥ 80% agreement with the reference annotation within the
> event-specific tolerance, which was derived empirically from a two-rater pilot
> ([n] sessions; onset [x] s, offset [y] s) rather than fixed a priori, following the
> approach of Wu et al. (2022). Ambiguous events were retained under a graded label and
> handled by sensitivity analysis (included vs excluded) rather than forced adjudication.
> Annotation reliability was assessed on [k] independently double-annotated sessions,
> reporting detection agreement (F1) and Bland-Altman limits of agreement for event timing
> and amplitude; agreement in the derived per-visit parameters was reported as ICC(2,1).
> Annotation limits of agreement were [z], substantially below the test-retest MDC95 of
> 1.784, indicating that the observed test-retest variability is not attributable to
> annotation error.

---

## 참고문헌

1. Wu J, Maurenbrecher H, Schaer A, et al. Human gait-labeling uncertainty and a hybrid
   model for gait segmentation. *Front Neurosci*. 2022. doi:10.3389/fnins.2022.976594
   (보행 이벤트 inter-labeler LoA: toe off 16 ms, heel strike 24 ms, heel off 72 ms,
   flat foot 80 ms. 이벤트 종류별 tolerance 유도 절차)
2. Vasseljen O, Dahl HH, Mork PJ, Torp HG. Muscle activity onset in the lumbar multifidus
   muscle recorded simultaneously by ultrasound imaging and intramuscular electromyography.
   *Clin Biomech*. 2006. doi:10.1016/j.clinbiomech.2006.05.003
   (육안 EMG onset 판정 SDD 21~24 ms. 단일 시행 불가, 반복 평균 필요)
3. Lacourse K, Delfrate J, Beaudry J, Peppard P, Warby SC. A sleep spindle detection
   algorithm that emulates human expert spindle scoring. *J Neurosci Methods*. 2019.
   doi:10.1016/j.jneumeth.2018.08.014
   (전문가 합의를 gold standard로, 사람과 알고리즘을 동일한 F1으로 비교)
4. Zhao R, Sun J, Zhang X, et al. Sleep spindle detection based on non-experts:
   A validation study. *PLoS One*. 2017. doi:10.1371/journal.pone.0177437
   (definite/indefinite 등급 라벨, 비전문가 합의 기준)
5. Rahimi S, Vadkertiova M, Joyce L, et al. Characterizing transition state in mouse
   vigilance with electroencephalogram-electromyogram hypnodensity. *Sleep Adv*. 2026.
   doi:10.1093/sleepadvances/zpag068
   (전문가 8명, 경계가 완만한 전이일수록 일치도 저하)
6. Bonett DG. Sample size requirements for estimating intraclass correlations with
   desired precision. *Stat Med*. 2002. (ICC 표본 수 산정)

---

## 변경 이력

| 날짜 | 내용 |
|---|---|
| 2026-08-18 | 최초 작성. `REVIEW_WORKFLOW.md`의 프로토콜·판정 규칙을 흡수하고, 조작적 정의·tolerance·rater 자격·신뢰도 설계를 신설 |
| 2026-08-18 | 구현 반영: `store_io.py`(EAG_RESULT_DIR 격리 + atomic write), 두 앱의 `--blank`, `reliability_pilot.py`(sessions/baseline/report). §5.3 절차와 §8.3을 실행 가능한 명령으로 교체 |
| 2026-08-19 | rater당 앱 2개(포트 4개)로 정정. 이전 예시는 rater마다 다른 작업을 시켜 리포트가 비었다. §8.2에 표본 제외 기준(`audit_ok` · `exclusions.json`)과 동결 상태 추가. `EAG_RESULT_DIR` 절대경로 명시 |
| 2026-08-19 | 채널 분모를 PASS 규칙으로 사전 동결(`channels` 서브커맨드, §8.2). 리포트에 커버리지 이행 점검과 제외 판단 kappa 추가(§8.4b) — 이전에는 한쪽만 작업한 채널이 조용히 빠져 LoA가 좁게 나왔다(시뮬레이션에서 반폭 0.060 vs 실제 0.174) |
