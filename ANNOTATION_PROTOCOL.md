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

**절차.**

1. 훈련을 마친 rater 2명이 **10세션**을 독립적으로 주석 (§8.3의 독립 관찰 조건 적용)
2. onset과 offset **각각** Bland-Altman LoA를 산출
3. LoA를 반올림한 값을 해당 이벤트 종류의 tolerance로 확정하고 이 문서에 기록
4. 확정된 tolerance는 이후 (a) rater 자격 판정(§6), (b) 신뢰도 매칭(§8),
   (c) auto vs manual 비교(§8)에 **동일하게** 쓴다

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

### 8.3 독립 관찰 확보

GUI는 자동 검출 결과를 초기값으로 그리고, 이미 확정된 세션은 `✅ [확정 ...s]`로 표시한다.
**두 rater가 같은 초기값에서 시작하면 둘 다 손을 안 댄 경우가 완전 일치로 잡혀,
재는 것이 사람의 일치도가 아니라 알고리즘의 결정론성이 된다.**

**코드 변경 없이 확보하는 방법** (둘 중 하나):

- (a) 신뢰도용 세션을 **미리 지정**해 본 작업에서 건너뛰게 하고, 마지막에 두 사람이
  각각 백지 상태에서 수행
- (b) `result/manual_offsets.json` · `manual_edges.json`을 다른 이름으로 옮겨둔 상태에서
  두 번째 rater가 수행한 뒤, 두 결과를 대조

어느 쪽이든 **본 작업 시작 전에 대상 세션 목록을 확정**해야 한다.

### 8.4 자동 수용 채널 층화

edge 작업의 상당 부분은 자동 결과를 그대로 받아들이는 것이다. 이를 섞어 계산하면
일치도가 부풀려진다. 따라서 **두 가지를 나눠 보고한다.**

- **전체(all-comers)**: 파이프라인 전체의 실제 신뢰도. 오차 예산(§8.5)에 쓴다
- **수정된 채널만**: 둘 중 한 명이라도 손댄 채널. 판단 작업 자체의 신뢰도

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

### 9.2 필수 개선 (작업 시작 전)

- **atomic write**: 현재 세 저장소 모두 `open(path,'w')` 직후 `json.dump`다. 쓰는 도중
  끊기거나 두 명이 동시에 저장하면 **파일이 통째로 유실된다.** 임시 파일에 쓰고
  `os.replace()`로 교체하고 백업을 회전한다. 신뢰도와 무관하게 데이터 보호를 위해 필요하다
- **작업 분할**: atomic write 적용 전까지는 세션을 나눠 맡아 동시 저장을 피한다

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
