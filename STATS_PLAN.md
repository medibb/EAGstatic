# EAG-GRF 통계분석 계획 (단계별 체중부하 dose-response)

## 1. 실험 설계 (데이터에서 확인된 프로토콜)

체중이동은 **graded staircase**로 진행됨. signed imbalance `(L-R)/(L+R)`가 세션마다 일관되게:

```
0 → +1.0 → +0.65 → +1.0 → 0 → +1.0 → -0.65 → +1.0 → -1.0 → +1.0 → 0
```

full-left(+1.0)를 기준으로 우측으로 **점점 큰 폭**으로 이동. 즉 각 GRF 전이의 **step 크기 `grf_step = to_level - from_level`가 부하 단계(dose)**이고, |grf_step|이 ~0.35 → 1.0 → 1.65 → 2.0으로 증가.

한 번의 전이(rise=부하, fall=이탈)마다 EAG는 knee-pair(onset+offset)로 반응 크기를 기록.
분석 단위: **transition 1개 x channel 1개 = 1 관측** (`grf_triggered_params_*.csv`의 matched 행).

## 2. 변수

| 역할 | 변수 | 설명 |
|------|------|------|
| 반응(Y) | `amplitude` | EAG 변화 크기 µV (offset_amp - onset_amp, 부호 포함). 주 분석은 `abs(amplitude)` |
| | `slope`, `transition_time` | 반응 속도/지속 |
| | `latency` | onset - GRF전이. 동시성(≈0) 검정용 |
| 용량(X) | `grf_step` | GRF 부하 단계 크기(dose). 주 분석은 `abs(grf_step)` |
| 요인 | `eag_direction`(rise/fall) = 부하 vs 이탈 | 방향별 반응 비대칭 검정 |
| | `channel`(1-8) | 전극 위치 |
| | `grf_from_level`,`grf_to_level` | 시작/도달 부하 수준 |
| 군집 | `subject`, `session` | 반복측정 random effect |
| 품질 | `channel_quality`(PASS/…), `channel_snr_db` | cross_params에서 병합, PASS만 1차 분석 |

## 3. 가설

- **H1 (dose-response)**: |EAG amplitude|는 |grf_step|에 비례 증가 (양의 기울기).
- **H2 (방향 비대칭)**: 부하(rise)와 이탈(fall)의 반응 크기/기울기가 다름.
- **H3 (동시성)**: latency 중앙값 ≈ 0 (EAG와 GRF 전이가 동시). 방향/부하와 무관.
- **H4 (채널 특이성)**: 특정 채널(전극 위치)이 더 큰 반응/기울기.

## 4. 분석 절차

1. **풀링·정제**: 모든 `grf_triggered_params_*.csv` 병합 → `matched==True` 유지 → `channel_quality==PASS` 필터(1차) → `abs_amp`,`abs_step` 파생. offset이 `needs_review`(large-offset 보류 등)인 세션은 플래그 컬럼으로 표시하고 민감도 분석에서 제외/포함 비교.
2. **기술통계**: |grf_step| 구간(bin: 0-0.5, 0.5-1.0, 1.0-1.5, 1.5-2.0) x eag_direction x channel 별 mean±sd, n.
3. **Dose-response**:
   - 1차: `abs_amp ~ abs_step` OLS(전체) + 방향별 층화. 기울기·R².
   - 혼합모형: `abs_amp ~ abs_step * eag_direction + C(channel) + (1|subject) + (1|session)` (statsmodels MixedLM; 없으면 subject별 OLS 기울기 → 1-sample t-test로 집단 추론).
4. **동시성 검정(H3)**: `latency` 1-sample Wilcoxon vs 0, 방향별. |grf_step|과 latency 상관.
5. **방향 비대칭(H2)**: rise vs fall |amp| Mann-Whitney; 상호작용항 유의성.
6. **채널(H4)**: channel별 |amp| Kruskal-Wallis + 사후. 채널별 dose 기울기 비교.
7. **민감도**: (a) PASS 채널만 vs 전체, (b) needs_review offset 세션 제외 vs 포함, (c) manual-edit 세션 반영 확인.

## 5. 산출물 (`result/stats/`)

- `grf_eag_pooled.csv` : 정제·병합된 관측 테이블 (분석 입력)
- `dose_response_by_direction.csv` : bin별 요약 + 회귀계수
- `latency_summary.csv`, `channel_summary.csv`
- `mixedmodel_summary.txt` : 혼합모형 결과 (가능 시)
- plots: `dose_response.png`(산점+적합, 방향별), `latency_hist.png`, `channel_box.png`

## 6. 실행 (manual review 이후)

```bash
# 0) offset review 확정 + edge 수동수정 (edge_app / edge_editor / offset_review)
# 1) 파라미터 재추출 (grf_step 컬럼 포함 최신 스키마)
python3 parameter_extractor.py --batch
# 2) 통계
python3 stats_grf_eag.py            # result/stats/ 생성
python3 stats_grf_eag.py --exclude-review   # needs_review offset 세션 제외 민감도
```

## 7. 주의

- `parameter_extractor.py --batch`를 review 확정 후 **다시 돌려야** manual offset/edge와 `grf_step` 신규 컬럼이 최종 반영됨 (현재 배치는 예비 실행).
- amplitude 부호: rise(부하)=+, fall(이탈)=-. 방향 통합 시 `abs_amp` 사용, 방향 효과는 요인으로.
- dose 축은 `abs_step`. from/to level로 "어느 부하 수준에서의 전이인가"(예: 0→1 vs 1→-1)도 2차 요인화 가능.
