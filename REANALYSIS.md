# 재분석 파이프라인 (annotation 확정 후)

**언제 쓰는가.** offset·edge 수동 확정이 끝나면 **현재 발표된 모든 수치가 바뀐다.**
Ch.4 4.3의 값은 전부 interim(자동 검출 기준)이고, 상위 이론 문서와 OBJ4에 인용된
실측 수치도 같은 파이프라인에서 나왔다. 이 문서는 **무엇을 어떤 순서로 다시 돌리고,
그 결과로 어느 문서의 어느 숫자를 고쳐야 하는지**를 한 곳에 적어 둔 것이다.

> 판정 기준은 `ANNOTATION_PROTOCOL.md`, 작업 절차는 `ANNOTATION_GUIDE.md`.
> 이 문서는 **확정 이후**만 다룬다.

---

## 0. 착수 조건

아래가 모두 참이어야 재분석을 시작한다. 하나라도 아니면 재분석분이 또 무효가 된다.

- [ ] **offset worklist 소진** — `python3 offset_review.py --list` 로 확정 수 확인
- [ ] **edge 확정 완료** — `python3 edge_editor.py list`
- [ ] **제외 라벨 정리** — `python3 exclusion_store.py --list`.
      폴더명 QC 접미사(`(분석안됨)` 등)가 `exclusions.json`으로 이관되었는지 확인
- [ ] **전처리 상수 불변** — `eag_analyzer.py` 의 `lowpass_cutoff=5.0`, `drift_method="detrend"`.
      바뀌었으면 **manual edge의 진폭이 무효**다 (`grf_triggered_annotator.py:1014`)
- [ ] **신뢰도 파일럿 tolerance 확정** — `ANNOTATION_PROTOCOL.md` §5.3 표가 채워졌는지

```bash
python3 run_pipeline.py status      # 백로그 한눈에
```

---

## 1. 실행 순서

**순서를 지킬 것.** 앞 단계 산출물을 뒤 단계가 읽는다.

```bash
cd /workspace/research/EAGstatic

# 1) 코호트 재구성 — 제외·재측정이 반영된 평면 미러와 manifest
python3 build_flat_view.py
#    확인: 방문/세션 수, axis_a·axis_b 적격 방문 수

# 2) 파라미터 재추출 (가장 오래 걸린다)
python3 parameter_extractor.py --batch
#    manual offset/edge/exclusion이 전부 반영된다
#    산출: result/phase1_params/grf_triggered_params.csv

# 3) 주 통계
python3 stats_grf_eag.py
#    ⚠️ stats_grf_eag.py:244 의 구 dose축(abs_step) LMM은 폐기 대상.
#       load_pct 모형만 남기고 abs_step 경로는 부록으로 내리거나 삭제할 것

# 4) 공변량 (성별·체성분)
python3 stats_covariates.py

# 5) ML — 부하 디코딩·방향·주파수대
python3 ml_decoder.py
python3 ml_direction.py
python3 ml_fbcsp.py
python3 ml_eegnet.py          # 있으면

# 6) 신뢰도 (파일럿 rater 작업이 끝난 뒤)
python3 reliability_pilot.py report --a rater_main --b rater_rel
python3 reliability_pilot.py report --a rater_main --b rater_ref

# 7) 기전 (재추출된 파라미터 기준으로 다시)
python3 plateau_decay.py hold --n 30
python3 plateau_decay.py grfrate --n 80
```

---

## 2. 재분석 후 반드시 다시 계산할 유도값

**아래는 파이프라인이 자동으로 내지 않는다.** 별도로 다시 뽑아 문서를 고쳐야 한다.

### 2.1 채널별 공간 구조 (OBJ4 대조에 쓰임)

```bash
python3 - <<'EOF'
import pandas as pd, numpy as np
d = pd.read_csv('result/phase1_params/grf_triggered_params.csv', low_memory=False)
d = d[(d.matched==True)&(d.accepted==True)&d.load_pct.notna()&d.amplitude.notna()]
d = d[~d.excluded.astype(str).str.lower().eq('true')]
ROW={1:'proximal',2:'proximal',5:'proximal',6:'proximal',
     3:'jointline',4:'jointline',7:'jointline',8:'jointline'}
COL={1:'medial',2:'medial',3:'medial',4:'medial',
     5:'lateral',6:'lateral',7:'lateral',8:'lateral'}
d['row']=d.channel.map(ROW); d['col']=d.channel.map(COL); d['absamp']=d.amplitude.abs()
sl = {ch: np.polyfit(g.load_pct,g.absamp,1)[0] for ch,g in d.groupby('channel')}
med=np.mean([sl[c] for c in (1,2,3,4)]); lat=np.mean([sl[c] for c in (5,6,7,8)])
prox=np.mean([sl[c] for c in (1,2,5,6)]); jl=np.mean([sl[c] for c in (3,4,7,8)])
print(f"이벤트 {len(d)}")
print(f"내측/외측 = {med/lat:.2f}   (내측 {med:.2f} / 외측 {lat:.2f})")
print(f"근위/관절선 = {prox/jl:.2f} (근위 {prox:.2f} / 관절선 {jl:.2f})")
EOF
```

→ 이 두 비가 **OBJ4 §3.4의 모델 대조 표**에 들어간다. 값이 바뀌면 그 절의 해석
(내측 우세의 소스 귀속, 소스 방향 혼합 판정)을 다시 확인해야 한다.

### 2.2 채널 SNR 분포 (OBJ4 역문제 상한에 쓰임)

```bash
python3 reliability_pilot.py channels --force   # 또는 전 세션 대상으로 확장
python3 -c "
import pandas as pd
d=pd.read_csv('result/reliability/pilot_channels.csv'); v=d.snr_db.dropna()
print(f'SNR 중앙 {v.median():.1f} · 사분위 {v.quantile(.25):.1f}~{v.quantile(.75):.1f} · 최대 {v.max():.1f}')"
```

→ **OBJ4 §3.5의 역문제 표는 측정 SNR 14 dB를 기준으로 계산됐다.**
SNR이 바뀌면 `volume_conductor.py localize --snr <새 값>` 을 다시 돌려 국소화 상한을
재산출해야 한다.

### 2.3 기전 판별 (크기 대 속도)

`plateau_decay.py grfrate` 재실행. 현재 값(크기 +40.5 µV, 속도 −5.2 µV, p=0.088)이
바뀌면 상위 문서 §1.2를 고친다.

---

## 3. 갱신 대상 문서 체크리스트

재분석 결과가 나오면 **아래 위치의 숫자를 전부 확인**한다. 놓치면 문서 간 불일치가 남는다.

### 3.1 Dissertation Ch.4 (`obsidian/claudeanswer/Lee_JH_PhD_Dissertation_EAG.md`)

- [ ] **4.3 전체** — 현재 interim 표시. 재산출 후 표시 제거
- [ ] 4.3.1 test-retest **ICC·MDC95** (현재 기울기 0.463 / 진폭 0.661 / MDC95 1.784)
- [ ] 4.3.2 **dose-response** 계수. 구 abs_step 모형(75.03) → load_pct 모형(1.591)으로
      이미 교체 권고됨. **분산성분 문장은 부등호가 뒤집히므로 반드시 재작성**
- [ ] 4.3.3~4.3.4 **ML 수치** (MAE 15.17 %BW, R² 0.504, 전이 15.16, AUC 0.942, CSP 0.547)
- [ ] 4.2.7 annotation reliability의 `[n]`·`[k]` 플레이스홀더 → 파일럿 결과
- [ ] 4.2.3 취득 문단의 **세션·채널 수** (구 "416 sessions, 3,328 channels")
- [ ] 4.2.6 / 4.3 서두의 **방문 수 불일치** (81 대 79)
- [ ] **참여자 수** — 제외 2명·복구 1명·재측정 6명 반영

### 3.2 OBJ2 노트 (`obsidian/📥️Inbox/} EAG OBJ2.md`)

- [ ] Stage 1.1 데이터 규모 (현재 "44명 / 95방문 / 1,215세션"은 재개 전 스냅샷)
- [ ] Stage 3.2 핵심 결과 포인트 10개 전부
- [ ] Gantt의 done 날짜

### 3.3 상위 이론 문서 (`obsidian/📥️Inbox/= 연골 전기생리와 EAG.md`)

**여기에 본 연구 실측이 여러 곳 인용돼 있다. 전부 재확인 대상이다.**

- [ ] §1.2 유지 구간 감쇠 (되돌아온 비율 중앙 −0.063, 11세션 424구간)
- [ ] §1.2 크기 대 속도 (크기 +40.5 / 속도 −5.2 µV, 3,496 이벤트)
- [ ] §1.3 체성분 상관 (체지방률 ρ=0.206 등, n=41)
- [ ] §4.2(c-2) montage 대조 (내측/외측 2.40, 근위/관절선 1.18, 70,566 이벤트)
- [ ] §4.2(f) 역문제 표의 기준 SNR (14 dB)
- [ ] §5 공간 구조 요약표

### 3.4 OBJ4 노트 (`obsidian/📥️Inbox/} EAG OBJ4.md`)

- [ ] §3.3 지방-진폭 실측 부호
- [ ] §3.4 실측 대조 표 (2.40 · 1.18)
- [ ] §3.5 역문제 표의 기준 SNR
- [ ] §4 초록의 해당 수치 전부

---

## 4. 재분석 후 검증 (숫자가 말이 되는지)

기계적으로 다시 돌리는 것만으로는 부족하다. 아래를 확인한다.

1. **이벤트 수가 늘었는가.** 수동 확정은 자동이 놓친 이벤트를 회수하므로 늘어야 정상.
   줄었다면 제외 라벨이 과하게 적용됐는지 본다
2. **offset 미확정 세션의 edge 통과율 격차가 사라졌는가.**
   확정 전 77.9% 대 88.1%였다. 확정 후에는 좁혀져야 한다
3. **dose-response 기울기의 방향과 규모가 유지되는가.** 크게 달라졌다면 수동 확정이
   체계적 편향을 넣었을 가능성을 의심한다 (특히 값 기준으로 검토 대상을 골랐다면)
4. **ICC가 올랐는가.** 오르는 것이 기대되나, **MDC95가 평균 기울기(1.45) 아래로
   내려가려면 ICC ≥ 0.645가 필요**하다. 산술적으로 확인할 것
5. **채널별 내측/외측 비가 2.4 근처를 유지하는가.** 크게 달라지면 OBJ4의 삼각검증
   논거가 흔들린다

---

## 5. 순서 요약

```
확정(offset → edge → 제외)
   ↓
build_flat_view → parameter_extractor --batch
   ↓
stats_grf_eag · stats_covariates · ML 4종
   ↓
유도값 재계산 (§2: 채널 비 · SNR · 기전)
   ↓
volume_conductor localize --snr <새 SNR>   ← OBJ4 상한 재산출
   ↓
문서 갱신 (§3 체크리스트 4곳)
   ↓
검증 (§4 다섯 항목)
```

---

## 변경 이력

| 날짜 | 내용 |
|---|---|
| 2026-08-19 | 최초 작성. 착수 조건, 실행 순서, 유도값 재계산, 문서 갱신 체크리스트, 사후 검증 |
