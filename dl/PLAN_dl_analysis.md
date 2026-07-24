# EAG-GRF 딥러닝 분석 계획

## 개요

GRF 이벤트 기반으로 EAG 신호를 추출하여 딥러닝 분석 수행.
변곡점/파라미터 추출 없이 raw filtered EAG를 직접 모델에 입력.

3단계 순차 탐색:
1. **Task A**: 동작 분류 (s/f/c) — EAG가 동작별 연골 부하 패턴을 반영하는가?
2. **Task B**: 체중부하 수준 예측 (crutch 세션, 20/50/80/100%) — EAG의 부하 민감도 정량화
3. **Task C**: EAG→GRF 재구성 — 두 신호 간 coupling 학습

환경: 여기서 프로토타입 (CPU, 경량 1D-CNN) → 본격 실험은 GPU

## 데이터 파이프라인 (공통)

### Step 1. 이벤트 기반 EAG 윈도우 추출

```
SyncAnalyzer (offset 정렬 완료)
  ↓
GRF 이벤트 감지 (onset_time, offset_time)
  ↓
각 이벤트:
  EAG window = [onset - 2초, onset + 13초] = 15초 × 250Hz = 3,750 samples
  GRF window = 같은 구간의 left_grf, right_grf
  ↓
  zero-padding으로 고정 길이
  ↓
출력: (n_events, 3750, 8) EAG + (n_events, 3750, 2) GRF
```

### Step 2. 라벨링

| Task | 라벨 소스 | 출력 |
|------|----------|------|
| A | session_name → s/f/c | 3 classes |
| B | c세션, 이벤트 순서 → 20/50/80/100% | 4 classes 또는 regression |
| C | GRF 시계열 자체 | regression (연속값) |

### Step 3. 데이터 분할

- Subject-level split (data leakage 방지)
- Train 12명 / Val 3명 / Test 3명
- GPU 실험 시: LOOCV (18-fold)

## 신규 파일

```
EAGstatic/
├── dl_dataset.py    # 이벤트 윈도우 추출 + PyTorch Dataset
├── dl_models.py     # 모델 정의 (1D-CNN, CNN-LSTM)
├── dl_train.py      # 학습 루프 + 평가
└── result/dl/       # 학습 결과
```

### dl_dataset.py

- `extract_event_windows()` → 전체 이벤트 윈도우 일괄 추출
- `EAGEventDataset` (PyTorch Dataset) — EAG window + label
- `create_subject_splits()` — subject-level 분할
- quality_flag 메타데이터 포함

### dl_models.py

**Model 1: EAG1DCNN** (프로토타입, CPU)
```
Conv1D(8→32, k=16, s=2) → BN → ReLU → MaxPool
Conv1D(32→64, k=8, s=2) → BN → ReLU → MaxPool
Conv1D(64→64, k=4) → BN → ReLU → AdaptiveAvgPool
FC(64 → n_classes)
파라미터: ~50K
```

**Model 2: EAGLSTM** (GPU)
```
Conv1D(8→32) → LSTM(64, bidirectional) → FC
```

**Model 3: EAGRegressor** (Task C, GRF 재구성)
```
encoder + FC → (batch, T, 2) 예측
Loss: MSE
```

### dl_train.py

```bash
# Task A: 동작 분류
python3 dl_train.py --task classify_action --model cnn --epochs 50

# Task B: 체중부하 예측
python3 dl_train.py --task classify_load --model cnn --epochs 50

# Task C: GRF 재구성
python3 dl_train.py --task regress_grf --model cnn --epochs 100
```

결과: `result/dl/task_{name}/metrics.csv`, `confusion_matrix.png`, `model_best.pt`

## 데이터 규모

| Task | 샘플 수 | 클래스 | 균형 |
|------|---------|--------|------|
| A | ~1,700 이벤트 | 3 (s/f/c) | 37/34/29% |
| B | ~480 이벤트 (c세션만) | 4 (20/50/80/100%) | ~25% 균등 |
| C | ~1,700 이벤트 | regression | N/A |

## 구현 배치

### 배치 1 (이번 세션)
1. dl_dataset.py — 데이터 추출 + Dataset
2. dl_models.py — 1D-CNN
3. dl_train.py — Task A 프로토타입

### 배치 2 (다음 세션)
4. Task B (체중부하 분류)
5. Task C (GRF 재구성)

### 배치 3 (GPU)
6. LSTM + LOOCV + 하이퍼파라미터 탐색

## 검증

- Task A/B: Accuracy, F1 (macro), confusion matrix
- Task C: R², MSE, 예측 vs 실측 overlay plot
- Baseline: 랜덤 분류기, 주파수 feature + SVM
