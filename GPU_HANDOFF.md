# GPU 서버 인계 프롬프트: EAG 목발분류 2차 실행

> 이 문서는 3090×2 우분투 서버의 Claude Code에게 주는 작업 지시서다. 처음부터 끝까지 읽고 순서대로 수행하라. 각 단계의 **확인 기준**을 통과하지 못하면 다음 단계로 가지 말고 사용자에게 보고하라. 질문은 모아서 한 번에 하고, 되돌릴 수 있는 일은 묻지 말고 진행하라.

## 0. 상황

- 이 서버: Ubuntu 26.04.1, RTX 3090 × 2. 호스트에 Docker + nvidia-container-toolkit. 연구는 **컨테이너 `~/projects/gpu-env`** 안에서 한다(공식 PyTorch 이미지, 호스트와 같은 UID/GID, Claude Code 포함, 자동 업데이트 꺼짐).
- 마운트: `~/projects` → 작업 폴더(쓰기 가능), `~/data` → **읽기 전용**, `~/.claude*` → 로그인 유지.
- 지금 네가 어디서 실행 중인지 먼저 확인하라: `[ -f /.dockerenv ] && echo container || echo host`. 이 문서의 실행 단계는 **컨테이너 안**에서 한다. 호스트에서 할 일은 rsync뿐이며, 그것은 사용자가 한다.
- 이 저장소(EAGstatic)는 NAS(Synology, CPU만)에서 개발·1차 실행된 코드다. 1차 결과는 `result/ml/crutch_*`에 있다. 이 서버의 역할은 **CPU로 비현실적이었던 나머지 계산**(seed 반복, 순열 100회)을 GPU로 끝내는 것이다. 코드 로직은 바꾸지 않는다.
- 연구 내용: 무릎 EAG(electroarthrography, 8채널 25 Hz) 신호로 체중부하 시 "목발(c) vs 반대다리(s, f)"를 분류하는 LOSO 실험. 1차 결론은 "모든 모델이 balanced accuracy 0.53~0.55, chance 위이지만 작음, 작은 CNN이 최고". 자세한 것은 `ml_crutch.py` docstring.

## 1. 하지 말 것

- `ml_crutch.py`, `epoch_extractor.py` 등 분석 로직 수정 금지. 버그를 발견하면 고치지 말고 보고.
- `result/` 아래 파일을 git에 추가하지 말 것(.gitignore 대상). 커밋·push 자체를 하지 말 것.
- `~/data`에 쓰지 말 것. 이 작업은 `~/data`를 쓰지 않는다.
- 데이터 파일의 `subject`, `subject_id` 열 값을 **화면에 출력하거나 보고서에 적지 말 것**. 사람 이름이 들어 있을 수 있다. 피험자 수(41)만 보고하라.
- 긴 작업이 도는 동안 컨테이너를 `stop`/`down` 하지 말 것. 프로세스가 죽는다.
- GPU 외 다른 방법(CPU fallback)으로 본 실행을 하지 말 것. CUDA가 안 되면 멈추고 보고.

## 2. 전제 확인

```bash
cd ~/projects/EAGstatic && git branch --show-current     # feature/aligner 여야 함
nvidia-smi --query-gpu=index,name,memory.total --format=csv
python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())"
ls ml_crutch.py ml_direction.py epoch_extractor.py run_crutch_gpu.sh setup_gpu.sh requirements-ml.txt
```

**확인 기준**: 브랜치 feature/aligner, GPU 2장, `device_count() == 2`, 파일 6개 모두 존재.
파일이 없으면 NAS 쪽 커밋·push가 안 된 것이다. "NAS에서 커밋·push가 필요합니다"라고 보고하고 멈춰라.

## 3. 데이터 확인

이 작업에 필요한 입력은 아래 6개(총 85 MB)뿐이다. 원본 29 GB `data/`는 필요 없고 여기 없는 것이 정상이다.

```bash
ls -la result/ml/epochs.npz result/ml/cycles.npz result/ml/events.csv result/ml/loso_predictions.csv \
       result/stats/grf_eag_pooled.csv result/stats/cov_per_subject.csv
ls result/ml/crutch_*_summary.json | wc -l      # NAS 1차 결과(선택). 있으면 재현 대조에 쓴다
```

**확인 기준**: 6개 모두 존재. 하나라도 없으면 사용자에게 아래 명령을 **호스트에서** 실행해 달라고 하고 멈춰라(컨테이너에는 NAS ssh 키가 없다).

```bash
rsync -avz --relative nas:/volume1/docker/claude-system/workspace/research/EAGstatic/./result/{ml/epochs.npz,ml/cycles.npz,ml/events.csv,ml/loso_predictions.csv,stats/grf_eag_pooled.csv,stats/cov_per_subject.csv} ~/projects/EAGstatic/
rsync -avz nas:/volume1/docker/claude-system/workspace/research/EAGstatic/result/ml/crutch_* ~/projects/EAGstatic/result/ml/
```

## 4. 환경 설치

```bash
./setup_gpu.sh --container
```

이 스크립트는 이미지의 torch를 그대로 두고(torch 줄 제외) 나머지 패키지를 `pip --user`로 설치한 뒤 GPU 2장에서 합성곱 연산을 검증한다. sudo 없이 돈다.

**확인 기준**: 마지막에 `GPU 연산 OK (2장)`가 출력. `AssertionError`가 나면 compose의 GPU 할당 문제이므로 보고.

## 5. smoke test (1~2분)

```bash
python3 ml_crutch.py --window event --model hybrid --subjects 8 --device cuda --tag smoke
rm -f result/ml/crutch_event_hybrid_smoke_*
```

**확인 기준**: `device=cuda (NVIDIA GeForce RTX 3090)` 줄과 `bacc_subject_mean:` 줄이 출력되고 Traceback이 없음. 8명 결과값 자체는 의미 없다.

## 6. 재현 확인 (~5분)

NAS 1차와 같은 seed로 event 창 CNN LOSO를 한 번 돌려 환경이 동일한지 본다.

```bash
python3 ml_crutch.py --window event --model cnn --seed 20260919 --device cuda --tag repro
python3 -c "import json; print(json.load(open('result/ml/crutch_event_cnn_repro_summary.json'))['bacc_subject_mean'])"
```

**확인 기준**: NAS 값 **0.547 ± 0.062**(subject mean ± sd)에 대해 bacc가 0.537~0.557 안. GPU 비결정성 때문에 셋째 자리는 다를 수 있다. 0.52 아래거나 0.58 위면 데이터나 코호트가 다른 것이다(피험자 수가 41이 아니면 `grf_eag_pooled.csv` 누락). 보고하고 멈춰라.

## 7. 본 실행 (2~4시간, 무인)

```bash
nohup ./run_crutch_gpu.sh > /dev/null 2>&1 &
```

GPU 0에서 event 창, GPU 1에서 cycle 창이 병렬로 돈다. 각각 seed 3개(20260919, 1, 2) × 모델 4종(cnn, cnn_lstm, cnn_tf, hybrid) → cnn·hybrid 순열 100회.

진행 확인:
```bash
tail -3 result/ml/crutch_gpu_event.log result/ml/crutch_gpu_cycle.log
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
```

두 로그에 각각 `EVENT DONE`, `CYCLE DONE`이 찍히면 끝. 시작 직후 사용자에게 "시작했고 N시간 뒤 확인하겠다"고 보고한 뒤, 끝날 때까지 기다리지 말고 종료하라. 사용자가 다시 부르면 §8로.

실패 징후: 로그에 `Traceback`, `CUDA out of memory`, 두 로그 중 하나만 진행. OOM이면 `run_crutch_gpu.sh`를 수정하지 말고 보고(배치 128·2만 파라미터 모델이라 OOM은 비정상이다).

## 8. 완료 후 정리

`result/ml/`에 생기는 파일(창 × 모델별):
- `crutch_{window}_{model}_s{seed}_summary.json / _per_subject.csv / _preds.npy` (seed 3개)
- `crutch_{window}_{cnn,hybrid}_perm100_summary.json / _null.txt / _null_partial.txt`

다음을 계산해 `result/ml/GPU_RUN_SUMMARY.md`로 저장하라(피험자 이름 금지, 숫자만):

1. **seed 평균표**: 창 × 모델별 bacc_subject_mean의 seed 3개 평균·sd·범위, auc 평균. (NAS 1차 `crutch_seed_summary.csv`와 같은 형식)
2. **순열표**: 창 × {cnn, hybrid}의 관측 bacc, null 평균·sd·최댓값, p_perm, z = (관측 − null평균)/null sd.
3. **NAS 대조**: `crutch_event_cnn_summary.json`(NAS, seed 20260919)과 `crutch_event_cnn_s20260919_summary.json`(여기)의 bacc 차이. 0.01 이내면 "재현됨".
4. 실행 시간(로그의 `###` 타임스탬프로), GPU별 최대 메모리.

판독 기준(1차 결론과의 일치 여부만 적어라):
- 모든 DL 모델 bacc 0.52~0.57 범위인가.
- cnn ≥ hybrid 인가(1차: 0.548 vs 0.543).
- 순열 p: 관측이 null 최댓값보다 크면 p = 1/101 = 0.0099.

마지막으로 사용자에게 보고할 것: 요약 표 2개, 재현 여부 한 줄, 그리고 **호스트에서** 실행할 되돌리기 명령:
```bash
rsync -avz ~/projects/EAGstatic/result/ml/crutch_* ~/projects/EAGstatic/result/ml/GPU_RUN_SUMMARY.md nas:/volume1/docker/claude-system/workspace/research/EAGstatic/result/ml/
```
NAS 쪽 문서(계획서 §8, 보고서) 갱신은 NAS의 Claude가 한다. 여기서는 하지 않는다.

## 9. 참고: 파일 역할

| 파일 | 역할 |
|---|---|
| `ml_crutch.py` | 본체. `--window event|cycle`, `--model cnn|cnn_lstm|cnn_tf|hybrid|csp|lgb|dur`, `--seed`, `--n-perm`, `--perm-offset`, `--device`, `--tag` |
| `ml_direction.py` | `ml_crutch`가 CSP 함수를 import. 직접 실행 안 함 |
| `run_crutch_gpu.sh` | 2-GPU 러너(이 문서 §7) |
| `setup_gpu.sh --container` | 환경 설치·검증(§4) |
| `ml_crutch_hetero.py`, `ml_crutch_session.py`, `ml_adherence_sweep.py`, `ml_crutch_report.py` | NAS에서 이미 실행한 후속 분석·그림. 여기서 돌릴 필요 없음 |
| `epoch_extractor.py` | 원본 29 GB에서 npz를 만드는 단계. 원본이 없으므로 여기서 실행 불가·불필요 |
| `REANALYSIS.md` | 전체 재분석 순서. 참고용 |

## 10. 보고 형식

각 단계가 끝날 때 한 줄씩:
```
[2/10 전제] OK: feature/aligner, GPU 2 (3090×2), torch 2.x+cu128
[3/10 데이터] OK: 6/6, NAS 1차 결과 17개
[4/10 환경] OK
[5/10 smoke] OK
[6/10 재현] OK: 0.549 (NAS 0.547)
[7/10 본실행] 시작 14:02, 예상 종료 17:00
```
막히면 그 단계 번호와 함께 오류 원문 5줄 이내, 그리고 네가 추정하는 원인 한 줄.
