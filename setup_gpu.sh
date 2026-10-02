#!/bin/bash
# 3090×2 우분투 박스 1회 세팅. 멱등: 다시 실행해도 안전.
#   (A) 컨테이너 안 (실제 세팅, ~/projects/gpu-env PyTorch 이미지):  ./setup_gpu.sh --container
#       torch·CUDA는 이미지에 있으므로 나머지 패키지만 pip 설치하고 GPU를 검증한다. sudo 불필요.
#   (B) bare metal (예비):  ./setup_gpu.sh   → apt + uv venv(3.12) + torch cu128
#   끝나면 result/ 파생물을 NAS에서 rsync 한 뒤  ./run_crutch_gpu.sh
set -e
cd "$(dirname "$0")"

if [ "$1" = "--container" ] || [ -f /.dockerenv ]; then
  echo "== 컨테이너 모드: 이미지의 torch 사용"
  python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA 불가: compose의 gpu 할당 확인'"
  # torch 줄만 빼고 설치 (이미지의 CUDA torch를 pip이 CPU 빌드로 덮어쓰지 않게)
  grep -hvE '^\s*(#|$)|^torch' requirements.txt requirements-ml.txt > /tmp/req-nontorch.txt
  pip install --user -q -r /tmp/req-nontorch.txt
  python3 - <<'EOF'
import torch, sklearn, lightgbm, scipy, pandas, numpy
n = torch.cuda.device_count()
print(f"torch {torch.__version__}  cuda {torch.version.cuda}  GPU {n}개:",
      [torch.cuda.get_device_name(i) for i in range(n)])
assert n >= 2, "GPU 2장이 안 보임: compose deploy.resources 또는 --gpus all 확인"
for i in range(n):
    x = torch.randn(64, 1, 8, 125, device=f'cuda:{i}')
    torch.nn.Conv2d(1, 8, (1, 13), padding=(0, 6)).to(f'cuda:{i}')(x)
print("GPU 연산 OK (2장)  sklearn", sklearn.__version__, "lightgbm", lightgbm.__version__)
EOF
  python3 -c "import matplotlib.font_manager as fm; fm._load_fontmanager(try_read_cache=False)" 2>/dev/null || true
  echo "== 다음: result/ 파생물 rsync(호스트에서) → smoke → nohup ./run_crutch_gpu.sh"
  exit 0
fi

echo "== 1. GPU 드라이버"
if ! command -v nvidia-smi >/dev/null; then
  echo "   nvidia-smi 없음 → sudo ubuntu-drivers install 후 재부팅하고 다시 실행"; exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

echo "== 2. 시스템 패키지"
sudo apt-get update -qq
sudo apt-get install -y -qq git rsync tmux python3-venv python3-dev build-essential fonts-nanum >/dev/null

echo "== 3. uv + 가상환경 (.venv, Python 3.12)"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d .venv ] || uv venv --python 3.12 .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "== 4. 패키지 (torch는 CUDA 12.8 빌드)"
uv pip install --upgrade pip >/dev/null
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt -r requirements-ml.txt

echo "== 5. 검증"
python - <<'EOF'
import torch, sklearn, lightgbm, scipy, pandas, numpy
n = torch.cuda.device_count()
print(f"torch {torch.__version__}  cuda {torch.version.cuda}  GPU {n}개:",
      [torch.cuda.get_device_name(i) for i in range(n)])
assert n >= 1, "CUDA GPU 미인식"
x = torch.randn(64, 1, 8, 125, device='cuda'); y = torch.nn.Conv2d(1, 8, (1, 13), padding=(0, 6)).cuda()(x)
print("GPU 연산 OK", tuple(y.shape))
print("sklearn", sklearn.__version__, "lightgbm", lightgbm.__version__, "pandas", pandas.__version__, "numpy", numpy.__version__)
EOF
# 한글 폰트(matplotlib 캐시 갱신)
python -c "import matplotlib.font_manager as fm; fm._load_fontmanager(try_read_cache=False)" 2>/dev/null || true

echo "== 6. 다음 단계"
cat <<'EOF'
   (a) NAS에서 파생물 가져오기 (85 MB, 원본 data/ 29 GB는 불필요):
       rsync -avz --relative nas:/volume1/docker/claude-system/workspace/research/EAGstatic/./result/{ml/epochs.npz,ml/cycles.npz,ml/events.csv,ml/loso_predictions.csv,stats/grf_eag_pooled.csv,stats/cov_per_subject.csv} .
       (1차 결과 비교용, 선택)  rsync -avz nas:.../result/ml/crutch_* result/ml/
   (b) smoke:  source .venv/bin/activate && python3 ml_crutch.py --window event --model hybrid --subjects 8 --device cuda
   (c) 본 실행: nohup ./run_crutch_gpu.sh > /dev/null 2>&1 &   (GPU0 event / GPU1 cycle, 2~4 h)
   (d) 결과 되돌리기: rsync -avz result/ml/crutch_* nas:/volume1/docker/claude-system/workspace/research/EAGstatic/result/ml/
EOF
