"""주석 저장소 3종(offset · edge · exclusion)의 공통 입출력.

두 가지를 제공한다.

1) **저장 위치 분리** — 환경변수 `EAG_RESULT_DIR`로 저장 루트를 바꾼다.
   신뢰도 하위연구에서 rater별·라운드별 결과를 본 분석과 격리하기 위한 것이다
   (`ANNOTATION_PROTOCOL.md` §8.3). 앱 코드를 고치지 않고 실행 시점에만 바꾼다:

       EAG_RESULT_DIR=result/reliability/rater_A python3 offset_app.py --port 8768 --blank

   `sync_analyzer`도 같은 모듈을 통해 offset을 조회하므로 자동으로 따라온다.
   지정하지 않으면 기존과 동일하게 `result/`를 쓴다.

2) **원자적 쓰기** — 임시 파일에 쓰고 `os.replace()`로 교체한다.
   기존 방식(`open(path,'w')` 직후 `json.dump`)은 쓰는 도중 중단되거나 두 사람이
   동시에 저장하면 파일이 통째로 유실된다. 확정값 수백 건이 한 번에 날아갈 수 있어
   신뢰도 연구와 무관하게 필요하다. 교체 직전 `.bak` 1세대를 남긴다.
"""

import contextlib
import json
import os
import tempfile

ENV_RESULT_DIR = 'EAG_RESULT_DIR'
_REPO_DIR = os.path.dirname(os.path.abspath(__file__))


def result_dir() -> str:
    """주석 저장소가 놓이는 디렉터리. 호출 시점에 환경변수를 읽는다."""
    return os.environ.get(ENV_RESULT_DIR) or os.path.join(_REPO_DIR, 'result')


def store_path(filename: str) -> str:
    return os.path.join(result_dir(), filename)


def load_json(path: str, default=None):
    if not os.path.exists(path):
        return {} if default is None else default
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json_atomic(path: str, data, backup: bool = True):
    """같은 디렉터리의 임시 파일에 쓰고 원자적으로 교체한다.

    같은 파일시스템이어야 `os.replace`가 원자적이므로 임시 파일을 대상 디렉터리에
    만든다. 교체 전에 기존 파일을 `.bak`으로 복사해 1세대를 남긴다.
    """
    d = os.path.dirname(path) or '.'
    os.makedirs(d, exist_ok=True)

    if backup and os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as src:
                prev = src.read()
            with open(path + '.bak', 'w', encoding='utf-8') as dst:
                dst.write(prev)
        except OSError:
            pass  # 백업 실패가 본 저장을 막지는 않는다

    # mkstemp는 0600으로 만든다. 공유 작업 환경이므로 기존 파일의 권한을 잇고,
    # 새 파일이면 0644로 맞춘다.
    try:
        mode = os.stat(path).st_mode & 0o777
    except OSError:
        mode = 0o644

    fd, tmp = tempfile.mkstemp(dir=d, prefix='.tmp_', suffix='.json')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
