"""주석 작업 범위(scope) 파일 로더 — 동결된 세션·채널 목록을 읽는다.

신뢰도 파일럿은 사전에 동결한 세션·채널만 작업해야 한다(ANNOTATION_PROTOCOL.md §8.2).
GUI가 전체 1200여 세션을 그대로 보여주면 rater가 대상을 찾기 어렵고, 범위 밖을 작업해도
막히지 않는다. 리포트가 사후에 경고하긴 하지만, 예방이 탐지보다 낫다.

`reliability_pilot.py`에 두지 않는 이유는 순환 import다. 그쪽은 세션 스캔을 위해
`offset_app`을 import하므로, 앱이 거꾸로 그것을 import할 수 없다. 이 모듈은 아무것도
import하지 않으므로 양쪽이 같은 정의를 공유할 수 있다.
"""

import csv
import os


def _truthy(v) -> bool:
    return str(v).strip().lower() in ('true', '1', 'yes')


def load_scope_sessions(path: str) -> set:
    """세션 범위 {(subject, session)}. `pilot_sessions.csv` 형식.

    파일이 없으면 **예외를 던진다.** 조용히 전체를 열어주면 rater가 범위를 벗어난 줄
    모르고 작업하게 되는데, 그게 애초에 막으려던 상황이다.
    """
    if not os.path.exists(path):
        raise SystemExit(f"[scope] 목록 파일 없음: {path}\n"
                         f"        먼저 `python3 reliability_pilot.py sessions` 를 실행하거나, "
                         f"전체 세션을 열려면 --only 없이 기동하세요.")
    with open(path, encoding='utf-8-sig') as f:
        return {(r['subject'], r['session']) for r in csv.DictReader(f)}


def load_scope_channels(path: str) -> dict:
    """채널 범위 {(subject, session): [1-based 채널 …]}. `pilot_channels.csv` 형식.

    `include` 가 참인 행만 담는다. 신호 품질(PASS)로 걸러진 채널은 본 분석에도
    들어가지 않으므로 rater에게 보여줄 이유가 없다.
    """
    if not os.path.exists(path):
        raise SystemExit(f"[scope] 채널 목록 파일 없음: {path}\n"
                         f"        먼저 `python3 reliability_pilot.py channels` 를 실행하세요.")
    out = {}
    with open(path, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if _truthy(r['include']):
                out.setdefault((r['subject'], r['session']), []).append(int(r['channel']))
    for k in out:
        out[k].sort()
    return out
