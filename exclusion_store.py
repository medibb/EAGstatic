"""
Exclusion Store — 분석에서 제외할 데이터의 라벨 관리 (result/exclusions.json)

offset/edge를 검토하다 보면 "이건 분석에 못 쓴다"고 판단되는 것이 나온다
(EAG 신호가 통째로 깨짐, 동기화 불가, 프로토콜 미시행 등).
지우거나 숨기지 않고 **라벨만 붙여** 두고, 통계 단계에서 걸러낸다.
판단 근거가 남아야 나중에 재검토하거나 사유별로 집계할 수 있기 때문이다.

구조 (offset_manager/edge_store와 같은 형태, channel 0 = 세션 전체):
{
  "(02.02_10)주창민_1": {
    "s1": {
      "0": {"reason": "동기화불가", "note": "...", "updated_at": "..."},   # 세션 전체 제외
      "3": {"reason": "노이즈",     "note": "...", "updated_at": "..."}    # ch3만 제외
    }
  }
}

사용:
  set_exclusion(subj, sess, 0, '동기화불가')      # 세션 전체
  set_exclusion(subj, sess, 3, '노이즈')          # 채널만
  is_excluded(subj, sess, 3)  -> dict | None      # 세션 제외도 함께 확인
"""

import json
import os
from datetime import datetime
from typing import Optional

from store_io import store_path, load_json, save_json_atomic

STORE_FILENAME = 'exclusions.json'


def default_path() -> str:
    """저장 위치. `EAG_RESULT_DIR`로 바꿀 수 있다 (store_io 참조)."""
    return store_path(STORE_FILENAME)


DEFAULT_PATH = default_path()  # 하위호환 (import 시점 해석)

# 사유는 자유 입력이지만, 집계가 가능하도록 표준 목록을 권장한다
REASONS = [
    '노이즈',        # EAG 신호 품질 불량 (드리프트·잡음으로 knee 판독 불가)
    '동기화불가',    # offset을 맞출 수 없음
    '프로토콜이상',  # 부하 4회 미시행·중단 등
    '기록오류',      # 장비/파일 문제
    '기타',
]

SESSION_SCOPE = 0        # channel 0 = 세션 전체


def load_exclusions(path: str = None) -> dict:
    return load_json(path or default_path())


def save_exclusions(d: dict, path: str = None):
    save_json_atomic(path or default_path(), d)


def set_exclusion(subject: str, session: str, channel: int = SESSION_SCOPE,
                  reason: str = '기타', note: str = '', path: str = None):
    d = load_exclusions(path)
    d.setdefault(subject, {}).setdefault(session, {})[str(int(channel))] = {
        'reason': reason or '기타',
        'note': note,
        'updated_at': datetime.now().isoformat(timespec='seconds'),
    }
    save_exclusions(d, path)


def clear_exclusion(subject: str, session: str, channel: int = SESSION_SCOPE,
                    path: str = None) -> bool:
    d = load_exclusions(path)
    sess = d.get(subject, {}).get(session, {})
    if str(int(channel)) in sess:
        del sess[str(int(channel))]
        if not sess:
            d[subject].pop(session, None)
        if not d.get(subject):
            d.pop(subject, None)
        save_exclusions(d, path)
        return True
    return False


def is_excluded(subject: str, session: str, channel: Optional[int] = None,
                path: str = None) -> Optional[dict]:
    """제외 여부. 채널을 주면 세션 전체 제외도 함께 본다.

    Returns: {'scope': 'session'|'channel', 'reason', 'note', 'updated_at'} 또는 None
    """
    sess = load_exclusions(path).get(subject, {}).get(session, {})
    if not sess:
        return None
    s = sess.get(str(SESSION_SCOPE))
    if s is not None:
        return {**s, 'scope': 'session'}
    if channel is not None:
        c = sess.get(str(int(channel)))
        if c is not None:
            return {**c, 'scope': 'channel'}
    return None


def excluded_map(path: str = None) -> dict:
    """(subject, session) -> {'session': entry|None, 'channels': {ch: entry}} 조회용."""
    out = {}
    for subj, sessions in load_exclusions(path).items():
        for sess, chans in sessions.items():
            e = {'session': None, 'channels': {}}
            for ch, v in chans.items():
                if int(ch) == SESSION_SCOPE:
                    e['session'] = v
                else:
                    e['channels'][int(ch)] = v
            out[(subj, sess)] = e
    return out


def list_all(path: str = None) -> list:
    rows = []
    for subj, sessions in load_exclusions(path).items():
        for sess, chans in sessions.items():
            for ch, v in chans.items():
                rows.append({'subject': subj, 'session': sess, 'channel': int(ch),
                             'scope': 'session' if int(ch) == SESSION_SCOPE else 'channel',
                             **v})
    return rows


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='분석 제외 라벨 관리')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--set', action='store_true')
    ap.add_argument('--clear', action='store_true')
    ap.add_argument('--subject'); ap.add_argument('--session')
    ap.add_argument('--channel', type=int, default=SESSION_SCOPE,
                    help='0=세션 전체 (기본), 1~8=해당 채널')
    ap.add_argument('--reason', default='기타', choices=REASONS + ['기타'])
    ap.add_argument('--note', default='')
    a = ap.parse_args()

    if a.list:
        rows = list_all()
        if not rows:
            print('제외 라벨 없음')
        for r in rows:
            tgt = '세션 전체' if r['scope'] == 'session' else f"ch{r['channel']}"
            print(f"  {r['subject']}/{r['session']} {tgt}: {r['reason']}"
                  f"{' — ' + r['note'] if r['note'] else ''} ({r['updated_at']})")
    elif a.set:
        if not (a.subject and a.session):
            ap.error('--subject --session 필요')
        set_exclusion(a.subject, a.session, a.channel, a.reason, a.note)
        print(f"✅ 제외: {a.subject}/{a.session} "
              f"{'세션 전체' if a.channel == 0 else f'ch{a.channel}'} ({a.reason})")
    elif a.clear:
        if not (a.subject and a.session):
            ap.error('--subject --session 필요')
        print('✅ 해제' if clear_exclusion(a.subject, a.session, a.channel) else '⚠️ 해당 항목 없음')
    else:
        ap.error('--list / --set / --clear 중 하나')
