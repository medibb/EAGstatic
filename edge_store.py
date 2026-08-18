"""
Manual Edge Store — GRF-triggered 파라미터 추출용 EAG edge(knee)의 수동 최종수정 저장소.

offset_manager.py(manual_offsets.json)와 같은 철학: 사람이 확정한 edge 목록이 최우선.
자동 검출(detect_eag_edges)이 부정확한 세션/채널에 대해, 검토 PNG를 보고 삭제/추가/이동으로
curated한 최종 edge 목록을 채널별로 JSON에 동결한다. parameter_extractor는 manual edge가
있으면 자동 검출을 무시하고 이 목록을 쓴다.

구조 (result/manual_edges.json):
{
  "(02.10_15)김영훈_1": {
    "s7": {
      "1": {                                  # channel (str)
        "edges": [                            # te_corr(보정된 EAG 시간축) 프레임
          {"onset_time":9.58,"onset_amp":-70.0,"offset_time":9.96,"offset_amp":-11.0},
          ...
        ],
        "offset_used": -0.124,                # 이 edge들이 정의된 corrected offset
        "note": "", "updated_at": "..."
      }
    }
  }
}
"""

import os
import json
from datetime import datetime
from typing import Optional, List, Dict

from store_io import store_path, load_json, save_json_atomic

STORE_FILENAME = 'manual_edges.json'


def default_path() -> str:
    """저장 위치. `EAG_RESULT_DIR`로 바꿀 수 있다 (store_io 참조)."""
    return store_path(STORE_FILENAME)


DEFAULT_PATH = default_path()  # 하위호환 (import 시점 해석)


def load_manual_edges(path: str = None) -> dict:
    return load_json(path or default_path())


def save_manual_edges(data: dict, path: str = None):
    save_json_atomic(path or default_path(), data)


def get_channel_edges(subject: str, session: str, channel: int,
                      path: str = None) -> Optional[List[Dict]]:
    """채널의 수동 edge 목록 반환 (없으면 None). 각 dict: onset/offset_time·amp."""
    d = load_manual_edges(path)
    entry = d.get(subject, {}).get(session, {}).get(str(channel))
    if entry and 'edges' in entry:
        return sorted(entry['edges'], key=lambda e: e['onset_time'])
    return None


def set_channel_edges(subject: str, session: str, channel: int,
                      edges: List[Dict], offset_used: float = 0.0,
                      note: str = "", path: str = None):
    d = load_manual_edges(path)
    d.setdefault(subject, {}).setdefault(session, {})[str(channel)] = {
        'edges': sorted(edges, key=lambda e: e['onset_time']),
        'offset_used': round(float(offset_used), 4),
        'updated_at': datetime.now().isoformat(timespec='seconds'),
        'note': note,
    }
    save_manual_edges(d, path)


def clear_channel_edges(subject: str, session: str, channel: int,
                        path: str = None) -> bool:
    d = load_manual_edges(path)
    sess = d.get(subject, {}).get(session, {})
    if str(channel) in sess:
        del sess[str(channel)]
        if not sess:
            d[subject].pop(session, None)
        if not d.get(subject):
            d.pop(subject, None)
        save_manual_edges(d, path)
        return True
    return False


def list_all(path: str = None) -> list:
    d = load_manual_edges(path)
    rows = []
    for subj, sessions in d.items():
        for sess, chans in sessions.items():
            for ch, entry in chans.items():
                rows.append({'subject': subj, 'session': sess, 'channel': ch,
                             'n_edges': len(entry.get('edges', [])),
                             'note': entry.get('note', '')})
    return rows
