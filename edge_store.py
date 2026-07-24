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

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), 'result', 'manual_edges.json')


def load_manual_edges(path: str = None) -> dict:
    path = path or DEFAULT_PATH
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_manual_edges(data: dict, path: str = None):
    path = path or DEFAULT_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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
