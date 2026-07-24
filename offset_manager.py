"""
Manual Offset Manager
- EAG-GRF 동기화의 수동 offset을 JSON으로 관리
- SyncAnalyzer에서 자동 조회하여 사용
"""

import os
import json
from datetime import datetime
from typing import Optional

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), 'result', 'manual_offsets.json')


def load_manual_offsets(path: str = None) -> dict:
    path = path or DEFAULT_PATH
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_manual_offsets(offsets: dict, path: str = None):
    path = path or DEFAULT_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(offsets, f, ensure_ascii=False, indent=2)


def get_manual_offset(subject: str, session: str, path: str = None) -> Optional[float]:
    offsets = load_manual_offsets(path)
    entry = offsets.get(subject, {}).get(session, {})
    if 'manual_offset' in entry:
        return float(entry['manual_offset'])
    return None


def set_manual_offset(subject: str, session: str, offset: float,
                      auto_offset: float = 0.0, auto_method: str = "",
                      note: str = "", path: str = None):
    offsets = load_manual_offsets(path)
    if subject not in offsets:
        offsets[subject] = {}
    offsets[subject][session] = {
        'manual_offset': offset,
        'auto_offset': auto_offset,
        'auto_method': auto_method,
        'updated_at': datetime.now().isoformat(timespec='seconds'),
        'note': note,
    }
    save_manual_offsets(offsets, path)


def clear_manual_offset(subject: str, session: str, path: str = None):
    offsets = load_manual_offsets(path)
    if subject in offsets and session in offsets[subject]:
        del offsets[subject][session]
        if not offsets[subject]:
            del offsets[subject]
        save_manual_offsets(offsets, path)
        return True
    return False


def list_all_offsets(path: str = None) -> list:
    """모든 manual offset을 flat list로 반환."""
    offsets = load_manual_offsets(path)
    rows = []
    for subject, sessions in offsets.items():
        for session, data in sessions.items():
            rows.append({
                'subject': subject,
                'session': session,
                **data,
            })
    return rows
