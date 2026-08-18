"""
Manual Offset Manager
- EAG-GRF 동기화의 수동 offset을 JSON으로 관리
- SyncAnalyzer에서 자동 조회하여 사용
"""

import os
import json
from datetime import datetime
from typing import Optional

from store_io import store_path, load_json, save_json_atomic

STORE_FILENAME = 'manual_offsets.json'


def default_path() -> str:
    """저장 위치. `EAG_RESULT_DIR`로 바꿀 수 있다 (store_io 참조)."""
    return store_path(STORE_FILENAME)


DEFAULT_PATH = default_path()  # 하위호환 (import 시점 해석)


def load_manual_offsets(path: str = None) -> dict:
    return load_json(path or default_path())


def save_manual_offsets(offsets: dict, path: str = None):
    save_json_atomic(path or default_path(), offsets)


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
