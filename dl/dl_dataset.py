"""
EAG-GRF Deep Learning Dataset
- GRF 이벤트 기반 EAG 윈도우 추출
- PyTorch Dataset 클래스
- Subject-level split

Usage:
    from dl_dataset import build_dataset, create_subject_splits
    dataset = build_dataset()
    train_ds, val_ds, test_ds = create_subject_splits(dataset)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from typing import List, Dict, Optional, Tuple

from sync_analyzer import SyncAnalyzer, find_all_pairs, SessionPair, SAMPLE_RATE
from eag_analyzer import EEG_CHANNELS, FilterConfig

# 윈도우 설정: onset 전 2초 ~ onset 후 13초 = 15초 = 3750 samples
WINDOW_PRE = 2.0    # seconds before onset
WINDOW_POST = 13.0  # seconds after onset
WINDOW_SAMPLES = int((WINDOW_PRE + WINDOW_POST) * SAMPLE_RATE)  # 3750

# 세션 타입 매핑
SESSION_TYPE_MAP = {'s': 0, 'f': 1, 'c': 2}
SESSION_TYPE_NAMES = ['side', 'front', 'crutch']


def extract_event_windows(pairs: List[SessionPair],
                          verbose: bool = True
                          ) -> Dict[str, np.ndarray]:
    """전체 세션에서 GRF 이벤트 기반 EAG/GRF 윈도우를 추출한다.

    Returns:
        dict with keys:
        - eag: (N, WINDOW_SAMPLES, 8) float32
        - grf: (N, WINDOW_SAMPLES, 2) float32 [left, right]
        - labels_action: (N,) int — 0=side, 1=front, 2=crutch
        - labels_load: (N,) int — 이벤트 순서 (c세션만 유효, 나머지 -1)
        - subjects: (N,) str — 피험자명
        - sessions: (N,) str — 세션명
        - event_ids: (N,) int
    """
    all_eag = []
    all_grf = []
    all_labels_action = []
    all_labels_load = []
    all_subjects = []
    all_sessions = []
    all_event_ids = []

    config = FilterConfig()
    config.lowpass_cutoff = 5.0
    config.start_time = 0.0

    for i, pair in enumerate(pairs):
        # 세션 타입 추출
        session_type = ''.join(c for c in pair.session_name if c.isalpha())
        if session_type not in SESSION_TYPE_MAP:
            continue

        try:
            sa = SyncAnalyzer(pair, config=config)
            sa.run_analysis()
        except Exception:
            continue

        if not sa.grf_events:
            continue

        # GRF를 EAG 시간축으로 보간
        grf_interp_left = np.interp(sa.unified_time_eag,
                                     sa.unified_time_grf, sa.grf_left)
        grf_interp_right = np.interp(sa.unified_time_eag,
                                      sa.unified_time_grf, sa.grf_right)

        for event_idx, event in enumerate(sa.grf_events):
            # 윈도우 시작/끝 (unified time)
            win_start = event.onset_time - WINDOW_PRE
            win_end = event.onset_time + WINDOW_POST

            # 인덱스 변환
            start_idx = int(win_start * SAMPLE_RATE)
            end_idx = start_idx + WINDOW_SAMPLES

            # 범위 체크
            if start_idx < 0 or end_idx > len(sa.unified_time_eag):
                continue

            # EAG 윈도우 (8ch)
            eag_window = sa.eag_filtered[start_idx:end_idx, :].copy()

            # GRF 윈도우 (2ch: left, right)
            grf_window = np.stack([
                grf_interp_left[start_idx:end_idx],
                grf_interp_right[start_idx:end_idx],
            ], axis=-1)

            # 길이 검증
            if eag_window.shape[0] != WINDOW_SAMPLES:
                continue

            all_eag.append(eag_window)
            all_grf.append(grf_window)
            all_labels_action.append(SESSION_TYPE_MAP[session_type])
            all_subjects.append(pair.subject_name)
            all_sessions.append(pair.session_name)
            all_event_ids.append(event.event_id)

            # 체중부하 라벨 (c세션만, 이벤트 순서로 추정)
            if session_type == 'c':
                # 프로토콜: 0%→20%→50%→80%→100% 순서
                # 이벤트 수에 따라 균등 분할
                n_events = len(sa.grf_events)
                if n_events >= 4:
                    load_idx = min(event_idx, 3)  # 0,1,2,3 → 20,50,80,100%
                else:
                    load_idx = int(event_idx / max(n_events - 1, 1) * 3)
                all_labels_load.append(load_idx)
            else:
                all_labels_load.append(-1)

        if verbose and (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(pairs)} sessions processed, "
                  f"{len(all_eag)} events extracted")

    if verbose:
        print(f"  총 {len(all_eag)} events from {len(pairs)} sessions")

    return {
        'eag': np.array(all_eag, dtype=np.float32),
        'grf': np.array(all_grf, dtype=np.float32),
        'labels_action': np.array(all_labels_action, dtype=np.int64),
        'labels_load': np.array(all_labels_load, dtype=np.int64),
        'subjects': np.array(all_subjects),
        'sessions': np.array(all_sessions),
        'event_ids': np.array(all_event_ids, dtype=np.int64),
    }


class EAGEventDataset(Dataset):
    """PyTorch Dataset for EAG event windows."""

    def __init__(self, data: Dict[str, np.ndarray],
                 task: str = 'classify_action',
                 normalize: bool = True):
        """
        Args:
            data: extract_event_windows() 결과
            task: 'classify_action', 'classify_load', 'regress_grf'
            normalize: 채널별 z-score 정규화
        """
        self.eag = torch.from_numpy(data['eag'])      # (N, T, 8)
        self.grf = torch.from_numpy(data['grf'])      # (N, T, 2)
        self.labels_action = torch.from_numpy(data['labels_action'])
        self.labels_load = torch.from_numpy(data['labels_load'])
        self.subjects = data['subjects']
        self.task = task

        if normalize:
            # 채널별 z-score (전체 데이터 기준)
            self.eag_mean = self.eag.mean(dim=(0, 1), keepdim=True)
            self.eag_std = self.eag.std(dim=(0, 1), keepdim=True).clamp(min=1e-6)
            self.eag = (self.eag - self.eag_mean) / self.eag_std

            self.grf_mean = self.grf.mean(dim=(0, 1), keepdim=True)
            self.grf_std = self.grf.std(dim=(0, 1), keepdim=True).clamp(min=1e-6)
            self.grf = (self.grf - self.grf_mean) / self.grf_std

    def __len__(self):
        return len(self.eag)

    def __getitem__(self, idx):
        # EAG: (T, 8) → (8, T) for Conv1D
        x = self.eag[idx].permute(1, 0)  # (8, T)

        if self.task == 'classify_action':
            y = self.labels_action[idx]
        elif self.task == 'classify_load':
            y = self.labels_load[idx]
        elif self.task == 'regress_grf':
            y = self.grf[idx].permute(1, 0)  # (2, T)
        else:
            raise ValueError(f"Unknown task: {self.task}")

        return x, y


def create_subject_splits(dataset: EAGEventDataset,
                          test_ratio: float = 0.17,
                          val_ratio: float = 0.17,
                          seed: int = 42
                          ) -> Tuple[Subset, Subset, Subset]:
    """Subject-level train/val/test split.

    Returns:
        (train_subset, val_subset, test_subset)
    """
    subjects = np.unique(dataset.subjects)
    rng = np.random.RandomState(seed)
    rng.shuffle(subjects)

    n = len(subjects)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))

    test_subjects = set(subjects[:n_test])
    val_subjects = set(subjects[n_test:n_test + n_val])
    train_subjects = set(subjects[n_test + n_val:])

    train_idx = [i for i, s in enumerate(dataset.subjects) if s in train_subjects]
    val_idx = [i for i, s in enumerate(dataset.subjects) if s in val_subjects]
    test_idx = [i for i, s in enumerate(dataset.subjects) if s in test_subjects]

    print(f"Split: train={len(train_idx)} ({len(train_subjects)}명), "
          f"val={len(val_idx)} ({len(val_subjects)}명), "
          f"test={len(test_idx)} ({len(test_subjects)}명)")

    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)


def build_dataset(data_dir: str = 'data',
                  task: str = 'classify_action',
                  cache_path: str = 'result/dl/event_windows.npz',
                  force_rebuild: bool = False) -> EAGEventDataset:
    """데이터셋을 빌드하거나 캐시에서 로드."""
    if os.path.exists(cache_path) and not force_rebuild:
        print(f"캐시 로드: {cache_path}")
        cached = dict(np.load(cache_path, allow_pickle=True))
        data = {k: cached[k] for k in cached}
    else:
        print("이벤트 윈도우 추출 중...")
        pairs = find_all_pairs(data_dir)
        # trial 세션 제외
        pairs = [p for p in pairs
                 if p.session_name and p.session_name[0] in ('s', 'f', 'c')
                 and not p.session_name.endswith('t')]
        # s로 시작하지만 st인 경우 제외
        pairs = [p for p in pairs
                 if not any(p.session_name.startswith(t) for t in ['st', 'ft', 'ct'])]
        print(f"대상 세션: {len(pairs)}개")

        data = extract_event_windows(pairs)

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, **data)
        print(f"캐시 저장: {cache_path}")

    # task에 따라 필터링
    if task == 'classify_load':
        mask = data['labels_load'] >= 0
        data = {k: v[mask] for k, v in data.items()}
        print(f"체중부하 분류용 필터링: {mask.sum()}개 이벤트 (crutch 세션만)")

    dataset = EAGEventDataset(data, task=task)
    print(f"Dataset: {len(dataset)} samples, task={task}")

    return dataset


if __name__ == '__main__':
    ds = build_dataset(task='classify_action')
    print(f"\nEAG shape: {ds.eag.shape}")
    print(f"Action labels: {np.bincount(ds.labels_action.numpy())}")
    print(f"  side={SESSION_TYPE_NAMES[0]}, front={SESSION_TYPE_NAMES[1]}, crutch={SESSION_TYPE_NAMES[2]}")

    train, val, test = create_subject_splits(ds)
    x, y = train[0]
    print(f"\nSample: x={x.shape}, y={y}")
