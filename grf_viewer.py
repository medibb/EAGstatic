"""
Kinvent K-Plate Force Plate GRF(지면반력) 시계열 시각화 도구
- 양발(왼발/오른발) 수직 하중 시계열 그래프
- CSV 파일의 raw 채널 데이터(CH1~4 합산)를 사용
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator
from pathlib import Path


# ==================== 한글 폰트 설정 ====================
def setup_korean_font():
    font_candidates = ['Malgun Gothic', 'NanumGothic', 'AppleGothic']
    for font_name in font_candidates:
        font_path = font_manager.findfont(font_name, fallback_to_default=False)
        if font_path and 'LastResort' not in font_path:
            plt.rcParams['font.family'] = font_name
            break
    plt.rcParams['axes.unicode_minus'] = False


def find_raw_data_start(filepath: str) -> int:
    """CSV에서 '가공되지 않은 채널 데이터' 이후 실제 데이터 시작 행 번호를 찾는다."""
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if '시간 (초)' in line and 'CHANNEL_1' in line:
                return i
    raise ValueError("raw 채널 데이터 헤더를 찾을 수 없습니다.")


def load_grf_data(filepath: str) -> pd.DataFrame:
    """CSV에서 GRF 시계열 데이터를 로드한다."""
    header_row = find_raw_data_start(filepath)

    df = pd.read_csv(filepath, skiprows=header_row, encoding='utf-8', index_col=False)

    # 빈 컬럼(Unnamed) 제거
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # 컬럼명 정리: 중복된 CHANNEL_1~4 → 왼쪽/오른쪽 구분
    cols = df.columns.tolist()
    new_cols = ['time']
    ch_count = 0
    for col in cols[1:]:
        if 'CHANNEL' in col:
            ch_count += 1
            if ch_count <= 4:
                new_cols.append(f'left_ch{ch_count}')
            else:
                new_cols.append(f'right_ch{ch_count - 4}')
        elif 'COP' in col or 'CoP' in col or 'cop' in col:
            new_cols.append(col)
        else:
            new_cols.append(col)

    df.columns = new_cols[:len(df.columns)]

    # 숫자형 변환
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=['time'], inplace=True)

    # GRF 합산 (4채널 합 = 해당 발의 수직 하중)
    df['left_grf'] = df[['left_ch1', 'left_ch2', 'left_ch3', 'left_ch4']].sum(axis=1)
    df['right_grf'] = df[['right_ch1', 'right_ch2', 'right_ch3', 'right_ch4']].sum(axis=1)
    df['total_grf'] = df['left_grf'] + df['right_grf']

    return df


def extract_body_weight(filepath: str) -> float:
    """CSV 상단에서 체중(kg) 정보를 추출한다."""
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if '체중' in line and '킬로그램' in line:
                parts = line.strip().split(',')
                try:
                    return float(parts[1])
                except (IndexError, ValueError):
                    pass
    return 0.0


def get_result_dir() -> str:
    """result 폴더 경로 반환"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, 'result')


def get_output_dir(filepath: str) -> Path:
    """CSV 파일 경로에서 result/피험자/ 저장 경로를 계산한다.

    OpenBCISession 폴더의 바로 상위를 피험자 폴더로 인식한다.
    예: data/2026 실험(02~)/주창민_1/OpenBCISession_.../ → result/주창민_1/
    """
    result_base = Path(get_result_dir())
    parts = Path(filepath).parts

    for i, part in enumerate(parts):
        if part.startswith('OpenBCISession'):
            if i >= 1:
                subject_name = parts[i - 1]
                return result_base / subject_name
            break

    return result_base / 'unknown'


def plot_grf(df: pd.DataFrame, title: str = "", body_weight: float = 0.0,
             save_path: str = None):
    """양발 GRF 시계열 그래프를 그린다."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    time = df['time']

    # 왼발
    axes[0].plot(time, df['left_grf'], color='#2196F3', linewidth=0.8)
    axes[0].set_ylabel('왼발 (kg)')
    axes[0].set_title('왼발 GRF')
    axes[0].grid(True, alpha=0.3)
    mean_left = df['left_grf'].mean()
    axes[0].axhline(y=mean_left, color='#2196F3', linestyle='--', alpha=0.5,
                    label=f'평균: {mean_left:.1f} kg')
    axes[0].legend(loc='upper right')

    # 오른발
    axes[1].plot(time, df['right_grf'], color='#F44336', linewidth=0.8)
    axes[1].set_ylabel('오른발 (kg)')
    axes[1].set_title('오른발 GRF')
    axes[1].grid(True, alpha=0.3)
    mean_right = df['right_grf'].mean()
    axes[1].axhline(y=mean_right, color='#F44336', linestyle='--', alpha=0.5,
                    label=f'평균: {mean_right:.1f} kg')
    axes[1].legend(loc='upper right')

    # 합계
    axes[2].plot(time, df['left_grf'], color='#2196F3', linewidth=0.6, alpha=0.7, label='왼발')
    axes[2].plot(time, df['right_grf'], color='#F44336', linewidth=0.6, alpha=0.7, label='오른발')
    axes[2].plot(time, df['total_grf'], color='#4CAF50', linewidth=0.8, label='합계')
    axes[2].set_ylabel('하중 (kg)')
    axes[2].set_xlabel('시간 (초)')
    axes[2].set_title('양발 GRF 비교')
    axes[2].grid(True, alpha=0.3)
    if body_weight > 0:
        axes[2].axhline(y=body_weight, color='gray', linestyle=':', alpha=0.5,
                        label=f'체중: {body_weight:.1f} kg')
    axes[2].legend(loc='upper right')

    # x축 설정
    duration = time.max() - time.min()
    if duration <= 10:
        axes[2].xaxis.set_major_locator(MultipleLocator(1))
    elif duration <= 60:
        axes[2].xaxis.set_major_locator(MultipleLocator(5))
    else:
        axes[2].xaxis.set_major_locator(MultipleLocator(10))

    fig_title = f'GRF 시계열 - {title}' if title else 'GRF 시계열'
    fig.suptitle(fig_title, fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"그래프 저장: {save_path}")

    plt.show()


def is_forceplate_csv(filepath: str) -> bool:
    """Kinvent forceplate CSV 파일인지 확인한다."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            return 'Kinvent' in first_line or 'K-Plate' in first_line
    except (UnicodeDecodeError, PermissionError):
        return False


def find_forceplate_csvs_in_dir(target_dir: str) -> list:
    """특정 폴더(재귀)에서 forceplate CSV 파일 경로 목록을 반환한다."""
    results = []
    for root, dirs, files in os.walk(target_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for f in sorted(files):
            if f.endswith('.csv') and not f.startswith('.'):
                full_path = os.path.join(root, f)
                if is_forceplate_csv(full_path):
                    results.append(full_path)
    return results


def list_subjects(data_dir: str) -> list:
    """data 폴더에서 피험자 폴더를 재귀적으로 찾는다 (OpenBCISession을 포함하는 폴더).

    Returns:
        List of (display_name, path) tuples
    """
    subjects = []

    def find_recursive(current_dir, prefix=""):
        try:
            items = sorted(os.listdir(current_dir))
        except PermissionError:
            return
        for item in items:
            if item.startswith('.'):
                continue
            item_path = os.path.join(current_dir, item)
            if not os.path.isdir(item_path):
                continue
            has_session = any(
                sub.startswith('OpenBCISession')
                for sub in os.listdir(item_path)
                if os.path.isdir(os.path.join(item_path, sub))
            )
            if has_session:
                display = f"{prefix}{item}" if prefix else item
                subjects.append((display, item_path))
            else:
                new_prefix = f"{prefix}{item}/" if prefix else f"{item}/"
                find_recursive(item_path, new_prefix)

    find_recursive(data_dir)
    return subjects


def list_sessions(subject_dir: str) -> list:
    """피험자 폴더 내 OpenBCISession 폴더 목록 반환"""
    sessions = []
    for item in sorted(os.listdir(subject_dir)):
        item_path = os.path.join(subject_dir, item)
        if os.path.isdir(item_path) and item.startswith('OpenBCISession'):
            sessions.append(item)
    return sessions


def interactive_select():
    """인터랙티브하게 피험자 → 세션 → forceplate CSV 파일을 선택한다."""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

    if not os.path.exists(data_dir):
        print(f"data 폴더가 없습니다: {data_dir}")
        return None

    # 1. 피험자 선택
    subjects = list_subjects(data_dir)
    if not subjects:
        print("data 폴더에 피험자 데이터가 없습니다.")
        return None

    print("\n" + "=" * 60)
    print("=== 피험자 선택 ===")
    print("=" * 60)
    for i, (display, path) in enumerate(subjects):
        fp_count = len(find_forceplate_csvs_in_dir(path))
        print(f"  {i}: {display} (forceplate {fp_count}개)")
    print(f"  a: 전체 분석 (모든 피험자)")
    print("=" * 60)

    while True:
        choice = input(f"피험자 번호를 선택하세요 (0-{len(subjects)-1}, a=전체): ").strip().lower()
        if choice == 'a':
            return find_forceplate_csvs_in_dir(data_dir)
        try:
            idx = int(choice)
            if 0 <= idx < len(subjects):
                break
            print(f"0-{len(subjects)-1} 사이의 숫자를 입력하세요.")
        except ValueError:
            print("올바른 숫자를 입력하세요.")

    _, subject_dir = subjects[idx]

    # 2. 세션 선택
    sessions = list_sessions(subject_dir)
    if not sessions:
        print(f"{os.path.basename(subject_dir)}에 세션 데이터가 없습니다.")
        return None

    print("\n" + "=" * 60)
    print(f"=== 세션 선택 ({os.path.basename(subject_dir)}) ===")
    print("=" * 60)
    for i, session in enumerate(sessions):
        session_dir = os.path.join(subject_dir, session)
        fp_count = len(find_forceplate_csvs_in_dir(session_dir))
        print(f"  {i}: {session} [forceplate {fp_count}개]")
    print(f"  a: 전체 분석 (이 피험자의 모든 세션)")
    print("=" * 60)

    while True:
        choice = input(f"세션 번호를 선택하세요 (0-{len(sessions)-1}, a=전체): ").strip().lower()
        if choice == 'a':
            return find_forceplate_csvs_in_dir(subject_dir)
        try:
            idx = int(choice)
            if 0 <= idx < len(sessions):
                break
            print(f"0-{len(sessions)-1} 사이의 숫자를 입력하세요.")
        except ValueError:
            print("올바른 숫자를 입력하세요.")

    session_name = sessions[idx]
    session_dir = os.path.join(subject_dir, session_name)

    # 3. CSV 파일 선택
    csv_files = find_forceplate_csvs_in_dir(session_dir)
    if not csv_files:
        print(f"{session_name}에 forceplate CSV 파일이 없습니다.")
        return None

    if len(csv_files) == 1:
        print(f"\n선택된 파일: {os.path.basename(csv_files[0])}")
        return csv_files[0]

    print("\n" + "=" * 60)
    print(f"=== CSV 파일 선택 ({session_name}) ===")
    print("=" * 60)
    for i, csv_file in enumerate(csv_files):
        print(f"  {i}: {os.path.basename(csv_file)}")
    print(f"  a: 전체 분석 (이 세션의 모든 파일)")
    print("=" * 60)

    while True:
        choice = input(f"파일 번호를 선택하세요 (0-{len(csv_files)-1}, a=전체): ").strip().lower()
        if choice == 'a':
            return csv_files
        try:
            idx = int(choice)
            if 0 <= idx < len(csv_files):
                return csv_files[idx]
            print(f"0-{len(csv_files)-1} 사이의 숫자를 입력하세요.")
        except ValueError:
            print("올바른 숫자를 입력하세요.")


def process_single_file(filepath: str):
    """단일 forceplate CSV 파일을 로드하고 시각화 및 저장한다."""
    print(f"\n로드 중: {os.path.basename(filepath)}")

    body_weight = extract_body_weight(filepath)
    if body_weight > 0:
        print(f"체중: {body_weight:.1f} kg")

    df = load_grf_data(filepath)
    duration = df['time'].max() - df['time'].min()
    print(f"데이터: {len(df)}샘플, {duration:.1f}초, 250Hz")
    print(f"왼발 평균: {df['left_grf'].mean():.1f} kg | 오른발 평균: {df['right_grf'].mean():.1f} kg")

    # 저장 경로 생성
    output_dir = get_output_dir(filepath)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = os.path.basename(filepath).replace('.csv', '')
    save_path = str(output_dir / f'{base_name}_grf.png')

    plot_grf(df, title=base_name, body_weight=body_weight, save_path=save_path)
    print(f"결과 저장 위치: {output_dir}")


def main():
    setup_korean_font()

    # 명령행 인자로 파일 경로를 받거나, 인터랙티브 선택
    if len(sys.argv) > 1:
        selected = sys.argv[1]
    else:
        selected = interactive_select()

    if selected is None:
        print("선택이 취소되었습니다.")
        return

    # 리스트(전체 분석) 또는 단일 파일 경로
    if isinstance(selected, list):
        if not selected:
            print("forceplate CSV 파일을 찾을 수 없습니다.")
            return
        print(f"\n총 {len(selected)}개 파일을 분석합니다.")
        for i, fp in enumerate(selected, 1):
            print(f"\n[{i}/{len(selected)}] ", end="")
            try:
                process_single_file(fp)
            except Exception as e:
                print(f"오류 발생: {e}")
    else:
        if not os.path.exists(selected):
            print(f"파일을 찾을 수 없습니다: {selected}")
            return
        process_single_file(selected)


if __name__ == '__main__':
    main()
