#!/usr/bin/env python3
"""Tier A 용적전도 시뮬레이터 — EAG 표면 전위의 forward 문제.

**목적.** 전극 배치를 해부학적 직관이 아니라 물리로 정하기 위한 최소 모델이다.
데이터 수집 전에 다음 네 가지를 답한다.

  1. 표면 전위가 소스 깊이·피하지방 두께에 따라 얼마나 감쇠하는가
  2. 국소 소스 하나가 만드는 표면 전위의 **폭**(point-spread) = 전극을 몇 mm
     떨어뜨려야 서로 다른 것을 보는가
  3. **골 차폐 가설**: 관절선이 이기는 이유가 "연골에 가까워서"인가,
     아니면 "고저항 뼈를 우회하지 않는 유일한 창이어서"인가
  4. **유효 rank**: N개 전극으로 독립적으로 복원 가능한 소스 성분이 몇 개인가.
     이것이 "왜 16채널인가"에 대한 답이다

**물리.** EAG 대역은 0.3~0.7 Hz로 준정적(quasi-static)이다. 용량성 항과 시간 항이
사라져 문제가 다음 하나로 줄어든다.

    ∇·(σ ∇φ) = -q          (q = 주입 전류 밀도, A/m³)
    피부 바깥으로 전류 없음  (Neumann, ∂φ/∂n = 0)

선형 타원형 PDE 하나이므로 EEG forward보다도 단순하다. 여기서는 3차원 유한차분
(면 전도도는 조화평균)으로 풀고, 순수 Neumann이라 해가 상수만큼 부정이므로
평균 0으로 고정한다.

**소스 모델.** 압박된 연골에서 간질액이 밀려나며 이온을 운반한다(streaming current).
Tier A에서는 이를 연골 두께 방향의 **전류 쌍극자**로 근사한다. 크기는 임의 단위이며
(상대 비교만 하므로), 방향은 관절면 법선이다.

**전도도 값은 자리표시자다.** `CONDUCTIVITY`의 값은 저주파 생체임피던스 문헌의
통상 범위를 반영했으나 **출처가 확정되지 않았다.** 확정 전에는 절대값이 아니라
**비율에서 나오는 결론**(감쇠 형태, PSF 폭, rank)만 인용할 것.
`--sigma-json`으로 교체할 수 있다.

사용:
    python3 volume_conductor.py map      # 표면 전위 지도 + PSF 폭
    python3 volume_conductor.py sweep    # 지방 두께·소스 깊이 스윕
    python3 volume_conductor.py shield   # 골 차폐 가설 검정
    python3 volume_conductor.py rank     # lead field SVD → 유효 rank
    python3 volume_conductor.py all
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import cg, LinearOperator

OUT_DIR = Path('result/volume_conductor')

# ==================== 전도도 (S/m, 저주파) ====================
# ⚠️ 자리표시자. 출처 확정 전에는 비율 기반 결론만 인용할 것.
CONDUCTIVITY = {
    'air':       0.0,      # 절연 (경계조건으로 처리)
    'skin':      0.33,     # 전처리·젤 도포 상태 가정
    'fat':       0.04,
    'muscle':    0.35,     # 등방 근사 (실제는 이방성)
    'bone':      0.008,    # 피질골. muscle 대비 약 1/44
    'marrow':    0.07,     # 해면골·골수
    'cartilage': 0.18,
    'synovial':  1.50,     # 관절액 (식염수 수준)
}

LABELS = {name: i for i, name in enumerate(CONDUCTIVITY)}


# ==================== 기하 ====================

@dataclass
class Geometry:
    """무릎 관절선 부근의 단순 층상 모델.

    좌표계 (mm):
      x  근위-원위 (사지 장축).  관절선은 x = x_joint 평면
      y  내측-외측
      z  피부에서 안쪽으로의 깊이.  z = 0 이 피부 표면

    x < x_joint - gap/2 는 대퇴골, x > x_joint + gap/2 는 경골이 차지한다.
    그 사이 틈(gap)에 연골 두 층과 관절액이 있다. 즉 **관절 간극이 고저항 뼈
    껍질을 관통하는 유일한 통로**이며, 이 구조가 골 차폐 가설을 검정한다.
    """
    nx: int = 60          # 격자 수
    ny: int = 60
    nz: int = 30
    h: float = 2.0        # 격자 간격 (mm)

    d_skin: float = 2.0   # 피부 두께
    d_fat: float = 6.0    # 피하지방 두께
    z_bone: float = 16.0  # 피부에서 뼈 앞면까지 (지방 아래는 근육·관절낭)

    x_joint: float = 60.0  # 관절선 위치
    gap: float = 8.0       # 관절 간극 폭 (연골 2층 + 관절액)
    cart: float = 2.5      # 편측 연골 두께

    no_bone: bool = False  # True면 뼈를 근육으로 치환 (차폐 대조군)

    # 내외측 비대칭. 내측 무릎은 피하지방이 얇고 외측은 두껍다.
    # None이면 d_fat 균일. 값을 주면 y를 따라 선형으로 변한다.
    d_fat_med: float = None   # y 낮은 쪽(내측)
    d_fat_lat: float = None   # y 높은 쪽(외측)

    @property
    def shape(self):
        return (self.nx, self.ny, self.nz)

    def coords(self):
        x = (np.arange(self.nx) + 0.5) * self.h
        y = (np.arange(self.ny) + 0.5) * self.h
        z = (np.arange(self.nz) + 0.5) * self.h
        return np.meshgrid(x, y, z, indexing='ij')


def build_labels(g: Geometry) -> np.ndarray:
    """조직 라벨 볼륨."""
    X, _, Z = g.coords()
    lab = np.full(g.shape, LABELS['muscle'], dtype=np.int8)

    lab[Z < g.d_skin] = LABELS['skin']
    if g.d_fat_med is None or g.d_fat_lat is None:
        fat_t = np.full(g.shape, g.d_fat)
    else:
        _, Y, _ = g.coords()
        frac = Y / (g.ny * g.h)
        fat_t = g.d_fat_med + (g.d_fat_lat - g.d_fat_med) * frac
    lab[(Z >= g.d_skin) & (Z < g.d_skin + fat_t)] = LABELS['fat']

    deep = Z >= g.z_bone
    dx = np.abs(X - g.x_joint)
    in_gap = dx < g.gap / 2

    bone_lab = LABELS['muscle'] if g.no_bone else LABELS['bone']
    lab[deep & ~in_gap] = bone_lab
    if not g.no_bone:
        # 뼈 안쪽은 해면골
        lab[deep & ~in_gap & (Z > g.z_bone + 8.0)] = LABELS['marrow']

    # 간극: 바깥쪽 연골 두 층, 가운데 관절액
    lab[deep & in_gap] = LABELS['synovial']
    lab[deep & in_gap & (dx > g.gap / 2 - g.cart)] = LABELS['cartilage']
    return lab


def build_sigma(lab: np.ndarray) -> np.ndarray:
    sig = np.zeros(lab.shape)
    for name, idx in LABELS.items():
        sig[lab == idx] = CONDUCTIVITY[name]
    return np.maximum(sig, 1e-6)   # 0 나눗셈 방지 (air는 쓰지 않음)


# ==================== 솔버 ====================

def _face_sigma(sig, axis):
    """인접 셀 사이 면 전도도 = 조화평균 (직렬 저항이 물리적으로 맞다)."""
    a = np.take(sig, np.arange(sig.shape[axis] - 1), axis=axis)
    b = np.take(sig, np.arange(1, sig.shape[axis]), axis=axis)
    return 2.0 * a * b / (a + b)


def build_operator(sig: np.ndarray, h_mm: float):
    """∇·(σ∇·) 의 7점 유한차분 행렬. h는 mm → m 변환."""
    h = h_mm * 1e-3
    nx, ny, nz = sig.shape
    n = nx * ny * nz
    idx = np.arange(n).reshape(sig.shape)

    rows, cols, vals = [], [], []
    diag = np.zeros(n)

    for axis in (0, 1, 2):
        fs = _face_sigma(sig, axis) / h ** 2
        lo = np.take(idx, np.arange(idx.shape[axis] - 1), axis=axis).ravel()
        hi = np.take(idx, np.arange(1, idx.shape[axis]), axis=axis).ravel()
        f = fs.ravel()
        rows.extend([lo, hi]); cols.extend([hi, lo]); vals.extend([f, f])
        np.add.at(diag, lo, -f)
        np.add.at(diag, hi, -f)

    rows = np.concatenate(rows + [np.arange(n)])
    cols = np.concatenate(cols + [np.arange(n)])
    vals = np.concatenate(vals + [diag])
    return coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


def far_boundary_mask(shape) -> np.ndarray:
    """원거리 경계(피부면 z=0 을 뺀 나머지 5면). 여기에 φ=0을 건다.

    피부면은 공기와 접하므로 **반드시 Neumann**(전류가 나가지 않음)이고, 이는 셀
    사이 면만 연결하는 유한차분 구성에서 자연히 만족된다. 나머지 5면은 사지가
    계속 이어지는 쪽이고 실제 레퍼런스도 원위(경골 10 cm)에 있으므로, 절연벽으로
    막아 장을 가두는 것보다 **원거리 접지**로 두는 편이 물리에 가깝다.
    """
    m = np.zeros(shape, dtype=bool)
    m[0, :, :] = m[-1, :, :] = True
    m[:, 0, :] = m[:, -1, :] = True
    m[:, :, -1] = True
    return m


def solve_potential(A, q, shape, tol=1e-10, maxiter=5000):
    """A φ = -q. 원거리 5면은 φ=0 (Dirichlet), 피부면은 Neumann.

    Dirichlet이 하나라도 있으면 계가 정부호가 되어 영공간 처리가 필요 없고
    CG 수렴도 빠르다.
    """
    fixed = far_boundary_mask(shape).ravel()
    free = ~fixed
    b = (-q.ravel().astype(float))[free]
    Aff = A[free][:, free].tocsr()
    d = np.abs(Aff.diagonal()); d[d == 0] = 1.0
    M = LinearOperator(Aff.shape, matvec=lambda v: v / d)
    try:
        x, info = cg(Aff, b, rtol=tol, maxiter=maxiter, M=M)
    except TypeError:                  # scipy < 1.12
        x, info = cg(Aff, b, tol=tol, maxiter=maxiter, M=M)
    phi = np.zeros(fixed.size)
    phi[free] = x
    return phi.reshape(shape), info


def dipole_source(g: Geometry, x_mm, y_mm, z_mm, moment=1.0, axis=0):
    """연골 두께 방향 전류 쌍극자. 인접 두 셀에 +I / -I 를 넣는다."""
    q = np.zeros(g.shape)
    i = int(np.clip(x_mm / g.h, 1, g.nx - 2))
    j = int(np.clip(y_mm / g.h, 1, g.ny - 2))
    k = int(np.clip(z_mm / g.h, 1, g.nz - 2))
    vol = (g.h * 1e-3) ** 3
    p = [i, j, k]; m = [i, j, k]
    p[axis] += 1; m[axis] -= 1
    q[tuple(p)] = +moment / vol
    q[tuple(m)] = -moment / vol
    return q


def surface_map(phi: np.ndarray) -> np.ndarray:
    """피부 표면(z=0 층)의 전위."""
    return phi[:, :, 0]


# ==================== 지표 ====================

def psf_width(surf, g: Geometry) -> dict:
    """표면 전위의 확산 폭. 절대값 최대의 절반이 되는 폭(FWHM 유사)."""
    a = np.abs(surf)
    pk = a.max()
    if pk <= 0:
        return {'fwhm_x_mm': None, 'fwhm_y_mm': None, 'peak': 0.0}
    i, j = np.unravel_index(a.argmax(), a.shape)
    out = {'peak': float(surf[i, j]),
           'peak_x_mm': float((i + 0.5) * g.h), 'peak_y_mm': float((j + 0.5) * g.h)}
    for lab, prof, n in (('x', a[:, j], g.nx), ('y', a[i, :], g.ny)):
        above = prof >= pk / 2
        out[f'fwhm_{lab}_mm'] = float(above.sum() * g.h)
    return out


def effective_rank(L: np.ndarray, snr_db: float = 20.0) -> dict:
    """lead field의 특이값 중 잡음 바닥 위에 있는 개수.

    이것이 'N개 전극으로 복원 가능한 소스 성분 수'이며, 전극을 늘려도 이 수가
    늘지 않으면 늘릴 이유가 (조건수 개선 외에는) 없다.
    """
    s = np.linalg.svd(L, compute_uv=False)
    thr = s[0] * 10 ** (-snr_db / 20.0)
    return {'singular_values': s.tolist(),
            'threshold': float(thr),
            'effective_rank': int((s > thr).sum()),
            'cond': float(s[0] / s[-1]) if s[-1] > 0 else float('inf')}


# ==================== 소스 파셀 · 전극 ====================

def cartilage_parcels(g: Geometry, n_par: int, side: str = 'both') -> list:
    """관절 간극 안의 연골을 y(내외측) 방향으로 n등분한 파셀 중심 좌표."""
    ys = np.linspace(0.15, 0.85, n_par) * g.ny * g.h
    z = g.z_bone + 6.0
    out = []
    for y in ys:
        if side in ('both', 'femoral'):
            out.append((g.x_joint - g.gap / 2 + g.cart / 2, y, z, 'fem'))
        if side in ('both', 'tibial'):
            out.append((g.x_joint + g.gap / 2 - g.cart / 2, y, z, 'tib'))
    return out


def electrode_grid(g: Geometry, n_x: int, n_y: int, span_x=60.0, span_y=80.0) -> list:
    """관절선을 중심으로 한 피부 전극 격자."""
    xs = g.x_joint + np.linspace(-span_x / 2, span_x / 2, n_x) if n_x > 1 else [g.x_joint]
    ys = np.linspace(-span_y / 2, span_y / 2, n_y) + g.ny * g.h / 2
    return [(float(x), float(y)) for x in xs for y in ys]


def sample_at(surf, g: Geometry, electrodes) -> np.ndarray:
    return np.array([surf[int(np.clip(x / g.h, 0, g.nx - 1)),
                          int(np.clip(y / g.h, 0, g.ny - 1))] for x, y in electrodes])


def _prepare(g: Geometry):
    lab = build_labels(g)
    sig = build_sigma(lab)
    return build_operator(sig, g.h), lab


# ==================== 서브커맨드 ====================

def cmd_map(args):
    g = Geometry(d_fat=args.fat)
    A, _ = _prepare(g)
    q = dipole_source(g, g.x_joint, g.ny * g.h / 2, g.z_bone + 6.0)
    phi, info = solve_potential(A, q, g.shape)
    surf = surface_map(phi)
    m = psf_width(surf, g)
    print(f"CG info={info}  (0이면 수렴)")
    print(f"표면 최대 전위 {m['peak']:.4g} (임의 단위), 위치 x={m['peak_x_mm']:.0f} y={m['peak_y_mm']:.0f} mm")
    print(f"**PSF 폭  x {m['fwhm_x_mm']:.0f} mm · y {m['fwhm_y_mm']:.0f} mm**")
    print(f"→ 전극 간격이 이보다 좁으면 두 전극은 사실상 같은 것을 본다")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / 'surface_map.npy', surf)
    json.dump({'geometry': asdict(g), 'psf': m, 'sigma': CONDUCTIVITY},
              open(OUT_DIR / 'map.json', 'w'), ensure_ascii=False, indent=2)
    print(f"→ {OUT_DIR}/map.json")


def cmd_sweep(args):
    rows = []
    print(f"{'지방(mm)':>9} {'소스깊이(mm)':>12} {'표면최대':>12} {'PSF-x(mm)':>10}")
    for fat in args.fat_list:
        for dz in args.depth_list:
            g = Geometry(d_fat=fat)
            A, _ = _prepare(g)
            q = dipole_source(g, g.x_joint, g.ny * g.h / 2, g.z_bone + dz)
            phi, _ = solve_potential(A, q, g.shape)
            m = psf_width(surface_map(phi), g)
            rows.append({'fat_mm': fat, 'src_depth_mm': dz,
                         'peak': abs(m['peak']), 'fwhm_x_mm': m['fwhm_x_mm']})
            print(f"{fat:9.1f} {dz:12.1f} {abs(m['peak']):12.4g} {m['fwhm_x_mm']:10.0f}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import csv
    with open(OUT_DIR / 'sweep.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    base = [r for r in rows if r['fat_mm'] == args.fat_list[0]][0]['peak']
    worst = [r for r in rows if r['fat_mm'] == args.fat_list[-1]][0]['peak']
    print(f"\n지방 {args.fat_list[0]}→{args.fat_list[-1]} mm 에서 진폭 비 = {worst / base:.3f}")
    print(f"→ {OUT_DIR}/sweep.csv")


def cmd_shield(args):
    """골 차폐 가설: 뼈를 근육으로 치환했을 때 표면 전위가 얼마나 커지는가.

    **같은 위치끼리** 비교해야 한다. 쌍극자가 사지 장축(피부에 접선) 방향이면
    소스 바로 위가 전위 null이므로, 그 지점 값만 보면 차폐가 아니라 lobe 구조를
    재게 된다(첫 구현의 오류).

    소스 방향은 불확실하므로 두 가지를 모두 돌린다.
      tangential : 연골 압박축(사지 장축) = 피부에 접선. 표면에 양극성 lobe 2개
      normal     : 피부 법선 방향. 소스 바로 위가 최대
    """
    print("골 차폐 가설: 관절선이 이기는 이유가 거리인가, 뼈를 우회하지 않는 창인가\n")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res = {}
    yc = None
    for axis, aname in ((0, 'tangential(장축)'), (2, 'normal(피부법선)')):
        maps = {}
        for no_bone in (False, True):
            g = Geometry(d_fat=args.fat, no_bone=no_bone)
            yc = g.ny * g.h / 2
            A, _ = _prepare(g)
            q = dipole_source(g, g.x_joint, yc, g.z_bone + 6.0, axis=axis)
            phi, _ = solve_potential(A, q, g.shape)
            maps['no_bone' if no_bone else 'bone'] = (surface_map(phi), g)

        row = {}
        for label, dx in (('관절선(0mm)', 0.0), (f'+{args.offset:.0f}mm(뼈 위)', args.offset)):
            vb = abs(sample_at(*maps['bone'][::-1][::-1][:1], [(maps['bone'][1].x_joint + dx, yc)])[0]) \
                if False else abs(sample_at(maps['bone'][0], maps['bone'][1],
                                            [(maps['bone'][1].x_joint + dx, yc)])[0])
            vn = abs(sample_at(maps['no_bone'][0], maps['no_bone'][1],
                               [(maps['no_bone'][1].x_joint + dx, yc)])[0])
            row[label] = {'뼈있음': vb, '뼈없음': vn, '차폐배수': vn / vb if vb else float('inf')}
        pb = np.abs(maps['bone'][0]).max()
        pn = np.abs(maps['no_bone'][0]).max()
        row['표면 최대'] = {'뼈있음': float(pb), '뼈없음': float(pn),
                         '차폐배수': float(pn / pb) if pb else float('inf')}
        res[aname] = row

        print(f"[소스 방향: {aname}]")
        for k, v in row.items():
            print(f"  {k:>16s}  뼈있음 {v['뼈있음']:.4g} · 뼈없음 {v['뼈없음']:.4g} "
                  f"· **차폐 {v['차폐배수']:.1f}배**")
        print()

    json.dump(res, open(OUT_DIR / 'shield.json', 'w'), ensure_ascii=False, indent=2)
    print("해석: 차폐배수가 위치마다 크게 다르면(특히 뼈 위에서 훨씬 크면)")
    print("      관절선 우세는 거리가 아니라 **창(window) 효과**다.")
    print(f"→ {OUT_DIR}/shield.json")


def cmd_rank(args):
    """전극 수별 유효 rank. '왜 16채널인가'에 대한 답."""
    g = Geometry(d_fat=args.fat)
    A, _ = _prepare(g)
    parcels = cartilage_parcels(g, args.parcels)
    print(f"소스 파셀 {len(parcels)}개 (연골 내외측 {args.parcels}등분 × 대퇴/경골)")

    cols = []
    for (x, y, z, side) in parcels:
        q = dipole_source(g, x, y, z)
        phi, _ = solve_potential(A, q, g.shape)
        cols.append(surface_map(phi))
    print("forward 계산 완료\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    print(f"{'전극수':>7} {'유효rank':>9} {'조건수':>12}  특이값(상위 6, σ1 대비)")
    for (nx_e, ny_e) in args.layouts:
        el = electrode_grid(g, nx_e, ny_e)
        L = np.stack([sample_at(s, g, el) for s in cols], axis=1)
        L = L - L.mean(axis=0, keepdims=True)       # 공통평균참조 (실제 기록과 동일)
        r = effective_rank(L, args.snr)
        s = np.array(r['singular_values']); s = s / s[0]
        out[f'{nx_e}x{ny_e}={len(el)}'] = r
        print(f"{len(el):7d} {r['effective_rank']:9d} {r['cond']:12.3g}  "
              + " ".join(f"{v:.3f}" for v in s[:6]))
    json.dump(out, open(OUT_DIR / 'rank.json', 'w'), ensure_ascii=False, indent=2)
    print(f"\n전극을 늘려도 유효 rank가 늘지 않으면, 추가 전극은 소스 분해능이 아니라"
          f"\n조건수·SNR·재참조 자유도를 위한 것이다. 그렇게 논문에 써야 한다.")
    print(f"→ {OUT_DIR}/rank.json")


def cmd_montage(args):
    """OBJ2 montage를 모델에 심고 채널별 예측을 실측과 대조한다.

    이 시뮬레이터는 지금까지 예측만 했고 검증이 없었다. 실측이 이미 있는 두 비를
    모델이 재현하는지 보면, 모델의 타당성과 소스 방향을 동시에 검정할 수 있다.

    **montage** (Ch.4 Table 4.1). 근위 줄 4개(Ch1,2,5,6)와 관절선 줄 4개(Ch3,4,7,8),
    내측 열(Ch1-4)과 외측 열(Ch5-8). 레퍼런스는 경골 원위 10 cm.
    모델에서는 원거리 경계가 φ=0 이므로 원위 레퍼런스가 자연히 근사된다.

    **실측 대조값** (70,566 이벤트)
        내측/외측 기울기비  1.85 / 0.77 = 2.40
        근위/관절선 기울기비 1.42 / 1.20 = 1.18

    **두 질문.**

    (a) 기하만으로 내외측 2.4배가 나오는가.
        좌우 소스를 **같은 크기**로 주고 비를 잰다. 해부학적 비대칭(내측 지방이 얇음)을
        넣어도 재현되지 않으면, 실측 2.4배는 **소스 강도 차이**로 귀속된다.
        그 경우 Mayr 2026의 접촉면적 증가비 15.0/6.7 = 2.24 와 대조할 수 있다.

    (b) 근위/관절선 비가 소스 방향을 가르는가.
        접선 쌍극자는 관절선이 null이라 비가 크게, 법선은 관절선이 최대라 비가 1보다
        작게 나와야 한다. 실측 1.18이 어느 쪽에 가까운지 본다.
    """
    OBS_ML, OBS_PJ = 2.40, 1.18
    yc = None
    print("실측 대조: 내측/외측 = 2.40 · 근위/관절선 = 1.18\n")
    out = {}
    for tag, (fm, fl) in (('지방 대칭', (None, None)),
                          ('지방 비대칭(내 3mm / 외 9mm)', (3.0, 9.0)),
                          ('지방 비대칭(내 4mm / 외 12mm)', (4.0, 12.0))):
        for axis, aname in ((0, 'tangential'), (2, 'normal')):
            g = Geometry(nx=90, h=2.0, x_joint=60.0, d_fat_med=fm, d_fat_lat=fl)
            yc = g.ny * g.h / 2
            A, _ = _prepare(g)
            y_med, y_lat = yc - 25.0, yc + 25.0
            el = {'prox_med': (g.x_joint - 25.0, y_med),
                  'prox_lat': (g.x_joint - 25.0, y_lat),
                  'jl_med':   (g.x_joint, y_med),
                  'jl_lat':   (g.x_joint, y_lat)}
            v = {}
            for src_name, y_src in (('med', y_med), ('lat', y_lat)):
                q = dipole_source(g, g.x_joint, y_src, g.z_bone + 6.0, axis=axis)
                phi, _ = solve_potential(A, q, g.shape)
                surf = surface_map(phi)
                for k, pos in el.items():
                    v[f'{src_name}->{k}'] = abs(sample_at(surf, g, [pos])[0])
            # 내측 소스가 내측 전극에, 외측 소스가 외측 전극에 만드는 값의 비
            ml = ((v['med->jl_med'] + v['med->prox_med']) /
                  (v['lat->jl_lat'] + v['lat->prox_lat']))
            # 근위/관절선 (내측 소스 기준)
            pj = v['med->prox_med'] / v['med->jl_med'] if v['med->jl_med'] else float('inf')
            out[f'{tag} | {aname}'] = {'medial_lateral': ml, 'prox_jointline': pj,
                                       'raw': {k: float(x) for k, x in v.items()}}
            print(f"[{tag:28s} | {aname:10s}]  내측/외측 {ml:5.2f}  ·  근위/관절선 {pj:5.2f}")
        print()

    print("=" * 68)
    print("해석")
    print("=" * 68)
    best_ml = max((abs(np.log(d['medial_lateral'] / OBS_ML)), k)
                  for k, d in out.items())
    ml_vals = {k: d['medial_lateral'] for k, d in out.items()}
    pj_vals = {k: d['prox_jointline'] for k, d in out.items()}
    print(f"내측/외측: 모델 범위 {min(ml_vals.values()):.2f}~{max(ml_vals.values()):.2f} "
          f"· 실측 {OBS_ML:.2f}")
    if max(ml_vals.values()) < OBS_ML * 0.7:
        print("  → **기하만으로는 실측 2.40을 못 만든다.** 차이는 소스 강도로 귀속되며,")
        print("     Mayr 2026의 접촉면적 증가비 2.24와 대조할 수 있다.")
    else:
        print("  → 기하가 상당 부분을 설명한다. 소스 귀속에 신중할 것.")
    print(f"\n근위/관절선: 실측 {OBS_PJ:.2f}")
    for k, val in pj_vals.items():
        print(f"  {k:45s} {val:5.2f}")
    print("  → 접선은 관절선이 null이라 비가 크고, 법선은 1 이하여야 한다.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump({'observed': {'medial_lateral': OBS_ML, 'prox_jointline': OBS_PJ},
               'model': out}, open(OUT_DIR / 'montage.json', 'w'),
              ensure_ascii=False, indent=2)
    print(f"\n→ {OUT_DIR}/montage.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sigma-json', help='전도도 값 교체 (JSON)')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('map', help='표면 전위 지도 + PSF 폭')
    p.add_argument('--fat', type=float, default=6.0)
    p.set_defaults(func=cmd_map)

    p = sub.add_parser('sweep', help='지방 두께·소스 깊이 스윕')
    p.add_argument('--fat-list', type=float, nargs='+', default=[2, 4, 6, 10],
                   help='피하지방 두께. z_bone(16mm)을 넘기면 근육층이 사라져 조성 변화와 섞인다')
    p.add_argument('--depth-list', type=float, nargs='+', default=[2.0, 6.0, 12.0],
                   help='관절 간극 안에서 소스가 놓이는 깊이(뼈 앞면 기준)')
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser('shield', help='골 차폐 가설 검정')
    p.add_argument('--fat', type=float, default=6.0)
    p.add_argument('--offset', type=float, default=30.0, help='관절선에서 옆으로(mm)')
    p.set_defaults(func=cmd_shield)

    p = sub.add_parser('rank', help='lead field SVD → 유효 rank')
    p.add_argument('--fat', type=float, default=6.0)
    p.add_argument('--parcels', type=int, default=4, help='내외측 등분 수')
    p.add_argument('--snr', type=float, default=20.0, help='잡음 바닥 (dB)')
    p.add_argument('--layouts', type=int, nargs='+', action='append',
                   default=None, metavar='NX NY')
    p.set_defaults(func=cmd_rank)

    p = sub.add_parser('montage', help='OBJ2 montage 예측 대 실측 대조 (모델 검증)')
    p.set_defaults(func=cmd_montage)

    p = sub.add_parser('all', help='map → sweep → shield → rank 일괄')
    p.set_defaults(func=None)

    a = ap.parse_args()
    if a.sigma_json:
        CONDUCTIVITY.update(json.load(open(a.sigma_json)))

    if a.cmd == 'all':
        for name, fn, kw in (('MAP', cmd_map, dict(fat=6.0)),
                             ('SWEEP', cmd_sweep, dict(fat_list=[2, 4, 6, 10, 16],
                                                       depth_list=[6.0])),
                             ('SHIELD', cmd_shield, dict(fat=6.0, offset=30.0)),
                             ('RANK', cmd_rank, dict(fat=6.0, parcels=4, snr=20.0,
                                                     layouts=None))):
            print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
            fn(argparse.Namespace(**kw))
        return

    if a.cmd == 'rank' and not a.layouts:
        a.layouts = [(1, 4), (2, 4), (2, 8), (4, 4), (4, 8)]
    elif a.cmd == 'rank':
        a.layouts = [tuple(v) for v in a.layouts]
    a.func(a)


if __name__ == '__main__':
    main()
