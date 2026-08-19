#!/usr/bin/env python3
"""부하 유지 구간의 EAG 거동 — streaming potential 대 고정전하밀도(FCD) 판별.

**왜 보는가.** EAG의 기전으로 두 가지가 제안된다.

  streaming potential : 간질액 **유동**이 이온을 나른다. 유동은 변형률(dε/dt)에
        비례하므로, 하중을 유지해 유동이 멎으면 전위가 **감쇠**해야 한다.
  고정전하밀도(FCD)   : 압박으로 물이 빠져 프로테오글리칸 음전하가 **농축**된다.
        이는 변형 자체(ε)의 함수이므로 하중이 유지되는 동안 **유지**된다.

따라서 유지 구간의 거동이 두 기전을 가른다.

  감쇠한다  → streaming 우세 (또는 poroelastic 시상수가 유지시간과 같은 규모)
  유지된다  → FCD 우세, 또는 streaming이되 시상수가 유지시간보다 훨씬 길다

**시상수 주의.** 연골의 poroelastic 시상수는 τ ≈ h²/(H_A·k) 규모다.
h≈2 mm, H_A≈0.5 MPa, k≈1e-15 m⁴/N·s를 넣으면 τ가 10³ s 이상이 된다.
**즉 5초 유지에서 감쇠가 없는 것은 FCD의 증거가 아니라, streaming이어도 예상되는
결과일 수 있다.** 감쇠 부재만으로 기전을 단정하면 안 된다. 그래서 이 도구는
두 번째 판별도 함께 낸다.

**두 번째 판별: 크기 대 속도.**

  streaming → 진폭이 부하 **변화 속도**에 의존해야 한다
  FCD       → 진폭이 부하 **크기**에만 의존해야 한다

두 예측변수를 같은 모형에 넣어 어느 쪽이 살아남는지 본다. 둘이 공선이면
(모든 전이가 비슷한 속도로 일어나면) 분리되지 않으므로, 속도의 분산을 먼저 보고한다.

**전처리 경고.** 파이프라인은 트레이스 전체에 선형 `detrend`를 건다. 유지 구간
수 초의 지수 감쇠는 전역 선형 성분과 달라 대부분 살아남지만, **완전히 무해하지는
않다.** `--raw` 로 detrend 없이 한 번 더 돌려 결과가 바뀌는지 확인할 것.

사용:
    python3 plateau_decay.py hold   --n 20     # 유지 구간 거동
    python3 plateau_decay.py rate              # 크기 대 속도 판별
"""

import argparse
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT_DIR = Path('result/mechanism')
PARAMS = Path('result/phase1_params/grf_triggered_params.csv')


def _quiet():
    import contextlib

    @contextlib.contextmanager
    def cm():
        old = sys.stdout
        sys.stdout = io.StringIO()
        try:
            yield
        finally:
            sys.stdout = old
    return cm()


# ==================== hold ====================

def cmd_hold(args):
    """부하 유지 구간에서 EAG가 감쇠하는지 본다.

    유지 구간은 '부하 시작 이벤트의 offset' 부터 '이탈 이벤트의 onset' 까지다.
    즉 EAG 전이가 끝나고 다음 전이가 시작되기 전, 부하가 걸린 채로 머무는 창이다.
    각 창을 시작값 기준으로 정규화하고 (a) 창 전체의 상대 변화, (b) 단일지수 적합의
    시상수를 낸다.
    """
    import pandas as pd
    from scipy.signal import detrend
    from scipy.optimize import curve_fit
    from sync_analyzer import find_session_pair, SyncAnalyzer
    import grf_triggered_annotator as G

    if not PARAMS.exists():
        raise SystemExit(f"{PARAMS} 없음")
    p = pd.read_csv(PARAMS, low_memory=False)
    p = p[(p['matched'] == True) & (p['accepted'] == True) & p['load_pct'].notna()]

    # 세션 목록 (파라미터에 등장하는 것 중 앞에서 n개)
    keys = p[['subject', 'session']].drop_duplicates().values.tolist()
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(keys))[:args.n]
    keys = [keys[i] for i in idx]

    from offset_app import scan_sessions
    with _quiet():
        allsess = {(s['subject'], s['session']): s['dir'] for s in scan_sessions()}

    rows = []
    for i, (subj, sess) in enumerate(keys, 1):
        sdir = allsess.get((subj, sess))
        if not sdir:
            continue
        print(f"[{i}/{len(keys)}] {subj} / {sess}", flush=True)
        try:
            pair = find_session_pair(sdir)
            if pair is None:
                continue
            with _quiet():
                sa = SyncAnalyzer(pair)
                off, trans, _s, _g = G.compute_offset(sa, 0, True)
            te = sa.unified_time_eag - off.residual
            fs = sa.eag.sample_rate
            sub = p[(p.subject == subj) & (p.session == sess)]

            for ch, gch in sub.groupby('channel'):
                sig = sa.eag_filtered[:, int(ch) - 1]
                if not args.raw:
                    sig = detrend(sig)
                for cyc, gc in gch.groupby('cycle_id'):
                    on = gc[gc.event_kind == 'on']
                    offv = gc[gc.event_kind == 'off']
                    if len(on) != 1 or len(offv) != 1:
                        continue
                    t0 = float(on.offset_time.iloc[0])     # 전이 끝
                    t1 = float(offv.onset_time.iloc[0])    # 다음 전이 시작
                    if not np.isfinite(t0) or not np.isfinite(t1) or t1 - t0 < args.min_hold:
                        continue
                    m = (te >= t0) & (te <= t1)
                    y = sig[m]
                    t = te[m] - t0
                    if len(y) < 10:
                        continue
                    amp = float(on.amplitude.iloc[0])       # 전이 크기 (부호 포함)
                    y0 = y[0]
                    # 전이 크기로 정규화한 상대 변화. +1 이면 전이만큼 되돌아왔다는 뜻
                    recov = (y[-1] - y0) / (-amp) if amp else np.nan
                    # 단일지수 적합
                    tau = np.nan
                    try:
                        f = lambda tt, a, tau_, c: a * np.exp(-tt / tau_) + c
                        pr, _ = curve_fit(f, t, y, p0=[y[0] - y[-1], max(1.0, (t1 - t0) / 2), y[-1]],
                                          maxfev=4000)
                        if 0.05 < pr[1] < 10 * (t1 - t0):
                            tau = float(pr[1])
                    except Exception:
                        pass
                    rows.append(dict(subject=subj, session=sess, channel=int(ch),
                                     cycle=int(cyc), hold_s=t1 - t0,
                                     load_pct=float(on.load_pct.iloc[0]),
                                     amp=amp, recov_frac=float(recov), tau_s=tau))
        except Exception as e:
            print(f"   실패: {type(e).__name__}: {e}")

    if not rows:
        raise SystemExit("유지 구간을 하나도 못 찾았다. --min-hold 를 낮춰볼 것.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / 'hold.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    d = pd.DataFrame(rows)
    print(f"\n유지 구간 {len(d)}개 · 세션 {d.session.nunique()} · 채널 {d.channel.nunique()}")
    print(f"유지 길이 중앙값 {d.hold_s.median():.2f} s  (사분위 {d.hold_s.quantile(.25):.2f}~{d.hold_s.quantile(.75):.2f})")
    r = d.recov_frac.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"\n**되돌아온 비율** (전이 크기 대비, 1.0 = 완전 복귀)")
    print(f"  중앙값 {r.median():+.3f} · 평균 {r.mean():+.3f} · 사분위 {r.quantile(.25):+.3f}~{r.quantile(.75):+.3f}")
    print(f"  |변화| < 0.1 인 비율 = {(r.abs() < 0.1).mean():.3f}   (유지 = 감쇠 없음)")
    print(f"  > 0.3 인 비율 = {(r > 0.3).mean():.3f}   (뚜렷한 감쇠)")
    tt = d.tau_s.dropna()
    if len(tt):
        print(f"\n적합된 시상수 중앙값 {tt.median():.2f} s (n={len(tt)}, 사분위 "
              f"{tt.quantile(.25):.2f}~{tt.quantile(.75):.2f})")
    print(f"\n부하 단계별 되돌아온 비율:")
    d['step'] = pd.qcut(d.load_pct, 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    for k, g in d.groupby('step', observed=True):
        print(f"  {k} (load {g.load_pct.median():5.1f}%)  중앙 {g.recov_frac.median():+.3f}  n={len(g)}")
    print(f"\n→ {OUT_DIR}/hold.csv")
    print("\n해석 지침: 되돌아온 비율이 0 근처면 유지(FCD 또는 긴 τ), 1 근처면 완전 감쇠(streaming).")
    print("           단, docstring의 τ 논의를 반드시 함께 읽을 것.")


# ==================== rate ====================

def cmd_rate(args):
    """진폭이 부하 **크기**에 의존하나, **변화 속도**에 의존하나.

    streaming은 속도(dε/dt)에, FCD는 크기(ε)에 의존한다. 두 예측변수를 같은
    모형에 넣어 어느 쪽이 살아남는지 본다.

    ⚠️ **속도 대리변수의 순환에 주의.** 파이프라인의 `slope`는 amplitude/transition_time
    이므로 종속변수를 포함한다. 여기서는 **부하 쪽 정보만**으로 속도를 만든다:
    rate = load_pct / transition_time. transition_time은 EAG 유래라 완전히 독립은
    아니지만 amplitude와는 구성상 독립이다. 더 깨끗한 대리변수는 GRF 램프 지속시간이며,
    파라미터 CSV에 없으므로 후속 과제로 남긴다.
    """
    import pandas as pd
    import statsmodels.formula.api as smf

    p = pd.read_csv(PARAMS, low_memory=False)
    d = p[(p['matched'] == True) & (p['accepted'] == True)
          & p['load_pct'].notna() & p['amplitude'].notna()
          & p['transition_time'].notna() & (p['transition_time'] > 0)].copy()
    d['absamp'] = d.amplitude.abs()
    d['rate'] = d.load_pct / d.transition_time
    d = d[np.isfinite(d.rate)]

    print(f"이벤트 {len(d)} · 피험자 {d.subject.nunique()}")
    print(f"\n전이 시간 분포 (s): 중앙 {d.transition_time.median():.3f} · "
          f"사분위 {d.transition_time.quantile(.25):.3f}~{d.transition_time.quantile(.75):.3f} "
          f"· 변동계수 {d.transition_time.std()/d.transition_time.mean():.2f}")
    print(f"속도 분포 (%BW/s): 중앙 {d.rate.median():.1f} · "
          f"사분위 {d.rate.quantile(.25):.1f}~{d.rate.quantile(.75):.1f}")
    r = np.corrcoef(d.load_pct, d.rate)[0, 1]
    print(f"\n**크기와 속도의 상관 r = {r:.3f}**")
    if abs(r) > 0.8:
        print("  → 공선성이 심해 두 효과를 분리하기 어렵다. 결과를 조심해서 읽을 것.")
    else:
        print("  → 분리 가능한 수준.")

    d['z_load'] = (d.load_pct - d.load_pct.mean()) / d.load_pct.std()
    d['z_rate'] = (d.rate - d.rate.mean()) / d.rate.std()
    print("\n=== LMM: absamp ~ z_load + z_rate + (1|subject) ===")
    m = smf.mixedlm("absamp ~ z_load + z_rate", d, groups=d['subject']).fit()
    print(m.summary().tables[1])
    b = m.params
    print(f"\n표준화 계수:  부하 크기 {b.get('z_load', np.nan):+.2f} µV  vs  "
          f"변화 속도 {b.get('z_rate', np.nan):+.2f} µV")
    if abs(b.get('z_load', 0)) > 2 * abs(b.get('z_rate', 0)):
        print("→ **크기 우세**. FCD(변형 의존) 기여가 크다는 쪽.")
    elif abs(b.get('z_rate', 0)) > 2 * abs(b.get('z_load', 0)):
        print("→ **속도 우세**. streaming(유동 의존) 기여가 크다는 쪽.")
    else:
        print("→ 두 효과가 비슷하다. 단일 기전으로 환원되지 않는다.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / 'rate_vs_magnitude.txt', 'w', encoding='utf-8') as f:
        f.write(str(m.summary()))
    print(f"\n→ {OUT_DIR}/rate_vs_magnitude.txt")


# ==================== grfrate ====================

def _ramp_rate(t, f, t_start, t_end, f0, f1, lo=0.10, hi=0.90):
    """10~90% 램프 속도. 전적으로 GRF에서만 나온다.

    시작 plateau f0에서 끝 plateau f1로 넘어가는 구간의 10%·90% 통과 시각을 찾아
    그 사이 기울기를 낸다. 문턱을 10/90으로 잡는 것은 plateau 근처의 잡음이
    통과 시각을 크게 흔들기 때문이다.
    """
    d = f1 - f0
    if not np.isfinite(d) or abs(d) < 1e-6:
        return np.nan, np.nan
    m = (t >= t_start) & (t <= t_end)
    if m.sum() < 5:
        return np.nan, np.nan
    tt, ff = t[m], f[m]
    # 진행 방향에 무관하게 0~1로 정규화
    prog = (ff - f0) / d
    i_lo = np.flatnonzero(prog >= lo)
    if not len(i_lo):
        return np.nan, np.nan
    i_hi = np.flatnonzero(prog[i_lo[0]:] >= hi)
    if not len(i_hi):
        return np.nan, np.nan
    t10, t90 = tt[i_lo[0]], tt[i_lo[0] + i_hi[0]]
    dt = t90 - t10
    if dt <= 0:
        return np.nan, np.nan
    return abs((hi - lo) * d) * 100.0 / dt, dt      # %BW/s, 램프 지속시간(s)


def cmd_grfrate(args):
    """**GRF에서만** 유도한 부하 변화 속도를 만들고, 크기와 분리해 진폭을 설명한다.

    `rate` 서브커맨드의 속도 대리변수(load_pct/transition_time)는 transition_time을
    통해 진폭과 구성상 결합돼 있어 음의 계수라는 인공물을 냈다. 여기서는 EAG를
    전혀 쓰지 않는다. 검사측 하중분율 f(t) = 검사측힘/(좌+우) 를 만들고, 각 이벤트의
    10~90% 램프 기울기를 속도로 쓴다.

      streaming potential → 진폭이 **속도**(dε/dt)를 따라야 한다
      고정전하밀도(FCD)   → 진폭이 **크기**(ε)를 따라야 한다
    """
    import pandas as pd
    from sync_analyzer import find_session_pair, SyncAnalyzer
    import grf_triggered_annotator as G
    from offset_app import scan_sessions

    p = pd.read_csv(PARAMS, low_memory=False)
    p = p[(p['matched'] == True) & (p['accepted'] == True) & p['amplitude'].notna()]
    keys = p[['subject', 'session']].drop_duplicates().values.tolist()
    rng = np.random.default_rng(args.seed)
    keys = [keys[i] for i in rng.permutation(len(keys))[:args.n]]
    with _quiet():
        allsess = {(s['subject'], s['session']): s['dir'] for s in scan_sessions()}

    rows = []
    for i, (subj, sess) in enumerate(keys, 1):
        sdir = allsess.get((subj, sess))
        if not sdir:
            continue
        print(f"[{i}/{len(keys)}] {subj} / {sess}", flush=True)
        try:
            pair = find_session_pair(sdir)
            if pair is None:
                continue
            with _quiet():
                sa = SyncAnalyzer(pair)
                signed = G.signed_imbalance(sa.grf_left, sa.grf_right)
                tg = sa.unified_time_grf
                rest, cycles, _ = G.detect_load_cycles_expected(
                    tg, signed, sa.grf_left, sa.grf_right)
            L = np.asarray(sa.grf_left, float); R = np.asarray(sa.grf_right, float)
            tot = L + R
            for c in cycles:
                if not np.isfinite(c.load_ratio) or not np.isfinite(c.rest_ratio):
                    continue
                test = R if c.test_side == 'R' else L
                with np.errstate(invalid='ignore', divide='ignore'):
                    f = np.where(tot > 1e-6, test / tot, np.nan)
                r_on, d_on = _ramp_rate(tg, f, c.onset_time - 0.5, c.offset_time,
                                        c.rest_ratio, c.load_ratio)
                r_off, d_off = _ramp_rate(tg, f, c.offset_time - 0.5, c.end_time + 0.5,
                                          c.load_ratio, c.rest_ratio)
                for kind, rr, dd in (('on', r_on, d_on), ('off', r_off, d_off)):
                    rows.append(dict(subject=subj, session=sess, cycle_id=c.cycle_id,
                                     event_kind=kind, grf_rate=rr, grf_ramp_s=dd,
                                     delta_pct=abs(c.load_ratio - c.rest_ratio) * 100.0,
                                     load_pct=c.load_pct))
        except Exception as e:
            print(f"   실패: {type(e).__name__}: {e}")

    if not rows:
        raise SystemExit("램프를 하나도 못 구했다.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = pd.DataFrame(rows)
    g.to_csv(OUT_DIR / 'grf_rate.csv', index=False)

    d = p.merge(g, on=['subject', 'session', 'cycle_id', 'event_kind'],
                how='inner', suffixes=('', '_grf'))
    d = d[d.grf_rate.notna() & (d.grf_rate > 0)].copy()
    d['absamp'] = d.amplitude.abs()
    print(f"\n병합된 이벤트 {len(d)} · 피험자 {d.subject.nunique()} · 세션 {d.session.nunique()}")
    print(f"GRF 램프 지속 (s): 중앙 {d.grf_ramp_s.median():.3f} · "
          f"사분위 {d.grf_ramp_s.quantile(.25):.3f}~{d.grf_ramp_s.quantile(.75):.3f}")
    print(f"GRF 속도 (%BW/s): 중앙 {d.grf_rate.median():.1f} · "
          f"사분위 {d.grf_rate.quantile(.25):.1f}~{d.grf_rate.quantile(.75):.1f} "
          f"· 변동계수 {d.grf_rate.std()/d.grf_rate.mean():.2f}")

    r = np.corrcoef(d.delta_pct, d.grf_rate)[0, 1]
    print(f"\n**크기(Δ%BW)와 속도(%BW/s)의 상관 r = {r:.3f}**")
    print("  → " + ("공선성이 심하다. 분리 해석 금지." if abs(r) > 0.8 else "분리 가능."))

    import statsmodels.formula.api as smf
    for col in ('delta_pct', 'grf_rate', 'grf_ramp_s'):
        d['z_' + col] = (d[col] - d[col].mean()) / d[col].std()
    print("\n=== LMM: absamp ~ z_delta_pct + z_grf_rate + (1|subject) ===")
    m = smf.mixedlm("absamp ~ z_delta_pct + z_grf_rate", d, groups=d['subject']).fit()
    print(m.summary().tables[1])
    b_mag = m.params.get('z_delta_pct', np.nan); b_rate = m.params.get('z_grf_rate', np.nan)
    print(f"\n표준화 계수:  크기 {b_mag:+.2f} µV  ·  속도 {b_rate:+.2f} µV")
    ratio = abs(b_mag) / abs(b_rate) if b_rate else np.inf
    print(f"크기/속도 = {ratio:.2f}")
    if ratio > 2:
        print("→ **크기 우세**. 변형 자체에 의존 = FCD 기여가 크다는 쪽.")
    elif ratio < 0.5:
        print("→ **속도 우세**. 유동에 의존 = streaming 기여가 크다는 쪽.")
    else:
        print("→ 두 항이 비슷하다. 단일 기전으로 환원되지 않는다.")
    print("\n⚠️ 속도가 크기와 강하게 얽혀 있으면(위 r 확인) 위 판정을 신뢰하지 말 것.")

    with open(OUT_DIR / 'grf_rate_lmm.txt', 'w', encoding='utf-8') as f:
        f.write(str(m.summary()))
    print(f"\n→ {OUT_DIR}/grf_rate.csv · {OUT_DIR}/grf_rate_lmm.txt")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('hold', help='유지 구간 감쇠')
    p.add_argument('--n', type=int, default=20, help='세션 표본 수')
    p.add_argument('--seed', type=int, default=20260819)
    p.add_argument('--min-hold', type=float, default=1.0, help='최소 유지 길이 (s)')
    p.add_argument('--raw', action='store_true', help='detrend 없이')
    p.set_defaults(func=cmd_hold)

    p = sub.add_parser('rate', help='크기 대 속도 판별 (EAG 유래 속도 — 오염됨, 참고용)')
    p.set_defaults(func=cmd_rate)

    p = sub.add_parser('grfrate', help='크기 대 속도 판별 (GRF 유래 속도 — 이쪽을 쓸 것)')
    p.add_argument('--n', type=int, default=60, help='세션 표본 수')
    p.add_argument('--seed', type=int, default=20260819)
    p.set_defaults(func=cmd_grfrate)

    a = ap.parse_args()
    a.func(a)


if __name__ == '__main__':
    main()
