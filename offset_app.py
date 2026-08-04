"""
Offset Alignment App — 브라우저 GUI로 GRF/EAG 위의 점을 직접 찍어 offset을 확정.

REVIEW_WORKFLOW Step 1(②③)의 "안 겹치면 offset 값을 판단 → --set" 과정을 대체한다.
눈금을 읽어 값을 계산할 필요 없이, **GRF 트레이스의 변곡점 1개 + EAG 트레이스의
대응 변곡점 1개를 클릭**하면 두 점이 겹치는 offset이 자동 계산되어 즉시 미리보기로
반영된다. 여러 쌍을 찍으면 중앙값(median)을 쓰므로 한 쌍이 부정확해도 강건하다.

좌표 규약 (grf_triggered_annotator와 동일):
  te_corr = te - residual,  corrected_offset = auto_offset + residual
  → GRF의 점 g 와 EAG의 점 e(te_corr 프레임)를 겹치려면
    new_offset = corrected_offset + (e - g)
  (offset 증가 = GRF를 오른쪽으로 = EAG를 왼쪽으로)

manual_offsets.json에 저장 → SyncAnalyzer가 자동 조회, parameter_extractor 재실행 시 반영.
수동 offset이 있는 세션은 residual 재계산을 끄고(recompute=False) 불러오므로
화면에서 본 정렬이 그대로 파이프라인 정렬이다 (parameter_extractor와 동일한 규칙).

⚠️ 안전: api-server(port 3002)와 무관한 별도 포트(기본 8766)만 사용. 3002 금지.

실행:
  python3 offset_app.py                      # http://127.0.0.1:8766
  python3 offset_app.py --host 0.0.0.0 --port 8766
외부(DDNS) 접속: code-server 내장 포트 프록시 경유 — 끝 슬래시 필수
  http://<code-server 호스트>/proxy/8766/     (예: medibb.synology.me:18440)
  API를 문서 기준 상대경로로 호출하므로 프록시 prefix 아래에서도 동작한다.
브라우저에서:
  - worklist/세션 드롭다운 선택(또는 경로 직접 입력) → Load
  - GRF 패널에서 변곡점 클릭 → EAG 패널에서 대응 변곡점 클릭 → 쌍 성립, 미리보기 이동
  - 휠=확대/축소, 드래그=이동, snap 체크 시 근처 변곡점으로 자동 흡착
  - 잘 겹치면 Save (manual_offsets.json 확정) / Clear manual (자동 복귀)
"""

import argparse
import contextlib
import csv
import io
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from sync_analyzer import find_session_pair, SyncAnalyzer
import grf_triggered_annotator as G
from eag_analyzer import get_data_dir
from offset_manager import (set_manual_offset, clear_manual_offset,
                            list_all_offsets, get_manual_offset)
import exclusion_store

MAX_POINTS = 20000    # 표시 다운샘플 목표 점 수 (클릭 정밀도 확보용으로 넉넉히)


@contextlib.contextmanager
def _quiet():
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old


def _ds(a, n=MAX_POINTS):
    a = np.asarray(a)
    step = max(1, len(a) // n)
    return a[::step]


def build_data(session_dir: str, channel: int = 1) -> dict:
    """세션 로드 → GRF/EAG 트레이스 + offset 진단 + match 프로파일."""
    from scipy.signal import detrend

    pair = find_session_pair(session_dir)
    if pair is None:
        raise FileNotFoundError(f"EAG+GRF 쌍 없음: {session_dir}")

    manual = get_manual_offset(pair.subject_name, pair.session_name)
    # 수동 offset이 있으면 그것이 최종값 → residual 재계산 금지 (parameter_extractor와 동일)
    recompute = manual is None

    with _quiet():
        sa = SyncAnalyzer(pair)
        off, trans, signed, grf_t = G.compute_offset(sa, channel - 1, recompute)

    te = sa.unified_time_eag - off.residual          # 보정된 EAG 시간축 (표시 프레임)
    eag = detrend(sa.eag_filtered[:, channel - 1])
    tg = sa.unified_time_grf

    edges = G.detect_eag_edges(te, eag, fs=sa.eag.sample_rate)
    eag_edge_c = np.array([e[0] for e in edges]) if edges else np.array([])
    # 프로파일은 미보정 프레임의 edge 열 기준 (compute_offset과 동일한 residual 좌표계)
    cand, prof, best_res, margin = G.offset_match_profile(grf_t, eag_edge_c + off.residual)

    r1 = lambda a: [round(float(x), 1) for x in a]
    r3 = lambda a: [round(float(x), 3) for x in a]

    exc = exclusion_store.is_excluded(pair.subject_name, pair.session_name)
    return {
        'session_dir': str(Path(session_dir)),
        'excluded': exc,
        'subject': pair.subject_name, 'session': pair.session_name, 'channel': channel,
        'n_channels': int(sa.eag_filtered.shape[1]),
        'auto_offset': round(float(off.auto_offset), 3),
        'residual': round(float(off.residual), 3),
        'corrected_offset': round(float(off.corrected_offset), 3),
        'method': off.method,
        'match_auto': round(float(off.match_rate_auto), 3),
        'match_corrected': round(float(off.match_rate_corrected), 3),
        'needs_review': bool(off.needs_review), 'reason': off.review_reason,
        'has_manual': manual is not None,
        'manual_offset': None if manual is None else round(float(manual), 3),
        'recomputed': bool(recompute),
        'te': r3(_ds(te)), 'eag': r1(_ds(eag)),
        'grf_t': r3(_ds(tg)), 'grf_signed': r3(_ds(signed)),
        'trans': r3(grf_t),
        'eag_edges': r3(eag_edge_c),
        'prof_res': r3(cand), 'prof_mr': r3(prof),
        'best_res': None if not np.isfinite(best_res) else round(float(best_res), 3),
        'margin': None if not np.isfinite(margin) else round(float(margin), 3),
    }


# ==================== 세션 탐색 ====================

def scan_sessions() -> list:
    """평면 미러 아래 모든 세션 (subject, session, dir) — 가벼운 파일명 스캔.

    subject를 세션 폴더의 부모 이름으로 잡으므로 반드시 평면 미러를 봐야 한다.
    원본 data/는 부모가 조건 폴더('1. Side')라 manual_offsets 키와 어긋난다.
    """
    out = []
    for p in sorted(Path(get_data_dir()).rglob('BrainFlow-RAW_*.csv')):
        d = p.parent
        name = d.name
        sess = name.rsplit('-', 1)[-1] if '-' in name else name
        out.append({'subject': d.parent.name, 'session': sess, 'dir': str(d)})
    # 중복 제거 (한 폴더에 RAW가 여러 개인 경우)
    seen, uniq = set(), []
    for r in out:
        if r['dir'] in seen:
            continue
        seen.add(r['dir'])
        uniq.append(r)
    return uniq


def load_worklist() -> list:
    p = Path('result/offset_review/worklist.csv')
    if not p.exists():
        return []
    rows = []
    with open(p, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append({'subject': row.get('subject', ''),
                         'session': row.get('session', ''),
                         'reason': row.get('reason', '')})
    return rows


def find_session_dir(subject: str, session: str) -> str:
    for r in scan_sessions():
        if r['subject'] == subject and r['session'] == session:
            return r['dir']
    return ''


def session_index() -> list:
    """드롭다운용: 전체 세션 + worklist 사유 + 수동확정 여부."""
    wl = {(r['subject'], r['session']): r['reason'] for r in load_worklist()}
    man = {(r['subject'], r['session']): r.get('manual_offset')
           for r in list_all_offsets()}
    exc = exclusion_store.excluded_map()
    rows = []
    for r in scan_sessions():
        key = (r['subject'], r['session'])
        e = exc.get(key, {})
        rows.append({**r, 'reason': wl.get(key, ''), 'in_worklist': key in wl,
                     'manual': man.get(key),
                     'excluded': (e.get('session') or {}).get('reason') if e.get('session') else None})
    # 검토 필요 세션을 위로
    rows.sort(key=lambda r: (not r['in_worklist'], r['subject'], r['session']))
    return rows


# ==================== HTTP ====================

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype='application/json'):
        b = body if isinstance(body, bytes) else body.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path in ('/', '/index.html'):
                return self._send(200, HTML, 'text/html; charset=utf-8')
            if u.path == '/api/sessions':
                return self._send(200, json.dumps(session_index(), ensure_ascii=False))
            if u.path == '/api/data':
                sdir = q.get('session', [''])[0]
                if not sdir and q.get('subject'):
                    sdir = find_session_dir(q['subject'][0], q.get('session_name', [''])[0])
                if not sdir:
                    raise ValueError('세션 경로 없음')
                ch = int(q.get('channel', ['1'])[0])
                return self._send(200, json.dumps(build_data(sdir, ch), ensure_ascii=False))
            return self._send(404, json.dumps({'error': 'not found'}))
        except Exception as e:
            return self._send(500, json.dumps({'error': f'{type(e).__name__}: {e}'},
                                              ensure_ascii=False))

    def do_POST(self):
        u = urlparse(self.path)
        ln = int(self.headers.get('Content-Length', 0))
        try:
            payload = json.loads(self.rfile.read(ln) or b'{}')
            if u.path == '/api/save':
                set_manual_offset(payload['subject'], payload['session'],
                                  round(float(payload['offset']), 4),
                                  auto_offset=float(payload.get('auto_offset', 0.0)),
                                  auto_method=str(payload.get('method', '')),
                                  note=payload.get('note', 'gui point-pick'))
                return self._send(200, json.dumps({'ok': True,
                                                   'offset': float(payload['offset'])}))
            if u.path == '/api/clear':
                ok = clear_manual_offset(payload['subject'], payload['session'])
                return self._send(200, json.dumps({'ok': bool(ok)}))
            if u.path == '/api/exclude':
                if payload.get('remove'):
                    ok = exclusion_store.clear_exclusion(payload['subject'], payload['session'])
                    return self._send(200, json.dumps({'ok': bool(ok), 'excluded': None}))
                exclusion_store.set_exclusion(
                    payload['subject'], payload['session'],
                    reason=payload.get('reason', '기타'), note=payload.get('note', ''))
                return self._send(200, json.dumps({'ok': True,
                                                   'excluded': payload.get('reason')}))
            return self._send(404, json.dumps({'error': 'not found'}))
        except Exception as e:
            return self._send(500, json.dumps({'error': f'{type(e).__name__}: {e}'},
                                              ensure_ascii=False))


HTML = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>EAG-GRF Offset Aligner</title><style>
 body{font-family:system-ui,-apple-system,sans-serif;margin:10px;background:#fafafa;color:#222}
 #bar,#bar2{display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-bottom:5px}
 input,select,button{font-size:13px;padding:3px 6px}
 button{cursor:pointer;border:1px solid #bbb;background:#fff;border-radius:4px}
 button:hover{background:#f0f0f0}
 button.on{background:#1f77b4;color:#fff;border-color:#1f77b4}
 button.primary{background:#2ca02c;color:#fff;border-color:#2ca02c}
 button.danger{background:#fff;color:#d62728;border-color:#d62728}
 #status{margin-left:6px;color:#333;font-size:13px}
 canvas{border:1px solid #ccc;background:#fff;display:block;touch-action:none;cursor:crosshair}
 #tip{font-size:12px;color:#666;margin:4px 0}
 #meta{font-size:13px;margin:6px 0;font-family:ui-monospace,monospace}
 .big{font-size:15px;font-weight:600}
 table{border-collapse:collapse;font-size:12px;margin-top:6px}
 td,th{border:1px solid #ddd;padding:2px 8px;text-align:right}
 th{background:#f2f2f2}
 .warn{color:#d62728}.ok{color:#2ca02c}
</style></head><body>

<div id="bar">
 <select id="sesslist" style="max-width:480px"></select>
 <select id="filter" title="목록 필터">
  <option value="all" selected>전체</option>
  <option value="review">검토대상 ▲</option>
  <option value="done">확정 ✅</option>
  <option value="todo">미확정</option>
  <option value="excluded">분석제외 ⛔</option>
 </select>
 <input id="sess" placeholder="세션 디렉터리 직접 입력" size="40">
 <label>ch <input id="ch" type="number" value="1" min="1" max="8" style="width:44px"></label>
 <button id="load">Load</button>
 <span id="status"></span>
</div>

<div id="bar2">
 <label>클릭 흡착
  <select id="snapmode">
   <option value="point" selected>가장 가까운 점</option>
   <option value="inflect">변곡점</option>
   <option value="free">자유(흡착 없음)</option>
  </select></label>
 <label><input type="checkbox" id="ovl" checked> overlay(EAG 위에 GRF)</label>
 <button id="undo">Undo pair</button>
 <button id="clearpairs">Clear pairs</button>
 <span style="border-left:1px solid #ccc;padding-left:8px">nudge</span>
 <button data-n="-0.1">-0.1</button><button data-n="-0.02">-0.02</button>
 <button data-n="0.02">+0.02</button><button data-n="0.1">+0.1</button>
 <button id="usebest">best-match 값 사용</button>
 <button id="resetprev">미리보기 초기화</button>
 <span style="border-left:1px solid #ccc;padding-left:8px">offset
  <input id="offin" type="number" step="0.01" style="width:80px"> s</span>
 <button id="applyin">적용</button>
 <button id="fit">전체보기</button>
 <button id="save" class="primary">Save (확정)</button>
 <button id="clearman" class="danger">Clear manual</button>
 <span style="border-left:1px solid #ccc;padding-left:8px">분석제외
  <select id="excreason">
   <option>노이즈</option><option>동기화불가</option><option>프로토콜이상</option>
   <option>기록오류</option><option>기타</option>
  </select></span>
 <button id="excl" class="danger" title="이 세션을 분석에서 제외 라벨링">⛔ 세션 제외</button>
</div>

<div id="tip">① GRF 패널(위)에서 기준 변곡점 클릭 → ② EAG 패널(아래)에서 대응 변곡점 클릭 → 쌍 성립 시 즉시 정렬 미리보기.
여러 쌍을 찍으면 중앙값 사용 · 휠=확대/축소 · 드래그=좌우 이동 · 맨 위는 match-rate 프로파일(클릭 시 그 값으로 이동)
<br>클릭 흡착: <b>가장 가까운 점</b>=모든 샘플 중 최근접(기본, 클릭한 자리에서 거의 안 움직임) · <b>변곡점</b>=근처 코너로 흡착(±0.3s까지 이동) · <b>자유</b>=클릭한 좌표 그대로</div>

<canvas id="cv" width="1420" height="720"></canvas>
<div id="meta"></div>
<div id="tbl"></div>

<script>
// API 경로는 현재 문서 기준 상대경로로 — code-server의 /proxy/<port>/ 중계 아래에서도 동작
const BASE=(()=>{let p=location.pathname; if(/\.[a-z]+$/i.test(p))p=p.replace(/[^/]*$/,'');
 return p.endsWith('/')?p:p+'/';})();
const api=p=>BASE+p;

const cv=document.getElementById('cv'), ctx=cv.getContext('2d');
const P={l:60,r:14};                 // 좌우 여백
const PROF={y:8,h:64}, GRF={y:96,h:180}, EAG={y:300,h:400};
let D=null;                          // 서버 데이터
let S=0;                             // 미리보기 shift (초). 표시 EAG = te - S, 유효 offset = corrected + S
let pairs=[];                        // [{g:GRF시각, e:EAG시각(te_corr 기준)}]
let pend={g:null,e:null};            // 진행 중인 쌍
let view=null;                       // [t0,t1] 표시 구간 (공유 x축)
let panning=null, moved=false, hoverX=null;

const W=()=>cv.width-P.l-P.r;
const span=()=>view[1]-view[0];
const X=t=>P.l+(t-view[0])/span()*W();
const Tinv=px=>view[0]+(px-P.l)/W()*span();
const med=a=>{const s=[...a].sort((x,y)=>x-y),n=s.length;return n?(n%2?s[(n-1)/2]:(s[n/2-1]+s[n/2])/2):0;};
const effOffset=()=>D?D.corrected_offset+S:0;

// ---- 데이터 헬퍼 ----
function nearestIdx(arr,t){let lo=0,hi=arr.length-1;while(lo<hi){const m=(lo+hi)>>1;arr[m]<t?lo=m+1:hi=m;}
 if(lo>0&&Math.abs(arr[lo-1]-t)<Math.abs(arr[lo]-t))lo--;return lo;}
function snapWin(){return Math.min(0.30,Math.max(0.03,span()/W()*10));}
// 흡착 모드: point=모든 샘플 중 가장 가까운 점 / inflect=근처 |2차미분| 최대점 / free=흡착 없음
function snapTo(tArr,vArr,t){
 const m=document.getElementById('snapmode').value;
 if(m==='free') return t;
 if(m==='point') return tArr[nearestIdx(tArr,t)];
 const w=snapWin(), i0=nearestIdx(tArr,t-w), i1=nearestIdx(tArr,t+w);
 if(i1-i0<5) return tArr[nearestIdx(tArr,t)];
 let best=-1,bv=-1;
 for(let i=i0+1;i<i1-1;i++){const d2=Math.abs(vArr[i+1]-2*vArr[i]+vArr[i-1]); if(d2>bv){bv=d2;best=i;}}
 return best<0?tArr[nearestIdx(tArr,t)]:tArr[best];
}
function rangeIn(tArr,vArr,t0,t1){
 let mn=Infinity,mx=-Infinity;
 const i0=nearestIdx(tArr,t0), i1=nearestIdx(tArr,t1);
 for(let i=i0;i<=i1;i++){if(vArr[i]<mn)mn=vArr[i];if(vArr[i]>mx)mx=vArr[i];}
 if(!isFinite(mn)||mn===mx){mn=(mn||0)-1;mx=(mx||0)+1;}
 const pad=(mx-mn)*0.08; return [mn-pad,mx+pad];
}
function matchAt(res){  // 프로파일 선형보간
 if(!D||!D.prof_res.length) return null;
 const c=D.prof_res; if(res<c[0]||res>c[c.length-1]) return null;
 const i=nearestIdx(c,res); return D.prof_mr[i];
}

// ---- 그리기 ----
function axisY(y0,h,lo,hi){return v=>y0+(hi-v)/(hi-lo)*h;}
function frame(y0,h,label){
 ctx.strokeStyle='#ddd';ctx.lineWidth=1;ctx.strokeRect(P.l,y0,W(),h);
 ctx.fillStyle='#888';ctx.font='11px sans-serif';ctx.textAlign='left';ctx.fillText(label,P.l+4,y0+13);
}
function trace(tArr,vArr,Y,color,lw,alpha){
 ctx.save();ctx.globalAlpha=alpha||1;ctx.strokeStyle=color;ctx.lineWidth=lw||1;ctx.beginPath();
 let started=false;
 const i0=Math.max(0,nearestIdx(tArr,view[0])-1), i1=Math.min(tArr.length-1,nearestIdx(tArr,view[1])+1);
 for(let i=i0;i<=i1;i++){const x=X(tArr[i]),y=Y(vArr[i]); started?ctx.lineTo(x,y):(ctx.moveTo(x,y),started=true);}
 ctx.stroke();ctx.restore();
}
function vline(x,y0,h,color,dash,lw){
 ctx.save();ctx.strokeStyle=color;ctx.lineWidth=lw||1;if(dash)ctx.setLineDash(dash);
 ctx.beginPath();ctx.moveTo(x,y0);ctx.lineTo(x,y0+h);ctx.stroke();ctx.restore();
}
function ticks(){
 const n=10,step=span()/n;
 ctx.fillStyle='#666';ctx.font='11px sans-serif';ctx.textAlign='center';
 for(let i=0;i<=n;i++){const t=view[0]+i*step,x=X(t);
  vline(x,GRF.y,GRF.h,'#f0f0f0');vline(x,EAG.y,EAG.h,'#f0f0f0');
  ctx.fillText(t.toFixed(1),x,EAG.y+EAG.h+13);}
 ctx.textAlign='left';ctx.fillText('time (s)',P.l,EAG.y+EAG.h+13);
}
function draw(){
 ctx.clearRect(0,0,cv.width,cv.height);
 if(!D)return;

 // (0) match-rate 프로파일
 frame(PROF.y,PROF.h,'match-rate profile (residual, s)');
 if(D.prof_res.length){
  const c=D.prof_res,lo=c[0],hi=c[c.length-1];
  const PX=r=>P.l+(r-lo)/(hi-lo)*W(), PY=axisY(PROF.y,PROF.h,0,1);
  ctx.strokeStyle='#7b3fa0';ctx.lineWidth=1.2;ctx.beginPath();
  c.forEach((r,i)=>{const x=PX(r),y=PY(D.prof_mr[i]);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();
  if(D.best_res!==null)vline(PX(D.best_res),PROF.y,PROF.h,'#2ca02c',[4,3],2);
  vline(PX(D.residual+S),PROF.y,PROF.h,'#d62728',null,2);          // 현재 미리보기
  vline(PX(D.residual),PROF.y,PROF.h,'#999',[2,3],1);              // 서버 corrected
  ctx.fillStyle='#666';ctx.font='11px sans-serif';ctx.textAlign='right';
  ctx.fillText('best(초록) / 현재(빨강)',P.l+W()-4,PROF.y+13);ctx.textAlign='left';
 }

 ticks();

 // (1) GRF 패널
 frame(GRF.y,GRF.h,'GRF signed imbalance — 여기서 기준점 클릭');
 const gr=rangeIn(D.grf_t,D.grf_signed,view[0],view[1]);
 const GY=axisY(GRF.y+6,GRF.h-12,gr[0],gr[1]);
 vline(P.l,GRF.y,0,'#000');
 ctx.strokeStyle='#eee';ctx.beginPath();ctx.moveTo(P.l,GY(0));ctx.lineTo(P.l+W(),GY(0));ctx.stroke();
 D.trans.forEach(t=>{const x=X(t);vline(x,GRF.y,GRF.h,'rgba(44,160,44,.45)',[4,4]);
                     vline(x,EAG.y,EAG.h,'rgba(44,160,44,.30)',[4,4]);});
 trace(D.grf_t,D.grf_signed,GY,'#2ca02c',1.4);

 // (2) EAG 패널 (te - S)
 frame(EAG.y,EAG.h,'EAG ch'+D.channel+' @ offset '+effOffset().toFixed(3)+'s — 여기서 대응점 클릭');
 const teS=D.te.map(t=>t-S);
 const er=rangeIn(teS,D.eag,view[0],view[1]);
 const EY=axisY(EAG.y+6,EAG.h-12,er[0],er[1]);
 if(document.getElementById('ovl').checked){       // 정규화 GRF 겹쳐 그리기
  const gsp=gr[1]-gr[0]||1, esp=er[1]-er[0]||1;
  const GO=v=>EY(er[0]+ (v-gr[0])/gsp*esp);
  trace(D.grf_t,D.grf_signed,GO,'#2ca02c',1.2,0.45);
 }
 trace(teS,D.eag,EY,'#1f77b4',1.1);
 D.eag_edges.forEach(t=>{const x=X(t-S);vline(x,EAG.y+EAG.h-14,14,'#1f77b4',null,1.5);});

 // (3) 확정된 쌍 + 진행 중 점
 pairs.forEach((p,k)=>{
  const xg=X(p.g), xe=X(p.e-S);
  vline(xg,GRF.y,GRF.h,'#ff7f0e',null,1.5); vline(xe,EAG.y,EAG.h,'#ff7f0e',null,1.5);
  ctx.fillStyle='#ff7f0e';ctx.font='10px sans-serif';ctx.textAlign='center';
  ctx.fillText('#'+(k+1),xg,GRF.y+GRF.h-3); ctx.fillText('#'+(k+1),xe,EAG.y+12);
  ctx.textAlign='left';
 });
 if(pend.g!==null){const x=X(pend.g);vline(x,GRF.y,GRF.h,'#e377c2',[6,3],2);
   ctx.fillStyle='#e377c2';ctx.textAlign='center';ctx.fillText('GRF 선택됨 → EAG 클릭',x,GRF.y+GRF.h-3);ctx.textAlign='left';}
 if(pend.e!==null){const x=X(pend.e-S);vline(x,EAG.y,EAG.h,'#e377c2',[6,3],2);
   ctx.fillStyle='#e377c2';ctx.textAlign='center';ctx.fillText('EAG 선택됨 → GRF 클릭',x,EAG.y+12);ctx.textAlign='left';}

 // (4) 마우스 커서 가이드 + 시각 표시
 if(hoverX!==null&&hoverX>=P.l&&hoverX<=P.l+W()){
  vline(hoverX,GRF.y,GRF.h,'rgba(0,0,0,.18)');vline(hoverX,EAG.y,EAG.h,'rgba(0,0,0,.18)');
  ctx.fillStyle='#333';ctx.font='11px ui-monospace,monospace';ctx.textAlign='left';
  ctx.fillText('t='+Tinv(hoverX).toFixed(3)+'s',Math.min(hoverX+5,P.l+W()-70),GRF.y+13);}

 renderMeta();
}

function renderMeta(){
 const mr=matchAt(D.residual+S);
 const dlt=S;
 document.getElementById('meta').innerHTML =
  `<span class="big">${D.subject} / ${D.session} · ch${D.channel}</span> &nbsp;|&nbsp; `+
  `auto=${D.auto_offset.toFixed(3)} · server corrected=${D.corrected_offset.toFixed(3)} (${D.method})`+
  (D.has_manual?` · <b class="ok">manual=${D.manual_offset.toFixed(3)} 확정됨</b>`:'')+
  (D.excluded?` · <b class="warn">⛔ 분석제외 (${D.excluded.reason}${D.excluded.note?': '+D.excluded.note:''})</b>`:'')+
  `<br><span class="big">최종 offset = ${effOffset().toFixed(3)} s</span>`+
  ` (미리보기 이동 ${dlt>=0?'+':''}${dlt.toFixed(3)} s)`+
  ` &nbsp;|&nbsp; match@현재 = ${mr===null?'—':(mr*100).toFixed(0)+'%'}`+
  ` · best-match res=${D.best_res===null?'—':D.best_res.toFixed(2)}`+
  (D.margin===null?'':` (margin ${D.margin.toFixed(2)})`)+
  (D.reason?` &nbsp;|&nbsp; <span class="warn">review: ${D.reason}</span>`:'');
 let h='<table><tr><th>#</th><th>GRF (s)</th><th>EAG (s)</th><th>Δ = EAG−GRF</th><th></th></tr>';
 pairs.forEach((p,k)=>{h+=`<tr><td>${k+1}</td><td>${p.g.toFixed(3)}</td><td>${p.e.toFixed(3)}</td>`+
   `<td>${(p.e-p.g)>=0?'+':''}${(p.e-p.g).toFixed(3)}</td>`+
   `<td><button onclick="delPair(${k})">삭제</button></td></tr>`;});
 if(pairs.length>1)h+=`<tr><th colspan="3">median</th><th>${S>=0?'+':''}${S.toFixed(3)}</th><th></th></tr>`;
 document.getElementById('tbl').innerHTML = pairs.length?h+'</table>':'';
}
function delPair(k){pairs.splice(k,1);recompute();}
window.delPair=delPair;
function recompute(){S=pairs.length?med(pairs.map(p=>p.e-p.g)):0;draw();}

// ---- 상호작용 ----
cv.addEventListener('mousedown',ev=>{
 if(!D)return;const r=cv.getBoundingClientRect();
 panning={x:ev.clientX-r.left,v0:[...view]};moved=false;
});
cv.addEventListener('mousemove',ev=>{
 if(!D)return;const r=cv.getBoundingClientRect(),mx=ev.clientX-r.left;hoverX=mx;
 if(panning){const dx=mx-panning.x;
  if(Math.abs(dx)>3){moved=true;const dt=dx/W()*(panning.v0[1]-panning.v0[0]);
   view=[panning.v0[0]-dt,panning.v0[1]-dt];}}
 draw();
});
window.addEventListener('mouseup',ev=>{
 if(!D||!panning){panning=null;return;}
 const r=cv.getBoundingClientRect(),mx=ev.clientX-r.left,my=ev.clientY-r.top;
 const wasPan=moved;panning=null;
 if(wasPan){draw();return;}
 if(mx<P.l||mx>P.l+W()){draw();return;}
 const t=Tinv(mx);
 if(my>=PROF.y&&my<=PROF.y+PROF.h&&D.prof_res.length){        // 프로파일 클릭 → 그 residual로
  const c=D.prof_res,lo=c[0],hi=c[c.length-1];
  const res=lo+(mx-P.l)/W()*(hi-lo); S=res-D.residual; pairs=[]; pend={g:null,e:null}; draw(); return;
 }
 if(my>=GRF.y&&my<=GRF.y+GRF.h){pend.g=snapTo(D.grf_t,D.grf_signed,t);}
 else if(my>=EAG.y&&my<=EAG.y+EAG.h){
  const teS=D.te.map(x=>x-S);
  pend.e=snapTo(teS,D.eag,t)+S;                                // te_corr 프레임으로 환산 저장
 } else {draw();return;}
 if(pend.g!==null&&pend.e!==null){pairs.push({g:pend.g,e:pend.e});pend={g:null,e:null};recompute();}
 else draw();
});
cv.addEventListener('wheel',ev=>{
 if(!D)return;ev.preventDefault();
 const r=cv.getBoundingClientRect(),mx=ev.clientX-r.left;
 if(mx<P.l||mx>P.l+W())return;
 const t=Tinv(mx),f=ev.deltaY>0?1.2:1/1.2;
 const a=(t-view[0])*f,b=(view[1]-t)*f;
 view=[t-a,t+b];draw();
},{passive:false});

document.querySelectorAll('button[data-n]').forEach(b=>b.onclick=()=>{
 if(!D)return;S+=parseFloat(b.dataset.n);pairs=[];draw();});
document.getElementById('undo').onclick=()=>{if(pend.g!==null||pend.e!==null){pend={g:null,e:null};draw();}
 else{pairs.pop();recompute();}};
document.getElementById('clearpairs').onclick=()=>{pairs=[];pend={g:null,e:null};recompute();};
document.getElementById('resetprev').onclick=()=>{pairs=[];pend={g:null,e:null};S=0;draw();};
document.getElementById('usebest').onclick=()=>{if(!D||D.best_res===null)return;pairs=[];S=D.best_res-D.residual;draw();};
document.getElementById('applyin').onclick=()=>{const v=parseFloat(document.getElementById('offin').value);
 if(!D||isNaN(v))return;pairs=[];S=v-D.corrected_offset;draw();};
document.getElementById('fit').onclick=()=>{if(!D)return;fitView();draw();};
document.getElementById('snapmode').onchange=draw;
document.getElementById('ovl').onchange=draw;
window.addEventListener('keydown',ev=>{
 if(ev.target.tagName==='INPUT')return;
 if(ev.key==='z'){document.getElementById('undo').click();}
 if(ev.key==='f'){document.getElementById('fit').click();}
});

function fitView(){const t0=Math.min(D.grf_t[0],D.te[0]-S),t1=Math.max(D.grf_t[D.grf_t.length-1],D.te[D.te.length-1]-S);
 view=[t0,t1];}
function status(t,cls){const s=document.getElementById('status');s.textContent=t;s.className=cls||'';}

async function load(){
 const s=document.getElementById('sess').value.trim(), ch=document.getElementById('ch').value;
 if(!s){status('세션을 선택하세요','warn');return;}
 status('loading...');
 const res=await fetch(api('api/data?session='+encodeURIComponent(s)+'&channel='+ch));
 const j=await res.json();
 if(j.error){status('ERR: '+j.error,'warn');return;}
 D=j;S=0;pairs=[];pend={g:null,e:null};
 document.getElementById('ch').max=D.n_channels;
 document.getElementById('offin').value=D.corrected_offset.toFixed(3);
 fitView();
 document.getElementById('excl').textContent = D.excluded?'⛔ 제외 해제':'⛔ 세션 제외';
 status(D.has_manual?'loaded (manual 확정본, residual 재계산 안 함)':'loaded');
 draw();
}
document.getElementById('load').onclick=load;

document.getElementById('save').onclick=async()=>{
 if(!D)return;const v=effOffset();
 if(!confirm(`${D.subject}/${D.session}\noffset = ${v.toFixed(3)} s 로 확정할까요?`))return;
 const res=await fetch(api('api/save'),{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({subject:D.subject,session:D.session,offset:v,
   auto_offset:D.auto_offset,method:D.method,note:'gui point-pick'})});
 const j=await res.json();
 if(j.error){status('ERR: '+j.error,'warn');return;}
 status('저장됨: '+v.toFixed(3)+'s → manual_offsets.json','ok');
 await load(); refreshList();
};
document.getElementById('excl').onclick=async()=>{
 if(!D)return; const on=!!D.excluded;
 const reason=document.getElementById('excreason').value;
 const note=on?'':(prompt('제외 사유 메모(선택):','')||'');
 if(!confirm(`${D.subject}/${D.session}\n${on?'분석제외를 해제할까요?':'이 세션을 분석에서 제외할까요? ('+reason+')'}`))return;
 const res=await fetch(api('api/exclude'),{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({subject:D.subject,session:D.session,reason,note,remove:on})});
 const j=await res.json(); if(j.error){status('ERR: '+j.error,'warn');return;}
 status(on?'분석제외 해제됨':'⛔ 분석제외로 표시됨','warn'); await load(); refreshList();};
document.getElementById('clearman').onclick=async()=>{
 if(!D)return;
 if(!confirm(`${D.subject}/${D.session}\n수동 offset을 제거하고 자동값으로 되돌릴까요?`))return;
 await fetch(api('api/clear'),{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({subject:D.subject,session:D.session})});
 status('manual offset 제거됨 (auto 복귀)');
 await load(); refreshList();
};

let SESSIONS=[];
function renderList(){
 const sel=document.getElementById('sesslist'), f=document.getElementById('filter').value;
 const cur=document.getElementById('sess').value;
 const keep=x=> f==='all'?true : f==='review'?x.in_worklist : f==='done'?x.manual!=null
              : f==='excluded'?x.excluded!=null : x.manual==null;
 const rows=SESSIONS.filter(keep), nDone=SESSIONS.filter(x=>x.manual!=null).length;
 sel.innerHTML=`<option value="">— 세션 선택 (표시 ${rows.length} / 전체 ${SESSIONS.length} · 확정 ${nDone}) —</option>`;
 rows.forEach(x=>{const o=document.createElement('option');o.value=x.dir;
  o.textContent=(x.excluded?'⛔ ':x.manual!=null?'✅ ':x.in_worklist?'▲ ':'　 ')+x.subject+' / '+x.session+
   (x.excluded?'  [제외:'+x.excluded+']':'')+
   (x.manual!=null?`  [확정 ${Number(x.manual)>=0?'+':''}${Number(x.manual).toFixed(2)}s]`:'')+
   (x.reason?'  · '+x.reason:'');
  sel.appendChild(o);});
 sel.value=cur;                      // 현재 세션이 필터에서 빠지면 빈 값이 되지만 #sess는 유지됨
}
function refreshList(){
 return fetch(api('api/sessions')).then(r=>r.json()).then(rows=>{SESSIONS=rows;renderList();});
}
document.getElementById('filter').onchange=renderList;
document.getElementById('sesslist').onchange=()=>{
 const v=document.getElementById('sesslist').value;
 if(v){document.getElementById('sess').value=v;load();}
};
refreshList();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description='EAG-GRF offset 정렬 GUI (점 찍어 맞추기)')
    ap.add_argument('--port', type=int, default=8766, help='포트 (기본 8766, 3002 금지)')
    ap.add_argument('--host', default='127.0.0.1')
    args = ap.parse_args()
    if args.port == 3002:
        raise SystemExit('port 3002는 api-server 전용이라 사용 금지')
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Offset Aligner: http://{args.host}:{args.port}  (Ctrl-C 종료)")
    print("  GRF 점 클릭 → EAG 대응점 클릭 → Save")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == '__main__':
    main()
