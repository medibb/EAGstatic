"""
Edge Annotation App — 가벼운 로컬 GUI (브라우저) EAG edge 수동 수정기.

의존성 없음(파이썬 표준 http.server만). edge_editor/edge_store 로직 재사용.
manual_edges.json에 바로 저장 → parameter_extractor가 자동 반영.

⚠️ 안전: api-server(port 3002)와 무관한 별도 포트(기본 8765)만 사용. 3002 금지.

실행:
  python3 edge_app.py                 # http://127.0.0.1:8765
  python3 edge_app.py --port 8890
외부(DDNS) 접속: code-server 내장 포트 프록시 경유 — 끝 슬래시 필수
  http://<code-server 호스트>/proxy/8765/     (예: medibb.synology.me:18440)
  API를 문서 기준 상대경로로 호출하므로 프록시 prefix 아래에서도 동작한다.
브라우저에서:
  - 세션 디렉터리 + 채널 입력 후 Load
  - knee 점을 드래그해 이동 / [Add]모드로 트레이스 2점 클릭해 edge 추가 / edge 선택 후 Delete
  - Snap 체크 시 corner 자동 정렬 / Save로 확정, Reset으로 자동검출 복귀
"""

import argparse
import csv
import json
import io
import sys
import contextlib
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sync_analyzer import find_session_pair, SyncAnalyzer
import grf_triggered_annotator as G
from eag_analyzer import get_data_dir
import edge_store
import exclusion_store


@contextlib.contextmanager
def _quiet():
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old


def _ds(a, step):
    a = np.asarray(a)
    return a[::step]


def build_data(session_dir: str, channel: int, recompute: bool = True) -> dict:
    from scipy.signal import detrend
    pair = find_session_pair(session_dir)
    if pair is None:
        raise FileNotFoundError(session_dir)
    with _quiet():
        sa = SyncAnalyzer(pair)
        off, trans, _s, _g = G.compute_offset(sa, channel - 1, recompute)
    te = sa.unified_time_eag - off.residual
    eag = detrend(sa.eag_filtered[:, channel - 1])
    tg = sa.unified_time_grf
    signed = G.signed_imbalance(sa.grf_left, sa.grf_right)

    # 프로토콜(체중부하 4 cycle × 2 = 8 이벤트) — edge 검출보다 먼저 anchor를 잡는다
    rest, cycles, cyc_info = G.detect_load_cycles_expected(
        tg, signed, sa.grf_left, sa.grf_right)
    anchors = G.cycles_to_transitions(cycles, trans)

    man = edge_store.get_channel_edges(pair.subject_name, pair.session_name, channel)
    if man is not None:
        edges = man
        source = 'manual'
    else:
        auto = G.detect_eag_edges_protocol(te, eag, anchors, fs=sa.eag.sample_rate)
        edges = [{'onset_time': e[0], 'onset_amp': e[1],
                  'offset_time': e[3], 'offset_amp': e[4]} for e in auto]
        source = 'auto'
    e_on = np.array([e['onset_time'] for e in edges]) if edges else np.array([])
    e_amp = np.array([e['offset_amp'] - e['onset_amp'] for e in edges]) if edges else np.array([])
    valid = G.validate_cycle_edges(cycles, e_on, e_amp, raw_trans=trans)

    exc = exclusion_store.is_excluded(pair.subject_name, pair.session_name, channel)
    step = max(1, len(te) // 20000)  # 표시 다운샘플 (확대 시 정밀도 확보용으로 넉넉히)
    r1 = lambda arr: [round(float(x), 1) for x in arr]
    r3 = lambda arr: [round(float(x), 3) for x in arr]
    return {
        'subject': pair.subject_name, 'session': pair.session_name, 'channel': channel,
        'corrected_offset': round(float(off.corrected_offset), 3), 'method': off.method,
        'source': source,
        'rest_level': round(float(rest), 3),
        'excluded': exc,
        'cycles': [{'id': c.cycle_id, 'onset': round(c.onset_time, 3),
                    'offset': round(c.offset_time, 3), 'load': round(c.load_level, 3),
                    'step': round(c.load_step, 3),
                    'end': round(c.end_time, 3) if np.isfinite(c.end_time) else round(c.offset_time, 3),
                    'load_pct': None if not np.isfinite(c.load_ratio) else round(c.load_pct, 1),
                    'test_side': c.test_side} for c in cycles],
        'cycle_search': cyc_info,
        'per_cycle': valid['per_cycle'], 'noise_idx': valid['noise_idx'],
        'n_measured_cycles': valid['n_measured_cycles'],
        'anchors': [{'t': round(a.time, 3), 'kind': 'on' if i % 2 == 0 else 'off',
                     'cycle': i // 2} for i, a in enumerate(anchors)],
        'valid': {k: valid[k] for k in ('ok', 'n_cycles', 'n_matched', 'n_events',
                                        'n_edges', 'n_measured_cycles', 'reasons',
                                        'priority', 'labels', 'n_single_sided')},
        'events': valid['events'],
        'expected_cycles': G.EXPECTED_CYCLES, 'expected_events': G.EXPECTED_EVENTS,
        'te': r3(_ds(te, step)), 'eag': r1(_ds(eag, step)),
        'grf_t': r3(_ds(tg, step)), 'grf_signed': r3(_ds(signed, step)),
        'trans': r3([t.time for t in trans]),
        'edges': [{'onset_time': round(e['onset_time'], 3), 'onset_amp': round(e['onset_amp'], 1),
                   'offset_time': round(e['offset_time'], 3), 'offset_amp': round(e['offset_amp'], 1)}
                  for e in edges],
    }


def load_worklist() -> list:
    p = Path('result/offset_review/worklist.csv')
    if not p.exists():
        return []
    out = []
    with open(p, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            out.append({'subject': row.get('subject', ''), 'session': row.get('session', ''),
                        'reason': row.get('reason', '')})
    return out


def find_session_dir(subject: str, session: str) -> str:
    # subject는 세션 폴더의 부모 이름 = 방문 폴더여야 하므로 평면 미러를 스캔한다.
    for p in Path(get_data_dir()).rglob('BrainFlow-RAW_*.csv'):
        if p.parent.parent.name == subject and p.parent.name.endswith('-' + session):
            return str(p.parent)
    return ''


def load_protocol_status() -> dict:
    """edge_review 배치 결과 → 세션별 (통과채널수, 전체채널수, cycle수).

    edge_review.py --dir data 로 생성. 없으면 빈 dict (드롭다운에 상태 미표시).
    """
    p = Path('result/edge_review/all_channels.csv')
    if not p.exists():
        return {}
    out = {}
    with open(p, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            key = (r['subject'], r['session'])
            e = out.setdefault(key, {'ok': 0, 'total': 0, 'cycles': None})
            e['total'] += 1
            if str(r['ok']).lower() == 'true':
                e['ok'] += 1
            if e['cycles'] is None:
                try: e['cycles'] = int(r['n_cycles'])
                except (ValueError, TypeError): pass
    return out


def session_index() -> list:
    """드롭다운용: 전체 세션 + worklist 사유 + 수동 edge 채널 + 수동 offset 확정 여부.

    worklist(needs_review)만이 아니라 data/ 아래 모든 세션을 대상으로 한다.
    edge가 이미 확정된 세션은 채널 목록과 edge 수를 함께 돌려준다.
    """
    from offset_app import scan_sessions          # 세션 스캔(파일명 기반, 가벼움) 공유
    from offset_manager import list_all_offsets

    wl = {(r['subject'], r['session']): r['reason'] for r in load_worklist()}
    proto = load_protocol_status()
    ed = {}
    for r in edge_store.list_all():
        ed.setdefault((r['subject'], r['session']), []).append(
            (int(r['channel']), int(r['n_edges'])))
    off = {(r['subject'], r['session']): r.get('manual_offset')
           for r in list_all_offsets()}
    exc = exclusion_store.excluded_map()

    rows = []
    for r in scan_sessions():
        key = (r['subject'], r['session'])
        chans = sorted(ed.get(key, []))
        p = proto.get(key)
        rows.append({**r,
                     'reason': wl.get(key, ''), 'in_worklist': key in wl,
                     'edge_channels': [c for c, _ in chans],
                     'edge_counts': [n for _, n in chans],
                     'manual_offset': off.get(key),
                     'proto_ok': None if p is None else p['ok'],
                     'proto_total': None if p is None else p['total'],
                     'proto_cycles': None if p is None else p['cycles'],
                     'excl_session': (exc.get(key, {}).get('session') or {}).get('reason'),
                     'excl_channels': sorted((exc.get(key, {}).get('channels') or {}).keys())})
    # 검토 대상 먼저, 그다음 피험자/세션 순
    rows.sort(key=lambda r: (not r['in_worklist'], r['subject'], r['session']))
    return rows


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
            if u.path == '/' or u.path == '/index.html':
                return self._send(200, HTML, 'text/html; charset=utf-8')
            if u.path == '/api/worklist':
                return self._send(200, json.dumps(load_worklist()))
            if u.path == '/api/sessions':
                return self._send(200, json.dumps(session_index(), ensure_ascii=False))
            if u.path == '/api/data':
                sdir = q.get('session', [''])[0]
                if not sdir and q.get('subject'):
                    sdir = find_session_dir(q['subject'][0], q.get('session_name', [''])[0])
                ch = int(q.get('channel', ['1'])[0])
                return self._send(200, json.dumps(build_data(sdir, ch)))
            if u.path == '/api/finddir':
                sdir = find_session_dir(q.get('subject', [''])[0], q.get('session_name', [''])[0])
                return self._send(200, json.dumps({'dir': sdir}))
            return self._send(404, json.dumps({'error': 'not found'}))
        except Exception as e:
            return self._send(500, json.dumps({'error': str(e)}))

    def do_POST(self):
        u = urlparse(self.path)
        ln = int(self.headers.get('Content-Length', 0))
        payload = json.loads(self.rfile.read(ln) or b'{}')
        try:
            if u.path == '/api/save':
                edge_store.set_channel_edges(
                    payload['subject'], payload['session'], int(payload['channel']),
                    payload['edges'], offset_used=payload.get('corrected_offset', 0.0),
                    note=payload.get('note', 'gui edit'))
                return self._send(200, json.dumps({'ok': True, 'n': len(payload['edges'])}))
            if u.path == '/api/exclude':
                ch = int(payload.get('channel', 0))
                if payload.get('remove'):
                    ok = exclusion_store.clear_exclusion(
                        payload['subject'], payload['session'], ch)
                    return self._send(200, json.dumps({'ok': bool(ok)}))
                exclusion_store.set_exclusion(
                    payload['subject'], payload['session'], ch,
                    reason=payload.get('reason', '기타'), note=payload.get('note', ''))
                return self._send(200, json.dumps({'ok': True}))
            if u.path == '/api/reset':
                ok = edge_store.clear_channel_edges(
                    payload['subject'], payload['session'], int(payload['channel']))
                return self._send(200, json.dumps({'ok': ok}))
            return self._send(404, json.dumps({'error': 'not found'}))
        except Exception as e:
            return self._send(500, json.dumps({'error': str(e)}))


HTML = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>EAG Edge Annotator</title><style>
 body{font-family:system-ui,sans-serif;margin:10px;background:#fafafa}
 #bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:6px}
 input,select,button{font-size:13px;padding:3px 6px}
 button{cursor:pointer;border:1px solid #bbb;background:#fff;border-radius:4px}
 button.on{background:#1f77b4;color:#fff}
 #status{margin-left:8px;color:#333}
 canvas{border:1px solid #ccc;background:#fff;display:block;touch-action:none}
 #tip{font-size:12px;color:#666;margin:4px 0}
 table{border-collapse:collapse;font-size:12px;margin-top:6px}
 td,th{border:1px solid #ddd;padding:2px 6px}
 #meta{font-size:13px;margin:6px 0;line-height:1.6}
 .ok{color:#2ca02c}.bad{color:#d62728}.warn{color:#c26a12}
 button.danger{color:#d62728;border-color:#d62728}
</style></head><body>
<div id="bar">
 <select id="wl" title="세션 목록" style="max-width:440px"></select>
 <select id="filter" title="목록 필터">
  <option value="all" selected>전체</option>
  <option value="review">검토대상 ▲</option>
  <option value="done">edge 확정 ✅</option>
  <option value="todo">edge 미확정</option>
  <option value="proto">프로토콜 미충족 ⚠️</option>
  <option value="excluded">분석제외 ⛔</option>
 </select>
 <input id="sess" placeholder="session dir (또는 목록 선택)" size="30">
 <label>ch <input id="ch" type="number" value="1" min="1" max="8" style="width:44px"></label>
 <button id="load">Load</button>
 <button id="addmode">Add mode</button>
 <button id="del">Delete sel</button>
 <label><input type="checkbox" id="snap" checked> snap</label>
 <button id="panL" title="왼쪽으로 이동 (← 키)">◀</button>
 <button id="panR" title="오른쪽으로 이동 (→ 키)">▶</button>
 <button id="evL" title="이전 이벤트로 (Shift+←)">◀이벤트</button>
 <button id="evR" title="다음 이벤트로 (Shift+→)">이벤트▶</button>
 <button id="fit">전체보기</button>
 <button id="denoise" title="anchor에서 먼 edge(부하 중간·휴식 구간)를 노이즈로 보고 일괄 삭제">노이즈 삭제</button>
 <button id="save">Save</button>
 <button id="reset">Reset→auto</button>
 <span style="border-left:1px solid #ccc;padding-left:8px">분석제외
  <select id="excreason">
   <option>노이즈</option><option>동기화불가</option><option>프로토콜이상</option>
   <option>기록오류</option><option>기타</option>
  </select></span>
 <button id="exclCh" class="danger">⛔ 채널 제외</button>
 <button id="exclSes" class="danger">⛔ 세션 제외</button>
 <span id="status"></span>
</div>
<div id="tip">드래그=knee 이동 · Add mode 후 트레이스 2점 클릭=edge 추가 · edge 클릭 선택 후 Delete/Del키 · rise=빨강 fall=초록
<br><b>휠=확대/축소</b>(커서 기준) · <b>빈 공간 드래그=좌우 이동</b>(Shift+드래그·휠버튼도 가능) · <b>Shift+휠</b>=좌우 이동
 · <b>← →</b> 이동, <b>Shift+← →</b> 이전/다음 이벤트로 점프, <b>Home/End</b> 처음/끝, <b>f</b> 전체보기 · 확대할수록 snap 범위도 좁아져 정밀해진다
<br>주황 음영=체중부하 cycle(프로토콜 4회) · 세로 점선=분석 anchor 8개(빨강=부하 시작, 파랑=이탈) · <b class="bad">굵은 빨강="누락"</b>=그 anchor에 knee가 없음 → 그 자리에 edge를 추가하세요</div>
<canvas id="cv" width="1400" height="600"></canvas>
<div id="meta"></div><div id="tbl"></div>
<script>
// API 경로는 현재 문서 기준 상대경로로 — code-server의 /proxy/<port>/ 중계 아래에서도 동작
const BASE=(()=>{let p=location.pathname; if(/\.[a-z]+$/i.test(p))p=p.replace(/[^/]*$/,'');
 return p.endsWith('/')?p:p+'/';})();
const api=p=>BASE+p;
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
let D=null, sel=-1, addPts=[], addMode=false, drag=null;
let view=null, pan=null;                    // view=[t0,t1] 표시 구간, pan=이동 중 상태
const M={l:55,r:15,t:15,b:34}, GH=140; // GRF strip height
function fitView(){view=[D.te[0], D.te[D.te.length-1]];}
function dataRange(){return [D.te[0], D.te[D.te.length-1]];}
function clampView(){                       // 데이터 밖으로 벗어나지 않게
 if(!D||!view)return; const [t0,t1]=dataRange(); let s=view[1]-view[0];
 if(s>=t1-t0){view=[t0,t1];return;}
 if(view[0]<t0)view=[t0,t0+s]; if(view[1]>t1)view=[t1-s,t1];}
function panBy(frac){                       // 보이는 폭의 frac 만큼 좌우 이동
 if(!D||!view)return; const s=view[1]-view[0];
 view=[view[0]+s*frac, view[1]+s*frac]; clampView(); draw();}
function centerOn(t){                       // 배율 유지한 채 t를 화면 중앙으로
 if(!D||!view)return; const s=view[1]-view[0];
 view=[t-s/2, t+s/2]; clampView(); draw();}
function gotoEvent(dir){                    // 이전/다음 anchor로 점프
 if(!D||!view)return; const ts=(D.anchors||[]).map(a=>a.t).sort((x,y)=>x-y);
 if(!ts.length)return; const c=(view[0]+view[1])/2, eps=1e-3;
 const cand = dir>0 ? ts.filter(t=>t>c+eps) : ts.filter(t=>t<c-eps);
 if(!cand.length)return;
 centerOn(dir>0?cand[0]:cand[cand.length-1]);}
function xr(){return view;}
function PW(){return cv.width-M.l-M.r;}
// y축은 보이는 구간 기준으로 자동 스케일 (확대 시 파형이 납작해지지 않게).
// draw() 시작에 한 번만 계산해 _yr에 캐시 — EY가 매번 전 구간을 스캔하면 O(n^2)가 된다.
let _yr=null;
function calcYR(){let[a,b]=xr(),i0=nearestIdx(a),i1=nearestIdx(b),mn=Infinity,mx=-Infinity;
  for(let i=i0;i<=i1;i++){if(D.eag[i]<mn)mn=D.eag[i];if(D.eag[i]>mx)mx=D.eag[i];}
  if(!isFinite(mn)||mn===mx){mn=(mn||0)-1;mx=(mx||0)+1;}
  let pad=(mx-mn)*0.06; return [mn-pad,mx+pad];}
function eagYR(){return _yr||(_yr=calcYR());}
function X(t){let[a,b]=xr();return M.l+(t-a)/(b-a)*PW();}
function Tinv(px){let[a,b]=xr();return a+(px-M.l)/PW()*(b-a);}
function vline(x,y0,h,color,dash,lw){
 ctx.save();ctx.strokeStyle=color;ctx.lineWidth=lw||1;if(dash)ctx.setLineDash(dash);
 ctx.beginPath();ctx.moveTo(x,y0);ctx.lineTo(x,y0+h);ctx.stroke();ctx.restore();}
function eagH(){return cv.height-M.t-M.b-GH;}
function EY(v){let[a,b]=eagYR();return M.t+(b-v)/(b-a)*eagH();}
function ampAt(t){let i=nearestIdx(t);return D.eag[i];}
function nearestIdx(t){let lo=0,hi=D.te.length-1;while(lo<hi){let m=(lo+hi)>>1;if(D.te[m]<t)lo=m+1;else hi=m;}
  if(lo>0&&Math.abs(D.te[lo-1]-t)<Math.abs(D.te[lo]-t))lo--; return lo;}
// snap 탐색 범위는 확대 배율에 따라 좁아진다 (확대할수록 정밀)
function snapWin(){let[a,b]=xr();return Math.min(0.30,Math.max(0.02,(b-a)/PW()*10));}
function snapCorner(t){ if(!document.getElementById('snap').checked) return [t,ampAt(t)];
  let w=snapWin(),i0=nearestIdx(t-w),i1=nearestIdx(t+w); if(i1-i0<5){let i=nearestIdx(t);return[D.te[i],D.eag[i]];}
  let best=i0,bv=-1; for(let i=i0+1;i<i1-1;i++){let d2=Math.abs(D.eag[i+1]-2*D.eag[i]+D.eag[i-1]); if(d2>bv){bv=d2;best=i;}} return [D.te[best],D.eag[best]];}
function draw(){ if(!D||!view)return; ctx.clearRect(0,0,cv.width,cv.height);
 _yr=calcYR();                              // 이번 프레임의 y범위 (view 변경 반영)
 // 시간축 눈금 (확대 시 현재 위치 파악용)
 let gy0=cv.height-M.b-GH, [a,b]=xr();
 ctx.fillStyle='#666';ctx.font='11px sans-serif';ctx.textAlign='center';
 for(let i=0;i<=10;i++){let t=a+(b-a)*i/10,x=X(t);
  ctx.strokeStyle='#f2f2f2';ctx.beginPath();ctx.moveTo(x,M.t);ctx.lineTo(x,cv.height-M.b);ctx.stroke();
  ctx.fillText(t.toFixed(2),x,cv.height-M.b+13);}
 ctx.textAlign='left';ctx.fillText('time (s)  span='+(b-a).toFixed(2)+'s',M.l,cv.height-M.b+27);
 // 체중부하 cycle 구간 음영 + 번호 (프로토콜: 4회)
 (D.cycles||[]).forEach(c=>{const x1=X(c.onset),x2=X(c.end??c.offset);
  ctx.fillStyle='rgba(255,127,14,.07)';ctx.fillRect(x1,M.t,x2-x1,cv.height-M.t-M.b);
  ctx.fillStyle='#c26a12';ctx.font='11px sans-serif';ctx.textAlign='center';
  ctx.fillText('부하'+(c.id+1)+' (step '+c.step.toFixed(2)+')',(x1+x2)/2,M.t+12);});
 // 분석 anchor 8개 (cycle당 부하 시작/종료) — 여기에 knee가 하나씩 있어야 한다
 (D.anchors||[]).forEach(a=>{const x=X(a.t);
  vline(x,M.t,cv.height-M.t-M.b,a.kind==='on'?'rgba(214,39,40,.55)':'rgba(31,119,180,.55)',[2,3],1.5);});
 // 매칭 실패한 anchor는 굵게 강조
 (D.events||[]).filter(e=>e.edge_idx<0).forEach(e=>{const x=X(e.grf_time);
  vline(x,M.t,cv.height-M.t-M.b,'#d62728',null,2.5);
  ctx.fillStyle='#d62728';ctx.font='bold 11px sans-serif';ctx.textAlign='center';
  ctx.fillText('누락 c'+(e.cycle+1)+e.kind,x,M.t+26);});
 ctx.textAlign='left';
 // GRF strip
 ctx.strokeStyle='#2ca02c';ctx.lineWidth=1;ctx.beginPath();
 for(let i=0;i<D.grf_t.length;i++){let x=X(D.grf_t[i]);let y=gy0+GH/2-(D.grf_signed[i])*(GH/2-6);i?ctx.lineTo(x,y):ctx.moveTo(x,y);}ctx.stroke();
 ctx.strokeStyle='#eee';ctx.beginPath();ctx.moveTo(M.l,gy0+GH/2);ctx.lineTo(cv.width-M.r,gy0+GH/2);ctx.stroke();
 // GRF transitions (both regions)
 for(const t of D.trans){let x=X(t);ctx.strokeStyle='rgba(44,160,44,.4)';ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(x,M.t);ctx.lineTo(x,cv.height-M.b);ctx.stroke();ctx.setLineDash([]);}
 // EAG
 ctx.strokeStyle='#1f77b4';ctx.lineWidth=1;ctx.beginPath();
 for(let i=0;i<D.te.length;i++){let x=X(D.te[i]),y=EY(D.eag[i]);i?ctx.lineTo(x,y):ctx.moveTo(x,y);}ctx.stroke();
 // edges
 const noise=new Set(D.noise_idx||[]);
 D.edges.forEach((e,k)=>{let amp=e.offset_amp-e.onset_amp;const nz=noise.has(k);
  let col=nz?'#aaa':(amp>0?'#d62728':'#2ca02c');
  let x1=X(e.onset_time),y1=EY(e.onset_amp),x2=X(e.offset_time),y2=EY(e.offset_amp);
  ctx.save();if(nz)ctx.setLineDash([3,3]);
  ctx.strokeStyle=col;ctx.lineWidth=k===sel?4:(nz?1.3:2.3);ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();ctx.restore();
  ctx.fillStyle=col;[[x1,y1],[x2,y2]].forEach(([x,y])=>{ctx.beginPath();ctx.arc(x,y,k===sel?7:(nz?3:5),0,7);ctx.fill();});
  ctx.fillStyle=nz?'#999':'#000';ctx.font='11px sans-serif';ctx.fillText('['+k+']',x1-16,y1-4);});
 // add points
 ctx.fillStyle='#ff7f0e';addPts.forEach(p=>{ctx.beginPath();ctx.arc(X(p[0]),EY(p[1]),5,0,7);ctx.fill();});
 renderTbl();
}
function hitKnee(mx,my){for(let k=0;k<D.edges.length;k++){let e=D.edges[k];
  for(const kn of ['on','off']){let t=kn==='on'?e.onset_time:e.offset_time,v=kn==='on'?e.onset_amp:e.offset_amp;
   if(Math.hypot(mx-X(t),my-EY(v))<9)return{k,kn};}}return null;}
function hitEdge(mx,my){for(let k=0;k<D.edges.length;k++){let e=D.edges[k];let x1=X(e.onset_time),x2=X(e.offset_time);
  if(mx>=Math.min(x1,x2)-4&&mx<=Math.max(x1,x2)+4){let y1=EY(e.onset_amp),y2=EY(e.offset_amp);if(my>Math.min(y1,y2)-16&&my<Math.max(y1,y2)+16)return k;}}return -1;}
cv.addEventListener('mousedown',ev=>{if(!D)return;let r=cv.getBoundingClientRect(),mx=ev.clientX-r.left,my=ev.clientY-r.top;
 if(ev.shiftKey||ev.button===1){pan={x:mx,v0:[...view]};ev.preventDefault();return;}   // 좌우 이동
 if(addMode){let t=Tinv(mx);let[st,sv]=snapCorner(t);addPts.push([st,sv]);
   if(addPts.length===2){addPts.sort((p,q)=>p[0]-q[0]);D.edges.push({onset_time:addPts[0][0],onset_amp:addPts[0][1],offset_time:addPts[1][0],offset_amp:addPts[1][1]});D.edges.sort((p,q)=>p.onset_time-q.onset_time);addPts=[];setAdd(false);}
   draw();return;}
 let h=hitKnee(mx,my); if(h){drag=h;sel=h.k;draw();return;}
 let ek=hitEdge(mx,my);
 if(ek>=0){sel=ek;draw();return;}
 pan={x:mx,v0:[...view],moved:false};    // 빈 공간 드래그 = 좌우 이동 (클릭만 하면 선택 해제)
 ev.preventDefault();});
cv.addEventListener('mousemove',ev=>{if(!D)return;let r=cv.getBoundingClientRect(),mx=ev.clientX-r.left;
 if(pan){const dx=mx-pan.x; if(Math.abs(dx)>3)pan.moved=true;
  let dt=dx/PW()*(pan.v0[1]-pan.v0[0]);view=[pan.v0[0]-dt,pan.v0[1]-dt];clampView();draw();return;}
 if(!drag)return;let t=Tinv(mx);
 let[st,sv]=snapCorner(t);let e=D.edges[drag.k];if(drag.kn==='on'){e.onset_time=st;e.onset_amp=sv;}else{e.offset_time=st;e.offset_amp=sv;}draw();});
window.addEventListener('mouseup',()=>{
 if(pan&&!pan.moved&&sel>=0){sel=-1;draw();}   // 빈 공간 클릭(이동 없음) = 선택 해제
 pan=null;
 if(drag){D.edges.sort((p,q)=>p.onset_time-q.onset_time);drag=null;draw();}});
// 휠 = 커서 위치 기준 가로축 확대/축소
cv.addEventListener('wheel',ev=>{if(!D||!view)return;ev.preventDefault();
 let r=cv.getBoundingClientRect(),mx=ev.clientX-r.left; if(mx<M.l||mx>M.l+PW())return;
 if(ev.shiftKey){panBy(ev.deltaY>0?0.2:-0.2);return;}   // Shift+휠 = 좌우 이동
 let t=Tinv(mx),f=ev.deltaY>0?1.2:1/1.2;
 view=[t-(t-view[0])*f, t+(view[1]-t)*f]; clampView(); draw();},{passive:false});
window.addEventListener('keydown',ev=>{if(ev.target.tagName==='INPUT')return;
 if(ev.key==='Delete'&&sel>=0){D.edges.splice(sel,1);sel=-1;draw();}
 if(ev.key==='f'&&D){fitView();draw();}
 if(ev.key==='ArrowLeft'){ev.preventDefault(); ev.shiftKey?gotoEvent(-1):panBy(-0.25);}
 if(ev.key==='ArrowRight'){ev.preventDefault(); ev.shiftKey?gotoEvent(1):panBy(0.25);}
 if(ev.key==='Home'&&D){const [t0]=dataRange(); centerOn(t0);}
 if(ev.key==='End'&&D){const r=dataRange(); centerOn(r[1]);}});
document.getElementById('panL').onclick=()=>panBy(-0.25);
document.getElementById('panR').onclick=()=>panBy(0.25);
document.getElementById('evL').onclick=()=>gotoEvent(-1);
document.getElementById('evR').onclick=()=>gotoEvent(1);
async function toggleExclude(scope){
 if(!D)return; const ch=scope==='session'?0:D.channel;
 const cur=D.excluded && ((scope==='session')===(D.excluded.scope==='session'));
 const reason=document.getElementById('excreason').value;
 const tgt=scope==='session'?'세션 전체':('ch'+D.channel);
 if(!confirm(`${D.subject}/${D.session} ${tgt}\n${cur?'분석제외를 해제할까요?':'분석에서 제외할까요? ('+reason+')'}`))return;
 const note=cur?'':(prompt('제외 사유 메모(선택):','')||'');
 const res=await fetch(api('api/exclude'),{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({subject:D.subject,session:D.session,channel:ch,reason,note,remove:!!cur})});
 const j=await res.json(); if(j.error){status('ERR '+j.error);return;}
 status(cur?'분석제외 해제됨':'⛔ 분석제외로 표시됨'); await load(); refreshList();}
document.getElementById('exclCh').onclick=()=>toggleExclude('channel');
document.getElementById('exclSes').onclick=()=>toggleExclude('session');
document.getElementById('fit').onclick=()=>{if(D){fitView();draw();}};
document.getElementById('denoise').onclick=()=>{if(!D)return;
 const nz=(D.noise_idx||[]).slice().sort((a,b)=>b-a);
 if(!nz.length){status('노이즈로 판정된 edge 없음');return;}
 nz.forEach(i=>D.edges.splice(i,1)); D.noise_idx=[]; sel=-1;
 status(nz.length+'개 삭제 — Save로 확정하세요'); draw();};
function setAdd(v){addMode=v;addPts=[];document.getElementById('addmode').classList.toggle('on',v);}
document.getElementById('addmode').onclick=()=>setAdd(!addMode);
document.getElementById('del').onclick=()=>{if(sel>=0){D.edges.splice(sel,1);sel=-1;draw();}};
document.getElementById('load').onclick=load;
async function load(){let s=document.getElementById('sess').value,ch=document.getElementById('ch').value;
 status('loading...');let res=await fetch(api('api/data?session='+encodeURIComponent(s)+'&channel='+ch));
 let j=await res.json(); if(j.error){status('ERR: '+j.error);return;} D=j;sel=-1;addPts=[];setAdd(false);fitView();
 const v=D.valid||{};
 document.getElementById('meta').innerHTML=
  `${D.subject}/${D.session} ch${D.channel} · source=${D.source} · offset corr=${D.corrected_offset} (${D.method})`+
  `<br><b class="${v.priority==='high'?'bad':(v.priority==='low'?'warn':'ok')}">`+
  `${v.priority==='high'?'⚠️ 검토 필요':(v.priority==='low'?'🟡 후순위 검토(한쪽만 채택)':'✅ 프로토콜 충족')}</b>`+
  (v.labels?` <span class="warn">${v.labels}</span>`:'')+
  (D.excluded?` <b class="bad">⛔ 분석제외(${D.excluded.scope==='session'?'세션':'채널'}: ${D.excluded.reason}${D.excluded.note?', '+D.excluded.note:''})</b>`:'')+
  ` — cycle ${v.n_cycles}/${D.expected_cycles} · 측정가능 cycle ${v.n_measured_cycles}/${v.n_cycles}`+
  ` · 이벤트 ${v.n_matched}/${v.n_events} · edge ${v.n_edges}개`+
  ((D.noise_idx||[]).length?` · <span class="bad">노이즈 후보 ${D.noise_idx.length}개</span>`:'')+
  (v.reasons?` · <span class="bad">${v.reasons}</span>`:'');
 const ex=D.excluded;
 document.getElementById('exclCh').textContent=(ex&&ex.scope==='channel')?'⛔ 채널 제외해제':'⛔ 채널 제외';
 document.getElementById('exclSes').textContent=(ex&&ex.scope==='session')?'⛔ 세션 제외해제':'⛔ 세션 제외';
 status('loaded '+D.edges.length+' edges');draw();}
document.getElementById('save').onclick=async()=>{if(!D)return;let res=await fetch(api('api/save'),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({subject:D.subject,session:D.session,channel:D.channel,corrected_offset:D.corrected_offset,edges:D.edges})});let j=await res.json();if(j.error){status('ERR '+j.error);return;}
 status('✅ saved '+j.n+' edges');await load();refreshList();};   // 재검증 위해 재로드
document.getElementById('reset').onclick=async()=>{if(!D)return;await fetch(api('api/reset'),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({subject:D.subject,session:D.session,channel:D.channel})});await load();refreshList();};
function status(t){document.getElementById('status').textContent=t;}
function renderTbl(){const noise=new Set(D.noise_idx||[]);
 const mt=new Map((D.events||[]).filter(e=>e.edge_idx>=0).map(e=>[e.edge_idx,'c'+(e.cycle+1)+(e.kind==='on'?'부하':'이탈')]));
 let h='<table><tr><th>id</th><th>onset</th><th>offset</th><th>amp</th><th>dir</th><th>판정</th></tr>';
 D.edges.forEach((e,k)=>{let amp=(e.offset_amp-e.onset_amp);const nz=noise.has(k);
  h+=`<tr style="${k===sel?'background:#eef':(nz?'color:#999':'')}"><td>${k}</td><td>${e.onset_time.toFixed(2)}</td><td>${e.offset_time.toFixed(2)}</td><td>${amp.toFixed(0)}</td><td>${amp>0?'rise':'fall'}</td><td>${nz?'노이즈':(mt.get(k)||'후보')}</td></tr>`;});
 h+='</table>';
 if(D.per_cycle&&D.per_cycle.length){
  h+='<table style="margin-left:12px"><tr><th>cycle</th><th>부하%</th><th>|amp| 부하</th><th>|amp| 이탈</th><th>대표 amp</th><th>비대칭</th><th>채택</th></tr>';
  D.per_cycle.forEach(p=>{const bad=p.n_events===0;
   const st=p.amp==null?'측정불가':(p.accepted==='both'?'양측':(p.accepted==='on'?'부하만':'이탈만'));
   h+=`<tr style="${bad?'color:#d62728':(p.single_sided?'color:#c26a12':'')}"><td>c${p.cycle+1}</td><td>${p.load_pct??'-'}</td><td>${p.amp_on??'-'}</td><td>${p.amp_off??'-'}</td><td><b>${p.amp??'-'}</b></td><td>${p.asymmetry??'-'}</td><td>${st}</td></tr>`;});
  h+='</table>';}
 document.getElementById('tbl').innerHTML='<div style="display:flex;gap:14px;align-items:flex-start">'+h+'</div>';}
// 세션 드롭다운 — worklist(needs_review)만이 아니라 data/ 아래 전체 세션.
// edge가 확정된 세션은 ✅와 채널 목록, offset 확정 세션은 off✅ 로 표시.
let SESSIONS=[];
function renderList(){
 const sel=document.getElementById('wl'), f=document.getElementById('filter').value;
 const cur=document.getElementById('sess').value;
 const done=x=>x.edge_channels.length>0;
 const proto=x=>x.proto_total&&x.proto_ok<x.proto_total;
 const isExc=x=>x.excl_session||(x.excl_channels&&x.excl_channels.length);
 const keep=x=> f==='all'?true : f==='review'?x.in_worklist : f==='done'?done(x)
              : f==='proto'?proto(x) : f==='excluded'?isExc(x) : !done(x);
 const rows=SESSIONS.filter(keep), nDone=SESSIONS.filter(done).length;
 sel.innerHTML=`<option value="">— 세션 선택 (표시 ${rows.length} / 전체 ${SESSIONS.length} · edge확정 ${nDone}) —</option>`;
 rows.forEach(x=>{const o=document.createElement('option');o.value=x.dir;
  const pr = x.proto_total ? (x.proto_ok===x.proto_total?'✅':'⚠️')+x.proto_ok+'/'+x.proto_total : '';
  const ex = x.excl_session?('제외:'+x.excl_session):(x.excl_channels&&x.excl_channels.length?('제외 ch'+x.excl_channels.join(',')):'');
  o.textContent=(x.excl_session?'⛔ ':done(x)?'✔ ':x.in_worklist?'▲ ':'　 ')+x.subject+' / '+x.session+
   (ex?'  ['+ex+']':'')+
   (pr?'  [프로토콜 '+pr+(x.proto_cycles!=null&&x.proto_cycles!==4?' cyc'+x.proto_cycles:'')+']':'')+
   (done(x)?'  [edge ch'+x.edge_channels.join(',')+']':'')+
   (x.manual_offset!=null?'  [off✅]':'')+
   (x.reason?'  · '+x.reason:'');
  sel.appendChild(o);});
 sel.value=cur;
}
function refreshList(){return fetch(api('api/sessions')).then(r=>r.json()).then(rows=>{SESSIONS=rows;renderList();});}
document.getElementById('filter').onchange=renderList;
document.getElementById('wl').onchange=()=>{const v=document.getElementById('wl').value;
 if(v){document.getElementById('sess').value=v;load();}};
refreshList();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description='EAG edge annotation GUI (local)')
    ap.add_argument('--port', type=int, default=8765, help='포트 (기본 8765, 3002 금지)')
    ap.add_argument('--host', default='127.0.0.1')
    args = ap.parse_args()
    if args.port == 3002:
        raise SystemExit('port 3002는 api-server 전용이라 사용 금지')
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Edge Annotator: http://{args.host}:{args.port}  (Ctrl-C 종료)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == '__main__':
    main()
