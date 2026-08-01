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
import edge_store


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

    man = edge_store.get_channel_edges(pair.subject_name, pair.session_name, channel)
    if man is not None:
        edges = man
        source = 'manual'
    else:
        auto = G.detect_eag_edges(te, eag, fs=sa.eag.sample_rate)
        edges = [{'onset_time': e[0], 'onset_amp': e[1],
                  'offset_time': e[3], 'offset_amp': e[4]} for e in auto]
        source = 'auto'

    step = max(1, len(te) // 6000)   # 표시 다운샘플 (~6000점)
    r1 = lambda arr: [round(float(x), 1) for x in arr]
    r3 = lambda arr: [round(float(x), 3) for x in arr]
    return {
        'subject': pair.subject_name, 'session': pair.session_name, 'channel': channel,
        'corrected_offset': round(float(off.corrected_offset), 3), 'method': off.method,
        'source': source,
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
    import csv
    out = []
    with open(p, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            out.append({'subject': row.get('subject', ''), 'session': row.get('session', ''),
                        'reason': row.get('reason', '')})
    return out


def find_session_dir(subject: str, session: str) -> str:
    for p in Path('data').rglob('BrainFlow-RAW_*.csv'):
        if p.parent.parent.name == subject and p.parent.name.endswith('-' + session):
            return str(p.parent)
    return ''


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
</style></head><body>
<div id="bar">
 <select id="wl" title="needs_review worklist"></select>
 <input id="sess" placeholder="session dir (또는 worklist 선택)" size="42">
 <label>ch <input id="ch" type="number" value="1" min="1" max="8" style="width:44px"></label>
 <button id="load">Load</button>
 <button id="addmode">Add mode</button>
 <button id="del">Delete sel</button>
 <label><input type="checkbox" id="snap" checked> snap</label>
 <button id="save">Save</button>
 <button id="reset">Reset→auto</button>
 <span id="status"></span>
</div>
<div id="tip">드래그=knee 이동 · Add mode 후 트레이스 2점 클릭=edge 추가 · edge 클릭 선택 후 Delete/Del키 · rise=빨강 fall=초록</div>
<canvas id="cv" width="1400" height="560"></canvas>
<div id="meta"></div><div id="tbl"></div>
<script>
// API 경로는 현재 문서 기준 상대경로로 — code-server의 /proxy/<port>/ 중계 아래에서도 동작
const BASE=(()=>{let p=location.pathname; if(/\.[a-z]+$/i.test(p))p=p.replace(/[^/]*$/,'');
 return p.endsWith('/')?p:p+'/';})();
const api=p=>BASE+p;
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
let D=null, sel=-1, addPts=[], addMode=false, drag=null;
const M={l:55,r:15,t:15,b:20}, GH=140; // GRF strip height
function xr(){return [D.te[0], D.te[D.te.length-1]];}
function eagYR(){let a=D.eag; return [Math.min(...a), Math.max(...a)];}
function X(t){let[a,b]=xr();return M.l+(t-a)/(b-a)*(cv.width-M.l-M.r);}
function Tinv(px){let[a,b]=xr();return a+(px-M.l)/(cv.width-M.l-M.r)*(b-a);}
function eagH(){return cv.height-M.t-M.b-GH;}
function EY(v){let[a,b]=eagYR();return M.t+(b-v)/(b-a)*eagH();}
function ampAt(t){let i=nearestIdx(t);return D.eag[i];}
function nearestIdx(t){let lo=0,hi=D.te.length-1;while(lo<hi){let m=(lo+hi)>>1;if(D.te[m]<t)lo=m+1;else hi=m;}return lo;}
function snapCorner(t){ if(!document.getElementById('snap').checked) return [t,ampAt(t)];
  let i0=nearestIdx(t-0.3),i1=nearestIdx(t+0.3); if(i1-i0<5)return[t,ampAt(t)];
  let best=i0,bv=-1; for(let i=i0+1;i<i1-1;i++){let d2=Math.abs(D.eag[i+1]-2*D.eag[i]+D.eag[i-1]); if(d2>bv){bv=d2;best=i;}} return [D.te[best],D.eag[best]];}
function draw(){ if(!D)return; ctx.clearRect(0,0,cv.width,cv.height);
 // GRF strip
 let gy0=cv.height-GH, [a,b]=xr();
 ctx.strokeStyle='#2ca02c';ctx.lineWidth=1;ctx.beginPath();
 for(let i=0;i<D.grf_t.length;i++){let x=X(D.grf_t[i]);let y=gy0+GH/2-(D.grf_signed[i])*(GH/2-6);i?ctx.lineTo(x,y):ctx.moveTo(x,y);}ctx.stroke();
 ctx.strokeStyle='#eee';ctx.beginPath();ctx.moveTo(M.l,gy0+GH/2);ctx.lineTo(cv.width-M.r,gy0+GH/2);ctx.stroke();
 // GRF transitions (both regions)
 for(const t of D.trans){let x=X(t);ctx.strokeStyle='rgba(44,160,44,.4)';ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(x,M.t);ctx.lineTo(x,cv.height-M.b);ctx.stroke();ctx.setLineDash([]);}
 // EAG
 ctx.strokeStyle='#1f77b4';ctx.lineWidth=1;ctx.beginPath();
 for(let i=0;i<D.te.length;i++){let x=X(D.te[i]),y=EY(D.eag[i]);i?ctx.lineTo(x,y):ctx.moveTo(x,y);}ctx.stroke();
 // edges
 D.edges.forEach((e,k)=>{let amp=e.offset_amp-e.onset_amp;let col=amp>0?'#d62728':'#2ca02c';
  let x1=X(e.onset_time),y1=EY(e.onset_amp),x2=X(e.offset_time),y2=EY(e.offset_amp);
  ctx.strokeStyle=col;ctx.lineWidth=k===sel?4:2.3;ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();
  ctx.fillStyle=col;[[x1,y1],[x2,y2]].forEach(([x,y])=>{ctx.beginPath();ctx.arc(x,y,k===sel?7:5,0,7);ctx.fill();});
  ctx.fillStyle='#000';ctx.font='11px sans-serif';ctx.fillText('['+k+']',x1-16,y1-4);});
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
 if(addMode){let t=Tinv(mx);let[st,sv]=snapCorner(t);addPts.push([st,sv]);
   if(addPts.length===2){addPts.sort((p,q)=>p[0]-q[0]);D.edges.push({onset_time:addPts[0][0],onset_amp:addPts[0][1],offset_time:addPts[1][0],offset_amp:addPts[1][1]});D.edges.sort((p,q)=>p.onset_time-q.onset_time);addPts=[];setAdd(false);}
   draw();return;}
 let h=hitKnee(mx,my); if(h){drag=h;sel=h.k;draw();return;}
 let ek=hitEdge(mx,my); sel=ek; draw();});
cv.addEventListener('mousemove',ev=>{if(!drag||!D)return;let r=cv.getBoundingClientRect();let t=Tinv(ev.clientX-r.left);
 let[st,sv]=snapCorner(t);let e=D.edges[drag.k];if(drag.kn==='on'){e.onset_time=st;e.onset_amp=sv;}else{e.offset_time=st;e.offset_amp=sv;}draw();});
window.addEventListener('mouseup',()=>{if(drag){D.edges.sort((p,q)=>p.onset_time-q.onset_time);drag=null;draw();}});
window.addEventListener('keydown',ev=>{if(ev.key==='Delete'&&sel>=0){D.edges.splice(sel,1);sel=-1;draw();}});
function setAdd(v){addMode=v;addPts=[];document.getElementById('addmode').classList.toggle('on',v);}
document.getElementById('addmode').onclick=()=>setAdd(!addMode);
document.getElementById('del').onclick=()=>{if(sel>=0){D.edges.splice(sel,1);sel=-1;draw();}};
document.getElementById('load').onclick=load;
async function load(){let s=document.getElementById('sess').value,ch=document.getElementById('ch').value;
 status('loading...');let res=await fetch(api('api/data?session='+encodeURIComponent(s)+'&channel='+ch));
 let j=await res.json(); if(j.error){status('ERR: '+j.error);return;} D=j;sel=-1;addPts=[];setAdd(false);
 document.getElementById('meta').textContent=`${D.subject}/${D.session} ch${D.channel} · source=${D.source} · offset corr=${D.corrected_offset} (${D.method})`;
 status('loaded '+D.edges.length+' edges');draw();}
document.getElementById('save').onclick=async()=>{if(!D)return;let res=await fetch(api('api/save'),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({subject:D.subject,session:D.session,channel:D.channel,corrected_offset:D.corrected_offset,edges:D.edges})});let j=await res.json();status(j.ok?('saved '+j.n+' edges'):('ERR '+j.error));};
document.getElementById('reset').onclick=async()=>{if(!D)return;await fetch(api('api/reset'),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({subject:D.subject,session:D.session,channel:D.channel})});load();};
function status(t){document.getElementById('status').textContent=t;}
function renderTbl(){let h='<table><tr><th>id</th><th>onset</th><th>offset</th><th>amp</th><th>dir</th></tr>';
 D.edges.forEach((e,k)=>{let amp=(e.offset_amp-e.onset_amp);h+=`<tr style="${k===sel?'background:#eef':''}"><td>${k}</td><td>${e.onset_time.toFixed(2)}</td><td>${e.offset_time.toFixed(2)}</td><td>${amp.toFixed(0)}</td><td>${amp>0?'rise':'fall'}</td></tr>`;});
 document.getElementById('tbl').innerHTML=h+'</table>';}
// worklist dropdown
fetch(api('api/worklist')).then(r=>r.json()).then(w=>{let sel=document.getElementById('wl');sel.innerHTML='<option value="">— worklist —</option>';
 w.forEach(x=>{let o=document.createElement('option');o.value=JSON.stringify(x);o.textContent=`${x.subject} ${x.session} · ${x.reason}`.slice(0,70);sel.appendChild(o);});
 sel.onchange=async()=>{if(!sel.value)return;let x=JSON.parse(sel.value);let r=await fetch(api('api/finddir?subject='+encodeURIComponent(x.subject)+'&session_name='+encodeURIComponent(x.session)));let j=await r.json();document.getElementById('sess').value=j.dir;};});
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
