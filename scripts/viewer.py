"""Latent Mischief local workbench. Standard library HTTP, no admin or frontend build."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import subprocess
import sys
import threading
from urllib.parse import parse_qs, urlparse

from storage import write_json

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
WEB = ROOT / "ui"
TOKEN = secrets.token_urlsafe(32)
JOBS = {}
JOB_LOCK = threading.Lock()
ANNOTATION_LOCK = threading.Lock()
PERSONA_LOCK = threading.Lock()
LIVE_LOCK = threading.Lock()
LIVE_PROCESS = None
LIVE_LOG = None
MAPPING_DESIGNS = {
    "pilot": ("configs/exploration.json", "scripts/explore.py", "Original six-channel pilot"),
    "replication": ("configs/completion-replication.json", "scripts/map_experiment.py", "Completion-candidate replication"),
    "expanded": ("configs/exploration-expanded.json", "scripts/map_experiment.py", "Twelve additional random channels"),
}
EXTRA_DESIGNS = {
    'joint': ('configs/joint-characters.json', 'scripts/joint_characters.py', 86),
    'characters': ('configs/characters.json', 'scripts/character_lab.py', 170),
    'replicate': ('configs/offset-replication.json', 'scripts/replicate_offsets.py', 158),
}


def read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def persona_catalog():
    catalog = read_json(RUNS / 'persona-catalog.json') or read_json(ROOT / 'experiments/personas/registry.json', {'recipes': [], 'runs': {}, 'run_refs': {}})
    overrides = read_json(RUNS / 'persona-names.json', {})
    for entry in catalog['recipes']:
        if entry['id'] in overrides:
            entry.update(overrides[entry['id']])
            entry['local_name_override'] = True
    return catalog


def performance_runs():
    result=[]
    for folder in sorted(RUNS.glob('*PERF-*'),reverse=True):
        metadata=read_json(folder/'metadata.json')
        if not metadata:
            cfg=read_json(folder/'source/experiment-config.json')
            identity=read_json(folder/'identity.json',{})
            if not cfg or not identity: continue
            metadata={'experiment':cfg['experiment_id'],'model':identity['model'],'runtime_attention':cfg['attention'],'experiment_config':cfg}
        rows=read_json(folder/'outputs.json',[])
        result.append({'id':folder.name,'metadata':metadata,'summary':read_json(folder/'summary.json'),
                       'rows':[{k:v for k,v in row.items() if k not in ('text','token_ids')} for row in rows],
                       'checks':read_json(folder/'checks.json'),'preflight':read_json(folder/'preflight.json'),
                       'interruption':read_json(folder/'interruption.json')})
    return result


def run_path(name):
    if not name or Path(name).name != name or "/" in name or "\\" in name or name in (".", ".."):
        raise ValueError("Invalid run ID")
    path = (RUNS / name).resolve()
    if path.parent != RUNS.resolve():
        raise ValueError("Invalid run path")
    return path


def runs():
    result = []
    for folder in sorted(RUNS.iterdir(), reverse=True):
        if not folder.is_dir() or not (folder / "metadata.json").is_file():
            continue
        outputs = read_json(folder / "outputs.json", [])
        summary = read_json(folder / "summary.json")
        checks = read_json(folder / "checks.json")
        metadata = read_json(folder / 'metadata.json', {})
        live = '-LIVE-' in folder.name
        result.append({"id": folder.name, "outputs": len(outputs),
                       'model_id':metadata.get('model',metadata.get('config',{})).get('model_id'),
                       'model_key':metadata.get('model_key','lfm3'),'experiment':metadata.get('experiment'),
                       "screen_conditions": len(read_json(folder / 'screening.json', [])) if 'CAL-' in folder.name else 0,
                       "status": "complete" if summary or (checks and "baseline" in folder.name) else ("interrupted" if (folder / 'interruption.json').is_file() else "partial"),
                       "kind": "live" if live else "performance" if "-PERF-" in folder.name else ("characters" if "CHAR-" in folder.name else ("calibration" if "CAL-" in folder.name else ("exploration" if "MAP-" in folder.name else ("personas" if "persona" in folder.name else "baseline"))))})
    return result


def job_states():
    states = []
    for job_id, job in JOBS.items():
        code = job["process"].poll()
        if code is not None and job["state"]["status"] in ("running", "cancelled") and "exit_code" not in job["state"]:
            status = job["state"]["status"]
            job["state"].update(status=status if status == "cancelled" else ("complete" if code == 0 else "failed"), exit_code=code)
            job["log"].close()
            write_json(job["folder"] / "job.json", job["state"])
            if code == 0:
                # CPU-only postprocessing; never rewrites raw evidence or publishes local notes.
                catalog_log = (job['folder'] / 'catalog.log').open('w', encoding='utf-8')
                try:
                    subprocess.Popen([sys.executable,str(ROOT/'scripts/catalog.py')],cwd=ROOT,
                        stdout=catalog_log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                finally:
                    catalog_log.close()
        states.append({**job["state"], "managed": True})
    history = RUNS / "jobs"
    if history.is_dir():
        for folder in history.iterdir():
            if not folder.is_dir() or folder.name in JOBS:
                continue
            saved = read_json(folder / "job.json")
            if isinstance(saved, dict):
                saved = {**saved, "managed": False}
                if saved.get("status") == "running":
                    saved["status"] = "unmanaged"
                states.append(saved)
    return sorted(states, key=lambda item: item.get("started", ""))


def job_folder(name):
    if not name or Path(name).name != name or "/" in name or "\\" in name or name in (".", ".."):
        raise ValueError("Invalid job ID")
    path = (RUNS / "jobs" / name).resolve()
    if path.parent != (RUNS / "jobs").resolve():
        raise ValueError("Invalid job path")
    return path


def gpu_job_running(ignore_pids=()):
    import psutil
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            if process.info['pid'] in ignore_pids: continue
            cmd = process.info["cmdline"] or []
            scripts = {Path(arg).name for arg in cmd}
            if {"personas.py", "explore.py", "map_experiment.py", "calibrate.py", "calibrated_map.py", "resume_calibrated.py", "character_lab.py", "joint_characters.py", "replicate_offsets.py", "performance.py", "live_worker.py", "qwen_eval.py"} & scripts or ("lab.py" in scripts and "smoke" in cmd):
                return True
        except (psutil.Error, OSError):
            continue
    return False


def live_process_ids():
    """Windows virtualenv launchers can own the actual Python worker as a child."""
    import psutil
    if not LIVE_PROCESS or LIVE_PROCESS.poll() is not None: return []
    try: return [LIVE_PROCESS.pid, *[p.pid for p in psutil.Process(LIVE_PROCESS.pid).children(recursive=True)]]
    except psutil.NoSuchProcess: return []


def start_live(body):
    global LIVE_PROCESS,LIVE_LOG
    models=read_json(ROOT/'configs/models.json',{})
    key=body.get('model_key'); identifier=body.get('recipe_id','default')
    if key not in models: raise ValueError('Choose a registered model')
    entry=next((x for x in persona_catalog()['recipes'] if x['id']==identifier),None)
    if identifier!='default' and entry is None: raise ValueError('Choose an existing model-specific recipe')
    if entry and entry['recipe']['model']['model_id']!=models[key]['model_id']: raise ValueError('Recipe belongs to a different model')
    prompt=body.get('prompt','');system=body.get('system','You are a helpful assistant.')
    tokens=body.get('max_new_tokens',256);thinking=body.get('thinking',False)
    if not isinstance(prompt,str) or not prompt.strip() or len(prompt)>1000000: raise ValueError('Enter a prompt (up to one million characters)')
    if not isinstance(system,str) or len(system)>16000 or type(tokens) is not int or not 16<=tokens<=2048 or type(thinking) is not bool: raise ValueError('Invalid generation settings')
    histories={}
    for name in ['history_default','history_edited']:
        history=body.get(name,[])
        if not isinstance(history,list) or len(history)>40 or any(not isinstance(x,dict) or x.get('role') not in ['user','assistant'] or not isinstance(x.get('content'),str) for x in history): raise ValueError('Invalid branch history')
        if sum(len(x['content']) for x in history)>1000000: raise ValueError('Conversation history too large')
        histories[name]=history
    with LIVE_LOCK:
        own_pid=LIVE_PROCESS.pid if LIVE_PROCESS and LIVE_PROCESS.poll() is None else None
        if gpu_job_running(ignore_pids=live_process_ids()): raise ValueError('A batch/benchmark is using the GPU. Wait for it to finish.')
        pending=[p for p in RUNS.glob('*-LIVE-*') if read_json(p/'job.json',{}).get('status') in ['queued','running']]
        if pending and own_pid is not None: raise ValueError('A live comparison is already running')
        if pending:
            for previous in pending: write_json(previous/'job.json',{'status':'interrupted','error':'Previous live worker ended; partial evidence preserved.'})
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'); identifier_run=stamp+'-LIVE-'+key
        folder=RUNS/identifier_run;folder.mkdir()
        write_json(folder/'job.json',{'status':'queued','model_key':key,'recipe_id':identifier})
        write_json(folder/'request.json',{'model_key':key,'recipe_id':identifier,'prompt':prompt,'system':system,'max_new_tokens':tokens,'thinking':thinking,'seed':42,**histories})
        if own_pid is None:
            LIVE_LOG=(RUNS/'live-worker.log').open('a',encoding='utf-8')
            LIVE_PROCESS=subprocess.Popen([sys.executable,str(ROOT/'scripts/live_worker.py')],cwd=ROOT,stdout=LIVE_LOG,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        return {'id':identifier_run,'status':'queued'}


def stop_live():
    global LIVE_PROCESS,LIVE_LOG
    with LIVE_LOCK:
        if LIVE_PROCESS and LIVE_PROCESS.poll() is None:
            import psutil
            processes=[psutil.Process(pid) for pid in live_process_ids()]
            for process in reversed(processes):
                try: process.terminate()
                except psutil.NoSuchProcess: pass
            psutil.wait_procs(processes,timeout=10)
            LIVE_PROCESS.wait(timeout=10)
        if LIVE_LOG: LIVE_LOG.close()
        LIVE_PROCESS=LIVE_LOG=None
        for folder in RUNS.glob('*-LIVE-*'):
            if read_json(folder/'job.json',{}).get('status') in ['queued','running']:
                write_json(folder/'job.json',{**read_json(folder/'job.json',{}),'status':'cancelled','error':'Stopped by user; partial outputs preserved.'})
    return {'stopped':True}


def mapping_designs():
    result = {}
    for key, (config_name, _script, label) in MAPPING_DESIGNS.items():
        cfg = read_json(ROOT / config_name)
        prompts = read_json(ROOT / cfg.get('prompt_file', 'prompts/open-exploration.json'))
        count = len(prompts)
        interventions = sum(len(x.get('gains', cfg.get('gains', []))) for x in cfg.get('candidates', []))
        interventions += len(cfg.get('layers', [])) * cfg.get('channels_per_layer', 0) * len(cfg.get('gains', []))
        control = cfg.get('random_controls')
        if control:
            interventions += control['count'] * len(control['gains'])
        result[key] = {'label': label, 'config': cfg, 'prompt_count': count,
                       'generation_count': count + 2 + count * interventions}
    return result


def calibrated_designs():
    cfg = read_json(ROOT / 'configs/calibrated-mapping.json')
    available = []
    for folder in sorted(RUNS.iterdir(), reverse=True):
        if not folder.is_dir() or 'CAL-' not in folder.name:
            continue
        summary = read_json(folder / 'summary.json', {})
        if summary.get('completed') is True and summary.get('selected_channels', 0) > 0 and all(summary.get('checks', {}).values()):
            metadata = read_json(folder / 'metadata.json', {})
            config = metadata.get('config', {})
            prompts = read_json(folder / 'source' / config.get('evaluation_prompts', 'missing'), [])
            units = read_json(folder / 'selected.json', [])
            available.append({'id': folder.name, 'selected_channels': summary['selected_channels'],
                              'generation_count': len(prompts) * (1 + len(units)) + 2, 'config': config})
    return {'config': cfg, 'available': available}


def start_job(body):
    mode = body.get("mode")
    if body.get('model_key','lfm3')!='lfm3': raise ValueError('These frozen research designs are LFM3-specific. Use Chat & Try for Qwen base versus published edit.')
    if mode not in ("smoke", "personas", "mapping", "calibration", "offsets", *EXTRA_DESIGNS):
        raise ValueError("Choose an allowlisted experiment mode")
    if LIVE_PROCESS and LIVE_PROCESS.poll() is None:
        if any(read_json(p/'job.json',{}).get('status') in ['queued','running'] for p in RUNS.glob('*-LIVE-*')):
            raise ValueError('A live comparison is running. Wait or stop it first.')
        stop_live()
    with JOB_LOCK:
        job_states()
        if any(j["state"]["status"] == "running" for j in JOBS.values()) or gpu_job_running():
            raise ValueError("An experiment is already running. Wait or stop the managed job first.")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        job_id = f"{stamp}-{mode}"
        folder = RUNS / "jobs" / job_id
        cfg = None
        calibration_id = None
        script_name = "scripts/personas.py"
        if mode in EXTRA_DESIGNS:
            config_name, script_name, _count = EXTRA_DESIGNS[mode]
            cfg = read_json(ROOT / config_name)
        if mode == 'calibration':
            cfg = read_json(ROOT / 'configs/calibrated-mapping.json')
            script_name = 'scripts/calibrate.py'
        if mode == 'offsets':
            calibration_id = body.get('calibration')
            available = {x['id']: x for x in calibrated_designs()['available']}
            if calibration_id not in available:
                raise ValueError('Choose a completed qualifying calibration from local history')
            cfg = available[calibration_id]['config']
        if mode == "mapping":
            design = body.get("design")
            if design not in MAPPING_DESIGNS:
                raise ValueError("Choose an allowlisted mapping design")
            config_name, script_name, _label = MAPPING_DESIGNS[design]
            cfg = read_json(ROOT / config_name)
        if mode == "personas":
            cfg = read_json(ROOT / "configs/personas.json")
            layers = body.get("layers", cfg["layers"])
            doses = body.get("doses", cfg["doses"])
            tokens = body.get("max_new_tokens", 96)
            controls = body.get("random_controls_per_layer", 2)
            if not isinstance(layers, list) or not 1 <= len(layers) <= 6 or any(type(x) is not int or not 0 <= x < 24 for x in layers):
                raise ValueError("Choose 1–6 distinct layers between 0 and 23")
            if len(set(layers)) != len(layers):
                raise ValueError("Layers must be distinct")
            if not isinstance(doses, list) or not 1 <= len(doses) <= 5 or any(type(x) not in (int, float) or not -1 <= x <= 1 for x in doses):
                raise ValueError("Choose 1–5 finite doses between -1 and 1")
            if type(tokens) is not int or not 32 <= tokens <= 256:
                raise ValueError("Token cap must be 32–256")
            if type(controls) is not int or not 1 <= controls <= 8:
                raise ValueError("Random controls must be 1–8 per layer")
            cfg.update(layers=layers, doses=doses, max_new_tokens=tokens, random_controls_per_layer=controls)
            cfg["random_control_dose"] = body.get("random_control_dose", 0.15)
            dose = cfg["random_control_dose"]
            if type(dose) not in (int, float) or not -1 <= dose <= 1:
                raise ValueError("Random control dose must be between -1 and 1")
        folder.mkdir(parents=True)
        command = [sys.executable, str(ROOT / "scripts/lab.py"), "smoke"]
        if cfg is not None:
            cfg_path = folder / "config.json"
            write_json(cfg_path, cfg)
            command = [sys.executable, str(ROOT / script_name), "--config", str(cfg_path)]
        if mode == 'offsets':
            command = [sys.executable, str(ROOT / 'scripts/calibrated_map.py'), '--calibration', calibration_id]
        log = (folder / "stdout.log").open("w", encoding="utf-8")
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        state = {"id": job_id, "mode": mode, "status": "running", "pid": process.pid,
                 "started": datetime.now(timezone.utc).isoformat(), "config": cfg,
                 "design": body.get("design") if mode == "mapping" else None,
                 "calibration": calibration_id}
        JOBS[job_id] = {"process": process, "state": state, "folder": folder, "log": log}
        write_json(folder / "job.json", state)
        return state


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def send(self, value, status=200, kind="application/json; charset=utf-8"):
        data = json.dumps(value).encode() if kind.startswith("application/json") else value
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(data)

    def valid_host(self):
        return self.headers.get("Host") in (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}")

    def do_GET(self):
        if not self.valid_host():
            return self.send({"error": "Localhost only"}, 403)
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/state":
                own_pid=LIVE_PROCESS.pid if LIVE_PROCESS and LIVE_PROCESS.poll() is None else None
                live_active=any(read_json(p/'job.json',{}).get('status') in ['queued','running'] for p in RUNS.glob('*-LIVE-*'))
                return self.send({"runs": runs(), "jobs": job_states(), "token": TOKEN,
                                  "gpu_busy": gpu_job_running(ignore_pids=live_process_ids() if own_pid and not live_active else []),
                                  "live_model_ready":bool(own_pid and not live_active), "defaults": read_json(ROOT / "configs/personas.json"),
                                  "mapping_designs": mapping_designs(), "calibrated_designs": calibrated_designs(),
                                  "extra_designs": {name:{'config':read_json(ROOT / paths[0]),'generation_count':paths[2]} for name,paths in EXTRA_DESIGNS.items()}})
            if parsed.path == '/api/personas':
                return self.send(persona_catalog())
            if parsed.path == '/api/models':
                return self.send(read_json(ROOT/'configs/models.json',{}))
            if parsed.path == '/api/live':
                name=parse_qs(parsed.query).get('id',[''])[0];folder=run_path(name)
                if '-LIVE-' not in name: raise ValueError('Invalid live comparison ID')
                state=read_json(folder/'job.json',{'status':'missing'})
                if 'branches' in state: state['branches']={branch:{k:v for k,v in data.items() if k not in ['input_token_ids','token_ids','messages']} for branch,data in state['branches'].items()}
                return self.send({'id':name,**state})
            if parsed.path == '/api/performance':
                return self.send({'runs':performance_runs(),'identity':read_json(RUNS/'model-identity.json')})
            if parsed.path == "/api/run":
                folder = run_path(parse_qs(parsed.query).get("id", [""])[0])
                if not (folder / "metadata.json").is_file():
                    return self.send({"error": "Run not found"}, 404)
                return self.send({"id": folder.name, "metadata": read_json(folder / "metadata.json"),
                                  "outputs": read_json(folder / "outputs.json", []),
                                  "checks": read_json(folder / "checks.json"),
                                  "summary": read_json(folder / "summary.json"),
                                  "profiles": read_json(folder / 'profiles.json', []),
                                  "rankings": read_json(folder / 'rankings.json', {}),
                                  "recipe_refs": persona_catalog().get('run_refs',{}).get(folder.name,{}),
                                  "interruption": read_json(folder / 'interruption.json'),
                                  "calibration": {'selected': read_json(folder / 'selected.json', []),
                                                  'activation_summary': read_json(folder / 'activation-summary.json'),
                                                  'screening': read_json(folder / 'screening.json', [])}
                                                  if 'CAL-' in folder.name else None,
                                  "review_order": read_json(folder / "review-order.json", []),
                                  "annotations": read_json(folder / "annotations.json", {})})
            if parsed.path == "/api/log":
                job_id = parse_qs(parsed.query).get("id", [""])[0]
                folder = JOBS[job_id]["folder"] if job_id in JOBS else job_folder(job_id)
                if not (folder / "job.json").is_file() or not (folder / "stdout.log").is_file():
                    return self.send({"error": "Job log not found"}, 404)
                text = (folder / "stdout.log").read_text(encoding="utf-8", errors="replace")
                return self.send({"text": text[-24000:]})
            files = {"/": ("index.html", "text/html; charset=utf-8"),
                     "/workflow.js": ("workflow.js", "text/javascript; charset=utf-8"),
                     "/bootstrap.js": ("bootstrap.js", "text/javascript; charset=utf-8"),
                     "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                     "/compare.js": ("compare.js", "text/javascript; charset=utf-8"),
                     "/style.css": ("style.css", "text/css; charset=utf-8")}
            if parsed.path in files:
                name, kind = files[parsed.path]
                return self.send((WEB / name).read_bytes(), kind=kind)
            return self.send({"error": "Not found"}, 404)
        except (ValueError, OSError) as error:
            return self.send({"error": str(error)}, 400)

    def do_POST(self):
        origin = self.headers.get("Origin")
        allowed_origins = {f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"}
        if not self.valid_host() or (origin and origin not in allowed_origins) or self.headers.get("X-Lab-Token") != TOKEN:
            return self.send({"error": "Local request token required"}, 403)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            limit=4_000_000 if self.path=='/api/live' else 16384
            if not 0 < length <= limit or not self.headers.get("Content-Type", "").startswith("application/json"):
                raise ValueError("Small JSON body required")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("JSON object required")
            if self.path == "/api/jobs":
                return self.send(start_job(body), 201)
            if self.path == '/api/live':
                return self.send(start_live(body),201)
            if self.path == '/api/live/stop':
                return self.send(stop_live())
            if self.path == '/api/personas':
                identifier=body.get('id'); catalog=persona_catalog()
                if identifier not in {x['id'] for x in catalog['recipes']}: raise ValueError('Unknown immutable recipe ID')
                name,status,note=body.get('observed_name',''),body.get('name_status','provisional'),body.get('naming_notes','')
                if not isinstance(name,str) or len(name)>80 or any(ord(c)<32 for c in name): raise ValueError('Use a short, single-line nickname')
                if status not in ['unreviewed','reference-only','provisional','replicated-description','rejected']: raise ValueError('Unknown review status')
                if not isinstance(note,str) or len(note)>4000: raise ValueError('Keep naming notes under 4000 characters')
                if status=='replicated-description' and not note.strip(): raise ValueError('Record the replication evidence and scope')
                with PERSONA_LOCK:
                    overrides=read_json(RUNS/'persona-names.json',{})
                    overrides[identifier]={'observed_name':name.strip(),'name_status':status,'naming_notes':note}
                    write_json(RUNS/'persona-names.json',overrides)
                return self.send({'saved':True,'id':identifier,'scope':'Local annotation; recipe and raw evidence unchanged'})
            if self.path == "/api/stop":
                job = JOBS.get(body.get("id"))
                if not job:
                    raise ValueError("Managed job not found")
                if job["process"].poll() is None:
                    job["process"].terminate()
                    job["state"]["status"] = "cancelled"
                    write_json(job["folder"] / "job.json", job["state"])
                return self.send({"jobs": job_states()})
            if self.path == "/api/annotations":
                folder = run_path(body.get("run", ""))
                if not (folder / "metadata.json").is_file():
                    raise ValueError("Unknown run")
                index = body.get("index")
                outputs = read_json(folder / "outputs.json", [])
                if type(index) is not int or not 0 <= index < len(outputs):
                    raise ValueError("Unknown output")
                label, note = body.get("label", "unreviewed"), body.get("note", "")
                if label not in ("unreviewed", "interesting", "no-effect", "damaged", "needs-review") or not isinstance(note, str) or len(note) > 4000:
                    raise ValueError("Invalid annotation")
                with ANNOTATION_LOCK:
                    annotations = read_json(folder / "annotations.json", {})
                    annotations[str(index)] = {"label": label, "note": note, "updated": datetime.now(timezone.utc).isoformat()}
                    write_json(folder / "annotations.json", annotations)
                return self.send({"saved": True})
            return self.send({"error": "Not found"}, 404)
        except (ValueError, TypeError, OSError) as error:
            return self.send({"error": str(error)}, 400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    RUNS.mkdir(exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Latent Mischief: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_live()
        server.server_close()


if __name__ == "__main__":
    main()
