"""API boundary and recording checks; no model download, GPU work or browser required."""
import importlib.util
from html.parser import HTMLParser
import json
from pathlib import Path
import tempfile
import threading
import unittest
import sys
import re
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
spec = importlib.util.spec_from_file_location("viewer", Path(__file__).resolve().parents[1] / "scripts/viewer.py")
viewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer)


class ViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.original_runs = viewer.RUNS
        viewer.RUNS = Path(cls.temp.name)
        run = viewer.RUNS / "test-run"
        run.mkdir()
        viewer.write_json(run / "metadata.json", {"experiment": "test"})
        viewer.write_json(run / "outputs.json", [{"text": "<script>untrusted model output</script>"}])
        cls.server = viewer.ThreadingHTTPServer(("127.0.0.1", 0), viewer.Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
        viewer.RUNS = cls.original_runs
        cls.temp.cleanup()

    def request(self, path, body=None, *, token=True, origin=None, host=None):
        headers = {}
        if body is not None:
            headers["Content-Type"] = "application/json"
            if token:
                headers["X-Lab-Token"] = viewer.TOKEN
        if origin:
            headers["Origin"] = origin
        if host:
            headers["Host"] = host
        request = Request(self.base + path, data=json.dumps(body).encode() if body is not None else None, headers=headers)
        try:
            with urlopen(request) as response:
                return response.status, json.load(response)
        except HTTPError as error:
            return error.code, json.load(error)

    def test_unknown_paths_and_traversal(self):
        self.assertEqual(self.request('/api/run?id=..%2F..')[0], 400)
        self.assertEqual(self.request('/api/run?id=missing')[0], 404)
        self.assertEqual(self.request('/does-not-exist')[0], 404)

    def test_assets_are_local_and_serve_expected_types(self):
        for path, mime in [('/', 'text/html'), ('/bootstrap.js', 'text/javascript'),
                           ('/compare.js', 'text/javascript'), ('/workflow.js', 'text/javascript'),
                           ('/app.js', 'text/javascript'), ('/style.css', 'text/css')]:
            with urlopen(self.base + path) as response:
                self.assertTrue(response.headers['Content-Type'].startswith(mime))
                self.assertIn("default-src 'self'", response.headers['Content-Security-Policy'])
                self.assertGreater(len(response.read()), 100)

    def test_every_script_referenced_by_page_is_served(self):
        html = (viewer.WEB / 'index.html').read_text(encoding='utf-8')
        scripts = re.findall(r'<script\b[^>]*\bsrc="([^"]+)"', html)
        self.assertTrue(scripts)
        for script in scripts:
            with self.subTest(script=script), urlopen(self.base + script) as response:
                self.assertEqual(response.status, 200)
                self.assertTrue(response.headers['Content-Type'].startswith('text/javascript'))
                self.assertNotEqual(response.read().lstrip()[:1], b'{')

    def test_static_workbench_controls_have_unique_html_targets(self):
        class IDs(HTMLParser):
            def __init__(self):
                super().__init__()
                self.ids = []
            def handle_starttag(self, tag, attrs):
                self.ids.extend(value for key, value in attrs if key == 'id')
        parser = IDs()
        parser.feed((viewer.WEB / 'index.html').read_text(encoding='utf-8'))
        references = set()
        for script in ['app.js', 'workflow.js']:
            references.update(re.findall(r'''\$\(["']([A-Za-z0-9_-]+)["']\)''', (viewer.WEB / script).read_text(encoding='utf-8')))
        self.assertEqual(len(parser.ids), len(set(parser.ids)))
        self.assertFalse(references - set(parser.ids), references - set(parser.ids))

    def test_local_host_and_post_origin(self):
        self.assertEqual(self.request('/api/state', host='attacker.invalid')[0], 403)
        self.assertEqual(self.request('/api/jobs', {"mode": "smoke"}, token=False)[0], 403)
        self.assertEqual(self.request('/api/jobs', {"mode": "smoke"}, origin='https://attacker.invalid')[0], 403)

    def test_annotation_preserves_raw_output(self):
        body = {"run": "test-run", "index": 0, "label": "needs-review", "note": "No claim of success."}
        self.assertEqual(self.request('/api/annotations', body)[0], 200)
        status, data = self.request('/api/run?id=test-run')
        self.assertEqual(status, 200)
        self.assertEqual(data['annotations']['0']['label'], 'needs-review')
        self.assertEqual(data['outputs'][0]['text'], '<script>untrusted model output</script>')
        body['index'] = 99
        self.assertEqual(self.request('/api/annotations', body)[0], 400)

    def test_rejects_arbitrary_commands_and_bad_sweeps(self):
        self.assertEqual(self.request('/api/jobs', {"mode": "shell", "command": "anything"})[0], 400)
        with patch.object(viewer, 'gpu_job_running', return_value=False):
            self.assertEqual(self.request('/api/jobs', {"mode": "personas", "layers": [-1]})[0], 400)
            self.assertEqual(self.request('/api/jobs', {"mode": "personas", "doses": [float('nan')]})[0], 400)
            self.assertEqual(self.request('/api/jobs', {"mode": "personas", "max_new_tokens": 100000})[0], 400)

    def test_busy_gpu_prevents_launch(self):
        with patch.object(viewer, 'gpu_job_running', return_value=True):
            code, data = self.request('/api/jobs', {"mode": "smoke"})
            self.assertEqual(code, 400)
            self.assertIn('already running', data['error'])

    def test_live_rejects_cross_model_recipe_and_invalid_settings(self):
        recipe={'id':'lfm-only','recipe':{'model':{'model_id':'LiquidAI/LFM3-1.7B'}}}
        with patch.object(viewer,'persona_catalog',return_value={'recipes':[recipe]}):
            code,data=self.request('/api/live',{'model_key':'qwen08','recipe_id':'lfm-only','prompt':'hello'})
            self.assertEqual(code,400)
            self.assertIn('different model',data['error'])
        for changes in [{'model_key':'unknown'},{'max_new_tokens':0},{'thinking':'false'},{'prompt':''},
                        {'history_default':[{'role':'system','content':'bad history'}]}]:
            body={'model_key':'lfm3','recipe_id':'default','prompt':'test',**changes}
            self.assertEqual(self.request('/api/live',body)[0],400)

    def test_batch_designs_do_not_silently_run_lfm_under_qwen(self):
        self.assertEqual(self.request('/api/jobs',{'mode':'smoke','model_key':'qwen08'})[0],400)

    def test_stopping_live_preserves_partial_reply(self):
        folder=viewer.RUNS/'test-LIVE-lfm3';folder.mkdir(exist_ok=True)
        viewer.write_json(folder/'job.json',{'status':'running','branches':{'baseline':{'text':'partial','partial':True}}})
        with patch.object(viewer,'LIVE_PROCESS',None),patch.object(viewer,'LIVE_LOG',None):
            self.assertEqual(self.request('/api/live/stop',{})[0],200)
        result=viewer.read_json(folder/'job.json')
        self.assertEqual(result['status'],'cancelled')
        self.assertEqual(result['branches']['baseline']['text'],'partial')

    def test_calibrated_launch_is_frozen_and_requires_completed_screen(self):
        class FakeProcess:
            pid = 98
            def poll(self): return 0
        with patch.object(viewer, 'gpu_job_running', return_value=False):
            self.assertEqual(self.request('/api/jobs', {'mode': 'offsets', 'calibration': '../../anything'})[0], 400)
            with patch.object(viewer.subprocess, 'Popen', return_value=FakeProcess()) as launch:
                status, job = self.request('/api/jobs', {'mode': 'calibration', 'max_new_tokens': 999})
                self.assertEqual(status, 201)
                self.assertEqual(job['config']['max_new_tokens'], 192)
                self.assertIn('calibrate.py', launch.call_args.args[0][1])
                viewer.job_states()
                viewer.JOBS.pop(job['id'])

    def test_saved_job_history_and_logs_survive_server_state(self):
        folder = viewer.RUNS / 'jobs' / 'historical-job'
        viewer.write_json(folder / 'job.json', {'id': 'historical-job', 'status': 'running'})
        (folder / 'stdout.log').write_text('Preserved partial experiment log.', encoding='utf-8')
        self.assertEqual(self.request('/api/log?id=historical-job')[1]['text'], 'Preserved partial experiment log.')
        self.assertEqual(self.request('/api/log?id=..%2F..')[0], 400)
        saved = next(x for x in viewer.job_states() if x['id'] == 'historical-job')
        self.assertEqual(saved['status'], 'unmanaged')
        self.assertFalse(saved['managed'])
        self.assertEqual(self.request('/api/stop', {'id': 'historical-job'})[0], 400)

    def test_mapping_designs_are_bounded_and_frozen(self):
        self.assertEqual(viewer.mapping_designs()['replication']['generation_count'], 51)
        self.assertEqual(viewer.mapping_designs()['expanded']['generation_count'], 77)
        with patch.object(viewer, 'gpu_job_running', return_value=False):
            self.assertEqual(self.request('/api/jobs', {'mode': 'mapping', 'design': '../../anything'})[0], 400)
            class FakeProcess:
                pid = 99
                def poll(self): return 0
            with patch.object(viewer.subprocess, 'Popen', return_value=FakeProcess()) as launch:
                status, job = self.request('/api/jobs', {'mode': 'mapping', 'design': 'replication', 'max_new_tokens': 999})
                self.assertEqual(status, 201)
                self.assertEqual(job['config']['max_new_tokens'], 192)
                self.assertIn('map_experiment.py', launch.call_args.args[0][1])
                viewer.job_states()
                viewer.JOBS.pop(job['id'])

    def test_managed_launch_saves_config_and_stop_preserves_evidence(self):
        class FakeProcess:
            pid = 12345
            code = None
            def poll(self):
                return self.code
            def terminate(self):
                self.code = -15
        process = FakeProcess()
        with patch.object(viewer, 'gpu_job_running', return_value=False), patch.object(viewer.subprocess, 'Popen', return_value=process) as launch:
            code, job = self.request('/api/jobs', {'mode': 'personas', 'layers': [13], 'doses': [0.15], 'max_new_tokens': 64})
            self.assertEqual(code, 201)
            cfg_path = viewer.RUNS / 'jobs' / job['id'] / 'config.json'
            self.assertEqual(viewer.read_json(cfg_path)['max_new_tokens'], 64)
            self.assertIn('--config', launch.call_args.args[0])
            self.assertEqual(self.request('/api/stop', {'id': job['id']})[0], 200)
            self.assertTrue(cfg_path.is_file())
            states = viewer.job_states()
            self.assertEqual(next(x for x in states if x['id'] == job['id'])['status'], 'cancelled')
        viewer.JOBS.pop(job['id'])


if __name__ == '__main__':
    unittest.main()
