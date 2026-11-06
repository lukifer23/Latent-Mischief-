"""API boundary and recording checks; no model download, GPU work or browser required."""
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
import sys
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
        for path, mime in [('/', 'text/html'), ('/app.js', 'text/javascript'), ('/style.css', 'text/css')]:
            with urlopen(self.base + path) as response:
                self.assertTrue(response.headers['Content-Type'].startswith(mime))
                self.assertIn("default-src 'self'", response.headers['Content-Security-Policy'])
                self.assertGreater(len(response.read()), 100)

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
