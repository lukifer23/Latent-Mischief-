from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from resource_guard import ResourceGuard,ResourceLimitError,thermal_warning,violation


class ResourceTests(unittest.TestCase):
    def policy(self):
        import json
        return json.loads((Path(__file__).resolve().parents[1]/'configs/qwen-mac-followup.json').read_text())['resources']

    def healthy(self):
        return {'available_bytes':5*2**30,'pressure_level':1,'thermal_warning':False,
                'swap_used_bytes':4*2**30,'process_rss_bytes':2*2**30,
                'mps_driver_allocated_bytes':2*2**30,'disk_free_bytes':20*2**30,
                'thermal_report':'normal','monotonic_seconds':time.perf_counter()}

    def test_limits_treat_existing_swap_separately_from_growth_and_use_real_headroom(self):
        row=self.healthy();policy=self.policy()
        self.assertIsNone(violation(row,policy,row['swap_used_bytes'],starting=True))
        for change in [{'available_bytes':2**30},{'pressure_level':2},{'thermal_warning':True},
                       {'thermal_warning':None},{'process_rss_bytes':5*2**30},
                       {'swap_used_bytes':row['swap_used_bytes']+257*2**20},
                       {'mps_driver_allocated_bytes':None},{'disk_free_bytes':2**30}]:
            self.assertIsNotNone(violation({**row,**change},policy,row['swap_used_bytes']))
        row['available_bytes']=3*2**30
        self.assertIsNone(violation(row,policy,row['swap_used_bytes']))
        self.assertIsNotNone(violation(row,policy,row['swap_used_bytes'],starting=True))

    def test_thermal_unknown_and_throttling_are_not_reported_healthy(self):
        self.assertFalse(thermal_warning('No thermal warning level has been recorded\nNo performance warning level has been recorded'))
        self.assertTrue(thermal_warning('Thermal_Level = 1\nCPU_Speed_Limit = 100'))
        self.assertTrue(thermal_warning('Thermal_Level = 0\nCPU_Speed_Limit = 80'))
        self.assertIsNone(thermal_warning('Unexpected telemetry output'))

    def test_stop_latches_after_pressure_recovers_and_preserves_resource_rows(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            guard=ResourceGuard(tmp,None,self.policy(),time.perf_counter()+60)
            with patch('resource_guard.snapshot',return_value=self.healthy()):guard.start()
            try:
                bad={**self.healthy(),'pressure_level':2}
                with patch('resource_guard.snapshot',return_value=bad):guard.sample()
                with patch('resource_guard.snapshot',return_value=self.healthy()):guard.sample()
                with self.assertRaisesRegex(ResourceLimitError,'pressure'):guard.check()
            finally:guard.close()
            rows=[json.loads(line) for line in (Path(tmp)/'resources.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),3)

    def test_stop_file_and_unavailable_sampler_stop_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp,'STOP').touch()
            guard=ResourceGuard(tmp,None,self.policy(),time.perf_counter()+60)
            with patch('resource_guard.snapshot',return_value=self.healthy()):guard.start()
            try:
                with self.assertRaisesRegex(ResourceLimitError,'STOP'):guard.check()
            finally:guard.close()
        with tempfile.TemporaryDirectory() as tmp:
            guard=ResourceGuard(tmp,None,self.policy(),time.perf_counter()+60)
            with patch('resource_guard.snapshot',side_effect=OSError('missing telemetry')):guard.start()
            try:
                with self.assertRaisesRegex(ResourceLimitError,'telemetry failed'):guard.check()
            finally:guard.close()


class CudaPolicyTests(unittest.TestCase):
    def policy(self):
        import json
        return json.loads((Path(__file__).resolve().parents[1]/'configs/lfm-phase-timing-cuda.json').read_text())['resources']|{'backend':'cuda'}

    def healthy(self):
        return {'available_bytes':30*2**30,'pressure_level':None,'thermal_warning':False,'swap_used_bytes':2**30,
                'process_rss_bytes':3*2**30,'peak_gpu_reserved_bytes':4*2**30,'gpu_free_bytes':12*2**30,
                'disk_free_bytes':50*2**30}

    def test_cuda_policy_has_no_pressure_level_but_enforces_ram_gpu_thermal_and_disk(self):
        policy=self.policy();row=self.healthy()
        self.assertTrue(policy['cuda_memory_fraction']<=.5)
        self.assertIsNone(violation(row,policy,row['swap_used_bytes'],starting=True))
        for change in [{'available_bytes':8*2**30},{'thermal_warning':None},{'thermal_warning':True},
                       {'peak_gpu_reserved_bytes':8*2**30},{'peak_gpu_reserved_bytes':None},
                       {'gpu_free_bytes':5*2**30},{'gpu_free_bytes':None},{'process_rss_bytes':9*2**30},
                       {'swap_used_bytes':row['swap_used_bytes']+300*2**20},{'disk_free_bytes':2**30}]:
            self.assertIsNotNone(violation({**row,**change},policy,row['swap_used_bytes'],starting=True),change)
        # the start-only floors relax while running
        row.update(available_bytes=8*2**30,gpu_free_bytes=5*2**30)
        self.assertIsNone(violation(row,policy,row['swap_used_bytes']))

    def test_cuda_thermal_parser_fails_closed(self):
        from resource_guard import cuda_thermal_warning
        head='temperature.gpu, a, b, c, d\n'
        self.assertFalse(cuda_thermal_warning(head+'40, Not Active, Not Active, Not Active, Not Active'))
        self.assertTrue(cuda_thermal_warning(head+'86, Not Active, Not Active, Not Active, Not Active'))
        self.assertTrue(cuda_thermal_warning(head+'40, Not Active, Active, Not Active, Not Active'))
        self.assertIsNone(cuda_thermal_warning(head+'40, [N/A], Not Active, Not Active, Not Active'))
        self.assertIsNone(cuda_thermal_warning(head+'40, Not Active'))
        self.assertIsNone(cuda_thermal_warning(''))


class RealTelemetryTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == 'win32', 'native Windows CUDA telemetry')
    def test_real_cuda_snapshot_accepts_path_roots_on_windows(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('no CUDA device')
        import runtime
        from resource_guard import snapshot
        with tempfile.TemporaryDirectory() as tmp:
            row=snapshot(Path(tmp),runtime.select('cuda'))
        self.assertIsInstance(row['thermal_warning'],bool)
        self.assertGreater(row['gpu_free_bytes'],0)
        self.assertGreater(row['disk_free_bytes'],0)
