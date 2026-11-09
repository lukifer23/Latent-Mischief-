from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import runtime
from runtime_identity import require_same_backend


class RuntimeTests(unittest.TestCase):
    def test_no_accelerator_requires_explicit_cpu_and_records_no_gpu_peaks(self):
        with patch.dict('os.environ', {}, clear=True), \
                patch('torch.cuda.is_available', return_value=False), \
                patch('torch.backends.mps.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'explicitly'):
                runtime.select()
            selected = runtime.select('cpu')
            self.assertEqual(selected.dtype_name, 'float32')
            with patch('torch.cuda.synchronize', side_effect=AssertionError('CUDA called')), \
                    patch('torch.cuda.reset_peak_memory_stats', side_effect=AssertionError('CUDA called')):
                selected.synchronize()
                selected.reset_peak_memory()
                selected.empty_cache()
                self.assertTrue(all(value is None for value in selected.memory().values()))

    def test_mps_metrics_are_current_values_and_never_cuda_peaks(self):
        selected = runtime.Runtime('mps', 'bfloat16')
        with patch('torch.mps.synchronize') as synchronize, \
                patch('torch.mps.current_allocated_memory', return_value=123), \
                patch('torch.mps.driver_allocated_memory', return_value=456), \
                patch('torch.cuda.max_memory_allocated', side_effect=AssertionError('CUDA called')):
            selected.synchronize()
            synchronize.assert_called_once()
            metrics = selected.memory()
            self.assertEqual(metrics['mps_current_allocated_bytes'], 123)
            self.assertEqual(metrics['mps_driver_allocated_bytes'], 456)
            self.assertIsNone(metrics['peak_gpu_allocated_bytes'])
            self.assertIsNone(metrics['peak_gpu_reserved_bytes'])

    def test_cuda_dispatch_keeps_device_and_existing_metric_meanings(self):
        selected = runtime.Runtime('cuda:1', 'bfloat16')
        with patch('torch.cuda.synchronize') as synchronize, \
                patch('torch.cuda.reset_peak_memory_stats') as reset, \
                patch('torch.cuda.max_memory_allocated', return_value=123) as allocated, \
                patch('torch.cuda.max_memory_reserved', return_value=456) as reserved:
            selected.synchronize()
            selected.reset_peak_memory()
            metrics = selected.memory()
            for call in (synchronize, reset, allocated, reserved):
                call.assert_called_once_with('cuda:1')
            self.assertEqual(metrics['peak_gpu_allocated_bytes'], 123)
            self.assertEqual(metrics['peak_gpu_reserved_bytes'], 456)

    def test_unavailable_requested_backend_and_bad_dtype_do_not_fallback(self):
        with patch('torch.backends.mps.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'MPS'):
                runtime.select('mps')
        with self.assertRaisesRegex(ValueError, 'LAB_DTYPE'):
            runtime.select('cpu', 'float64')

    def test_historical_calibration_rejects_mps_dtype_or_policy_drift(self):
        cuda = {'backend':'cuda', 'dtype':'bfloat16', 'mps_cpu_fallback':'0', 'mps_fast_math':'0'}
        require_same_backend({}, {'runtime':cuda})
        for change in ({'backend':'mps'}, {'dtype':'float32'}, {'mps_cpu_fallback':'1'}):
            with self.assertRaisesRegex(ValueError, 'Backend/dtype'):
                require_same_backend({}, {'runtime':{**cuda, **change}})


if __name__ == '__main__':
    unittest.main()
