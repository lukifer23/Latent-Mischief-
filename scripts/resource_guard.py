"""Conservative macOS resource sampling and latched cooperative experiment stops."""
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time

import psutil


class ResourceLimitError(RuntimeError):
    pass


def thermal_warning(text):
    """Unknown telemetry fails closed; never infer a measured temperature."""
    levels = re.findall(r'(?:Thermal|Performance)_Level\s*=\s*(\d+)', text, re.I)
    speeds = re.findall(r'(?:CPU|GPU)_(?:Speed|Scheduler)_Limit\s*=\s*(\d+)', text, re.I)
    if any(int(x) > 0 for x in levels) or any(int(x) < 100 for x in speeds):
        return True
    if 'No thermal warning level has been recorded' in text and 'No performance warning level has been recorded' in text:
        return False
    return False if levels and speeds else None


def cuda_thermal_warning(text):
    """Parse one `nvidia-smi --query-gpu=temperature.gpu,<throttle reasons>` CSV. Unknown fails closed."""
    lines = [x for x in text.strip().splitlines() if x.strip()]
    if len(lines) < 2:
        return None
    header = [x.strip() for x in lines[0].split(',')]
    values = [x.strip() for x in lines[-1].split(',')]
    if len(values) != len(header) or len(values) < 2:
        return None
    try:
        temperature = float(values[0])
    except ValueError:
        return None
    reasons = values[1:]
    if any(x not in ('Active', 'Not Active') for x in reasons):
        return None
    return temperature >= 85 or any(x == 'Active' for x in reasons)


NVIDIA_QUERY = ('temperature.gpu,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.sw_thermal_slowdown,'
                'clocks_throttle_reasons.hw_slowdown,clocks_throttle_reasons.hw_power_brake_slowdown')


def cuda_snapshot(root, selected, thermal=None):
    """Windows/CUDA telemetry. There is no macOS-style pressure level; RAM, swap and GPU headroom are used."""
    import torch
    if thermal is None:
        thermal = subprocess.run(['nvidia-smi', '--query-gpu=' + NVIDIA_QUERY, '--format=csv'], capture_output=True,
                                 text=True, check=True, timeout=5).stdout
    memory = psutil.virtual_memory()
    free, total = torch.cuda.mem_get_info(selected.device)
    return {'monotonic_seconds': time.perf_counter(), 'available_bytes': memory.available,
            'swap_used_bytes': psutil.swap_memory().used, 'process_rss_bytes': psutil.Process().memory_info().rss,
            'pressure_level': None, 'thermal_warning': cuda_thermal_warning(thermal), 'thermal_report': thermal,
            'disk_free_bytes': psutil.disk_usage(root).free, 'gpu_free_bytes': free, 'gpu_total_bytes': total,
            **selected.memory()}


def snapshot(root, selected, thermal=None):
    if selected is not None and selected.backend == 'cuda':
        return cuda_snapshot(root, selected, thermal)
    pressure = subprocess.run(['/usr/sbin/sysctl', '-n', 'kern.memorystatus_vm_pressure_level'],
                              capture_output=True, text=True, check=True, timeout=2)
    if thermal is None:
        thermal = subprocess.run(['/usr/bin/pmset', '-g', 'therm'], capture_output=True,
                                 text=True, check=True, timeout=2).stdout
    memory = psutil.virtual_memory()
    return {'monotonic_seconds': time.perf_counter(), 'available_bytes': memory.available,
            'swap_used_bytes': psutil.swap_memory().used, 'process_rss_bytes': psutil.Process().memory_info().rss,
            'pressure_level': int(pressure.stdout.strip()), 'thermal_warning': thermal_warning(thermal),
            'thermal_report': thermal, 'disk_free_bytes': psutil.disk_usage(root).free, **selected.memory()}


def violation(row, policy, initial_swap, *, starting=False):
    minimum = policy['start_min_available_bytes' if starting else 'running_min_available_bytes']
    cuda = policy.get('backend') == 'cuda'
    if not cuda and row['pressure_level'] != policy['max_pressure_level']:
        return 'macOS memory pressure is elevated or unknown'
    if policy['require_no_thermal_warning'] and row['thermal_warning'] is not False:
        return 'thermal/performance warning or unavailable thermal telemetry'
    if row['available_bytes'] < minimum:
        return 'available system memory below the frozen floor'
    if row['swap_used_bytes'] - initial_swap > policy['max_swap_growth_bytes']:
        return 'system swap growth exceeds the frozen allowance'
    if row['process_rss_bytes'] > policy['max_process_rss_bytes']:
        return 'process RSS exceeds the frozen ceiling'
    if cuda:
        if row.get('peak_gpu_reserved_bytes') is None or row['peak_gpu_reserved_bytes'] > policy['max_cuda_reserved_bytes']:
            return 'CUDA reserved memory unavailable or exceeds the frozen ceiling'
        if row.get('gpu_free_bytes') is None or row['gpu_free_bytes'] < policy['start_min_gpu_free_bytes' if starting else 'running_min_gpu_free_bytes']:
            return 'free GPU memory below the frozen floor'
    elif row['mps_driver_allocated_bytes'] is None or row['mps_driver_allocated_bytes'] > policy['max_mps_driver_bytes']:
        return 'MPS driver allocation unavailable or exceeds the frozen ceiling'
    if row['disk_free_bytes'] < policy['minimum_disk_free_bytes']:
        return 'free disk space below the frozen floor'
    return None


class ResourceGuard:
    def __init__(self, folder, selected, policy, deadline, *, initial_swap=None):
        self.folder = Path(folder)
        self.selected, self.policy, self.deadline = selected, policy, deadline
        self.initial_swap = initial_swap
        self.reason = None
        self.stage = 'starting'
        self.last = None
        self.started = time.perf_counter()
        self.closed = threading.Event()
        self.thread = None
        self.stream = None

    def sample(self, *, starting=False):
        try:
            # Thermal status is refreshed every five seconds; memory every second.
            thermal = self.last['thermal_report'] if self.last and time.perf_counter() - self.thermal_time < 5 else None
            row = snapshot(self.folder, self.selected, thermal)
            if thermal is None:
                self.thermal_time = time.perf_counter()
            if self.initial_swap is None:
                self.initial_swap = row['swap_used_bytes']
            row.update(stage=self.stage, elapsed_seconds=time.perf_counter() - self.started,
                       swap_growth_bytes=row['swap_used_bytes'] - self.initial_swap)
            self.last = row
            reason = violation(row, self.policy, self.initial_swap, starting=starting)
            if (self.folder / 'STOP').exists():
                reason = 'local STOP file requested cancellation'
            if time.perf_counter() >= self.deadline:
                reason = 'cumulative wall budget exhausted'
            if reason and self.reason is None:
                self.reason = reason
            self.stream.write(json.dumps(row, allow_nan=False) + '\n')
            self.stream.flush()
            os.fsync(self.stream.fileno())
        except Exception as error:
            if self.reason is None:
                self.reason = 'resource telemetry failed: ' + str(error)

    def start(self):
        self.stream = (self.folder / 'resources.jsonl').open('a', encoding='utf-8')
        self.sample(starting=True)
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
        return self

    def _monitor(self):
        while not self.closed.wait(self.policy['sample_interval_seconds']):
            self.sample()

    def check(self):
        if self.reason:
            raise ResourceLimitError(self.reason)
        if time.perf_counter() >= self.deadline:
            raise ResourceLimitError('cumulative wall budget exhausted')

    def close(self):
        self.closed.set()
        if self.thread:
            self.thread.join(timeout=6)
            if self.thread.is_alive():
                raise RuntimeError('Resource sampler did not exit within telemetry timeout')
        if self.stream:
            self.stream.close()
