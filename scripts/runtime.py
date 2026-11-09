"""Explicit local device policy and honest backend-specific timing/memory metrics."""
from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Runtime:
    device: str
    dtype_name: str

    @property
    def backend(self):
        return self.device.split(':')[0]

    @property
    def dtype(self):
        import torch
        return getattr(torch, self.dtype_name)

    def synchronize(self):
        import torch
        if self.backend == 'cuda':
            torch.cuda.synchronize(self.device)
        elif self.backend == 'mps':
            torch.mps.synchronize()

    def empty_cache(self):
        import torch
        if self.backend == 'cuda':
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
        elif self.backend == 'mps':
            torch.mps.empty_cache()

    def reset_peak_memory(self):
        import torch
        if self.backend == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)

    def memory(self):
        import torch
        # Keep existing CUDA field meanings. MPS has no public peak/reset API in
        # the pinned torch; current driver allocation is not a CUDA reserved peak.
        result = {'peak_gpu_allocated_bytes': None, 'peak_gpu_reserved_bytes': None,
                  'mps_current_allocated_bytes': None, 'mps_driver_allocated_bytes': None}
        if self.backend == 'cuda':
            result.update(peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                          peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(self.device))
        elif self.backend == 'mps':
            result.update(mps_current_allocated_bytes=torch.mps.current_allocated_memory(),
                          mps_driver_allocated_bytes=torch.mps.driver_allocated_memory())
        return result

    def identity(self):
        return {'device': self.device, 'backend': self.backend, 'dtype': self.dtype_name,
                'mps_cpu_fallback': os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0'),
                'mps_fast_math': os.environ.get('PYTORCH_MPS_FAST_MATH', '0')}


def select(device=None, dtype=None):
    """Prefer CUDA then MPS. A CPU run must be explicitly selected by the caller."""
    import torch
    requested = device or os.environ.get('LAB_DEVICE', 'auto')
    if requested == 'auto':
        if torch.cuda.is_available():
            requested = 'cuda'
        elif torch.backends.mps.is_available():
            requested = 'mps'
        else:
            raise RuntimeError('No accelerator available. Choose LAB_DEVICE=cpu explicitly for CPU inference.')
    parsed = torch.device(requested)
    if parsed.type not in ('cuda', 'mps', 'cpu'):
        raise ValueError('LAB_DEVICE must be auto, cuda, cuda:N, mps or cpu')
    if parsed.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA backend is unavailable')
        index = parsed.index if parsed.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise ValueError('Requested CUDA device is outside the available device count')
        requested = f'cuda:{index}'
    else:
        if parsed.index not in (None, 0):
            raise ValueError('CPU and MPS support only device index zero')
        requested = parsed.type
    if parsed.type == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('Requested MPS backend is unavailable')
    dtype_name = dtype or os.environ.get('LAB_DTYPE') or ('float32' if parsed.type == 'cpu' else 'bfloat16')
    if dtype_name not in ('bfloat16', 'float16', 'float32'):
        raise ValueError('LAB_DTYPE must be bfloat16, float16 or float32')
    runtime = Runtime(requested, dtype_name)
    # Fail before downloading/loading weights if this dtype cannot execute here.
    probe = torch.ones(2, device=runtime.device, dtype=runtime.dtype)
    if probe.sum().item() != 2:
        raise RuntimeError('Device/dtype probe failed')
    return runtime


def for_model(model):
    return Runtime(str(model.device), str(model.dtype).removeprefix('torch.'))


def require_cuda_protocol(name):
    runtime = select()
    if runtime.backend != 'cuda' or runtime.dtype_name != 'bfloat16':
        raise ValueError(f'{name} is a frozen CUDA/BF16 protocol. Predeclare a separate backend experiment; do not relabel or resume it on Mac.')
    return runtime
