"""Three-point stencil with replicated halo cells."""
import numpy as np
from numba import cuda

@cuda.jit
def stencil_1d(d_in, d_out, N):
    idx = cuda.grid(1)
    if idx < N:
        left = d_in[idx - 1] if idx > 0 else d_in[0]
        right = d_in[idx + 1] if idx < N - 1 else d_in[N - 1]
        d_out[idx] = 0.25 * left + 0.5 * d_in[idx] + 0.25 * right

def cpu_stencil(arr):
    padded = np.pad(arr, (1, 1), mode='edge')
    return 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]

def run_stencil(h_in):
    h_in = np.asarray(h_in, dtype=np.float32)
    if h_in.ndim != 1:
        raise ValueError('Expected a 1D array')
    if h_in.size == 0:
        return h_in.copy()
    d_in = cuda.to_device(h_in)
    d_out = cuda.device_array_like(d_in)
    threads = 256
    blocks = (h_in.size + threads - 1) // threads
    stencil_1d[blocks, threads](d_in, d_out, h_in.size)
    return d_out.copy_to_host()

if __name__ == '__main__':
    N = 10007
    data = np.sin(np.linspace(0, 10, N)).astype(np.float32)
    actual = run_stencil(data)
    expected = cpu_stencil(data)
    delta = np.max(np.abs(actual - expected))
    assert np.allclose(actual, expected, atol=1e-4)
    print(f'TASK 2 PASSED: MAX DELTA = {delta:.8g}')
