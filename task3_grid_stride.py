"""Scale a large array with a fixed launch grid and grid-stride loop."""
import numpy as np
from numba import cuda

@cuda.jit
def grid_stride_scale_kernel(d_arr, factor, N):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, N, stride):
        d_arr[i] = d_arr[i] * factor

def run_grid_stride(h_arr, factor):
    h_arr = np.asarray(h_arr, dtype=np.float32)
    if h_arr.ndim != 1:
        raise ValueError('Expected a 1D array')
    if h_arr.size == 0:
        return h_arr.copy()
    d_arr = cuda.to_device(h_arr)
    grid_stride_scale_kernel[64, 256](d_arr, np.float32(factor), h_arr.size)
    return d_arr.copy_to_host()

if __name__ == '__main__':
    N = 2 ** 24
    factor = 4.25
    actual = run_grid_stride(np.ones(N, dtype=np.float32), factor)
    assert np.allclose(actual, factor)
    print(f'TASK 3 PASSED: {N} elements, factor={factor}, 64 blocks x 256 threads')
