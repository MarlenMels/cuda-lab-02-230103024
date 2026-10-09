"""Compare uniform, interleaved and warp-aligned CUDA branching."""
import time
import numpy as np
from numba import cuda

N = 1 << 20
ITERATIONS = 1000
THREADS = 256

@cuda.jit
def uniform_kernel(data, out):
    idx = cuda.grid(1)
    if idx < data.size:
        x = data[idx]
        for _ in range(ITERATIONS):
            x = x * 1.00001 + 0.00001
        out[idx] = x

@cuda.jit
def interleaved_kernel(data, out):
    idx = cuda.grid(1)
    if idx < data.size:
        x = data[idx]
        if idx % 2 == 0:
            for _ in range(ITERATIONS):
                x = x * 1.00001 + 0.00001
        else:
            for _ in range(ITERATIONS):
                x = (x - 0.00001) / 1.00001
        out[idx] = x

@cuda.jit
def warp_aligned_kernel(data, out):
    idx = cuda.grid(1)
    if idx < data.size:
        x = data[idx]
        warp_id = idx // 32
        if warp_id % 2 == 0:
            for _ in range(ITERATIONS):
                x = x * 1.00001 + 0.00001
        else:
            for _ in range(ITERATIONS):
                x = (x - 0.00001) / 1.00001
        out[idx] = x

def benchmark(kernel, d_in, d_out, blocks):
    kernel[blocks, THREADS](d_in, d_out)
    cuda.synchronize()
    trials = []
    for _ in range(10):
        start = time.perf_counter()
        kernel[blocks, THREADS](d_in, d_out)
        cuda.synchronize()
        trials.append((time.perf_counter() - start) * 1000)
    return sum(trials) / len(trials)

def main():
    data = np.full(N, 0.5, dtype=np.float32)
    d_in = cuda.to_device(data)
    d_out = cuda.device_array_like(d_in)
    blocks = (N + THREADS - 1) // THREADS
    print('GPU:', cuda.get_current_device().name)
    print('N:', N, 'iterations:', ITERATIONS, 'trials:', 10)
    for label, kernel in [('Uniform', uniform_kernel),
                          ('Interleaved', interleaved_kernel),
                          ('Warp-aligned', warp_aligned_kernel)]:
        print(f'{label}: {benchmark(kernel, d_in, d_out, blocks):.4f} ms')

if __name__ == '__main__':
    main()
