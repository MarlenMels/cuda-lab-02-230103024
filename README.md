# CUDA Lab 02: Advanced Geometries & Stencils

Student ID: 230103024

Allocated GPU Node: NVIDIA Tesla T4

CUDA Compute Capability: 7.5

Official Verification Token: 2BD6875964FCD0091C5A


## Task 1: Warp Divergence Microbenchmark

This task compares three different branching patterns on the GPU.

The first kernel uses the same arithmetic path for every thread.

The second kernel uses interleaved branching based on whether the thread index is even or odd. Because neighboring threads inside the same warp execute different branches, this creates warp divergence.

The third kernel makes the branch decision using the warp ID. All 32 threads in one warp follow the same path, so there is no divergence inside a warp.

The benchmark uses a warm-up launch and then measures the average kernel execution time over 10 runs. Host-to-device and device-to-host transfers are not included in the timing.

### Benchmark Results

| Kernel | Branching Type | Average Time |
|---|---|---:|
| Kernel A | Uniform Path | INSERT_TIME ms |
| Kernel B | Full Divergence - Interleaved | INSERT_TIME ms |
| Kernel C | Warp-Aligned Branching | INSERT_TIME ms |

Kernel B is expected to be affected the most by warp divergence because even and odd threads inside the same warp take different execution paths.

Kernel C avoids intra-warp divergence because the branch is selected by warp ID instead of individual thread ID.


## Task 2: 1D Boundary Stencil

This task implements a 3-point smoothing stencil:

output[i] = 0.25 * left + 0.5 * center + 0.25 * right

Boundary clamping is used for the first and last elements. For the left boundary, the first value is reused as the missing left neighbor. For the right boundary, the last value is reused.

The GPU implementation was compared with a NumPy CPU reference using np.pad with edge mode.

The verification test passed successfully:

[PASS] Task 2 (1D Stencil & Clamping)


## Task 3: Grid-Stride Scaling

This task uses a grid-stride loop to process a vector that is much larger than the number of CUDA threads launched.

Configuration:

threads_per_block = 256

blocks_per_grid = 64

Total CUDA threads = 16,384

The kernel starts each thread at its global index and continues processing elements using the total grid size as the stride.

This allows a fixed number of CUDA threads to process an arbitrary number of elements without creating one thread for every array element.

The verification test passed successfully:

[PASS] Task 3 (Grid-Stride Scaling)


## Task 4: 2D Sobel Horizontal Filter

This task implements the Sobel-X operator using a 2D CUDA grid.

Input matrix size:

2048 x 2048

CUDA block size:

16 x 16

The Sobel-X kernel used is:

[-1  0  +1]
[-2  0  +2]
[-1  0  +1]

Only internal pixels are processed with the convolution. Pixels on the outer border are set to 0.0 to prevent out-of-bounds memory access.

The grid dimensions are calculated dynamically from the image dimensions.

The verification test passed successfully:

[PASS] Task 4 (2D Sobel Horizontal)


## Verification

The provided verify_submission.py script was executed on an NVIDIA Tesla T4 GPU.

Results:

[PASS] Task 2 (1D Stencil & Clamping)

[PASS] Task 3 (Grid-Stride Scaling)

[PASS] Task 4 (2D Sobel Horizontal)

VERIFICATION SUCCESSFUL

Official Submission Token:

2BD6875964FCD0091C5A


## Conclusion

In this lab I implemented and tested four CUDA programming concepts using Numba CUDA.

Task 1 demonstrates how branch organization can cause or avoid warp divergence.

Task 2 implements a boundary-safe 1D stencil.

Task 3 uses a grid-stride loop to process a large vector with a limited number of GPU threads.

Task 4 uses a 2D CUDA grid to apply a Sobel-X filter to an image matrix.

The automated verification script successfully passed Tasks 2, 3 and 4.