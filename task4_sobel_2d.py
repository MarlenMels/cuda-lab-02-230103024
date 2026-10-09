"""2D Sobel-X filter, with zeroed image borders."""
import numpy as np
from numba import cuda

@cuda.jit
def sobel_x_kernel(d_in, d_out, rows, cols):
    col, row = cuda.grid(2)
    if row < rows and col < cols:
        if 0 < row < rows - 1 and 0 < col < cols - 1:
            d_out[row, col] = (
                -d_in[row - 1, col - 1] + d_in[row - 1, col + 1]
                -2.0 * d_in[row, col - 1] + 2.0 * d_in[row, col + 1]
                -d_in[row + 1, col - 1] + d_in[row + 1, col + 1]
            )
        else:
            d_out[row, col] = 0.0

def run_sobel(h_img):
    h_img = np.asarray(h_img, dtype=np.float32)
    if h_img.ndim != 2:
        raise ValueError('Expected a 2D image')
    rows, cols = h_img.shape
    if rows == 0 or cols == 0:
        return np.zeros_like(h_img)
    d_in = cuda.to_device(h_img)
    d_out = cuda.device_array_like(d_in)
    threads = (16, 16)
    blocks = ((cols + 15) // 16, (rows + 15) // 16)
    sobel_x_kernel[blocks, threads](d_in, d_out, rows, cols)
    return d_out.copy_to_host()

if __name__ == '__main__':
    img = np.ones((2048, 2048), dtype=np.float32)
    actual = run_sobel(img)
    assert np.all(actual == 0)
    print('TASK 4 PASSED: 2048 x 2048 Sobel-X, all borders zero')
