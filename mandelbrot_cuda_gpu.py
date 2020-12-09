# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 14:44:19 2018

@author: peter
"""

import numpy as np
from numpy import NaN
from numba import cuda
import time
import matplotlib.pyplot as plt


@cuda.jit(device=True)
def mandelbrot_calculation(c_real,c_imag,max_iter):
    real = c_real
    imag = c_imag
    for n in range(max_iter):
        real2 = real*real
        imag2 = imag*imag
        if real2 + imag2 > 4.0:
            return n
        imag = 2* real*imag + c_imag
        real = real2 - imag2 + c_real
    return max_iter




@cuda.jit
def mandel_kernel(min_x, max_x, min_y, max_y, x_size, y_size, image, iters):


    pixel_size_x = (max_x - min_x) / x_size
    pixel_size_y = (max_y - min_y) / y_size

    start_x, start_y = cuda.grid(2)


    start_x = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    start_y = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y
    grid_x = cuda.gridDim.x * cuda.blockDim.x;
    grid_y = cuda.gridDim.y * cuda.blockDim.y;

    for x in range(start_x, x_size, grid_x):
      real = min_x + x * pixel_size_x
      for y in range(start_y, y_size, grid_y):
        imag = min_y + y * pixel_size_y
        image[y, x] = mandelbrot_calculation(real, imag, iters)


def plot_mandelbrot(min_x, max_x, min_y, max_y, Z_temp, dt):
    plt.imshow(Z_temp, cmap = plt.cm.prism, interpolation = None, extent = (min_x, max_x, min_y, max_y))
    plt.xlabel("Re(c), using gpu time: %f s" % dt)
    plt.ylabel("Im(c), max iter =300")
    plt.title( "mandelbrot set, image size (x,y): 4096 x 4096 pixels")
    plt.savefig("mandelbrot_python_gpu.png")
    plt.show()
    plt.close()


#initializations of constants
def main():
    x_size = 4096
    y_size = 4096
    block_dim = (32, 32)
    grid_dim = (128,128)
    x_min = -2
    x_max = 0.5
    y_min = -1
    y_max = 1
    max_iter = 300
    image = np.zeros((y_size, x_size), dtype = np.uint32)


# start calculations

    d_image = cuda.to_device(image)
    start = time.time()
    mandel_kernel[grid_dim, block_dim](x_min,x_max, y_min, y_max, x_size, y_size, d_image, max_iter)
    dt = time.time() - start
    d_image.copy_to_host(image)

    plot_mandelbrot(x_min, x_max, y_min, y_max, image, dt)


main()
