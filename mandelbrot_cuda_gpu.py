# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 14:44:19 2018
Updated on 9 December 2020
adhering to Py_lint standards

@author: peter
"""
import time
import numpy as np
from numba import cuda
import matplotlib.pyplot as plt


@cuda.jit(device=True)
def mandelbrot_calculation(c_real,c_imag,max_iter):
    """
    calculation of mandelbrot set formula using the Numba package and 
    the included cuda support functions to use the GPU of the machine
    """
    real = c_real
    imag = c_imag
    for i in range(max_iter):
        real2 = real*real
        imag2 = imag*imag
        if real2 + imag2 > 4.0:
            return i
        imag = 2* real*imag + c_imag
        real = real2 - imag2 + c_real
    return max_iter


@cuda.jit
def mandel_kernel(min_x, max_x, min_y, max_y, x_size, y_size, image, iters):
    """
    calclation of the mandelbrot image in a 2-D array
    """
    pixel_size_x = (max_x - min_x) / x_size
    pixel_size_y = (max_y - min_y) / y_size
    start_x, start_y = cuda.grid(2)
    start_x = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    start_y = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y
    grid_x = cuda.gridDim.x * cuda.blockDim.x
    grid_y = cuda.gridDim.y * cuda.blockDim.y

    for i in range(start_x, x_size, grid_x):
        real = min_x + i * pixel_size_x
        for j in range(start_y, y_size, grid_y):
            imag = min_y + j * pixel_size_y
            image[j, i] = mandelbrot_calculation(real, imag, iters)


def plot_mandelbrot(min_x, max_x, min_y, max_y, image_temp, elapsed_time):
    """
    plotting the calculated mandelbrot set and writing it to file
    """
    plt.imshow(image_temp, cmap = plt.cm.prism, interpolation = None, \
                extent = (min_x, max_x, min_y, max_y))
    plt.xlabel("Re(c), using gpu optimization time: %f s" % elapsed_time)
    plt.ylabel("Im(c), max iter =300")
    plt.title( "mandelbrot set, image size (x,y): 4096 x 4096 pixels")
    plt.savefig("mandelbrot_gpu_optimization.png")
    plt.show()
    plt.close()


#initializations of constants
def main():
    """
    Main function
    """
    X_SIZE = 4096
    Y_SIZE = 4096
    block_dim = (32, 32)
    grid_dim = (128,128)
    X_MIN = -2
    X_MAX = .5
    Y_MIN = -1
    Y_MAX = 1
    MAX_ITERATIONS = 300
    image = np.zeros((Y_SIZE, X_SIZE), dtype = np.uint32)


# start calculations

    d_image = cuda.to_device(image)
    start = time.time()
    mandel_kernel[grid_dim, block_dim](X_MIN,X_MAX, Y_MIN, Y_MAX, X_SIZE, Y_SIZE, \
                    d_image, MAX_ITERATIONS)
    time_elapsed = time.time() - start
    d_image.copy_to_host(image)

    plot_mandelbrot(X_MIN,X_MAX, Y_MIN, Y_MAX, image,time_elapsed)


main()
