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
def mandel_kernel(im_rect, image_array, im_size, iters):
    """
    calclation of the mandelbrot image in a 2-D array
    """
    pixel_size_x = (im_rect[1] - im_rect[0]) / im_size[0]
    pixel_size_y = (im_rect[3] - im_rect[2]) / im_size[1]
    start_x, start_y = cuda.grid(2)
    start_x = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    start_y = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y
    grid_x = cuda.gridDim.x * cuda.blockDim.x
    grid_y = cuda.gridDim.y * cuda.blockDim.y

    for i in range(start_x, im_size[0], grid_x):
        real = im_rect[0] + i * pixel_size_x
        for j in range(start_y, im_size[1], grid_y):
            imag = im_rect[2] + j * pixel_size_y
            image_array[j, i] = mandelbrot_calculation(real, imag, iters)

def plot_mandelbrot(image_rect, im_size, image_temp, elapsed_time, iterations):
    """
    plotting the calculated mandelbrot set and writing it to file
    """
    image_dimension = str(im_size[0]) + " x " + str(im_size[1])
    plt.imshow(image_temp, cmap = plt.cm.prism, interpolation = None, \
                extent = (image_rect[0], image_rect[1], \
                image_rect[2], image_rect[3]))
    plt.xlabel("Re(c), using gpu calculation time: {0}".format(elapsed_time))
    plt.ylabel("Im(c), max iter: {0}:".format(iterations))
    plt.title( "mandelbrot set, image size (x,y): {0}".format(image_dimension))
    plt.savefig("mandelbrot_gpu_optimization.png")
    plt.show()
    plt.close()


#initializations of constants
def main():
    """
    Main function
    """

    image_rectangle = np.array([-2, 0.5, -1, 1])
    image_size = np.array([4096,4096])

    block_dim = (32, 32)
    grid_dim = (128,128)
    max_iterations = 300
    image = np.zeros((image_size[0], image_size[1]), dtype = np.uint32)


# start calculations

    d_image = cuda.to_device(image)
    start = time.time()
    mandel_kernel[grid_dim, block_dim](image_rectangle, d_image, \
                    image_size, max_iterations)
    time_elapsed = time.time() - start
    d_image.copy_to_host(image)

    plot_mandelbrot(image_rectangle, image_size, image,time_elapsed, max_iterations)


main()
