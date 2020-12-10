"""
Created on Fri Oct 19 14:44:19 2018
Updated on 9 December 2020
adhering to Py_lint standards

@author: peter
"""

import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Qt5Agg")

def mandelbrot_calculation(c_real,c_imag,max_iter):
    """
    calculation of mandelbrot set formula
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


def mandelbrot_set(image_rect, max_iter):
    """
    calclation of the mandelbrot image in a 2-D array
    """
    stepsize_x = (image_rect['x_axis_max'] - image_rect['x_axis_min']) / image_rect['x_size']
    stepsize_y = (image_rect['y_axis_max'] - image_rect['y_axis_min']) / image_rect['y_size']
    x_axis_array = np.arange(image_rect['x_axis_min'], image_rect['x_axis_max'], stepsize_x)
    y_axis_array = np.arange(image_rect['y_axis_min'], image_rect['y_axis_max'], stepsize_y)
    image = np.zeros((len(y_axis_array), len(x_axis_array)))

    for j, y_coord in enumerate(y_axis_array):
        for i, x_coord in enumerate(x_axis_array):
            image[j,i] = mandelbrot_calculation(x_coord,y_coord, max_iter)
    return image

def plot_mandelbrot(image_rect, image_temp, elapsed_time, iterations):
    """
    plotting the calculated mandelbrot set and writing it to file
    """
    image_dimension = str(image_rect['x_size']) + " x " + str(image_rect['y_size'])
    plt.imshow(image_temp, cmap = plt.cm.prism, interpolation = None, \
                extent = (image_rect['x_axis_min'], image_rect['x_axis_max'], \
                image_rect['y_axis_min'], image_rect['y_axis_max']))
    plt.xlabel("Re(c), using no optimization time: {0}".format(elapsed_time))
    plt.ylabel("Im(c), max iter: {0}:".format(iterations))
    plt.title( "mandelbrot set, image size (x,y): {0}".format(image_dimension))
    plt.savefig("mandelbrot_no_optimization.png")
    plt.show()
    plt.close()

def main():
    """
    Main function
    """
#initialization of constants


    max_iterations = 300
    image_rectangle = { 'x_axis_min':-2,\
                        'x_axis_max': 0.5, \
                        'y_axis_min': -1, \
                        'y_axis_max': 1, \
                        'x_size': 4096, \
                        'y_size': 4096 }
# start calculation
    start = time.time()
    image = mandelbrot_set(image_rectangle, max_iterations)
    time_elapsed = time.time() - start


#plot image in window
    plot_mandelbrot(image_rectangle, image, time_elapsed, max_iterations)

main()
