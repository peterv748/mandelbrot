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


def mandelbrot_set(xmin,xmax,ymin,ymax,x_size, y_size, max_iter):
    """
    calclation of the mandelbrot image in a 2-D array
    """
    stepsize_x = (xmax - xmin)/x_size
    stepsize_y = (ymax - ymin)/y_size
    x_axis_array = np.arange(xmin, xmax, stepsize_x)
    y_axis_array = np.arange(ymin, ymax, stepsize_y)
    image = np.zeros((len(y_axis_array), len(x_axis_array)))

    for j, y_coord in enumerate(y_axis_array):
        for i, x_coord in enumerate(x_axis_array):
            image[j,i] = mandelbrot_calculation(x_coord,y_coord, max_iter)
    return image

def plot_mandelbrot(min_x, max_x, min_y, max_y, image_temp, elapsed_time):
    """
    plotting the calculated mandelbrot set and writing it to file
    """

    plt.imshow(image_temp, cmap = plt.cm.prism, interpolation = None, \
                extent = (min_x, max_x, min_y, max_y))
    plt.xlabel("Re(c), using no optimization time: %f s" % elapsed_time)
    plt.ylabel("Im(c), max iter =300")
    plt.title( "mandelbrot set, image size (x,y): 4096 x 4096 pixels")
    plt.savefig("mandelbrot_no_optimization.png")
    plt.show()
    plt.close()

def main():
    """
    Main function
    """
#initialization of constants

    x_min = -2
    x_max = .5
    y_min = -1
    y_max = 1
    x_size = 4096
    y_size = 4096
    max_iterations = 300

# start calculation
    start = time.time()
    image = mandelbrot_set(x_min,x_max,y_min,y_max,x_size, y_size, max_iterations)
    time_elapsed = time.time() - start


#plot image in window
    plot_mandelbrot(x_min,x_max,y_min,y_max, image, time_elapsed)

main()
