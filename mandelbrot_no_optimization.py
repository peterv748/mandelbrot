"""
Created on Fri Oct 19 14:44:19 2018
Updated on 9 December 2020
adhering to Pylint standards

@author: peter
"""

import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Qt5Agg")

def mandelbrot_calculation(c_real,c_imag,max_iter):
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
    stepsize_x = (xmax - xmin)/x_size
    stepsize_y = (ymax - ymin)/y_size
    x_axis_array = np.arange(xmin, xmax, stepsize_x)
    y_axis_array = np.arange(ymin, ymax, stepsize_y)
    image = np.zeros((len(y_axis_array), len(x_axis_array)))

    for j, y in enumerate(y_axis_array):
        for i, x in enumerate(x_axis_array):
            image[j,i] = mandelbrot_calculation(x,y, max_iter)
    return image

def plot_mandelbrot(min_x, max_x, min_y, max_y, Z_temp, dt):
    plt.imshow(Z_temp, cmap = plt.cm.prism, interpolation = None, extent = (min_x, max_x, min_y, max_y))
    plt.xlabel("Re(c), using no optimization time: %f s" % dt)
    plt.ylabel("Im(c), max iter =300")
    plt.title( "mandelbrot set, image size (x,y): 4096 x 4096 pixels")
    plt.savefig("mandelbrot_no_optimization.png")
    plt.show()
    plt.close()

def main():

#initialisation of constants

    X_MIN = -2
    X_MAX = .5
    Y_MIN = -1
    Y_MAX = 1
    X_SIZE = 4096
    Y_SIZE = 4096
    MAX_ITERATIONS = 300

# start calculation
    start = time.time()
    image = mandelbrot_set(X_MIN,X_MAX,Y_MIN,Y_MAX,X_SIZE, Y_SIZE, MAX_ITERATIONS)
    time_elapsed = time.time() - start


#plot image in window
    plot_mandelbrot(X_MIN,X_MAX,Y_MIN,Y_MAX, image, time_elapsed)

main()