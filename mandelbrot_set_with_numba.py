"""
Created on Fri Oct 19 14:44:19 2018
Updated on 9 December 2020
adhering to Py_lint standards

@author: peter
"""

import time
import numpy as np
from numba.experimental import jitclass
from numba import int64, float64
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Qt5Agg")


spec = [('min_x', float64),('max_x',float64), \
        ('min_y', float64),('max_y', float64),('size_x',int64), \
        ('size_y',int64),('max_iterations',int64)]
@jitclass(spec)
class Mandelbrot():
    """
    mandelbrot class
    """
    def __init__(self,x_min,x_max,y_min,y_max,x_size, y_size, max_iter):
        """
        init of class variables
        """
        self.min_x = x_min
        self.max_x = x_max
        self.min_y = y_min
        self.max_y = y_max
        self.size_x = x_size
        self.size_y = y_size
        self.max_iterations = max_iter

    def mandelbrot_calculation(self,c_real,c_imag):
        """
        calculation of mandelbrot set formula 
        """
        real = c_real
        imag = c_imag
        for i in range(self.max_iterations):
            real2 = real*real
            imag2 = imag*imag
            if real2 + imag2 > 4.0:
                return i
            imag = 2* real*imag + c_imag
            real = real2 - imag2 + c_real
        return self.max_iterations


    def mandelbrot_set(self):
        """
        calclation of the mandelbrot image in a 2-D array
        """
        stepsize_x = (self.max_x - self.min_x)/self.size_x
        stepsize_y = (self.max_y - self.min_y)/self.size_y
        x_axis_array = np.arange(self.min_x, self.max_x, stepsize_x)
        y_axis_array = np.arange(self.min_y, self.max_y, stepsize_y)
        image_array = np.zeros((len(y_axis_array), len(x_axis_array)))
        for j, y in enumerate(y_axis_array):
            for i, x in enumerate(x_axis_array):
                image_array[j,i] = self.mandelbrot_calculation(x,y)
        return image_array


def plot_mandelbrot(min_x, max_x, min_y, max_y, image_temp, elapsed_time):
    """
    plotting the calculated mandelbrot set and writing it to file
    """
    
    plt.imshow(image_temp, cmap = plt.cm.prism, \
                interpolation = None, extent = (min_x, max_x, min_y, max_y))
    plt.xlabel("Re(c), optimization using numba jit compiler time: %f s" % elapsed_time)
    plt.ylabel("Im(c), max iter =300")
    plt.title( "mandelbrot set, image size (x,y): 4096 x 4096 pixels")
    plt.savefig("mandelbrot_optimization_using_numba_jit.png")
    plt.show()
    plt.close()


def main():
    """
    Main function
    """
    X_MIN = -2
    X_MAX = .5
    Y_MIN = -1
    Y_MAX = 1
    X_SIZE = 4096
    Y_SIZE = 4096
    MAX_ITERATIONS = 300

    mandelbrot_object = Mandelbrot(X_MIN,X_MAX,Y_MIN,Y_MAX,X_SIZE, Y_SIZE, MAX_ITERATIONS)

    start = time.time()
    image = mandelbrot_object.mandelbrot_set()
    time_elapsed = time.time() - start

    plot_mandelbrot(X_MIN,X_MAX,Y_MIN,Y_MAX, image, time_elapsed)

main()
