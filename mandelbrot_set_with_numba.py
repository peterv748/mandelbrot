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
    def __init__(self,image_rect, im_size, max_iter):
        """
        init of class variables
        """
        self.min_x = image_rect[0]
        self.max_x = image_rect[1]
        self.min_y = image_rect[2]
        self.max_y = image_rect[3]
        self.size_x = im_size[0]
        self.size_y = im_size[1]
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
        
        for j, y_coord in enumerate(y_axis_array):
            for i, x_coord in enumerate(x_axis_array):
                image_array[j,i] = self.mandelbrot_calculation(x_coord,y_coord)
        return image_array


def plot_mandelbrot(image_rect, im_size, image_temp, elapsed_time, iterations):
    """
    plotting the calculated mandelbrot set and writing it to file
    """
    image_dimension = str(im_size[0]) + " x " + str(im_size[1])
    plt.imshow(image_temp, cmap = plt.cm.prism, interpolation = None, \
                extent = (image_rect[0], image_rect[1], \
                image_rect[2], image_rect[3]))
    plt.xlabel("Re(c), using jit optimization time: {0}".format(elapsed_time))
    plt.ylabel("Im(c), max iter: {0}:".format(iterations))
    plt.title( "mandelbrot set, image size (x,y): {0}".format(image_dimension))
    plt.savefig("mandelbrot_numba_jit_optimization.png")
    plt.show()
    plt.close()


def main():
    """
    Main function
    """
    image_rectangle = np.array([-2, 0.5, -1, 1])

    image_size = np.array([4096,4096])

    max_iterations = 300

    mandelbrot_object = Mandelbrot(image_rectangle, image_size, max_iterations)
    start = time.time()
    image = mandelbrot_object.mandelbrot_set()
    time_elapsed = time.time() - start

    plot_mandelbrot(image_rectangle, image_size, image, time_elapsed, max_iterations)

main()
