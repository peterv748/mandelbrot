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

class Mandelbrot():
    """
    mandelbrot class
    """

    def __init__(self,image_rect, max_iter):
        """
        init of class variables
        """
        self.min_x = image_rect['x_axis_min']
        self.max_x = image_rect['x_axis_max']
        self.min_y = image_rect['y_axis_min']
        self.max_y = image_rect['y_axis_max']
        self.size_x = image_rect['x_size']
        self.size_y = image_rect['y_size']
        self.max_iterations = max_iter

    def mandelbrot_calculation(self, c_real,c_imag):
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

    mandelbrot_object = Mandelbrot(image_rectangle, max_iterations)
    start = time.time()
    image = mandelbrot_object.mandelbrot_set()
    time_elapsed = time.time() - start


#plot image in window
    plot_mandelbrot(image_rectangle, image, time_elapsed, max_iterations)

main()
