"""
    mandelbrot class
"""

import numpy as np

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