"""
    mandelbrot class
"""
import numpy as np
import complex_calculation_mandelbrot

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
                image_array[j,i] = \
                complex_calculation_mandelbrot.complex_mandelbrot_calculation(x_coord,y_coord, self.max_iterations)
        return image_array
