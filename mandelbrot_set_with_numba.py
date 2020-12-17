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
import draw_mandelbrot
import class_Mandelbrot


spec = [('min_x', float64),('max_x',float64), \
        ('min_y', float64),('max_y', float64),('size_x',int64), \
        ('size_y',int64),('max_iterations',int64)]
@jitclass(spec)
class OptimizedMandelbrot(class_Mandelbrot.Mandelbrot):
    """
    Mandelbrot set calculation using numba jit optimization
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

    def optimized_mandelbrot_set(self):
        """
        calculate the mandelbrot set
        """
        return self.mandelbrot_set()

def main():
    """
    Main function
    """
    image_rectangle = np.array([-2, 0.5, -1, 1])

    image_size = np.array([4096,4096])

    max_iterations = 300

    mandelbrot_object = OptimizedMandelbrot(image_rectangle, image_size, max_iterations)
    start = time.time()
    image = mandelbrot_object.optimized_mandelbrot_set()
    time_elapsed = time.time() - start

    draw_mandelbrot.plot_mandelbrot(image_rectangle, image_size, image, \
                                    time_elapsed, max_iterations)

main()
