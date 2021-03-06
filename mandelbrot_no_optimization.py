"""
Created on Fri Oct 19 14:44:19 2018
Updated on 17 December 2020
adhering to Py_lint standards
removing duplicate code by creating separate modules

@author: peter
"""

import time
import numpy as np
import draw_mandelbrot
import class_mandelbrot_v1


def main():
    """
    Main function
    """

    max_iterations = np.int64(300)
    image_rectangle = np.array([-2, 0.5, -1, 1], dtype=np.float64)
    image_size = np.array([4096,4096], dtype=np.int64)

    mandelbrot_object = class_mandelbrot_v1.Mandelbrot(image_rectangle, image_size, max_iterations)

    start = time.time()
    image = mandelbrot_object.mandelbrot_set()
    time_elapsed = time.time() - start


#plot image in window
    draw_mandelbrot.plot_mandelbrot(image_rectangle, image_size, image, \
                                    time_elapsed, max_iterations)

main()
