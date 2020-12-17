"""
    plotting the calculated mandelbrot set and writing it to file
"""

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Qt5Agg")

def plot_mandelbrot(image_rect, im_size, image_temp, elapsed_time, iterations):
    """
    plotting the calculated mandelbrot set and writing it to file
    """
    image_dimension = str(im_size[0]) + " x " + str(im_size[1])
    plt.imshow(image_temp, cmap = plt.prism(), interpolation = None, \
                extent = (image_rect[0], image_rect[1], \
                image_rect[2], image_rect[3]))
    plt.xlabel("Re(c), using jit optimization time: {0}".format(elapsed_time))
    plt.ylabel("Im(c), max iter: {0}:".format(iterations))
    plt.title( "mandelbrot set, image size (x,y): {0}".format(image_dimension))
    plt.savefig("mandelbrot_numba_jit_optimization.png")
    plt.show()
    plt.close()

if __name__ == "__main__":
    import numpy as np

    MAXIMUM_ITERATIONS = 300
    image_rectangle = np.array([-2, 0.5, -1, 1])
    image_size = np.array([4096,4096])

    stepsize_x = (image_rectangle[1] - image_rectangle[0])/image_size[0]
    stepsize_y = (image_rectangle[3] - image_rectangle[2])/image_size[1]
    x_axis_array = np.arange(image_rectangle[0],image_rectangle[1], stepsize_x)
    y_axis_array = np.arange(image_rectangle[2], image_rectangle[3], stepsize_y)
    image_array = np.zeros((len(y_axis_array), len(x_axis_array)))
    TIME_ELAPSED = 10
    plot_mandelbrot(image_rectangle, image_size, image_array, TIME_ELAPSED, MAXIMUM_ITERATIONS)
    