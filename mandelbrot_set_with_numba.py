import time
import numpy as np
from numba.experimental import jitclass
from numba import int64, float64
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Qt5Agg")


spec = [('min_x', float64),('max_x',float64),('min_y', int64),('max_y', int64),('size_x',int64),('size_y',int64),('maxIter',int64)]
@jitclass(spec)
class Mandelbrot():

    def __init__(self,x_min,x_max,y_min,y_max,x_size, y_size, max_iter):
        self.min_x = x_min
        self.max_x = x_max
        self.min_y = y_min
        self.max_y = y_max
        self.size_x = x_size
        self.size_y = y_size
        self.max_Iter = max_iter

    def mandelbrot_calculation(self,c_real,c_imag):
        real = c_real
        imag = c_imag
        for n in range(self.max_Iter):
            real2 = real*real
            imag2 = imag*imag
            if real2 + imag2 > 4.0:
                return n
            imag = 2* real*imag + c_imag
            real = real2 - imag2 + c_real
            return self.max_Iter


    def mandelbrot_set(self):
        stepsize_x = (self.max_x - self.min_x)/self.size_x
        stepsize_y = (self.max_y - self.min_y)/self.size_y
        X = np.arange(self.min_x, self.max_x, stepsize_x)
        Y = np.arange(self.min_y, self.max_y, stepsize_y)
        Z = np.zeros((len(Y), len(X)))
        for iy, y in enumerate(Y):
            for ix, x in enumerate(X):
                Z[iy,ix] = self.mandelbrot_calculation(x,y)
        return Z


def plot_mandelbrot(min_x, max_x, min_y, max_y, Z_temp, dt):
    plt.imshow(Z_temp, cmap = plt.cm.prism, interpolation = None, extent = (min_x, max_x, min_y, max_y))
    plt.xlabel("Re(c), using numba jit compiler time: %f s" % dt)
    plt.ylabel("Im(c), max iter =300")
    plt.title( "mandelbrot set, image size (x,y): 4096 x 4096 pixels")
    plt.savefig("mandelbrot_python_optimize_numba_jit.png")
    plt.show()
    plt.close()


def main():

    x_min = -2
    x_max = .5
    y_min = -1
    y_max = 1
    x_size = 4096
    y_size = 4096
    max_iter = 300

    mandelbrot_object = Mandelbrot(x_min,x_max,y_min,y_max,x_size, y_size, max_iter)

    start = time.time()
    Z = mandelbrot_object.mandelbrot_set()
    dt = time.time() - start

    plot_mandelbrot(x_min, x_max, y_min, y_max, Z, dt)

main()
