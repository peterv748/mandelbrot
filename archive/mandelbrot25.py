
import numpy as np
from numpy import NaN
from numba import jit
import matplotlib.pyplot as plt
import time

@jit 
def mandelbrot(creal,cimag,maxiter):
    real = creal
    imag = cimag
    for n in range(maxiter):
        real2 = real*real
        imag2 = imag*imag
        if real2 + imag2 > 4.0:
            return n
        imag = 2* real*imag + cimag
        real = real2 - imag2 + creal       
    return NaN

@jit
def mandelbrot_set(xmin,xmax,ymin,ymax,stepsize,maxiter):
        X = np.arange(xmin, xmax, stepsize)
        Y = np.arange(ymin, ymax, stepsize)
        Z = np.zeros((len(Y), len(X)))
 
        for iy, y in enumerate(Y):
                for ix, x in enumerate(X):
                        Z[iy,ix] = mandelbrot(x,y, maxiter)

        return (Z)

xmin = -2
xmax = .5
ymin = -1
ymax = 1
stepsize = .0005
maxiter = 300

start = time.time()
Z = mandelbrot_set(xmin, xmax, ymin, ymax, stepsize, maxiter)
dt = time.time() - start

print ("Mandelbrot created on CPU in %f s" % dt)
plt.imshow(Z, cmap = plt.cm.prism, interpolation = 'none', extent = (xmin, xmax, ymin, ymax))
plt.xlabel("Re(c)")
plt.ylabel("Im(c)")
plt.savefig("mandelbrot_python_optimaze_cuda_jit.png")
plt.show()
