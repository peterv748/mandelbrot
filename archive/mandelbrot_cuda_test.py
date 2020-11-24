# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 14:44:19 2018

@author: peter
"""

import numpy as np
from numpy import NaN
from numba import cuda
import matplotlib.pyplot as plt
import time

@cuda.jit(device=True) 
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
    return maxiter




@cuda.jit
def mandel_kernel(min_x, max_x, min_y, max_y, stepsize, image, iters):
  height = image.shape[0]
  width = image.shape[1]
 
  pixel_size_x = (max_x - min_x) / width
  pixel_size_y = (max_y - min_y) / height
  
  startX, startY = cuda.grid(2)


  startX = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
  startY = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y
  gridX = cuda.gridDim.x * cuda.blockDim.x;
  gridY = cuda.gridDim.y * cuda.blockDim.y;

  for x in range(startX, width, gridX):
    real = min_x + x * pixel_size_x
    for y in range(startY, height, gridY):
      imag = min_y + y * pixel_size_y 
      image[y, x] = mandelbrot(real, imag, iters)


gimage = np.zeros((4000, 5000), dtype = np.uint8)
print ("height : %f " % gimage.shape[0], "width : %f" % gimage.shape[1])
blockdim = (64, 16)
griddim = (64,32)

xmin = -2
xmax = 0.5
ymin = -1
ymax = 1
stepsize = 0.0005
maxiter = 300

start = time.time()
d_image = cuda.to_device(gimage)
mandel_kernel[griddim, blockdim](xmin,xmax, ymin, ymax, stepsize, d_image, maxiter) 
d_image.to_host()
dt = time.time() - start

print ("Mandelbrot created on GPU in %f s" % dt)

plt.imshow(d_image, cmap = plt.cm.prism, interpolation = 'none', extent = (xmin, xmax, ymin, ymax))
plt.xlabel("Re(c)")
plt.ylabel("Im(c)")
plt.savefig("mandelbrot_python_optimize_cuda_gpu.png")
plt.show()
plt.close()
