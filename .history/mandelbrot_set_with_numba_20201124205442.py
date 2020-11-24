from matplotlib.pyplot import axes
import numpy as np
from numpy import NaN
from numba.experimental import jitclass
from numba import int64, float64
import time
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Qt5Agg")


spec = [('MinX', float64),('MaxX',float64),('MinY', int64),('MaxY', int64),('Size_X',int64),('Size_Y',int64),('MaxIter',int64)]
@jitclass(spec)
class Mandelbrot():    
        
        def __init__(self,xmin,xmax,ymin,ymax,X_size, Y_size, maxiter):
                self.MinX = xmin
                self.MaxX = xmax
                self.MinY = ymin
                self.MaxY = ymax
                self.Size_X = X_size
                self.Size_Y = Y_size
                self.MaxIter = maxiter
       
        def MandelbrotCalculation(self,creal,cimag):
                real = creal
                imag = cimag
                for n in range(self.MaxIter):
                        real2 = real*real
                        imag2 = imag*imag
                        if real2 + imag2 > 4.0:
                                return n
                        imag = 2* real*imag + cimag
                        real = real2 - imag2 + creal       
                return self.MaxIter

      
        def Mandelbrot_set(self):
                stepsize_x = (self.MaxX - self.MinX)/self.Size_X
                stepsize_y = (self.MaxY - self.MinY)/self.Size_Y
                X = np.arange(self.MinX, self.MaxX, stepsize_x)
                Y = np.arange(self.MinY, self.MaxY, stepsize_y)
                Z = np.zeros((len(Y), len(X)))
                for iy, y in enumerate(Y):
                        for ix, x in enumerate(X):
                                Z[iy,ix] = self.MandelbrotCalculation(x,y)
                return (Z)
        
                
def Plot_Mandelbrot(MinX, MaxX, MinY, MaxY, Ztemp, dt): 
               plt.imshow(Ztemp, cmap = plt.cm.prism, interpolation = None, extent = (MinX, MaxX, MinY, MaxY))
               plt.xlabel("Re(c), using numba jit compiler time: %f s" % dt)
               plt.ylabel("Im(c), max iter =300")
               plt.title( "mandelbrot set, image size (x,y): 4096 x 4096 pixels")
               plt.savefig("mandelbrot_python_optimize_numba_jit.png")
               plt.show()
               plt.close()

  
def main():
        
        xmin = -2
        xmax = .5
        ymin = -1
        ymax = 1
        X_size = 4096
        Y_size = 4096
        maxiter = 300

        mandelbrotObject = Mandelbrot(xmin,xmax,ymin,ymax,X_size, Y_size, maxiter)

        start = time.time()
        Z = mandelbrotObject.Mandelbrot_set()
        dt = time.time() - start

        Plot_Mandelbrot(xmin, xmax, ymin, ymax, Z, dt)
        

main()
