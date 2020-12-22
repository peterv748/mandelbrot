"""
 calculation of mandelbrot set formula
"""
from numba import jit_module

def complex_mandelbrot_calculation(c_real,c_imag, iterations):
    """
    calculation of mandelbrot set formula
    """
    real = c_real
    imag = c_imag


    for i in range(iterations):
        real2 = real*real
        imag2 = imag*imag
        if real2 + imag2 > 4.0:
            return i
        imag = 2* real*imag + c_imag
        real = real2 - imag2 + c_real
    return int(iterations)

jit_module(nopython=True)

if __name__ == "__main__":
    REAL = 0.3
    IMAG = 0.5
    MAXIMUM_ITERATIONS = 200

    print(complex_mandelbrot_calculation(REAL, IMAG, MAXIMUM_ITERATIONS))
