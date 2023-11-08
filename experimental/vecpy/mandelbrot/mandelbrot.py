'''Compute the mandelbrot set, see
    * https://mathworld.wolfram.com/MandelbrotSet.html
    * https://en.wikipedia.org/wiki/Mandelbrot_set
'''

import sys
import numpy as np
from matplotlib import pyplot as plt

MAX_NORM_SQUARED = 4

def mandel_recurrence(c_re, c_im, max_iters):
    '''
    Computes the Mandelbrot recurrence equation:

        z_{n+1} = z_{n}^2 + c,

    where z_0 = c. Returns either max_iters or the number of iterations
    requires to reach MAX_NORM_SQUARED.
    '''
    z_re = c_re
    z_im = c_im
    for i in range(0, max_iters):
        if (z_re * z_re + z_im * z_im) > MAX_NORM_SQUARED:
            break

        new_re = z_re*z_re - z_im * z_im
        new_im = 2 * z_re * z_im

        z_re = c_re + new_re
        z_im = c_im + new_im

    return i

def mandelbrot(x_0, y_0, x_1, y_1, width, height, max_iterations):
    '''
    Computes the Mandelbrot set for a square plane between (x0, y0) and (x1,
    y1). The number of points in X and Y axis are equal `width` and `height`,
    respectively.
    '''
    dx = (x_1 - x_0) / width
    dy = (y_1 - y_0) / height

    # Init the output
    output = np.zeros((width, height))

    for j in range(0, height):
        for i in range (0, width):
            x = x_0 + i * dx
            y = y_0 + j * dy

            output[i, j] = mandel_recurrence(x, y, max_iterations)

    return output

def main():
    '''Main entry point'''
    # TODO: These inputs should be script parameters
    data = mandelbrot(-1, -1, 1, 1, 100, 100, 1000)

    # Print the camputed values (helpful when comparing against the MLIR
    # implementation)
    np.set_printoptions(threshold=sys.maxsize)
    print(data)

    # Display the image
    plt.imshow(data, interpolation='nearest')
    plt.show()

main()
