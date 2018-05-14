import numpy as np
cimport cython
from libc.math cimport sqrt

DTYPE = np.float64

# 2D rbf evaluation at given points - "vectorised"
# possible Cythonisation - increase speed by using static typing??
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def rbf_2(double d, double x0, double [:] y0, double xi, double yi):
    # define some C vars to hold data
    cdef double x, y, r, tmp
    cdef int i
    cdef int y_range = y0.size

    # use memoryview of output
    output_np = np.zeros(y0.size, dtype = DTYPE)
    cdef double [:] output_c = output_np

    # perform the calculation on the C array
    x = x0 - xi
    for i in range(y_range):
        y = y0[i] - yi
        r = sqrt(x*x + y*y)
        tmp = max(0, 1 - d*r)
        output_c[i] = tmp*tmp*tmp*tmp*(4*d*r + 1)

    return output_np


# rbf evaluations at arbitrary test points -- for manufacture sol
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def rbf_2_mat(double d, double [:] x0, double [:] y0, double [:] xi, double [:] yi):
    # define some C vars to hold data
    cdef int num_centres = xi.size
    cdef int num_evals = x0.size
    cdef double x, y, r, tmp

    # use memory view of output
    output_np = np.zeros([num_evals, num_centres], dtype = DTYPE)
    cdef double [:, :] output_c = output_np

    # perform the calculation on the C array
    for i in range(num_evals):
        for j in range(num_centres):
            x = x0[i] - xi[j]
            y = y0[i] - yi[j]
            r = sqrt(x*x + y*y)
            tmp = max(0, 1 - d*r)
            output_c[i,j] = (tmp*tmp*tmp*tmp) * (4*d*r + 1)
    return output_np


# gradient components of the 2D basis functions
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def gradrbf_x(double d, double x0, double [:] y0, double xi, double yi):
    # define some C vars to hold data
    cdef double x, y, r, tmp
    cdef int i
    cdef int y_range = y0.size

    # use memoryview of output
    output_np = np.zeros(y0.size, dtype = DTYPE)
    cdef double [:] output_c = output_np

    # perform the calculation on the C array
    x = x0 - xi
    for i in range(y0.size):
        y = y0[i] - yi
        r = sqrt(x*x + y*y)
        tmp = max(0, 1 - d*r)
        output_c[i] = -20*d*d*x*tmp*tmp*tmp

    return output_np


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def gradrbf_y(double d, double x0, double [:] y0, double xi, double yi):
    # define some C vars to hold data
    cdef double x, y, r, tmp
    cdef int i
    cdef int y_range = y0.size

    # use memoryview of output
    output_np = np.zeros(y0.size, dtype = DTYPE)
    cdef double [:] output_c = output_np

    # perform the calculation on the C array
    x = x0 - xi
    for i in range(y0.size):
        y = y0[i] - yi
        r = sqrt(x*x + y*y)
        tmp = max(0, 1 - d*r)
        output_c[i] = -20*d*d*y*tmp*tmp*tmp

    return output_np
