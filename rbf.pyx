import numpy as np
cimport cython
from libc.math cimport sqrt, fabs

DTYPE = np.float64

# 2D rbf evaluation at given point
cdef double rbf_2(double d, double x0, double y0, double xi, double yi):
    # define some variables
    cdef double x, y, r, tmp

    # evaluate the RBF
    x = x0 - xi
    y = y0 - yi
    r = sqrt(x*x + y*y)
    tmp = max(0, 1 - d*r)

    return tmp*tmp*tmp*tmp*(4*d*r + 1)

# 1D rbf evaluation at given point
cdef double rbf_1(double d, double r0, double ri):
    # define variables
    cdef double r, tmp

    # evaluate the rbf
    r = fabs(r0 - ri)
    tmp = max(0, 1 - d*r)

    return tmp*tmp*tmp*tmp*(4*d*r + 1)



# rbf evaluations at arbitrary test points -- for manufacture sol
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def rbf_2_mat(double d, double [:] x0, double [:] y0, double [:] xi, double [:] yi):
    # define some C vars to hold data
    cdef int num_centres = xi.size
    cdef int num_evals = x0.size
    cdef double x, y, r, tmp

    # use memory view
    output_np = np.zeros([num_evals, num_centres], dtype = DTYPE)
    cdef double [:, :] output_c = output_np
    cdef double [:] x0_c = x0
    cdef double [:] y0_c = y0
    cdef double [:] xi_c = xi
    cdef double [:] yi_c = yi

    # perform the calculation on the C array
    for i in range(num_evals):
        for j in range(num_centres):
            x = x0_c[i] - xi_c[j]
            y = y0_c[i] - yi_c[j]
            r = sqrt(x*x + y*y)
            tmp = max(0, 1 - d*r)
            output_c[i,j] = (tmp*tmp*tmp*tmp) * (4*d*r + 1)
    return output_np


# TO-DO: check this!
# 1D of above
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def rbf_1_mat(double d, double [:] r0, double [:] ri):
    # define some C vars to hold data
    cdef int num_centres = ri.size
    cdef int num_evals = r0.size
    cdef double r, tmp

    # use memory view
    output_np = np.zeros([num_evals, num_centres], dtype = DTYPE)
    cdef double [:, :] output_c = output_np
    cdef double [:] r0_c = r0
    cdef double [:] ri_c = ri

    # perform calculation on the C array
    for i in range(num_evals):
        for j in range(num_centres):
            r = fabs(r0_c[i] - ri_c[j])
            tmp = max(0, 1 - d*r)
            output_c[i, j] = (tmp*tmp*tmp*tmp) * (4*d*r + 1)
    return output_np


# gradient components of the 2D basis functions
cdef double gradrbf_x(double d, double x0, double y0, double xi, double yi):
    # define some variables 
    cdef double x, y, r, tmp
    
    # evaluate the gradient
    x = x0 - xi
    y = y0 - yi
    r = sqrt(x*x + y*y)
    tmp = max(0, 1 - d*r)

    return -20*d*d*x*tmp*tmp*tmp


cdef double gradrbf_y(double d, double x0, double y0, double xi, double yi):
    # define some variables
    cdef double x, y, r, tmp
    
    # evaluate the gradient
    x = x0 - xi
    y = y0 - yi
    r = sqrt(x*x + y*y)
    tmp = max(0, 1 - d*r)

    return -20*d*d*y*tmp*tmp*tmp

# gradient of the 1D basis function
cdef double gradrbf_1(double d, double r0, double ri):
    # define variables
    cdef double r, tmp

    # evaluate the gradient
    r = r0 - ri
    tmp = max(0, 1 - d*fabs(r))

    return -20*d*d*r*tmp*tmp*tmp

