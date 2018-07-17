import numpy as np
from rbf cimport rbf_1, rbf_2, gradrbf_1, gradrbf_x, gradrbf_y
cimport cython

DTYPE = np.float64

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
# vectorised for matrix generation
def a_integrand_2d_mat(double [:] y0, double [:] x0, double yi, double xi, double yj, double xj, double d):
    cdef int y0_length = len(y0), x0_length = len(x0)
    cdef int i, j
    cdef double non_grad_term, grad_x_term, grad_y_term
    
    outputmat_np = np.zeros((x0_length, y0_length))
    cdef double [:, :] outputmat_c = outputmat_np

    cdef double [:] x0_c = x0
    cdef double [:] y0_c = y0
    
    for i in range(x0_length):
        for j in range(y0_length):
            non_grad_term = rbf_2(d, x0_c[i], y0_c[j], xi, yi) * rbf_2(d, x0_c[i], y0_c[j], xj, yj)
            grad_x_term = gradrbf_x(d, x0_c[i], y0_c[j], xi, yi) * gradrbf_x(d, x0_c[i], y0_c[j], xj, yj)
            grad_y_term = gradrbf_y(d, x0_c[i], y0_c[j], xi, yi) * gradrbf_y(d, x0_c[i], y0_c[j], xj, yj)
            outputmat_c[i,j] = grad_x_term + grad_y_term + non_grad_term 
    
    return outputmat_np


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
# vectorised for matrix generation
def a_boundary_integrand_2d_mat(double [:] y0, double [:] x0, double yi, double xi, double yj, 
                                double xj, double d, double sigma, double theta, double bdy):
    cdef int y0_length = len(y0), x0_length = len(x0)
    cdef int i, j
    cdef double non_grad_term, grad_u_term, grad_v_term
    cdef double sign = 1.
    
    outputmat_np = np.zeros((x0_length, y0_length))
    cdef double [:, :] outputmat_c = outputmat_np

    cdef double [:] x0_c = x0
    cdef double [:] y0_c = y0

    if bdy == 0. or bdy == 1.:
        gradrbf = gradrbf_x
    else:
        gradrbf = gradrbf_y

    if bdy == 0. or bdy == 2.:
        sign = -1.

    for i in range(x0_length): # v
        for j in range(y0_length): # u
            non_grad_term = sigma * rbf_2(d, x0_c[i], y0_c[j], xi, yi) * rbf_2(d, x0_c[i], y0_c[j], xj, yj)
            grad_v_term = theta * sign * gradrbf(d, x0_c[i], y0_c[j], xi, yi) * rbf_2(d, x0_c[i], y0_c[j], xj, yj)
            grad_u_term = rbf_2(d, x0_c[i], y0_c[j], xi, yi) * sign * gradrbf(d, x0_c[i], y0_c[j], xj, yj)
            outputmat_c[i,j] = grad_v_term - grad_u_term + non_grad_term 
    
    return outputmat_np


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def a_integrand_1d_mat(double [:] r0, double ri, double rj, double d):
    cdef int r0_length = len(r0)
    cdef int i
    cdef double non_grad_term, grad_term

    outputmat_np = np.zeros((r0_length))
    cdef double [:] outputmat_c = outputmat_np

    cdef double [:] r0_c = r0

    for i in range(r0_length):
        non_grad_term = rbf_1(d, r0_c[i], ri) * rbf_1(d, r0_c[i], rj)
        grad_term = gradrbf_1(d, r0_c[i], ri) * gradrbf_1(d, r0_c[i], rj)
        outputmat_c[i] = grad_term + non_grad_term

    return outputmat_np


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
# vectorised for matrix generation
def F_integrand_2d_mat(double [:] y0, double [:] x0, double yi, double xi, f, double d):
    cdef int y0_length = len(y0), x0_length = len(x0)
    cdef int i, j
    cdef double fterm, rbfterm
    
    outputmat_np = np.zeros((x0_length, y0_length))
    cdef double [:, :] outputmat_c = outputmat_np

    cdef double [:] x0_c = x0
    cdef double [:] y0_c = y0
    
    for i in range(x0_length):
        for j in range(y0_length):
            fterm = f(x0_c[i], y0_c[j])
            rbfterm = rbf_2(d, x0_c[i], y0_c[j], xi, yi)
            outputmat_c[i,j] = fterm*rbfterm
    
    return outputmat_np


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def F_dirichlet_boundary_integrand_2d_mat(double [:] y0, double [:] x0, 
    double yi, double xi, g, double d, double theta, double sigma, double bdy):
    cdef int y0_length = len(y0), x0_length = len(x0)
    cdef int i, j
    cdef double gterm, rbfterm, gradterm
    cdef double sign = 1.
    
    outputmat_np = np.zeros((x0_length, y0_length))
    cdef double [:, :] outputmat_c = outputmat_np

    cdef double [:] x0_c = x0
    cdef double [:] y0_c = y0

    if bdy == 0. or bdy == 1.:
        gradrbf = gradrbf_x
    else:
        gradrbf = gradrbf_y

    if bdy == 0. or bdy == 2.:
        sign = -1.
    
    for i in range(x0_length):
        for j in range(y0_length):
            gterm = g(x0_c[i], y0_c[j])
            rbfterm = rbf_2(d, x0_c[i], y0_c[j], xi, yi)
            gradterm = gradrbf(d, x0_c[i], y0_c[j], xi, yi)
            outputmat_c[i,j] = (sign * theta * gterm * gradterm) + (sigma * gterm * rbfterm) 
    
    return outputmat_np


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def F_neumann_boundary_integrand_2d_mat(double [:] y0, double [:] x0, double yi, double xi, g, double d, double theta, double bdy):
    cdef int y0_length = len(y0), x0_length = len(x0)
    cdef int i, j
    cdef double gterm, rbfterm, gradterm
    cdef double sign = 1.
    
    outputmat_np = np.zeros((x0_length, y0_length))
    cdef double [:, :] outputmat_c = outputmat_np

    cdef double [:] x0_c = x0
    cdef double [:] y0_c = y0
    
    for i in range(x0_length):
        for j in range(y0_length):
            gterm = g(x0_c[i], y0_c[j], bdy)
            rbfterm = rbf_2(d, x0_c[i], y0_c[j], xi, yi)
            outputmat_c[i,j] = gterm * rbfterm 
    
    return outputmat_np


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def F_integrand_1d_mat(double [:] r0, double ri, f, double d):
    cdef int r0_length = len(r0)
    cdef int i
    cdef double fterm, rbfterm

    outputmat_np = np.zeros((r0_length))
    cdef double [:] outputmat_c = outputmat_np

    cdef double [:] r0_c = r0

    for i in range(r0_length):
        fterm = f(r0_c[i])
        rbfterm = rbf_1(d, r0_c[i], ri)
        outputmat_c[i] = fterm*rbfterm

    return outputmat_np


def point_eval_1d(double r0, double ri, double d):
    return rbf_1(d, r0, ri)


def gradpoint_eval_1d(double r0, double ri, double d):
    return gradrbf_1(d, r0, ri)
