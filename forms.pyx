import numpy as np
from rbf cimport rbf_2, gradrbf_x, gradrbf_y
cimport cython

DTYPE = np.float64

 # 2d bilinear form --  if we could get just the general form from the user
 # maybe we can abstract these parts away from the user into a "forms.py" module?
# def a_integrand_2d(y0, x0, yi, xi, yj, xj, d):
#     non_grad_term = rbf_2(d, x0, y0, xi, yi) * rbf_2(d, x0, y0, xj, yj)
#     grad_x_term = gradrbf_x(d, x0, y0, xi, yi) * gradrbf_x(d, x0, y0, xj, yj)
#     grad_y_term = gradrbf_y(d, x0, y0, xi, yi) * gradrbf_y(d, x0, y0, xj, yj)
#     return grad_x_term + grad_y_term + non_grad_term


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


# # 2d linear form -- can refactor with RBF object?
# def F_integrand_2d(y0, x0, yi, xi, rbf, f, d):
#     return f(x0, y0) * rbf(d, x0, y0, xi, yi)


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


# # 1d bilinear form
# def a_integrand_1d(r0, ri, rj, rbf, grad_rbf, d):
#     non_grad_term = rbf(d, r0, ri) * rbf(d, r0, rj)
#     grad_term = grad_rbf(d, r0, ri) * grad_rbf(d, r0, rj)
#     return grad_term + non_grad_term


# # 1d linear form
# def F_integrand_1d(r0, ri, rbf, f, d):
#     return f(r0) * rbf(d, r0, ri)