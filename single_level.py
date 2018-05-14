import numpy as np
import time
from rbf import *
from quadrature import gauleg


 # 2d bilinear form --  if we could get just the general form from the user
 # maybe we can abstract these parts away from the user into a "forms.py" module?
def a_integrand_2d(y0, x0, yi, xi, yj, xj, rbf, gradrbf_x, gradrbf_y, d):
    non_grad_term = rbf(d, x0, y0, xi, yi) * rbf(d, x0, y0, xj, yj)
    grad_x_term = gradrbf_x(d, x0, y0, xi, yi) * gradrbf_x(d, x0, y0, xj, yj)
    grad_y_term = gradrbf_y(d, x0, y0, xi, yi) * gradrbf_y(d, x0, y0, xj, yj)
    return grad_x_term + grad_y_term + non_grad_term


# vectorised for matrix generation
def a_integrand_2d_mat(y0, x0, yi, xi, yj, xj, rbf, gradrbf_x, gradrbf_y, d):
    outputmat = np.zeros((len(x0),len(y0)))
    for i in range(len(y0)):
        non_grad_term = rbf(d, x0[i], y0, xi, yi) * rbf(d, x0[i], y0, xj, yj)
        grad_x_term = gradrbf_x(d, x0[i], y0, xi, yi) * gradrbf_x(d, x0[i], y0, xj, yj)
        grad_y_term = gradrbf_y(d, x0[i], y0, xi, yi) * gradrbf_y(d, x0[i], y0, xj, yj)
        outputmat[i,:] = grad_x_term + grad_y_term + non_grad_term 
    return outputmat


# 2d linear form -- can refactor with RBF object?
def F_integrand_2d(y0, x0, yi, xi, rbf, f, d):
    return f(x0, y0) * rbf(d, x0, y0, xi, yi)


# vectorised for matrix generation
def F_integrand_2d_mat(y0, x0, yi, xi, rbf, f, d):
    outputmat = np.zeros((len(x0), len(y0)))
    for i in range(len(y0)):
        fterm = f(x0[i], y0)
        rbfterm = rbf(d, x0[i], y0, xi, yi)
        outputmat[i,:] = fterm*rbfterm
    return outputmat


# 1d bilinear form
def a_integrand_1d(r0, ri, rj, rbf, grad_rbf, d):
    non_grad_term = rbf(d, r0, ri) * rbf(d, r0, rj)
    grad_term = grad_rbf(d, r0, ri) * grad_rbf(d, r0, rj)
    return grad_term + non_grad_term


# 1d linear form
def F_integrand_1d(r0, ri, rbf, f, d):
    return f(r0) * rbf(d, r0, ri)


# Cythonise -- static typing? nested loop unrolling? function evaluations?
def build_matrix_problem(N, xcentres, ycentres, pts, weights, f):
	N2 = N*N
	rhs_vec_1 = np.zeros(N2)
	A_mat_1 = np.zeros((N2,25))
	for i in range(N**2):
	    a1 = max(-1, xcentres[i] - 1.0/0.7)
	    b1 = min(1, xcentres[i] + 1.0/0.7)
	    a2 = max(-1, ycentres[i] - 1.0/0.7)
	    b2 = min(1, ycentres[i] + 1.0/0.7)
	    scale_2 = ((b2-a2)/2)
	    scale_1 = ((b1 - a1)/2)
	    rhs_vec_1[i] = scale_1 * scale_2 * np.einsum('i,i',weights, 
	                np.einsum('ij,j->i',F_integrand_2d_mat(scale_2*pts + (a2+b2)/2, scale_1*pts + (a1+b1)/2,ycentres[i],
	                                                     xcentres[i],rbf_2,f,0.7), weights))
	    for j in range(i+1):
	        if (np.sqrt((xcentres[i]-xcentres[j])**2 + (ycentres[i] - ycentres[j])**2) > 2./0.7):
	            A_mat_1[i,j] = 0
	        else:
	            A1 = max(-1, max(xcentres[i], xcentres[j])-1./0.7)
	            B1 = min(1, min(xcentres[i], xcentres[j])+1./0.7)
	            A2 = max(-1, max(ycentres[i], ycentres[j]) - 1./0.7)
	            B2 = min(1, min(ycentres[i], ycentres[j]) + 1./0.7)
	            Scale_m2 = ((B2 - A2)/2)
	            Scale_m1 = ((B1 - A1)/2)
	            A_mat_1[i,j] = Scale_m1 * Scale_m2 * np.einsum('i,i',weights,
	            np.einsum('ij,j->i', a_integrand_2d_mat(Scale_m2*pts + (A2+B2)/2, Scale_m1*pts + (A1+B1)/2,
	                    ycentres[i], xcentres[i], ycentres[j], xcentres[j],rbf_2,gradrbf_x,gradrbf_y,0.7),
	                      weights))

	return A_mat_1, rhs_vec_1


def main():
	start = time.time()
	# generate the centres -- this can easily be changed for different rectangular domains
	# needs more thought for non-rectangles!
	N = 5
	xcentres, ycentres = np.meshgrid(np.linspace(-1,1,N),np.linspace(-1,1,N))
	xcentres = xcentres.reshape(N*N,)
	ycentres = ycentres.reshape(N*N,)

	# vectorise creation of the matrix problem, use Gauss-Legendre quadrature to integrate
	f = lambda x, y: np.cos(np.pi*x)*np.cos(np.pi*y)

	# could pick a different number of points depending on accuracy needed
	pts, weights = gauleg(200)

	# make the full A
	A_mat_1, rhs_vec_1 = build_matrix_problem(N, xcentres, ycentres, pts, weights, f)
	A_mat_1_full = A_mat_1 + A_mat_1.transpose() - np.diag(np.diag(A_mat_1))

	# and solve the linear system
	c = np.linalg.solve(A_mat_1_full, rhs_vec_1)

	# exact solution lambda function
	u = lambda x, y: np.cos(np.pi*x)*np.cos(np.pi*y)/(2*np.pi**2 + 1) # vectorised

	# evaluate the solution at a set of points
	test_pt_ct = 40
	test_grid = np.linspace(-1, 1, test_pt_ct)
	x_test_pts, y_test_pts = np.meshgrid(test_grid, test_grid)
	x_test_pts = x_test_pts.reshape(test_pt_ct**2,)
	y_test_pts = y_test_pts.reshape(test_pt_ct**2,)
	exact_sol = u(x_test_pts, y_test_pts).reshape((test_pt_ct**2,))

	# evaluate the basis functions and numerical soln at the same set of points
	RBF_vals = rbf_2_mat(0.7, x_test_pts, y_test_pts, xcentres.reshape((N*N,)), ycentres.reshape((N*N,)))
	numerical_sol = np.dot(RBF_vals,c)

	# find the error in the numerical solutions
	print('RMS Error = {:.6E}'.format(np.linalg.norm(exact_sol-numerical_sol, 2)/test_pt_ct))
	print('Maximum Error = {:.6E}'.format(np.linalg.norm(exact_sol - numerical_sol, np.inf)))
	print('Total time taken was {:.3E}'.format(time.time()-start))

	return


if __name__ == '__main__':
	main()



