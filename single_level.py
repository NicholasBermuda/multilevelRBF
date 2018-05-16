import numpy as np
import time
from rbf import rbf_2_mat
from forms import a_integrand_2d_mat, F_integrand_2d_mat
from quadrature import gauleg

def build_matrix_problem(N, xcentres, ycentres, pts, weights, f, delta):
    N2 = N*N
    rhs_vec = np.zeros(N2)
    A_mat = np.zeros((N2,N2))
    for i in range(N2):
        a1 = max(-1, xcentres[i] - 1.0/delta)
        b1 = min(1, xcentres[i] + 1.0/delta)
        a2 = max(-1, ycentres[i] - 1.0/delta)
        b2 = min(1, ycentres[i] + 1.0/delta)
        scale_2 = ((b2-a2)/2)
        scale_1 = ((b1 - a1)/2)
        rhs_vec[i] = scale_1 * scale_2 * np.einsum('i,i',weights, 
                    np.einsum('ij,j->i',F_integrand_2d_mat(scale_2*pts + (a2+b2)/2, scale_1*pts + (a1+b1)/2,ycentres[i],
                                                         xcentres[i],f,delta), weights))
        for j in range(i+1):
            if (np.sqrt((xcentres[i]-xcentres[j])**2 + (ycentres[i] - ycentres[j])**2) > 2./delta):
                A_mat[i,j] = 0
            else:
                A1 = max(-1, max(xcentres[i], xcentres[j])-1./delta)
                B1 = min(1, min(xcentres[i], xcentres[j])+1./delta)
                A2 = max(-1, max(ycentres[i], ycentres[j]) - 1./delta)
                B2 = min(1, min(ycentres[i], ycentres[j]) + 1./delta)
                Scale_m2 = ((B2 - A2)/2)
                Scale_m1 = ((B1 - A1)/2)
                bilin_mat = a_integrand_2d_mat(Scale_m2*pts + (A2+B2)/2, Scale_m1*pts + (A1+B1)/2, ycentres[i], 
                                    xcentres[i], ycentres[j], xcentres[j], delta)
                A_mat[i,j] = Scale_m1 * Scale_m2 * np.einsum('i,i',weights, np.einsum('ij,j->i',bilin_mat, weights))

    return A_mat, rhs_vec


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
    pts, weights = gauleg(100) # number related to no. of basis funcs/support radius
    # rethink number of points as we do multilevel

    # pick a value for delta
    delta = 0.7

    # make the full A
    A_mat, rhs_vec = build_matrix_problem(N, xcentres, ycentres, pts, weights, f, delta)
    # print(A_mat[0:4,0:4])
    A_mat_full = A_mat + A_mat.transpose() - np.diag(np.diag(A_mat))

    # and solve the linear system
    c = np.linalg.solve(A_mat_full, rhs_vec)
    # print(np.linalg.cond(A_mat_full))

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
    RBF_vals = rbf_2_mat(delta, x_test_pts, y_test_pts, xcentres.reshape((N*N,)), ycentres.reshape((N*N,)))
    numerical_sol = np.dot(RBF_vals,c)

    # find the error in the numerical solutions
    print('RMS Error = {:.6E}'.format(np.linalg.norm(exact_sol-numerical_sol, 2)/test_pt_ct))
    print('Maximum Error = {:.6E}'.format(np.linalg.norm(exact_sol - numerical_sol, np.inf)))
    print('Total time taken was {:.3f}'.format(time.time()-start))

    return


if __name__ == '__main__':
    main()



