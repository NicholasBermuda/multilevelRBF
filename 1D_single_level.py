import numpy as np
import time
from rbf import rbf_1_mat
from quadrature import gauleg
from build_mat import build_1d_neumann_problem, build_1d_dirichlet_problem
import matplotlib.pyplot as plt

def homogeneous_dirichlet():
    start = time.time()

    # generate the centres
    N = 9
    centres = np.linspace(-1, 1, N)

    # forcing term
    f = lambda r: np.sin(np.pi*r)

    # use Gauss-Legendre quadrature to integrate:
    pts, weights = gauleg(30)

    # pick a value for delta
    delta = 0.7

    # make the matrix system
    A_mat, rhs_vec = build_1d_dirichlet_problem(N, centres, pts, weights, f, delta)
    A_mat_full = A_mat + A_mat.transpose() - np.diag(np.diag(A_mat))

    # and solve the system
    c = np.linalg.solve(A_mat_full, rhs_vec)
    print('Condition number of stiffness matrix is {:.3E}'.format(np.linalg.cond(A_mat_full)))

    # exact solution lambda function
    u = lambda r: np.sin(np.pi*r)/(np.pi*np.pi + 1)

    # evaluate the solution at a set of test points
    test_pt_ct = 40
    test_pts = np.linspace(-1, 1, test_pt_ct)
    exact_sol = u(test_pts)

    # evaluate the basis functions at the test points
    RBF_vals = rbf_1_mat(delta, test_pts, centres)
    numerical_sol = np.dot(RBF_vals, c)

    # find the error in the numerical solutions
    print('RMS Error = {:.6E}'.format(np.linalg.norm(exact_sol-numerical_sol, 2)/test_pt_ct))
    print('Maximum Error = {:.6E}'.format(np.linalg.norm(exact_sol - numerical_sol, np.inf)))
    print('Total time taken was {:.3f} seconds'.format(time.time()-start))

    plt.plot(test_pts, exact_sol)
    plt.plot(test_pts, numerical_sol)
    plt.show()


def inhomogeneous_dirichlet():
    start = time.time()

    bdy = (-np.pi/2, np.pi/2)
    bc = (-1.0, 1.0)

    # generate the centres
    N = 9
    centres = np.linspace(bdy[0], bdy[1], N)

    # forcing term
    f = lambda r: 2*np.sin(r)

    # use Gauss-Legendre quadrature to integrate:
    pts, weights = gauleg(30)

    # pick a value for delta
    delta = 0.7

    # make the matrix system
    A_mat, rhs_vec = build_1d_dirichlet_problem(N, centres, pts, weights, f, delta, bdy=bdy, bc=(-1.0, 1.0))
    A_mat_full = A_mat + A_mat.transpose() - np.diag(np.diag(A_mat))

    # and solve the system
    c = np.linalg.solve(A_mat_full, rhs_vec)
    print('Condition number of stiffness matrix is {:.3E}'.format(np.linalg.cond(A_mat_full)))

    # exact solution lambda function
    u = lambda r: np.sin(r)

    # evaluate the solution at a set of test points
    test_pt_ct = 40
    test_pts = np.linspace(bdy[0], bdy[1], test_pt_ct)
    exact_sol = u(test_pts)

    # evaluate the basis functions at the test points
    RBF_vals = rbf_1_mat(delta, test_pts, centres)
    numerical_sol = np.dot(RBF_vals, c)

    # find the error in the numerical solutions
    print('RMS Error = {:.6E}'.format(np.linalg.norm(exact_sol-numerical_sol, 2)/test_pt_ct))
    print('Maximum Error = {:.6E}'.format(np.linalg.norm(exact_sol - numerical_sol, np.inf)))
    print('Total time taken was {:.3f} seconds'.format(time.time()-start))

    plt.plot(test_pts, exact_sol)
    plt.plot(test_pts, numerical_sol)
    plt.show()

def homogeneous_neumann():
    start = time.time()

    # generate the centres
    N = 9
    centres = np.linspace(-1, 1, N)

    # forcing term
    f = lambda r: np.cos(np.pi*r)

    # use Gauss-Legendre quadrature to integrate:
    pts, weights = gauleg(30)

    # pick a value for delta
    delta = 0.7

    # make the matrix system
    A_mat, rhs_vec = build_1d_neumann_problem(N, centres, pts, weights, f, delta)
    A_mat_full = A_mat + A_mat.transpose() - np.diag(np.diag(A_mat))

    # and solve the system
    c = np.linalg.solve(A_mat_full, rhs_vec)
    print('Condition number of stiffness matrix is {:.3E}'.format(np.linalg.cond(A_mat_full)))

    # exact solution lambda function
    u = lambda r: np.cos(np.pi*r)/(np.pi*np.pi + 1)

    # evaluate the solution at a set of test points
    test_pt_ct = 40
    test_pts = np.linspace(-1, 1, test_pt_ct)
    exact_sol = u(test_pts)

    # evaluate the basis functions at the test points
    RBF_vals = rbf_1_mat(delta, test_pts, centres)
    numerical_sol = np.dot(RBF_vals, c)

    # find the error in the numerical solutions
    print('RMS Error = {:.6E}'.format(np.linalg.norm(exact_sol-numerical_sol, 2)/test_pt_ct))
    print('Maximum Error = {:.6E}'.format(np.linalg.norm(exact_sol - numerical_sol, np.inf)))
    print('Total time taken was {:.3f} seconds'.format(time.time()-start))

    plt.plot(test_pts, exact_sol)
    plt.plot(test_pts, numerical_sol)
    plt.show()


if __name__ == '__main__':
    homogeneous_neumann()