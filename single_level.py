import numpy as np
import time
from rbf import rbf_2_mat
from quadrature import gauleg
from build_mat import build_2d_neumann_problem, build_2d_dirichlet_problem 
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def homogeneous_dirichlet():
    start = time.time()

    bdy = [-1., 1., -1., 1.]

    # generate the centres -- this can easily be changed for different rectangular domains
    # needs more thought for non-rectangles!
    N = 17
    xcentres, ycentres = np.meshgrid(np.linspace(bdy[0], bdy[1], N),np.linspace(bdy[2], bdy[3], N))
    xcentres = xcentres.reshape(N*N,)
    ycentres = ycentres.reshape(N*N,)

    # forcing term
    f = lambda x, y: np.sin(np.pi*x)*np.sin(np.pi*y)

    # use Gauss-Legendre quadrature to integrate overlapping supports:
    # faster to generate the pts and weights just once and shift/scale as needed
    # could pick a different number of points depending on accuracy needed
    pts, weights = gauleg(30) # number related to no. of basis funcs/support radius
    # rethink number of points as we do multilevel

    # pick a value for delta
    delta = 0.7

    homog_bc = lambda x, y: 0.0
    # homog_bc = lambda x, y: np.sin(np.pi*x)*np.sin(np.pi*y)/(2*np.pi**2 + 1)

    # make the full A
    # print('Not yet implemented')
    # raise Exception
    A_mat, rhs_vec = build_2d_dirichlet_problem(N, xcentres, ycentres, pts, weights, f, delta, homog_bc)
    A_mat_full = A_mat + A_mat.transpose() - np.diag(np.diag(A_mat))

    # and solve the linear system
    c = np.linalg.solve(A_mat_full, rhs_vec)
    print('Condition number of stiffness matrix is {:.3E}'.format(np.linalg.cond(A_mat_full)))

    # exact solution lambda function
    u = lambda x, y: np.sin(np.pi*x)*np.sin(np.pi*y)/(2*np.pi**2 + 1) # vectorised

    # evaluate the solution at a set of points
    test_pt_ct = 40
    x_test_pts_g, y_test_pts_g = np.meshgrid(np.linspace(bdy[0], bdy[1], test_pt_ct), np.linspace(bdy[2], bdy[3], test_pt_ct))
    x_test_pts = x_test_pts_g.reshape(test_pt_ct**2,)
    y_test_pts = y_test_pts_g.reshape(test_pt_ct**2,)
    exact_sol = u(x_test_pts, y_test_pts).reshape((test_pt_ct**2,))

    # evaluate the basis functions and numerical soln at the same set of points
    RBF_vals = rbf_2_mat(delta, x_test_pts, y_test_pts, xcentres.reshape((N*N,)), ycentres.reshape((N*N,)))
    numerical_sol = np.dot(RBF_vals, c)
    rms_err = (exact_sol - numerical_sol)
    rms_err_m = rms_err.reshape((test_pt_ct, test_pt_ct))

    # find the error in the numerical solutions
    print('RMS Error = {:.6E}'.format(np.linalg.norm(exact_sol - numerical_sol, 2) / test_pt_ct))
    print('Maximum Error = {:.6E}'.format(np.linalg.norm(exact_sol - numerical_sol, np.inf)))
    print('Total time taken was {:.3f} seconds'.format(time.time() - start))

    # should move the following to a plotting module
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    # plt.plot(xcentres, ycentres,[0]*N*N,'k*')
    surf = ax.plot_surface(x_test_pts_g, y_test_pts_g, 
        numerical_sol.reshape((test_pt_ct,test_pt_ct)), cmap = cm.coolwarm,
        linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    errsurf = ax.plot_surface(x_test_pts_g, y_test_pts_g, 
        rms_err.reshape((test_pt_ct,test_pt_ct)), cmap = cm.coolwarm,
        linewidth=0, antialiased=False)
    fig.colorbar(errsurf, shrink=0.5, aspect=5)
    plt.savefig('helmholtz_test.eps', format='eps', dpi=1000)
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title('Error: Exact - Numerical')
    errconts = plt.contourf(x_test_pts_g, y_test_pts_g, rms_err_m,15, cmap = cm.BrBG)
    fig.colorbar(errconts, shrink=0.5, aspect=5)
    ax = fig.add_subplot(1,2,1)
    ax.set_title('Numerical Solution')
    conts = plt.contourf(x_test_pts_g, y_test_pts_g, numerical_sol.reshape((test_pt_ct,test_pt_ct)),15,cmap=cm.viridis)
    fig.colorbar(conts, shrink=0.5, aspect=5)
    plt.show()

    return


def inhomogeneous_dirichlet():
    start = time.time()

    # choose a different boundary for testing
    bdy = (-1./2, 1./2, -1./2, 1./2)
    
    # and use the solution as boundary condition -- this works fine for testing
    def inhomog_bc(x, y):
        return np.sin(np.pi*x)*np.sin(np.pi*y)/(2*np.pi**2 + 1)


    # generate the centres -- this can easily be changed for different rectangular domains
    # needs more thought for non-rectangles!
    N = 17
    xcentres, ycentres = np.meshgrid(np.linspace(bdy[0], bdy[1], N), np.linspace(bdy[2], bdy[3], N))
    xcentres = xcentres.reshape(N*N,)
    ycentres = ycentres.reshape(N*N,)

    # forcing term
    f = lambda x, y: np.sin(np.pi*x)*np.sin(np.pi*y)

    # use Gauss-Legendre quadrature to integrate overlapping supports:
    # faster to generate the pts and weights just once and shift/scale as needed
    # could pick a different number of points depending on accuracy needed
    pts, weights = gauleg(30) # number related to no. of basis funcs/support radius
    # rethink number of points as we do multilevel

    # pick a value for delta
    delta = 1.183

    # make the full A
    # print('Not yet implemented')
    # raise Exception
    A_mat, rhs_vec = build_2d_dirichlet_problem(N, xcentres, ycentres, pts, weights, f, delta, inhomog_bc)
    A_mat_full = A_mat + A_mat.transpose() - np.diag(np.diag(A_mat))

    # and solve the linear system
    c = np.linalg.solve(A_mat_full, rhs_vec)
    print('Condition number of stiffness matrix is {:.3E}'.format(np.linalg.cond(A_mat_full)))

    # exact solution lambda function
    u = lambda x, y: np.sin(np.pi*x)*np.sin(np.pi*y)/(2*np.pi**2 + 1) # vectorised

    # evaluate the solution at a set of points
    test_pt_ct = 40
    x_test_pts_g, y_test_pts_g = np.meshgrid(np.linspace(bdy[0], bdy[1], test_pt_ct), np.linspace(bdy[2], bdy[3], test_pt_ct))
    x_test_pts = x_test_pts_g.reshape(test_pt_ct**2,)
    y_test_pts = y_test_pts_g.reshape(test_pt_ct**2,)
    exact_sol = u(x_test_pts, y_test_pts).reshape((test_pt_ct**2,))

    # evaluate the basis functions and numerical soln at the same set of points
    RBF_vals = rbf_2_mat(delta, x_test_pts, y_test_pts, xcentres.reshape((N*N,)), ycentres.reshape((N*N,)))
    numerical_sol = np.dot(RBF_vals, c)
    rms_err = (exact_sol - numerical_sol)
    rms_err_m = rms_err.reshape((test_pt_ct, test_pt_ct))

    # find the error in the numerical solutions
    print('RMS Error = {:.6E}'.format(np.linalg.norm(exact_sol - numerical_sol, 2) / test_pt_ct))
    print('Maximum Error = {:.6E}'.format(np.linalg.norm(exact_sol - numerical_sol, np.inf)))
    print('Total time taken was {:.3f} seconds'.format(time.time() - start))

    # should move the following to a plotting module
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    # plt.plot(xcentres, ycentres,[0]*N*N,'k*')
    surf = ax.plot_surface(x_test_pts_g, y_test_pts_g, 
        numerical_sol.reshape((test_pt_ct,test_pt_ct)), cmap = cm.coolwarm,
        linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    errsurf = ax.plot_surface(x_test_pts_g, y_test_pts_g, 
        rms_err.reshape((test_pt_ct,test_pt_ct)), cmap = cm.coolwarm,
        linewidth=0, antialiased=False)
    fig.colorbar(errsurf, shrink=0.5, aspect=5)
    plt.savefig('helmholtz_test.eps', format='eps', dpi=1000)
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title('Error: Exact - Numerical')
    errconts = plt.contourf(x_test_pts_g, y_test_pts_g, rms_err_m,15, cmap = cm.BrBG)
    fig.colorbar(errconts, shrink=0.5, aspect=5)
    ax = fig.add_subplot(1,2,1)
    ax.set_title('Numerical Solution')
    conts = plt.contourf(x_test_pts_g, y_test_pts_g, numerical_sol.reshape((test_pt_ct,test_pt_ct)),15,cmap=cm.viridis)
    fig.colorbar(conts, shrink=0.5, aspect=5)
    plt.show()

    return


def inhomogeneous_neumann():
    start = time.time()

    bdy = [-1.0, 1.0, -1.0, 1.0]

    # this is a bc for a unit square
    def bc(x, y, which):
        factor = np.pi/(2*np.pi**2 + 1)

        if which == 0.0 or which == 2.0: # bottom and left
            sign = 1.
        else: # top and right
            sign = -1.

        if which == 2. or which == 3.: # bottom and top
            val = y
        else:
            val = x

        return sign*factor*np.sin(np.pi*val)


    # generate the centres -- this can easily be changed for different rectangular domains
    # needs more thought for non-rectangles!
    N = 17
    xcentres, ycentres = np.meshgrid(np.linspace(bdy[0], bdy[1], N),np.linspace(bdy[2], bdy[3], N))
    xcentres = xcentres.reshape(N*N,)
    ycentres = ycentres.reshape(N*N,)

    # forcing term
    f = lambda x, y: np.cos(np.pi * x) * np.cos(np.pi * y)

    # use Gauss-Legendre quadrature to integrate overlapping supports:
    # faster to generate the pts and weights just once and shift/scale as needed
    # could pick a different number of points depending on accuracy needed
    pts, weights = gauleg(30) # number related to no. of basis funcs/support radius
    # rethink number of points as we do multilevel

    # pick a value for delta
    delta = 1.183

    # make the full A
    A_mat, rhs_vec = build_2d_neumann_problem(N, xcentres, ycentres, pts, weights, f, delta, bc, bdy)
    A_mat_full = A_mat + A_mat.transpose() - np.diag(np.diag(A_mat))

    # and solve the linear system
    c = np.linalg.solve(A_mat_full, rhs_vec)
    print('Condition number of stiffness matrix is {:.3E}'.format(np.linalg.cond(A_mat_full)))

    # exact solution lambda function
    u = lambda x, y: np.cos(np.pi*x)*np.cos(np.pi*y)/(2*np.pi**2 + 1) # vectorised

    # evaluate the solution at a set of points
    test_pt_ct = 40
    x_test_pts_g, y_test_pts_g = np.meshgrid(np.linspace(bdy[0], bdy[1], test_pt_ct), np.linspace(bdy[2], bdy[3], test_pt_ct))
    x_test_pts = x_test_pts_g.reshape(test_pt_ct**2,)
    y_test_pts = y_test_pts_g.reshape(test_pt_ct**2,)
    exact_sol = u(x_test_pts, y_test_pts).reshape((test_pt_ct**2,))

    # evaluate the basis functions and numerical soln at the same set of points
    RBF_vals = rbf_2_mat(delta, x_test_pts, y_test_pts, xcentres.reshape((N*N,)), ycentres.reshape((N*N,)))
    numerical_sol = np.dot(RBF_vals, c)
    rms_err = (exact_sol - numerical_sol)
    rms_err_m = rms_err.reshape((test_pt_ct, test_pt_ct))

    # find the error in the numerical solutions
    print('RMS Error = {:.6E}'.format(np.linalg.norm(exact_sol - numerical_sol, 2) / test_pt_ct))
    print('Maximum Error = {:.6E}'.format(np.linalg.norm(exact_sol - numerical_sol, np.inf)))
    print('Total time taken was {:.3f} seconds'.format(time.time() - start))

    # should move the following to a plotting module
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    # plt.plot(xcentres, ycentres,[0]*N*N,'k*')
    surf = ax.plot_surface(x_test_pts_g, y_test_pts_g, 
        numerical_sol.reshape((test_pt_ct,test_pt_ct)), cmap = cm.coolwarm,
        linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    errsurf = ax.plot_surface(x_test_pts_g, y_test_pts_g, 
        rms_err.reshape((test_pt_ct,test_pt_ct)), cmap = cm.coolwarm,
        linewidth=0, antialiased=False)
    fig.colorbar(errsurf, shrink=0.5, aspect=5)
    plt.savefig('helmholtz_test.eps', format='eps', dpi=1000)
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title('Error: Exact - Numerical')
    errconts = plt.contourf(x_test_pts_g, y_test_pts_g, rms_err_m,15, cmap = cm.BrBG)
    fig.colorbar(errconts, shrink=0.5, aspect=5)
    ax = fig.add_subplot(1,2,1)
    ax.set_title('Numerical Solution')
    conts = plt.contourf(x_test_pts_g, y_test_pts_g, numerical_sol.reshape((test_pt_ct,test_pt_ct)),15,cmap=cm.viridis)
    fig.colorbar(conts, shrink=0.5, aspect=5)
    plt.show()


    return

    
def homogeneous_neumann():
    start = time.time()

    bdy = [-1., 1., -1., 1.]

    # generate the centres -- this can easily be changed for different rectangular domains
    # needs more thought for non-rectangles!
    N = 17
    xcentres, ycentres = np.meshgrid(np.linspace(-1, 1, N),np.linspace(-1, 1, N))
    xcentres = xcentres.reshape(N*N,)
    ycentres = ycentres.reshape(N*N,)

    # forcing term
    f = lambda x, y: np.cos(np.pi * x) * np.cos(np.pi * y)

    # use Gauss-Legendre quadrature to integrate overlapping supports:
    # faster to generate the pts and weights just once and shift/scale as needed
    # could pick a different number of points depending on accuracy needed
    pts, weights = gauleg(30) # number related to no. of basis funcs/support radius
    # rethink number of points as we do multilevel

    # pick a value for delta
    delta = 1.183

    # make the full A
    A_mat, rhs_vec = build_2d_neumann_problem(N, xcentres, ycentres, pts, weights, f, delta)
    A_mat_full = A_mat + A_mat.transpose() - np.diag(np.diag(A_mat))

    # and solve the linear system
    c = np.linalg.solve(A_mat_full, rhs_vec)
    print('Condition number of stiffness matrix is {:.3E}'.format(np.linalg.cond(A_mat_full)))

    # exact solution lambda function
    u = lambda x, y: np.cos(np.pi*x)*np.cos(np.pi*y)/(2*np.pi**2 + 1) # vectorised

    # evaluate the solution at a set of points
    test_pt_ct = 40
    x_test_pts_g, y_test_pts_g = np.meshgrid(np.linspace(bdy[0], bdy[1], test_pt_ct), np.linspace(bdy[2], bdy[3], test_pt_ct))
    x_test_pts = x_test_pts_g.reshape(test_pt_ct**2,)
    y_test_pts = y_test_pts_g.reshape(test_pt_ct**2,)
    exact_sol = u(x_test_pts, y_test_pts).reshape((test_pt_ct**2,))

    # evaluate the basis functions and numerical soln at the same set of points
    RBF_vals = rbf_2_mat(delta, x_test_pts, y_test_pts, xcentres.reshape((N*N,)), ycentres.reshape((N*N,)))
    numerical_sol = np.dot(RBF_vals, c)
    rms_err = (exact_sol - numerical_sol)
    rms_err_m = rms_err.reshape((test_pt_ct, test_pt_ct))

    # find the error in the numerical solutions
    print('RMS Error = {:.6E}'.format(np.linalg.norm(exact_sol - numerical_sol, 2) / test_pt_ct))
    print('Maximum Error = {:.6E}'.format(np.linalg.norm(exact_sol - numerical_sol, np.inf)))
    print('Total time taken was {:.3f} seconds'.format(time.time() - start))

    # should move the following to a plotting module
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    # plt.plot(xcentres, ycentres,[0]*N*N,'k*')
    surf = ax.plot_surface(x_test_pts_g, y_test_pts_g, 
        numerical_sol.reshape((test_pt_ct,test_pt_ct)), cmap = cm.coolwarm,
        linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    errsurf = ax.plot_surface(x_test_pts_g, y_test_pts_g, 
        rms_err.reshape((test_pt_ct,test_pt_ct)), cmap = cm.coolwarm,
        linewidth=0, antialiased=False)
    fig.colorbar(errsurf, shrink=0.5, aspect=5)
    plt.savefig('helmholtz_test.eps', format='eps', dpi=1000)
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title('Error: Exact - Numerical')
    errconts = plt.contourf(x_test_pts_g, y_test_pts_g, rms_err_m,15, cmap = cm.BrBG)
    fig.colorbar(errconts, shrink=0.5, aspect=5)
    ax = fig.add_subplot(1,2,1)
    ax.set_title('Numerical Solution')
    conts = plt.contourf(x_test_pts_g, y_test_pts_g, numerical_sol.reshape((test_pt_ct,test_pt_ct)),15,cmap=cm.viridis)
    fig.colorbar(conts, shrink=0.5, aspect=5)
    plt.show()


    return


if __name__ == '__main__':
    inhomogeneous_dirichlet()



