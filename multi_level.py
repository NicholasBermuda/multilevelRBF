import numpy as np
import time
from rbf import rbf_2_mat
from quadrature import gauleg
from build_mat import build_2d_neumann_problem, non_square_2d_neumann_matrix

def homogeneous_neumann():
    start = time.time()
    # first implement Algorithm 2 from (Wendland 1999)
    m = 3 # number of levels
    N = 5 # number of RBFs on the coarsest grid
    delta = 0.7 # RBF support on coarsest grid

    bdy = [-1.0, 1.0, -1.0, 1.0]

    # forcing term
    f = lambda x, y: np.cos(np.pi*x)*np.cos(np.pi*y)

    # quadrature points and weights
    pts, weights = gauleg(30)

    all_N = np.array([2**(i+2) + 1 for i in range(m)]) # all the Ns needed
     # all the deltas needed -- dunno if this is even a good set of choices but YOLO
    all_delta = np.array([delta*1.3**(i) for i in range(m)])

    # to have a numerical soln we'll need to know where to evaluate the solution?
    # for the purposes of nailing down the algorithm, let's just pick something rn
    
    # exact solution lambda function
    u = lambda x, y: np.cos(np.pi*x)*np.cos(np.pi*y)/(2*np.pi**2 + 1) # vectorised

    # evaluate the solution at a set of points
    test_pt_ct = 50
    test_grid = np.linspace(-1, 1, test_pt_ct)
    x_test_pts_g, y_test_pts_g = np.meshgrid(test_grid, test_grid)
    x_test_pts = x_test_pts_g.reshape(test_pt_ct**2, )
    y_test_pts = y_test_pts_g.reshape(test_pt_ct**2, )
    exact_sol = u(x_test_pts, y_test_pts).reshape((test_pt_ct**2, ))

    # set v_0 = 0, storage for v_{k-1}
    num_sol = np.zeros(test_pt_ct * test_pt_ct)
    # old_num_sol = np.zeros(test_pt_ct  *test_pt_ct)

    # A_km1 = 0
    c_km1 = 0

    for k in range(m):
        print('Currently on level {}'.format(k))
        this_N = all_N[k]
        this_delta = all_delta[k]

        # generate the rbf centres for this level
        xcentres, ycentres = np.meshgrid(np.linspace(bdy[0], bdy[1], this_N), np.linspace(bdy[2], bdy[3], this_N))
        xcentres = xcentres.reshape(this_N * this_N, )
        ycentres = ycentres.reshape(this_N * this_N, )
        
        # find uk in Vk with a(uk, v) = f(v) - a(v_{k-1}, v) forall v in Vk
        # i.e. solve c_k = A_k\(f_k - A_k-1*c_k-1)
        A_k, rhs_k = build_2d_neumann_problem(this_N, xcentres, ycentres, pts, weights, f, this_delta)
        A_k = A_k + A_k.transpose() - np.diag(np.diag(A_k))
        if k >= 1:
            old_rect_A = non_square_2d_neumann_matrix(old_xcentres, old_ycentres, xcentres, ycentres, pts, weights, this_delta)
        else:
            old_rect_A = 0
        c_k = np.linalg.solve(A_k, rhs_k - np.dot(old_rect_A, c_km1))

        # now generate the numerical solution at the desired points
        RBF_vals_k = rbf_2_mat(delta, x_test_pts, y_test_pts, xcentres.reshape((this_N * this_N, )), ycentres.reshape((this_N * this_N, )))
        u_k = np.dot(RBF_vals_k, c_k)

        # set vk = vkm1 + uk, update
        num_sol = num_sol + u_k

        c_km1 = c_k
        old_xcentres = xcentres
        old_ycentres = ycentres


        pass

    # RBF_vals = rbf_2_mat(delta, x_test_pts, y_test_pts, xcentres.reshape((N * N, )), ycentres.reshape((N * N, )))
    # num_sol = np.dot(RBF_vals, c_k)

    rms_err = (exact_sol-num_sol)
    rms_err_m = rms_err.reshape((test_pt_ct,test_pt_ct))

    # find the error in the numerical solutions
    print('RMS Error = {:.6E}'.format(np.linalg.norm(exact_sol-num_sol, 2)/test_pt_ct))
    print('Maximum Error = {:.6E}'.format(np.linalg.norm(exact_sol - num_sol, np.inf)))
    print('Total time taken was {:.3f} seconds'.format(time.time()-start))

    return

if __name__ == '__main__':
    homogeneous_neumann()
