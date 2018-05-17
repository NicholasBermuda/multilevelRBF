import numpy as np
import time
from rbf import rbf_2_mat
from quadrature import gauleg
from build_mat import build_matrix_problem

def main():
	# first implement Algorithm 2 from (Wendland 1999)
	m = 4 # number of levels
	N = 5 # number of RBFs on the coarsest grid
	delta = 0.7 # RBF support on coarsest grid

	# forcing term
	f = lambda x, y: np.cos(np.pi*x)*np.cos(np.pi*y)

	# quadrature points and weights
	pts, weights = gauleg(100)

	all_N = np.array([N*2**i for i in range(m)]) # all the Ns needed
	 # all the deltas needed -- dunno if this is even a good set of choices but YOLO
	all_delta = np.array([0.7+0.3*i for i in range(m)])

	# to have a numerical soln we'll need to know where to evaluate the solution?
	# for the purposes of nailing down the algorithm, let's just pick something rn
	
	# exact solution lambda function
    u = lambda x, y: np.cos(np.pi*x)*np.cos(np.pi*y)/(2*np.pi**2 + 1) # vectorised

    # evaluate the solution at a set of points
    test_pt_ct = 50
    test_grid = np.linspace(-1, 1, test_pt_ct)
    x_test_pts_g, y_test_pts_g = np.meshgrid(test_grid, test_grid)
    x_test_pts = x_test_pts_g.reshape(test_pt_ct**2,)
    y_test_pts = y_test_pts_g.reshape(test_pt_ct**2,)
    exact_sol = u(x_test_pts, y_test_pts).reshape((test_pt_ct**2,))

    # set v0 = 0
	num_sol = np.zeros(test_pt_ct*test_pt_ct)

	for k in range(m):
		this_N = all_N[k]
		this_delta = all_delta[k]
		
		# find uk in Vk with a(uk, v) = f(v) - a(vk, v) forall v in Vk
		# i.e. solve c_k = A_k\(f_k - A_k-1*c_k-1)
		# still need to find A_k, f_k:
		A_k, rhs_k = build_matrix_problem(this_N, xcentres, ycentres, pts, weights, f, this_delta)
		A_k = A_k + A_k.transpose() - np.diag(np.diag(A_k))
		
		# set vk = vkm1 + uk
		pass


	return

if __name__ == '__main__':
	main()
