# multilevelRBF
A Python/Cython implementation of the multilevel RBF Galerkin Method by [(Wendland 1998)](https://www.researchgate.net/publication/2296448_Numerical_Solution_of_Variational_Problems_by_Radial_Basis_Functions), which we try to extend to a broader range of problems in 1D and 2D and to Dirichlet problems.

### Please note that this code is a work in progress and may change dramatically day-to-day!

All code is mine unless otherwise noted

Files included:
--------------------
* `quadrature.py` - finds Gauss-Legendre quadrature points and weights for numerical integration. Originally written by Kathryn Gillow in MATLAB.
* `rbf.pyx` - Cython code to evaluate the RBFs in 1D and 2D
* `rbf.pxd` - Cython header file for RBFs
* `forms.pyx` - Cython code to build matrices from linear/bilinear forms
* `single_level.py` - Python implementation of Ch. 45 of [(Fasshauser 2009)](https://uk.mathworks.com/support/books/book48894.html).
* `1D_single_level.py` - 1D modified Helmholtz problem on the interval [-1, 1]. working homogeneous Neumann b.c., homogeneous Dirichlet in progress!
* `setup.py` - used to Cythonise things
* `build_mat.py` - assembles the matrix problem
* `multi_level.py` - multilevel method for the (2D) modified Helmholtz problem on the unit square with homogeneous Neumann conditions

Installation and Use
--------------------
The current instructions will allow you to run, for example, `single_level.py` with Cythonized RBF and form evaluation. This will eventually become a more well-structured module :)

	git clone https://github.com/NicholasBermuda/multilevelRBF.git
	cd multilevelRBF
	python setup.py build_ext --inplace
	python single_level.py

To-Do and Wish List
---------------------
There are a number of components left to build for this project. The following is a scratch list for features and ideas that will (definitely maybe) be included in the future.

* plotting module
* Dirichlet b.c.
* alter arguments/structure of `build_matrix_problem` for parallelisation?
* an abstract class for the problem/solution?? could have
	* attributes like: RHS, numerical soln, no. RBFs on coarsest level, no. levels etc.
	* methods like: solve, plot etc.
* further speed boosts
	* parallelisation of construction of stiffness matrix
	* can we refactor RBF code so that it only takes r = sqrt(x*x + y*y) to facilitate memoisation?
* irregular domains
	* characteristic function?
* PDE agnosticity - let the user specify the forms and/or give a variety of form to build from