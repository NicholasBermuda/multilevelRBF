# multilevelRBF
Implementation of my MMSC dissertation research. An implementation of the multilevel RBF Galerkin Method by [(Wendlund 1998)](https://www.researchgate.net/publication/2296448_Numerical_Solution_of_Variational_Problems_by_Radial_Basis_Functions), which we try to extend to a broader range of problems in 1D and 2D and to Dirichlet problems.

### Please note that this code is a work in progress and may change dramatically day-to-day!

All code is mine unless otherwise noted

Files included:
--------------------
* `quadrature.py` - finds Gauss-Legendre quadrature points and weights for numerical integration. Originally written by Kathryn Gillow in MATLAB.
* `rbf.pyx` - Cython code to evaluate the RBFs
* `rbf.pxd` - Cython header file for RBFs
* `forms.pyx` - Cython code to build matrices from linear/bilinear forms
* `single_level.py` - Python implementation of Ch. 45 of [(Fasshauser 2009)](https://uk.mathworks.com/support/books/book48894.html).
* `setup.py` - used to Cythonise things

Installation and Use
--------------------
The current instructions will allow you to run `single_level.py` with Cythonized RBF and form evaluation. This will eventually become a more well-structured module :)

	git clone https://github.com/NicholasBermuda/multilevelRBF.git
	cd multilevelRBF
	python setup.py build_ext --inplace
	python single_level.py

