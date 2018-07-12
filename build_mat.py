import numpy as np
from forms import point_eval_1d, gradpoint_eval_1d
from forms import a_integrand_2d_mat, F_integrand_2d_mat, F_integrand_1d_mat, a_integrand_1d_mat


def build_2d_dirichlet_problem(N, xcentres, ycentres, pts, weights, f, delta, bdy = (-1.0, 1.0, -1.0, 1.0), bc = (0.0, 0.0, 0.0, 0.0)):
    N2 = N * N
    rhs_vec = np.zeros(N2)
    A_mat = np.zeros((N2,N2))
    theta = -1
    sigma = 100
    for i in range(N2):
        a1 = max(bdy[0], xcentres[i] - 1.0/delta)
        b1 = min(bdy[1], xcentres[i] + 1.0/delta)
        a2 = max(bdy[2], ycentres[i] - 1.0/delta)
        b2 = min(bdy[3], ycentres[i] + 1.0/delta)
        scale_2 = ((b2 - a2)/2)
        scale_1 = ((b1 - a1)/2)
        rhs_vec[i] = scale_1 * scale_2 * np.einsum('i,i',weights, 
                    np.einsum('ij,j->i', F_integrand_2d_mat(scale_2*pts + (a2 + b2)/2, scale_1*pts + (a1+b1)/2, ycentres[i],
                                                         xcentres[i], f, delta), weights))
        # # now add the RHS terms of the bcs
        # rhs_vec[i] += bc[0] * 
        # rhs_vec[i] += bc[1] * 
        # rhs_vec[i] += bc[2] * 
        # rhs_vec[i] += bc[3] * 
        for j in range(i+1):
            if np.sqrt((xcentres[i] - xcentres[j])**2 + (ycentres[i] - ycentres[j])**2) > 2./delta:
                A_mat[i,j] = 0
            else:
                A1 = max(bdy[0], max(xcentres[i], xcentres[j]) - 1./delta)
                B1 = min(bdy[1], min(xcentres[i], xcentres[j]) + 1./delta)
                A2 = max(bdy[2], max(ycentres[i], ycentres[j]) - 1./delta)
                B2 = min(bdy[3], min(ycentres[i], ycentres[j]) + 1./delta)
                Scale_m2 = ((B2 - A2)/2)
                Scale_m1 = ((B1 - A1)/2)
                # first the standard bilinear form
                bilin_mat = a_integrand_2d_mat(Scale_m2 * pts + (A2 + B2)/2, Scale_m1 * pts + (A1 + B1)/2, ycentres[i], 
                                    xcentres[i], ycentres[j], xcentres[j], delta)
                A_mat[i, j] = Scale_m1 * Scale_m2 * np.einsum('i,i', weights, np.einsum('ij,j->i', bilin_mat, weights))
                # # now the boundary term and the penalty terms
                # # decompose the boundary into 4 straight lines, then do 1d integrals along the boundaries
                # # right bdy: 1 x [-1, 1], \hat{n} = \hat{x}: dot(grad(u), n) = du/dx
                # A_mat[i, j] += boundary_int(v, du/dx) + theta * boundary_int(u, dv/dx)
                # # left bdy: -1 x [-1, 1], \hat{n} = -\hat{x}: dot(grad(u), n) = -du/dx
                # A_mat[i, j] += boundary_int(v, -du/dx) + theta * boundary_int(u, -dv/dx)
                # # top bdy: [-1, 1] x 1, \hat{n} = \hat{y}: dot(grad(u), n) = du/dy
                # A_mat[i, j] += boundary_int(v, du/dy) + theta * boundary_int(u, dv/dy)
                # # bottom bdy: [-1, 1] x -1, \hat{n} = -\hat{y}: dot(grad(u), n) = -du/dy
                # A_mat[i, j] += boundary_int(v, -du/dy) + theta * boundary_int(u, -dv/dy)

    return A_mat, rhs_vec


def build_2d_neumann_problem(N, xcentres, ycentres, pts, weights, f, delta, bdy = (-1.0, 1.0, -1.0, 1.0)):
    N2 = N * N
    rhs_vec = np.zeros(N2)
    A_mat = np.zeros((N2,N2))
    for i in range(N2):
        a1 = max(bdy[0], xcentres[i] - 1.0/delta)
        b1 = min(bdy[1], xcentres[i] + 1.0/delta)
        a2 = max(bdy[2], ycentres[i] - 1.0/delta)
        b2 = min(bdy[3], ycentres[i] + 1.0/delta)
        scale_2 = ((b2 - a2)/2)
        scale_1 = ((b1 - a1)/2)
        rhs_vec[i] = scale_1 * scale_2 * np.einsum('i,i',weights, 
                    np.einsum('ij,j->i', F_integrand_2d_mat(scale_2*pts + (a2 + b2)/2, scale_1*pts + (a1+b1)/2, ycentres[i],
                                                         xcentres[i], f, delta), weights))
        for j in range(i+1):
            if np.sqrt((xcentres[i] - xcentres[j])**2 + (ycentres[i] - ycentres[j])**2) > 2./delta:
                A_mat[i,j] = 0
            else:
                A1 = max(bdy[0], max(xcentres[i], xcentres[j]) - 1./delta)
                B1 = min(bdy[1], min(xcentres[i], xcentres[j]) + 1./delta)
                A2 = max(bdy[2], max(ycentres[i], ycentres[j]) - 1./delta)
                B2 = min(bdy[3], min(ycentres[i], ycentres[j]) + 1./delta)
                Scale_m2 = ((B2 - A2)/2)
                Scale_m1 = ((B1 - A1)/2)
                bilin_mat = a_integrand_2d_mat(Scale_m2 * pts + (A2 + B2)/2, Scale_m1 * pts + (A1 + B1)/2, ycentres[i], 
                                    xcentres[i], ycentres[j], xcentres[j], delta)
                A_mat[i, j] = Scale_m1 * Scale_m2 * np.einsum('i,i', weights, np.einsum('ij,j->i', bilin_mat, weights))

    return A_mat, rhs_vec


def build_1d_neumann_problem(N, centres, pts, weights, f, delta, bdy = (-1.0, 1.0)):
    rhs_vec = np.zeros(N)
    A_mat = np.zeros((N,N))
    for i in range(N): # i is for v discretisation
        a = max(bdy[0], centres[i] - 1.0/delta)
        b = min(bdy[1], centres[i] + 1.0/delta)
        scale = (b - a)/2
        rhs_vec[i] = scale * np.einsum('i,i', F_integrand_1d_mat(scale*pts + (a + b)/2, centres[i], f, delta), weights)
        for j in range(i + 1): # j is for u discretisation
            if abs(centres[i] - centres[j]) > 2./delta:
                A_mat[i, j] = 0
            else:
                A = max(bdy[0], max(centres[i], centres[j]) - 1./delta)
                B = min(bdy[1], min(centres[i], centres[j]) + 1./delta)
                SCALE = (B - A)/2
                bilin_mat = a_integrand_1d_mat(SCALE*pts + (A + B)/2, centres[i], centres[j], delta)
                A_mat[i, j] = SCALE * np.einsum('i,i', bilin_mat, weights)

    return A_mat, rhs_vec



def build_1d_dirichlet_problem(N, centres, pts, weights, f, delta, bdy = (-1.0, 1.0), bc = (0, 0)):
    # homogeneous b.c. by default, use the symmetric bilinear form and sigma to try to force coercivity
    theta = -1
    sigma = 50
    rhs_vec = np.zeros(N)
    A_mat = np.zeros((N,N))
    for i in range(N): # for v discretisation
        a = max(bdy[0], centres[i] - 1.0/delta)
        b = min(bdy[1], centres[i] + 1.0/delta)
        scale = (b - a)/2
        # rhs unchanged for homogeneous b.c.
        rhs_vec[i] = scale * np.einsum('i,i', F_integrand_1d_mat(scale*pts + (a + b)/2, centres[i], f, delta), weights)
        rhs_vec[i] += bc[1] * (theta * gradpoint_eval_1d(bdy[1], centres[i], delta) + sigma * point_eval_1d(bdy[1], centres[i], delta))
        rhs_vec[i] += bc[0] * (-theta * gradpoint_eval_1d(bdy[0], centres[i], delta) + sigma * point_eval_1d(bdy[0], centres[i], delta))
        for j in range(i + 1): # for u discretisation
            if abs(centres[i] - centres[j]) > 2./delta:
                A_mat[i, j] = 0
            else:
                # first the bilinear form part
                A = max(bdy[0], max(centres[i], centres[j]) - 1./delta)
                B = min(bdy[1], min(centres[i], centres[j]) + 1./delta)
                SCALE = (B - A)/2
                bilin_mat = a_integrand_1d_mat(SCALE * pts + (A + B)/2, centres[i], centres[j], delta)
                A_mat[i, j] = SCALE * np.einsum('i,i', bilin_mat, weights)
                # now the boundary terms and theta/sigma part
                # - (u' v)(1) - theta*(u v')(1)
                A_mat[i, j] -= gradpoint_eval_1d(bdy[1], centres[j], delta)  * point_eval_1d(bdy[1], centres[i], delta)
                A_mat[i, j] += theta * gradpoint_eval_1d(bdy[1], centres[i], delta)  * point_eval_1d(bdy[1], centres[j], delta)
                # + (u' v)(-1) + theta*(u v')(-1)
                A_mat[i, j] += gradpoint_eval_1d(bdy[0], centres[j], delta)  * point_eval_1d(bdy[0], centres[i], delta)
                A_mat[i, j] -= theta * gradpoint_eval_1d(bdy[0], centres[i], delta)  * point_eval_1d(bdy[0], centres[j], delta)
                # + (sigma u v)(1) - (sigma u v)(-1)
                A_mat[i, j] += sigma * point_eval_1d(bdy[1], centres[j], delta)  * point_eval_1d(bdy[1], centres[i], delta)
                A_mat[i, j] += sigma * point_eval_1d(bdy[0], centres[i], delta)  * point_eval_1d(bdy[0], centres[j], delta)

    return A_mat, rhs_vec


def non_square_2d_neumann_matrix(xcentres1, ycentres1, xcentres2, ycentres2, pts, weights, delta):
    N1 = len(xcentres1)
    N2 = len(xcentres2)
    A_mat = np.zeros((N2, N1))

    for i in range(N2):
        for j in range(N1):
            if (np.sqrt((xcentres2[i] - xcentres1[j])**2 + (ycentres2[i] - ycentres1[j])**2) > 2./delta):
                A_mat[i,j] = 0
            else:
                A1 = max(-1, max(xcentres2[i], xcentres1[j]) - 1./delta)
                B1 = min(1, min(xcentres2[i], xcentres1[j]) + 1./delta)
                A2 = max(-1, max(ycentres2[i], ycentres1[j]) - 1./delta)
                B2 = min(1, min(ycentres2[i], ycentres1[j]) + 1./delta)

                Scale_m2 = ((B2 - A2)/2)
                Scale_m1 = ((B1 - A1)/2)
                bilin_mat = a_integrand_2d_mat(Scale_m2 * pts + (A2 + B2)/2, Scale_m1 * pts + (A1 + B1)/2, ycentres2[i], 
                                    xcentres2[i], ycentres1[j], xcentres1[j], delta)
                A_mat[i,j] = Scale_m1 * Scale_m2 * np.einsum('i,i', weights, np.einsum('ij,j->i', bilin_mat, weights))
    return A_mat
