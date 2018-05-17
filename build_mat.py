import numpy as np
from forms import a_integrand_2d_mat, F_integrand_2d_mat

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