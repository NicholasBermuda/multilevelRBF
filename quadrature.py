import numpy as np

# find the Gauss-Legendre quadrature points (Kathryn's script)
# possible Cythonisation here? we only need to call it once though
def gauleg(n):
    x = np.zeros(n)
    w = np.zeros(n)
    
    eps = 3e-15
    m = (n + 1.) / 2.
    for i in range(1, int(m+1)):
        z = np.cos(np.pi*(i-0.25)/(n+0.5))
        z1 = z+10*eps
        while(abs(z - z1) > eps):
            p1 = 1.
            p2 = 0.
            for j in range(1, n+1):
                p3 = p2
                p2 = p1
                p1 = ((2*j-1)*z*p2 - (j-1)*p3)/float(j)
            pp = n*(z*p1 - p2)/(z*z - 1)
            z1 = z
            z = z1-p1/pp
        x[i-1] = -z
        x[n-i] = z
        w[i-1] = 2/((1-z**2)*pp**2)
        w[n-i]=w[i-1]
    return (x, w)