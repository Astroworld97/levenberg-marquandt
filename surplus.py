# import numpy as np
# from numpy import inf
# import scipy
# from scipy import linalg

# #r = residual vector r(x)
# #J = Jacobian of r(x)
# #x = float; starting point of the iteration process
# def levenberg_marquardt (r, J, x, Delta = 100, Delta_max = 10000, eta = 0.0001, sigma = 0.1, nmax = 500, tol_abs = 10∗∗(−7), tol_rel= 10∗∗(−7), eps = 10∗∗(−3), Scaling = False ):

#     def f(x):
#         return 0.5*np.linalg.norm(r(x))**2
#     def gradf(x):
#         return np.dot(np.transpose(J(x)), r(x))

# for itr in range(max_iter):
#         old = new
#         J = Jacobian(f, old, x)
#         dy = y - f(old, x)
#         matrix = J.T@J
#         D = create_D(matrix)
#         new = old + np.linalg.inv(J.T@J + lamb*D)@J.T@dy
#         if np.linalg.norm(old-new) < tol:
#             break
# return new