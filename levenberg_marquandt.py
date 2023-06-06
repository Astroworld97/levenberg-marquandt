import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

def function3(a, b, X):
  return a - (1/b)*X[:,0]**2-X[:,1]**2

f3 = function3

#generate data from real model
x1 = np.linspace(-5, 5, 50)
x2 = np.linspace(-5, 5, 50)
X1, X2 = np.meshgrid(x1,  x2)
X = np.column_stack([X1.ravel(), X2.ravel()])
y = f3(5,4,X) + np.random.normal(0,1,size=len(X))

def create_D(matrix):
    # Get the diagonal by indexing with a single index
    diagonal = matrix.diagonal()
    # Create an all-zero matrix of the same shape
    new_matrix = np.zeros_like(matrix)
    # Set the diagonal of the new matrix
    np.fill_diagonal(new_matrix, diagonal)
    return new_matrix

def Jacobian(f, c, x): #In the provided code, b represents a vector of parameters. It is used as an input to the function f to calculate the Jacobian matrix.
    eps = 1e-6
    grads = []
    for i in range(len(c)):
        t = np.zeros_like(c).astype(float) #creates an array of zeros with the same shape and data type as a given input array
        t[i] = t[i] + eps
        grad = (f(c + t, x) - f(c - t, x))/(2*eps)
        grads.append(grad)
    return np.column_stack(grads)

def lm(f, x, y, c0, tol, max_iter):#NOTE: f is the function to input in the Jacobian. y is f(x).
    nu = 10 #"The choice of is arbitrary; 10 has been found in practice to be a good choice." -Marquardt
    lamb = 10**(-2)
    old = new = c0 #c's
    om = float('inf')
    for itr in range(max_iter):
        old = new
        lamb_a = lamb/nu
        lamb_b = lamb
        J = Jacobian(f, old, x)
        dy = y - f(old, x)
        matrix = J.T@J
        D = create_D(matrix)
        delta_m_1 = np.linalg.inv(J.T@J + lamb_a*D)@J.T@dy
        delta_m_2 = np.linalg.inv(J.T@J + lamb_b*D)@J.T@dy
        om_1_left = dy-J@delta_m_1
        om_1_right = dy-J@delta_m_1
        om_1 = om_1_left.T@om_1_right
        om_2_left = dy-J@delta_m_2
        om_2_right = dy-J@delta_m_2
        om_2 = om_2_left.T@om_2_right
        if om_1 <= om:
            lamb = lamb_a
            om = om_1
        elif om_1 > om and om_2 <= om:
            lamb = lamb_b
            om = om_2
        elif om_1 > om and om_2 > om:
            delta_curr = np.linalg.inv(J.T@J + lamb*D)@J.T@dy
            om_curr_left = dy-J@delta_curr
            om_curr_right = dy-J@delta_curr
            om_curr = om_curr_left.T@om_curr_right
            whileTaken = False
            while round(om,2) < round(om_curr,2):
                whileTaken = True
                lamb = lamb*nu
                delta_curr = np.linalg.inv(J.T@J + lamb*D)@J.T@dy
                om_curr_left = dy-J@delta_curr
                om_curr_right = dy-J@delta_curr
                om_curr = om_curr_left.T@om_curr_right
            if whileTaken:
                om = om_curr
        new = old + np.linalg.inv(J.T@J + lamb*D)@J.T@dy
        print(itr)
        print(new)
        print(lamb)
    return new

def function4(b,X):
    return b[0] - (1/b[1])*X[:,0]**2-(1/b[2])*X[:,1]**2

f4 = function4

bs = lm(f4, X, y, np.array([3,1,1]), 1e-5, 10)
print(bs)