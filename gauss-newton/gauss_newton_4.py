#uses more general jacobian and Gauss-Newton, where both use a vector of parameters instead of just two
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

def Jacobian(f, b, x): #In the provided code, b represents a vector of parameters. It is used as an input to the function f to calculate the Jacobian matrix.
    eps = 1e-6
    grads = []
    for i in range(len(b)):
        t = np.zeros_like(b).astype(float) #creates an array of zeros with the same shape and data type as a given input array
        t[i] = t[i] + eps
        grad = (f(b + t, x) - f(b - t, x))/(2*eps)
        grads.append(grad)
    return np.column_stack(grads)

def Gauss_Newton(f, x, y, b0, tol, max_iter): #note that b0 is the initial guess for b; aka, the vector of coefficients, called c in Yan-Bin's notes
    old = new = b0
    for itr in range(max_iter):
        old = new
        J = Jacobian(f, old, x)
        dy = y - f(old, x)
        new = old + np.linalg.inv(J.T@J)@J.T@dy
        if np.linalg.norm(old-new) < tol:
            break
    return new
    
def function4(b,X):
    return b[0] - (1/b[1])*X[:,0]**2-(1/b[2])*X[:,1]**2

f4 = function4

bs = Gauss_Newton(f4, X, y, np.array([3,1,1]), 1e-5, 10)
print(bs)