from scipy.optimize import least_squares
import numpy as np

def function3(a, b, X):
  return a - (1/b)*X[:,0]**2-X[:,1]**2

f3 = function3

def function4(b,X):
    return b[0] - (1/b[1])*X[:,0]**2-(1/b[2])*X[:,1]**2

f4 = function4

def func(b, X, y):
    return f4(b, X) - y

#generate data from real model
x1 = np.linspace(-5, 5, 50)
x2 = np.linspace(-5, 5, 50)
X1, X2 = np.meshgrid(x1,  x2)
X = np.column_stack([X1.ravel(), X2.ravel()])
y = f3(5,4,X) + np.random.normal(0,1,size=len(X))

res = least_squares(func, np.array([3, 1, 1]), args=(X,y))
print(res.x)
print(res.jac)