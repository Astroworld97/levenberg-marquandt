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

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], y, c = y, marker='o')
# plt.show()

def Jacobian(f, a, b, x):
  eps = 1e-6
  grad_a = (f(a+eps, b, x) - f(a - eps, b, x))/(2*eps)
  grad_b = (f(a, b+eps, x) - f(a, b - eps, x))/(2*eps)
  return np.column_stack([grad_a, grad_b])

def Gauss_Newton(f, x, y, a0, b0, tol, max_iter):
  old = new = np.array([a0, b0])
  for itr in range(max_iter):
    old = new
    J = Jacobian(f, old[0], old[1], x)
    dy = y - f(old[0], old[1], x)
    new = old + np.linalg.inv(J.T@J)@J.T@dy
    if np.linalg.norm(old-new) < tol:
      break
  return new

a,b = Gauss_Newton(f3, X, y, 3, 1, 1e-5, 10)
y_hat = f3(a, b, X)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X[:,0].reshape(50,50), X[:,1].reshape(50,50), y_hat.reshape(50,50), cmap=cm.coolwarm)
ax.scatter(X[:,0], X[:,1], y, c = y, marker='o')
plt.show()