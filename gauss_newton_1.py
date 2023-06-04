import numpy as np
import matplotlib.pyplot as plt

#generate function

def function1(a, b, x):
  return a*x/(b+x)

f1 = function1

#generate data from real model y = 2x/(3+x)

x = np.linspace(0, 5, 50)
y = f1(2,3,x) + np.random.normal(0,0.1,size=50)

plt.scatter(x,y)

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

a, b = Gauss_Newton(f1, x, y, 5, 1, 1e-5, 10)

y_hat = f1(a,b,x)

plt.scatter(x, y)
plt.plot(x, y_hat)
plt.show()