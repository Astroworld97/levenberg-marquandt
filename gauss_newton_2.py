import numpy as np
import matplotlib.pyplot as plt

def function2(a, b, x):
  return (1/np.sqrt(2*np.pi*b))*np.exp(-0.5*(1/b)*(x-a)**2)

f2 = function2

x = np.linspace(0, 5, 50)
y = f2(2.5, 0.5, x) + np.random.normal(0, 0.01, size=50)

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

a,b = Gauss_Newton(f2, x, y, 3, 1, 1e-5, 10)
y_hat = f2(a,b,x)

plt.scatter(x,y)
plt.plot(x, y_hat)
plt.show()