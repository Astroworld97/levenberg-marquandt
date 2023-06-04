import numpy as np

def Jacobian(f, b, x): #In the provided code, b represents a vector of parameters. It is used as an input to the function f to calculate the Jacobian matrix.
    eps = 1e-6
    grads = []
    for i in range(len(b)):
        t = np.zeros_like(b).astype(float) #creates an array of zeros with the same shape and data type as a given input array
        t[i] = t[i] + eps
        grad = (f(b + t, x) - f(b - t, x))/(2*eps)
        grads.append(grad)
    return np.column_stack(grads)

def Gauss_Newton(f, x, y, b0, tol, max_iter):
    old = new = b0
    for itr in range(max_iter):
        old = new
        J = Jacobian(f, old, x)
        dy = y - f(old, x)
        new = old + np.linalg.inv(J.T@J)@J.T@dy
        if np.linalg.norm(old-new) < tol:
            break
        return new