import numpy as np

def f(c, x):
    return c[0] * x + c[1] * np.sin(x)

def Jacobian(f, b, x): #In the provided code, b represents a vector of parameters. It is used as an input to the function f to calculate the Jacobian matrix.
    eps = 1e-6
    grads = []
    for i in range(len(b)):
        t = np.zeros_like(b).astype(float) #creates an array of zeros with the same shape and data type as a given input array
        t[i] = t[i] + eps
        grad = (f(b + t, x) - f(b - t, x))/(2*eps)
        grads.append(grad)
    return np.column_stack(grads)

x = np.array([1, 2, 3, 4, 5])  # Input vector
c = np.array([2, 3])  # Parameter vector

Jacobian = Jacobian(f, c, x)
print(Jacobian)

