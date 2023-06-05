import numpy as np

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
    old = new = c0
    for itr in range(max_iter):
        old = new
        J = Jacobian(f, old, x)
        dy = y - f(old, x)
        matrix = J.T@J
        D = create_D(matrix)
        new = old + np.linalg.inv(J.T@J + lamb*D)@J.T@dy
        if np.linalg.norm(old-new) < tol:
            break
        return new