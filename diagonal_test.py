import numpy as np

# Create a matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Get the diagonal using np.diagonal()
diagonal = np.diagonal(matrix)
print(diagonal)  # Output: [1 5 9]

# Get the diagonal by indexing with a single index
diagonal = matrix.diagonal()
print(diagonal)  # Output: [1 5 9]

# Create an all-zero matrix of the same shape
new_matrix = np.zeros_like(matrix)

# Set the diagonal of the new matrix
np.fill_diagonal(new_matrix, diagonal)

print(new_matrix)

