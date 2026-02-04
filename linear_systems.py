import numpy as np

def gaussian_elimination(A, b):
    """
    Solves a system of linear equations Ax = b using Gaussian Elimination.

    Args:
        A (np.array): The coefficient matrix (n x n).
        b (np.array): The right-hand side vector (n x 1).

    Returns:
        np.array: The solution vector x (n x 1).

    Raises:
        ValueError: If A is not a square matrix, or if dimensions of A and b do not match.
        ValueError: If the matrix is singular (no unique solution).
    """
    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError("Coefficient matrix A must be square.")
    if b.shape[0] != n:
        raise ValueError("Dimension mismatch between matrix A and vector b.")

    # Create an augmented matrix [A | b]
    M = np.concatenate((A, b.reshape(-1, 1)), axis=1).astype(float)

    # Forward Elimination
    for i in range(n):
        # Find pivot: Find the row with the largest absolute value in the current column
        # to improve numerical stability (partial pivoting)
        max_row = i
        for k in range(i + 1, n):
            if abs(M[k, i]) > abs(M[max_row, i]):
                max_row = k
        M[[i, max_row]] = M[[max_row, i]] # Swap rows

        # Check for singular matrix
        if M[i, i] == 0:
            raise ValueError("Matrix is singular or ill-conditioned. No unique solution exists.")

        # Make the diagonal element 1 and eliminate other entries below
        for k in range(i + 1, n):
            factor = M[k, i] / M[i, i]
            M[k, i:] = M[k, i:] - factor * M[i, i:]

    # Back Substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (M[i, -1] - np.dot(M[i, i+1:n], x[i+1:n])) / M[i, i]

    return x

# NumPy's linalg.solve will be used directly in the example file for comparison.