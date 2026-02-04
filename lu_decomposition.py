import numpy as np
# SciPy's LU decomposition functions will be used directly in the example file for comparison.

def lu_decompose(A):
    """
    Performs LU decomposition of a matrix A into L (lower triangular) and U (upper triangular)
    matrices using Doolittle's method (assuming no pivoting for simplicity, or implicit pivoting).

    Args:
        A (np.array): The square matrix to decompose.

    Returns:
        tuple: A tuple containing:
            - L (np.array): The lower triangular matrix.
            - U (np.array): The upper triangular matrix.

    Raises:
        ValueError: If A is not a square matrix.
        ValueError: If a zero pivot is encountered (matrix is singular or requires pivoting).
    """
    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError("Matrix A must be square for LU decomposition.")

    L = np.eye(n)  # Initialize L as an identity matrix
    U = A.astype(float).copy() # Initialize U as a copy of A (ensure float type)

    for i in range(n):
        if U[i, i] == 0:
            raise ValueError(f"Zero pivot encountered at row {i}, column {i}. Pivoting may be required or matrix is singular.")

        for j in range(i + 1, n):
            factor = U[j, i] / U[i, i]
            L[j, i] = factor
            U[j, i:] = U[j, i:] - factor * U[i, i:]

    return L, U

def lu_solve_forward_substitution(L, b):
    """
    Solves Ly = b for y using forward substitution.

    Args:
        L (np.array): The lower triangular matrix.
        b (np.array): The right-hand side vector.

    Returns:
        np.array: The solution vector y.
    """
    n = L.shape[0]
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    return y

def lu_solve_backward_substitution(U, y):
    """
    Solves Ux = y for x using backward substitution.

    Args:
        U (np.array): The upper triangular matrix.
        y (np.array): The intermediate vector from forward substitution.

    Returns:
        np.array: The solution vector x.
    """
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x

def lu_solve(L, U, b):
    """
    Solves Ax = b using the LU decomposed matrices L and U.

    Args:
        L (np.array): The lower triangular matrix from LU decomposition.
        U (np.array): The upper triangular matrix from LU decomposition.
        b (np.array): The right-hand side vector.

    Returns:
        np.array: The solution vector x.
    """
    y = lu_solve_forward_substitution(L, b)
    x = lu_solve_backward_substitution(U, y)
    return x