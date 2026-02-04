# numerical_methods_repository/eigenvalue_methods.py
import numpy as np

def power_method(A, num_iterations=100, tolerance=1e-6):
    """
    Finds the dominant eigenvalue and its corresponding eigenvector
    of a square matrix A using the Power Method.
    ... (diğer kodunuz) ...
    """
    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError("Matrix A must be square.")

    x = np.random.rand(n)
    x = x / np.linalg.norm(x)

    eigenvalue = 0.0

    for i in range(num_iterations):
        x_new = np.dot(A, x)

        if np.linalg.norm(x_new) == 0:
            raise RuntimeError("Power Method failed: Resulting vector is zero. Dominant eigenvalue might be zero.")

        eigenvalue_new = np.dot(x, x_new) / np.dot(x, x)

        x_new_normalized = x_new / np.linalg.norm(x_new)

        # Check for convergence (both eigenvalue and eigenvector)
        # Check eigenvalue convergence
        if abs(eigenvalue_new - eigenvalue) < tolerance:
            # Additionally, check eigenvector convergence for robustness
            if np.linalg.norm(x_new_normalized - x) < tolerance or np.linalg.norm(x_new_normalized + x) < tolerance:
                return eigenvalue_new, x_new_normalized

        eigenvalue = eigenvalue_new
        x = x_new_normalized

    raise RuntimeError("Power Method did not converge within the maximum number of iterations.")