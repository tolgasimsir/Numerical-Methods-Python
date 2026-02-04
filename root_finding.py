# numerical_methods_repository/root_finding.py

import numpy as np

def bisection_method(func, a, b, tol=1e-6, max_iter=100):
    """
    Finds the root of a function using the Bisection Method.
    Returns the estimated root and a list of (c, f(c)) values for plotting.

    Args:
        func (callable): The function f(x) for which to find the root.
                         It should return a single float value.
        a (float): The lower bound of the interval.
        b (float): The upper bound of the interval.
        tol (float, optional): The tolerance (stopping criterion). Defaults to 1e-6.
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.

    Returns:
        tuple: (float, list): The estimated root of the function and a list of
                               tuples (c_i, f(c_i)) for each iteration.

    Raises:
        ValueError: If the function has the same sign at the endpoints a and b,
                    indicating that a root may not exist in the interval or
                    the method may not converge.
        ValueError: If the method does not converge within max_iter.
    """
    if func(a) * func(b) >= 0:
        raise ValueError("Function has same signs at endpoints a and b. Bisection method may not converge.")

    c = a # Initialize c
    iterations_data = [] # Store (c_i, f(c_i)) for plotting

    for i in range(max_iter):
        c = (a + b) / 2.0
        f_c = func(c) # Calculate f(c) once

        iterations_data.append((c, f_c)) # Store iteration data

        if f_c == 0 or abs(b - a) / 2.0 < tol:
            return float(c), iterations_data # Return root and all iteration data

        if f_c * func(a) < 0:
            b = c
        else:
            a = c
    
    # If max_iter is reached, raise an error indicating non-convergence
    # The error message should describe the state but the function should not return data in this case.
    # The calling function (app.py) should handle the non-convergence.
    raise ValueError(f"Bisection method did not converge within {max_iter} iterations. Last approximation: {c:.6f}")


def newton_raphson_method(func, deriv, initial_guess, tol=1e-6, max_iter=100):
    """
    Finds the root of a function using the Newton-Raphson Method.
    Returns the estimated root and a list of (x_k, f(x_k)) values for plotting.

    Args:
        func (callable): The function f(x) for which to find the root.
        deriv (callable): The derivative of the function f'(x).
        initial_guess (float): The initial guess for the root.
        tol (float, optional): The tolerance (stopping criterion). Defaults to 1e-6.
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.

    Returns:
        tuple: (float, list): The estimated root of the function and a list of
                               tuples (x_k, f(x_k)) for each iteration.

    Raises:
        ValueError: If the derivative is zero or very close to zero at any iteration,
                    or if the method does not converge within max_iter.
    """
    x = float(initial_guess)
    iterations_data = [(x, func(x))] # Store initial guess data

    for i in range(max_iter):
        f_x = func(x)
        f_prime_x = deriv(x)

        if abs(f_prime_x) < 1e-12:
            raise ValueError(f"Derivative is zero or too close to zero at x = {x:.6f}. Newton-Raphson method cannot proceed.")

        x_new = x - f_x / f_prime_x
        
        # Store new iteration data before checking convergence
        iterations_data.append((x_new, func(x_new)))

        if abs(x_new - x) < tol:
            return float(x_new), iterations_data # Return root and all iteration data

        x = x_new

    # If max_iter is reached, raise an error indicating non-convergence
    # The error message should describe the state but the function should not return data in this case.
    raise ValueError(f"Newton-Raphson method did not converge within {max_iter} iterations. Last approximation: {x:.6f}")