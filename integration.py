import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, trapezoid # BURAYI DÜZELTTİK: trapz -> trapezoid

def trapezoidal_rule(func, a, b, n):
    """
    Approximates the definite integral of a function using the Trapezoidal Rule.

    Args:
        func (function): The function to integrate.
        a (float): The lower limit of integration.
        b (float): The upper limit of integration.
        n (int): The number of subintervals (trapezoids).

    Returns:
        float: The approximated definite integral.
    """
    if n <= 0:
        raise ValueError("Number of subintervals (n) must be positive.")
    
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = func(x)
    
    # Sum of the areas of the trapezoids
    integral = (h / 2) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])
    return integral

def simpsons_rule(func, a, b, n):
    """
    Approximates the definite integral of a function using Simpson's Rule.

    Args:
        func (function): The function to integrate.
        a (float): The lower limit of integration.
        b (float): The upper limit of integration.
    n (int): The number of subintervals (must be an even integer).

    Returns:
        float: The approximated definite integral.
    
    Raises:
        ValueError: If n is not a positive even integer.
    """
    if n <= 0 or n % 2 != 0:
        raise ValueError("Number of subintervals (n) must be a positive even integer for Simpson's Rule.")
    
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = func(x)
    
    integral = (h / 3) * (y[0] + 2 * np.sum(y[2:n:2]) + 4 * np.sum(y[1:n:2]) + y[n])
    return integral

# SciPy's quad and trapezoid will be used directly in the example file for comparison.