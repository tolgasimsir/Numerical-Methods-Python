import numpy as np
import matplotlib.pyplot as plt

def forward_difference(func, x, h=1e-6):
    """
    Approximates the derivative of a function using the forward difference method.

    Args:
        func (function): The function for which to approximate the derivative.
        x (float): The point at which to evaluate the derivative.
        h (float, optional): The step size. Defaults to 1e-6.

    Returns:
        float: The approximated derivative.
    """
    return (func(x + h) - func(x)) / h

def backward_difference(func, x, h=1e-6):
    """
    Approximates the derivative of a function using the backward difference method.

    Args:
        func (function): The function for which to approximate the derivative.
        x (float): The point at which to evaluate the derivative.
        h (float, optional): The step size. Defaults to 1e-6.

    Returns:
        float: The approximated derivative.
    """
    return (func(x) - func(x - h)) / h

def central_difference(func, x, h=1e-6):
    """
    Approximates the derivative of a function using the central difference method.

    Args:
        func (function): The function for which to approximate the derivative.
        x (float): The point at which to evaluate the derivative.
        h (float, optional): The step size. Defaults to 1e-6.

    Returns:
        float: The approximated derivative.
    """
    return (func(x + h) - func(x - h)) / (2 * h)