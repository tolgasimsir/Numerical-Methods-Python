import numpy as np

def golden_section_search(func, a, b, tol=1e-6, max_iter=100):
    """
    Finds the minimum of a unimodal function within a given interval [a, b]
    using the Golden Section Search method.

    Args:
        func (function): The function to minimize.
        a (float): The lower bound of the search interval.
        b (float): The upper bound of the search interval.
        tol (float): The desired tolerance for the interval width.
        max_iter (int): The maximum number of iterations.

    Returns:
        float: The estimated x-value at which the minimum occurs.

    Raises:
        ValueError: If the initial interval [a, b] is not valid.
    """
    if a >= b:
        raise ValueError("Invalid interval: 'a' must be less than 'b'.")

    # Golden ratio constant
    phi = (1 + np.sqrt(5)) / 2
    resphi = 2 - phi # 1/phi

    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)

    f1 = func(x1)
    f2 = func(x2)

    for i in range(max_iter):
        if abs(b - a) < tol:
            break

        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + resphi * (b - a)
            f1 = func(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = b - resphi * (b - a)
            f2 = func(x2)

    return (a + b) / 2

# SciPy's optimize.minimize_scalar and minimize will be used directly in the example file.