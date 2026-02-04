import numpy as np
from scipy.integrate import odeint # SciPy's ODE solver for comparison

def euler_method(func, y0, t, *args):
    """
    Solves an Ordinary Differential Equation using Euler's method.

    dy/dt = func(y, t, *args)
    y(t0) = y0

    Args:
        func (callable): Function that computes the derivative dy/dt.
                         Signature: func(y, t, *args)
        y0 (float or array_like): Initial condition(s).
        t (array_like): A sequence of time points for which to solve for y.
        *args: Optional arguments to pass to func.

    Returns:
        np.array: Array of y values corresponding to each time point in t.
    """
    n_points = len(t)
    # Handle scalar y0 as a single-element array for consistency
    y = np.zeros((n_points, len(np.atleast_1d(y0))))
    y[0] = y0 if np.isscalar(y0) else np.array(y0)

    for i in range(n_points - 1):
        dt = t[i+1] - t[i]
        y[i+1] = y[i] + dt * func(y[i], t[i], *args)

    return y.squeeze() # Squeeze to remove single dimensions if y0 was scalar

# SciPy's odeint will be used directly in the example file for comparison.