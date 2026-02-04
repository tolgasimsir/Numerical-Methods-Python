import numpy as np
import matplotlib.pyplot as plt
import math

def calculate_machine_epsilon():
    """
    Calculates the machine epsilon for floating-point numbers.
    Machine epsilon is the smallest number that, when added to 1.0,
    yields a result different from 1.0.
    """
    epsilon = 1.0
    while (1.0 + epsilon) != 1.0:
        epsilon /= 2.0
    return epsilon * 2.0

def illustrate_floating_point_precision():
    """
    Illustrates the limitations of floating-point precision with simple examples.
    This function no longer prints directly to Streamlit.
    Its output would typically be used programmatically or by a calling function
    that handles display.
    """
    # Example 1: Basic addition with small numbers
    a = 0.1
    b = 0.2
    c = 0.3
    # print(f"0.1 + 0.2 = {a + b}")
    # print(f"Is (0.1 + 0.2) == 0.3? { (a + b) == c }")
    # print(f"Difference: { (a + b) - c }")

    # Example 2: Subtraction of nearly equal numbers (catastrophic cancellation)
    x = 1.0
    y = 1.0 + 1e-15 # A very small difference
    # print(f"y - x = {y - x}")
    # print(f"Expected difference (1e-15): {1e-15}")

    # Example 3: Large numbers and small numbers
    large_num = 1e15
    small_num = 1e-5
    # print(f"({large_num} + {small_num}) - {large_num} = {(large_num + small_num) - large_num}")
    # print(f"Expected: {small_num}")

    # Machine Epsilon
    mach_eps = np.finfo(float).eps
    # print(f"Machine Epsilon (Numpy): {mach_eps}")
    # print(f"Is (1.0 + machine_epsilon) == 1.0? { (1.0 + mach_eps) == 1.0 }")
    # print(f"Is (1.0 + machine_epsilon/2) == 1.0? { (1.0 + mach_eps/2) == 1.0 }")
    # print(f"Smallest number that makes a difference from 1.0: {mach_eps}")
    pass # This function now effectively does nothing visible without a caller

def illustrate_truncation_error():
    """
    Illustrates truncation error using the Taylor series expansion of sin(x).
    The more terms we use, the smaller the truncation error.
    This function now returns only the Matplotlib figure.
    """
    x = np.pi / 4 # x = 45 degrees
    actual_sin_x = np.sin(x)

    # Taylor series for sin(x) = x - x^3/3! + x^5/5! - x^7/7! + ...
    def taylor_sin(val, num_terms):
        result = 0
        for n in range(num_terms):
            term = ((-1)**n) * (val**(2*n + 1)) / math.factorial(2*n + 1)
            result += term
        return result

    num_terms = np.arange(1, 15)
    truncation_errors = []

    for terms in num_terms:
        approx_sin_x = taylor_sin(x, terms)
        abs_error = abs(actual_sin_x - approx_sin_x)
        truncation_errors.append(abs_error)

    # Plotting the truncation error
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(num_terms, truncation_errors, marker='o', linestyle='-', color='b')
    ax.set_title('Truncation Error vs. Number of Terms in Taylor Series for sin(x)')
    ax.set_xlabel('Number of Terms')
    ax.set_ylabel('Absolute Truncation Error')
    ax.set_yscale('log') # Log scale for better visualization of decreasing error
    ax.grid(True)
    plt.tight_layout()

    return fig # Return the Matplotlib figure