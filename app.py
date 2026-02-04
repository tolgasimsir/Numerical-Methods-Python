# app.py

import streamlit as st
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import re

# Add the project root directory to the system path
# This assumes app.py is in the root directory of your project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
sys.path.insert(0, project_root)

# Import all necessary functions from your custom numerical methods repository
# Make sure your numerical_methods_repository folder is at the same level as app.py
try:
    # --- BURAYA YENİ IMPORT'LAR EKLENDİ ---
    from numerical_methods_repository import error_analysis
    from numerical_methods_repository.root_finding import bisection_method, newton_raphson_method
    from numerical_methods_repository.differentiation import forward_difference, backward_difference, central_difference
    from numerical_methods_repository.integration import trapezoidal_rule, simpsons_rule
    from numerical_methods_repository.linear_systems import gaussian_elimination
    from numerical_methods_repository.lu_decomposition import lu_decompose, lu_solve
    from numerical_methods_repository.optimization import golden_section_search
    from numerical_methods_repository.ode_solvers import euler_method
    from numerical_methods_repository import interpolation_methods 
    # YENİ EKLENEN: Eigenvalue Methods
    from numerical_methods_repository.eigenvalue_methods import power_method # Bu satır EKLENDİ
except ImportError as e:
    st.error(f"Failed to load numerical methods library. Check your folder structure: {e}")
    st.stop() # Stop the application

# --- Function Parsing and Safe Function Creation ---

# Supported mathematical functions and their numpy equivalents
SUPPORTED_FUNCTIONS = {
    'sin': np.sin,
    'cos': np.cos,
    'tan': np.tan,
    'exp': np.exp,
    'log': np.log,  # Natural logarithm (ln)
    'log10': np.log10, # Base-10 logarithm
    'sqrt': np.sqrt,
    'abs': np.abs,
    'pi': np.pi,
    'e': np.e
}

def create_safe_function(func_str, variables=['x']):
    """
    Creates a safe Python function from a user-provided string.
    Uses eval(), but in a processed and restricted environment.
    Supports multiple variables (x, y, t) via the 'variables' list.
    Ensures the return value is always a float or a numpy array.
    """

    # Remove spaces and convert to lowercase
    func_str = func_str.replace(" ", "").lower()

    # Replace supported functions and constants with their numpy equivalents
    for func_name, func_obj in SUPPORTED_FUNCTIONS.items():
        if func_name in ['pi', 'e']: # Special case for constants
            func_str = func_str.replace(func_name, f"np.{func_name}")
        else: # For functions
            func_str = re.sub(r'\b' + re.escape(func_name) + r'\(', f'np.{func_name}(', func_str)

    # Replace power operator ^ with Python's ** (user might type ^)
    func_str = func_str.replace('^', '**')

    # Convert all numeric literals to float (e.g., "2*x" -> "2.0*x")
    # This prevents numeric literals from staying as integers and causing numpy type errors
    def convert_numbers_to_float(match):
        return str(float(match.group(0)))

    func_str = re.sub(r'\b\d+(\.\d*)?([eE][+\-]?\d+)?\b', convert_numbers_to_float, func_str)

    try:
        # Create a safe global and local environment
        # Setting __builtins__ to None disables dangerous built-in functions
        # Only np and math libraries are accessible
        safe_globals = {
            'np': np,
            'math': math,
            '__builtins__': None
        }

        # Dynamically create a lambda expression based on the variables
        lambda_args = ", ".join(variables)
        lambda_expr = f"lambda {lambda_args}: {func_str}"
        compiled_func = eval(lambda_expr, safe_globals)

        def final_callable(*args_tuple):
            try:
                # If there's a single argument and it's a numpy array, pass it directly
                if len(args_tuple) == 1 and isinstance(args_tuple[0], np.ndarray):
                    result = compiled_func(args_tuple[0])
                else:
                    # For scalar inputs or multiple arguments (e.g., func(y, t) in ODE)
                    # Convert arguments to float.
                    processed_args = [float(arg) for arg in args_tuple]
                    result = compiled_func(*processed_args)

                # Ensure the returned result is strictly a float or a numpy array
                if isinstance(result, (np.ndarray, list, tuple)):
                    # If it's an array, ensure it's not a single-element array that should be a float
                    if len(result) == 1 and isinstance(result[0], (int, float, np.number)):
                        return float(result[0])
                    return result # Return as is if it's a multi-element array for vector operations
                elif isinstance(result, (int, float, np.number)): # np.number covers numpy float/int types
                    return float(result)
                else:
                    raise TypeError(f"Function returned an unsupported type: {type(result).__name__}. Expected a numeric value or numpy array.")

            except Exception as e:
                # Provide a more understandable error message to the user
                raise ValueError(f"Error evaluating function '{func_str}' with arguments {args_tuple}: {e}")

        return final_callable

    except (SyntaxError, NameError, TypeError) as e:
        raise ValueError(f"Invalid function expression: '{func_str}'. Please check syntax and supported operations. Error: {e}")
    except Exception as e:
        raise ValueError(f"An unexpected error occurred while processing function '{func_str}': {e}")


# --- Streamlit Interface ---

st.set_page_config(layout="wide", page_title="Numerical Methods Library")

st.title("🔢 Numerical Methods Library")
st.markdown("""
This application helps you solve mathematical problems using various numerical methods
(root finding, differentiation, integration, optimization, linear systems, ODE solvers, eigenvalue problems, interpolation).
""")

# Sidebar menu
st.sidebar.header("Method Selection")
method_choice = st.sidebar.radio(
    "Please select a method:",
    [
        "Root Finding",
        "Numerical Differentiation",
        "Numerical Integration",
        "Solve Linear Systems",
        "LU Decomposition",
        "Optimization",
        "ODE Solving",
        "Interpolation",
        "Eigenvalue Problems" # Bu satır YENİ EKLENDİ
    ]
)

if method_choice == "Root Finding":
    st.header("🔍 Root Finding Methods")
    st.markdown("Find the roots of the equation $f(x)=0$ using one of two methods.")

    func_str = st.text_input("Enter the function f(x) (e.g., `x**2 - 2`, `cos(x) - x`):", 'x**2 - 2')

    try:
        func = create_safe_function(func_str, variables=['x'])

        sub_method = st.radio("Root Finding Method:", ("Bisection Method", "Newton-Raphson Method"))

        if sub_method == "Bisection Method":
            col1, col2 = st.columns(2)
            a = col1.number_input("Lower Bound 'a'", value=1.0, step=0.1)
            b = col2.number_input("Upper Bound 'b'", value=2.0, step=0.1)
            tol = st.number_input("Tolerance (e.g., 1e-6)", value=1e-6, format="%.7f")

            if st.button("Find Root (Bisection)"):
                try:
                    val_a = func(a)
                    val_b = func(b)

                    if val_a * val_b >= 0:
                        st.error(f"Error: Function values at bounds 'a' ({val_a:.4e}) and 'b' ({val_b:.4e}) must have opposite signs for Bisection Method.")
                    else:
                        root, iterations_data = bisection_method(func, a, b, tol=tol)
                        st.success(f"**Bisection Method Result:** Estimated Root = `{root:.6f}`")

                        final_func_root_val = func(root)
                        st.info(f"f(root) = `{final_func_root_val:.6e}`")

                        # --- Grafik Ekleme: Bisection Method ---
                        st.subheader("Convergence Plot")
                        fig, ax = plt.subplots(figsize=(10, 6))

                        # Plot the function itself
                        x_plot = np.linspace(min(a, root - 0.5), max(b, root + 0.5), 500)
                        y_plot = np.array([func(x_val) for x_val in x_plot])
                        ax.plot(x_plot, y_plot, label=f'$f(x) = {func_str}$', color='blue', alpha=0.7)
                        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8) # x-axis

                        # Plot iteration points
                        iter_x = [data[0] for data in iterations_data]
                        iter_y = [data[1] for data in iterations_data]
                        ax.plot(iter_x, iter_y, 'ro-', markersize=6, alpha=0.8, label='Iteration Points (x_k, f(x_k))')

                        # Plot the found root
                        ax.plot(root, final_func_root_val, 'gx', markersize=10, markeredgewidth=2, label=f'Found Root ({root:.4f})')

                        ax.set_title('Bisection Method Convergence')
                        ax.set_xlabel('x')
                        ax.set_ylabel('f(x)')
                        ax.legend()
                        ax.grid(True)
                        st.pyplot(fig)
                        plt.clf()
                        plt.close(fig)
                        # --- Grafik Ekleme Bitti ---

                except ValueError as ve:
                    st.error(f"Calculation Error: {ve}")
                    # st.exception(ve) # Hata mesajını daha kısa tutmak için stack trace'i kaldırdık
                except TypeError as te:
                    st.error(f"Type Error during calculation: {te}")
                    # st.exception(te)
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
                    # st.exception(e)

        elif sub_method == "Newton-Raphson Method":
            deriv_str = st.text_input("Enter the derivative function f'(x) (e.g., `2*x`, `-sin(x) - 1`):", '2*x')
            initial_guess = st.number_input("Initial Guess:", value=1.5, step=0.1)
            tol = st.number_input("Tolerance (e.g., 1e-6)", value=1e-6, format="%.7f")

            if st.button("Find Root (Newton-Raphson)"):
                try:
                    deriv = create_safe_function(deriv_str, variables=['x'])
                    root, iterations_data = newton_raphson_method(func, deriv, initial_guess, tol=tol)
                    st.success(f"**Newton-Raphson Method Result:** Estimated Root = `{root:.6f}`")
                    st.info(f"f(root) = `{func(root):.6e}`")

                    # --- Grafik Ekleme: Newton-Raphson Method ---
                    st.subheader("Convergence Plot")
                    fig, ax = plt.subplots(figsize=(10, 6))

                    # Plot the function itself
                    # Determine a reasonable plot range based on initial guess and root
                    x_min_plot = min(initial_guess, root) - 1.0
                    x_max_plot = max(initial_guess, root) + 1.0
                    x_plot = np.linspace(x_min_plot, x_max_plot, 500)
                    y_plot = np.array([func(x_val) for x_val in x_plot])
                    ax.plot(x_plot, y_plot, label=f'$f(x) = {func_str}$', color='blue', alpha=0.7)
                    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8) # x-axis

                    # Plot iteration points
                    iter_x = [data[0] for data in iterations_data]
                    iter_y = [data[1] for data in iterations_data]
                    ax.plot(iter_x, iter_y, 'ro-', markersize=6, alpha=0.8, label='Iteration Points (x_k, f(x_k))')

                    # Plot the found root
                    ax.plot(root, func(root), 'gx', markersize=10, markeredgewidth=2, label=f'Found Root ({root:.4f})')

                    ax.set_title('Newton-Raphson Method Convergence')
                    ax.set_xlabel('x')
                    ax.set_ylabel('f(x)')
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)
                    plt.clf()
                    plt.close(fig)
                    # --- Grafik Ekleme Bitti ---

                except ValueError as ve:
                    st.error(f"Calculation Error: {ve}")
                    # st.exception(ve)
                except TypeError as te:
                    st.error(f"Type Error during calculation: {te}")
                    # st.exception(te)
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
                    # st.exception(e)

    except ValueError as ve:
        st.error(f"Function Definition Error: {ve}")
    except Exception as e:
        st.error(f"A general error occurred: {e}")

elif method_choice == "Numerical Differentiation":
    st.header("📈 Numerical Differentiation")
    st.markdown("Calculate the derivative of a function at a specific point.")

    func_str = st.text_input("Enter the function f(x) (e.g., `x**3 + 2*x`, `sin(x)`):", 'x**3 + 2*x')

    try:
        func = create_safe_function(func_str, variables=['x'])

        x_val = st.number_input("Enter x value at which to differentiate:", value=1.0, step=0.1)
        h_val = st.number_input("Enter step size 'h' (e.g., 1e-6):", value=1e-6, format="%.7f")

        if st.button("Calculate Derivative"):
            try:
                fwd_diff = forward_difference(func, x_val, h_val)
                bwd_diff = backward_difference(func, x_val, h_val)
                cnt_diff = central_difference(func, x_val, h_val)

                st.write(f"**Forward Difference Derivative:** `{fwd_diff:.8f}`")
                st.write(f"**Backward Difference Derivative:** `{bwd_diff:.8f}`")
                st.write(f"**Central Difference Derivative:** `{cnt_diff:.8f}`")

                # --- Grafik Ekleme: Numerical Differentiation ---
                st.subheader("Approximation Visualization")
                fig, ax = plt.subplots(figsize=(10, 6))

                # Plot the function
                x_plot = np.linspace(x_val - 2*h_val, x_val + 2*h_val, 100)
                y_plot = np.array([func(x) for x in x_plot])
                ax.plot(x_plot, y_plot, label=f'$f(x) = {func_str}$', color='blue', alpha=0.7)

                # Plot the point of interest
                ax.plot(x_val, func(x_val), 'ro', markersize=8, label=f'Point $x = {x_val}$')

                # Plot tangent lines based on derivatives (approximations)
                x_fwd_line = np.array([x_val, x_val + h_val])
                y_fwd_line = func(x_val) + fwd_diff * (x_fwd_line - x_val)
                ax.plot(x_fwd_line, y_fwd_line, 'g--', label=f'Forward Difference Slope ({fwd_diff:.4f})')

                x_bwd_line = np.array([x_val - h_val, x_val])
                y_bwd_line = func(x_val) + bwd_diff * (x_bwd_line - x_val)
                ax.plot(x_bwd_line, y_bwd_line, 'm--', label=f'Backward Difference Slope ({bwd_diff:.4f})')

                x_cnt_line = np.array([x_val - h_val, x_val + h_val])
                y_cnt_line = func(x_val) + cnt_diff * (x_cnt_line - x_val)
                ax.plot(x_cnt_line, y_cnt_line, 'c--', label=f'Central Difference Slope ({cnt_diff:.4f})')


                ax.set_title('Numerical Differentiation Approximation')
                ax.set_xlabel('x')
                ax.set_ylabel('f(x)')
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
                plt.clf()
                plt.close(fig)
                # --- Grafik Ekleme Bitti ---

            except ValueError as ve:
                st.error(f"Calculation Error: {ve}")
                # st.exception(ve)
            except TypeError as te:
                st.error(f"Type Error during calculation: {te}")
                # st.exception(te)
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                # st.exception(e)
    except ValueError as ve:
        st.error(f"Function Definition Error: {ve}")

elif method_choice == "Numerical Integration":
    st.header("📊 Numerical Integration")
    st.markdown("Calculate the integral of a function over a specified interval.")

    func_str = st.text_input("Enter the function f(x) to integrate (e.g., `x**2`, `exp(-x**2)`):", 'x**2')

    try:
        func = create_safe_function(func_str, variables=['x'])

        col1, col2 = st.columns(2)
        a = col1.number_input("Lower limit 'a':", value=0.0, step=0.1)
        b = col2.number_input("Upper limit 'b':", value=1.0, step=0.1)
        n = st.number_input("Number of subintervals 'n' (must be even for Simpson's):", value=100, step=1, format="%d")

        if st.button("Calculate Integral"):
            try:
                trap_integral = trapezoidal_rule(func, a, b, n)
                st.write(f"**Trapezoidal Rule Integral:** `{trap_integral:.6f}`")

                sim_integral = None # Initialize
                if n % 2 != 0:
                    st.warning("Warning: Number of subintervals 'n' for Simpson's Rule should be even. Simpson's Rule will not be calculated.")
                else:
                    sim_integral = simpsons_rule(func, a, b, n)
                    st.write(f"**Simpson's Rule Integral:** `{sim_integral:.6f}`")

                # --- Grafik Ekleme: Numerical Integration ---
                st.subheader("Integration Visualization")
                fig, ax = plt.subplots(figsize=(10, 6))

                x_plot_smooth = np.linspace(a, b, 500) # For smooth function plot
                y_plot_smooth = np.array([func(x) for x in x_plot_smooth])
                ax.plot(x_plot_smooth, y_plot_smooth, label=f'$f(x) = {func_str}$', color='blue', alpha=0.7)

                # Fill the area under the curve (actual integral representation)
                ax.fill_between(x_plot_smooth, 0, y_plot_smooth, color='skyblue', alpha=0.3, label='Area Under Curve') 
                
                if n <= 100: 
                    x_segments = np.linspace(a, b, n + 1)
                    y_segments = np.array([func(x) for x in x_segments])
                    
                    for i in range(n):
                        # Draw trapezoid segments
                        xs = [x_segments[i], x_segments[i+1], x_segments[i+1], x_segments[i]]
                        ys = [0, 0, y_segments[i+1], y_segments[i]]
                        ax.fill(xs, ys, 'lightgreen', alpha=0.4, edgecolor='green', linewidth=0.5) 
                    
                    ax.plot(x_segments, y_segments, 'go', markersize=4, label='Approximation Points')

                ax.set_title('Numerical Integration (Approximation of Area)')
                ax.set_xlabel('x')
                ax.set_ylabel('f(x)')
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
                plt.clf()
                plt.close(fig)
                # --- Grafik Ekleme Bitti ---

            except ValueError as ve:
                st.error(f"Calculation Error: {ve}")
                # st.exception(ve)
            except TypeError as te:
                st.error(f"Type Error during calculation: {te}")
                # st.exception(te)
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                # st.exception(e)
    except ValueError as ve:
        st.error(f"Function Definition Error: {ve}")


elif method_choice == "Solve Linear Systems":
    st.header("🧮 Solve Linear Systems (Ax = b)")
    st.markdown("Solve linear equation systems of the form $Ax=b$ using Gaussian Elimination.")
    st.info("Enter matrix A with rows separated by `;` and elements by `,` (e.g., `1,2;3,4`). Enter vector b with elements separated by `,` (e.g., `5,6`).")

    A_input = st.text_area("Matrix A:", "1,2;3,4")
    b_input = st.text_input("Vector b:", "5,6")

    if st.button("Solve System"):
        try:
            A_rows = A_input.split(';')
            A_list = []
            for row_str in A_rows:
                A_list.append([float(x.strip()) for x in row_str.split(',')])
            A = np.array(A_list)
            b = np.array([float(x.strip()) for x in b_input.split(',')])

            if A.shape[0] != A.shape[1]:
                st.error("Error: Matrix A must be square.")
            elif A.shape[0] != b.shape[0]:
                st.error("Error: Dimension mismatch between A and b.")
            else:
                x_gauss = gaussian_elimination(A.copy(), b.copy())
                st.success(f"**Solution x:** `{x_gauss}`")
                st.info(f"Verification (A @ x): `{A @ x_gauss}`")
                st.info(f"Error (A@x - b) norm: `{np.linalg.norm(A @ x_gauss - b):.6e}`")
        except ValueError as ve:
            st.error(f"Input Error: {ve}. Please check matrix and vector format.")
            # st.exception(ve)
        except TypeError as te:
            st.error(f"Type Error during calculation: {te}")
            # st.exception(te)
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            # st.exception(e)

elif method_choice == "LU Decomposition":
    st.header("🧩 LU Decomposition & Solve (Ax = b)")
    st.markdown("Decompose a matrix into its LU factors and solve the system.")
    st.info("Enter matrix A with rows separated by `;` and elements by `,` (e.g., `2,1,1;4,-6,0;-2,7,2`). Enter vector b with elements separated by `,` (e.g., `5,-2,9`).")

    A_input = st.text_area("Matrix A:", "2,1,1;4,-6,0;-2,7,2")
    b_input = st.text_input("Vector b:", "5,-2,9")

    if st.button("Decompose LU and Solve"):
        try:
            A_rows = A_input.split(';')
            A_list = []
            for row_str in A_rows:
                A_list.append([float(x.strip()) for x in row_str.split(',')])
            A = np.array(A_list)
            b = np.array([float(x.strip()) for x in b_input.split(',')])

            if A.shape[0] != A.shape[1]:
                st.error("Error: Matrix A must be square for LU decomposition.")
            elif A.shape[0] != b.shape[0]:
                st.error("Error: Dimension mismatch between A and b.")
            else:
                L, U = lu_decompose(A.copy())
                st.subheader("L Matrix:")
                st.write(L)
                st.subheader("U Matrix:")
                st.write(U)
                st.info(f"Verification (L @ U) - A norm: `{np.linalg.norm((L @ U) - A):.6e}`")

                x_lu = lu_solve(L, U, b.copy())
                st.success(f"**Solution x (using LU):** `{x_lu}`")
                st.info(f"Verification (A @ x): `{A @ x_lu}`")
                st.info(f"Error (A@x - b) norm: `{np.linalg.norm(A @ x_lu - b):.6e}`")
        except ValueError as ve:
            st.error(f"Input Error: {ve}. Please check matrix and vector format.")
            # st.exception(ve)
        except TypeError as te:
            st.error(f"Type Error during calculation: {te}")
            # st.exception(te)
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            # st.exception(e)

elif method_choice == "Optimization":
    st.header("📉 Optimization (Golden Section Search)")
    st.markdown("Find the minimum of a function over a specified interval.")

    func_str = st.text_input("Enter the function f(x) to minimize (e.g., `x**2 + 5 * sin(x)`):", 'x**2 + 5 * sin(x)')

    try:
        func = create_safe_function(func_str, variables=['x'])

        col1, col2 = st.columns(2)
        a = col1.number_input("Lower Bound 'a':", value=-4.0, step=0.1)
        b = col2.number_input("Upper Bound 'b':", value=2.0, step=0.1)
        tol = st.number_input("Tolerance (e.g., 1e-6):", value=1e-6, format="%.7f")

        if st.button("Find Minimum"):
            try:
                min_x = golden_section_search(func, a, b, tol=tol)
                min_y = func(min_x)
                st.success(f"**Golden Section Search:** Minimum at `x = {min_x:.6f}`, `f(x) = {min_y:.6f}`")

                fig, ax = plt.subplots(figsize=(8, 5))
                x_vals = np.linspace(a, b, 500)
                y_vals = np.array([func(val) for val in x_vals])
                ax.plot(x_vals, y_vals, label='Function')
                ax.plot(min_x, min_y, 'ro', markersize=8, label='Minimum Found')
                ax.set_title('Optimization Result')
                ax.set_xlabel('x')
                ax.set_ylabel('f(x)')
                ax.grid(True)
                ax.legend()
                st.pyplot(fig)
                plt.clf()
                plt.close(fig)
            except ValueError as ve:
                st.error(f"Calculation Error: {ve}")
                # st.exception(ve)
            except TypeError as te:
                st.error(f"Type Error during calculation: {te}")
                # st.exception(te)
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                # st.exception(e)
    except ValueError as ve:
        st.error(f"Function Definition Error: {ve}")

elif method_choice == "ODE Solving":
    st.header("🌊 Ordinary Differential Equation (ODE) Solving (Euler Method)")
    st.markdown("Solve ODEs of the form $\\frac{dy}{dt} = f(y, t)$ using Euler's Method.")

    func_str = st.text_input("Enter the derivative function f(y, t) (e.g., `-0.1 * y`, `y * (1 - y/10)`):", '-0.1 * y')

    try:
        # Use create_safe_function for ODEs with y and t variables
        func_ode = create_safe_function(func_str, variables=['y', 't'])

        y0 = st.number_input("Initial condition y0:", value=10.0, step=0.1)
        col1, col2 = st.columns(2)
        t_start = col1.number_input("Start time:", value=0.0, step=0.1)
        t_end = col2.number_input("End time:", value=10.0, step=0.1)
        num_points = st.number_input("Number of time points:", value=100, step=1, format="%d")

        if st.button("Solve ODE"):
            try:
                t = np.linspace(t_start, t_end, num_points)
                y_solution = euler_method(func_ode, y0, t)

                st.success(f"**Solution (at t={t_end:.2f}):** `y = {y_solution[-1]:.6f}`")

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(t, y_solution, 'b.-', label='Euler Method')
                ax.set_title('ODE Solution (Euler Method)')
                ax.set_xlabel('Time (t)')
                ax.set_ylabel('y(t)')
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
                plt.clf()
                plt.close(fig)
            except ValueError as ve:
                st.error(f"Calculation Error: {ve}")
                # st.exception(ve)
            except TypeError as te:
                st.error(f"Type Error during calculation: {te}")
                # st.exception(te)
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                # st.exception(e)
    except ValueError as ve:
        st.error(f"Function Definition Error: {ve}")

# !!! DİKKAT: Interpolation method_choice EKLENDİ !!!
elif method_choice == "Interpolation":
    interpolation_methods.run_interpolation_app() # interpolation_methods.py dosyasındaki fonksiyonu çağırıyoruz

# --- YENİ EKLENEN: Eigenvalue Problems Bölümü ---
elif method_choice == "Eigenvalue Problems":
    st.header("⚛️ Eigenvalue Problems (Power Method)")
    st.markdown("Find the dominant eigenvalue and its corresponding eigenvector of a square matrix using the Power Method.")
    st.info("Enter matrix A with rows separated by `;` and elements by `,` (e.g., `2,1;1,2`).")

    A_input_eigen = st.text_area("Matrix A:", "2,1;1,2")
    
    col1_eigen, col2_eigen = st.columns(2)
    num_iterations_eigen = col1_eigen.number_input("Maximum Iterations:", value=100, step=1, format="%d")
    tolerance_eigen = col2_eigen.number_input("Tolerance (e.g., 1e-6):", value=1e-6, format="%.7f")

    if st.button("Find Dominant Eigenvalue"):
        try:
            A_rows_eigen = A_input_eigen.split(';')
            A_list_eigen = []
            for row_str in A_rows_eigen:
                A_list_eigen.append([float(x.strip()) for x in row_str.split(',')])
            A_eigen = np.array(A_list_eigen)

            if A_eigen.shape[0] != A_eigen.shape[1]:
                st.error("Error: Matrix A must be square for Eigenvalue Problems.")
            else:
                dominant_val, dominant_vec = power_method(A_eigen.copy(), num_iterations=num_iterations_eigen, tolerance=tolerance_eigen)
                
                st.success(f"**Dominant Eigenvalue:** `{dominant_val:.6f}`")
                st.success(f"**Corresponding Eigenvector:** `{dominant_vec}`")

                st.info("Verification (A @ x ≈ λ * x):")
                st.write(f"A @ x = `{np.dot(A_eigen, dominant_vec)}`")
                st.write(f"λ * x = `{dominant_val * dominant_vec}`")
                st.info(f"Difference Norm (||A @ x - λ * x||): `{np.linalg.norm(np.dot(A_eigen, dominant_vec) - (dominant_val * dominant_vec)):.6e}`")

        except ValueError as ve:
            st.error(f"Input Error: {ve}. Please check matrix format and ensure it's a square matrix.")
        except RuntimeError as re:
            st.error(f"Calculation Error: {re}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

st.sidebar.markdown("---")
st.sidebar.info("This application is designed for exploring and testing numerical methods.")