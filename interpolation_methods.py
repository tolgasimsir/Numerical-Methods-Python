import numpy as np
from scipy.interpolate import interp1d, CubicSpline
import streamlit as st
import matplotlib.pyplot as plt

def linear_interpolation(x_known, y_known, x_interp):
    """
    Performs linear interpolation.
    x_known: Known x-coordinates (list or numpy array)
    y_known: Known y-coordinates (list or numpy array)
    x_interp: x-value(s) at which to interpolate
    """
    # Ensure inputs are numpy arrays for robust calculations
    x_known = np.array(x_known)
    y_known = np.array(y_known)
    x_interp = np.array(x_interp)

    # Sort data to ensure correct interpolation, especially for interp1d
    sort_indices = np.argsort(x_known)
    x_known = x_known[sort_indices]
    y_known = y_known[sort_indices]

    # Create a linear interpolation function
    f_linear = interp1d(x_known, y_known, kind='linear', fill_value="extrapolate")
    return f_linear(x_interp)

def cubic_spline_interpolation(x_known, y_known, x_interp):
    """
    Performs cubic spline interpolation.
    x_known: Known x-coordinates (list or numpy array)
    y_known: Known y-coordinates (list or numpy array)
    x_interp: x-value(s) at which to interpolate
    """
    x_known = np.array(x_known)
    y_known = np.array(y_known)
    x_interp = np.array(x_interp)

    sort_indices = np.argsort(x_known)
    x_known = x_known[sort_indices]
    y_known = y_known[sort_indices]

    # Create a cubic spline interpolation function
    f_cubic = CubicSpline(x_known, y_known, bc_type='natural') # natural boundary conditions
    return f_cubic(x_interp)

def run_interpolation_app():
    st.title("Interpolation Techniques")
    st.subheader("Estimate values between known data points")

    st.write("Enter your known data points (x, y). Use commas to separate values (e.g., 1,2,3 for x values and 10,20,30 for y values).")

    col1, col2 = st.columns(2)
    with col1:
        x_input = st.text_input("Enter x values (comma-separated):", "0, 1, 2, 3, 4")
    with col2:
        y_input = st.text_input("Enter y values (comma-separated):", "0, 1, 4, 9, 16")

    x_interp_single_input = st.number_input("Enter x value to interpolate:", value=2.5, format="%.2f")

    method = st.radio("Select Interpolation Method:", ("Linear Interpolation", "Cubic Spline Interpolation"))

    if st.button("Interpolate"):
        try:
            x_known = np.array([float(val.strip()) for val in x_input.split(',')])
            y_known = np.array([float(val.strip()) for val in y_input.split(',')])

            if len(x_known) != len(y_known):
                st.error("Error: Number of x values must match number of y values.")
            elif len(x_known) < 2:
                st.error("Error: At least 2 data points are required for interpolation.")
            elif len(x_known) < 4 and method == "Cubic Spline Interpolation":
                st.warning("Warning: Cubic Spline generally performs better with at least 4 data points. For fewer points, linear interpolation might be sufficient or Cubic Spline behavior might be less intuitive.")
                if len(x_known) < 2: # CubicSpline requires at least 2 points
                    st.error("Error: Cubic Spline requires at least 2 data points.")
                    return

            if x_interp_single_input < np.min(x_known) or x_interp_single_input > np.max(x_known):
                st.warning(f"Warning: The interpolation point {x_interp_single_input} is outside the range of known x values ({np.min(x_known)} - {np.max(x_known)}). Extrapolation will be performed.")

            interp_y = None
            if method == "Linear Interpolation":
                interp_y = linear_interpolation(x_known, y_known, x_interp_single_input)
                st.success(f"Linear Interpolation Result at x={x_interp_single_input}: y = {interp_y:.6f}")
            elif method == "Cubic Spline Interpolation":
                interp_y = cubic_spline_interpolation(x_known, y_known, x_interp_single_input)
                st.success(f"Cubic Spline Interpolation Result at x={x_interp_single_input}: y = {interp_y:.6f}")

            # Plotting the results
            st.subheader("Interpolation Visualization")
            fig, ax = plt.subplots(figsize=(10, 6))

            # Sort known points for plotting
            sort_indices = np.argsort(x_known)
            x_known_sorted = x_known[sort_indices]
            y_known_sorted = y_known[sort_indices]

            ax.plot(x_known_sorted, y_known_sorted, 'o', label='Known Data Points', color='blue')

            # Create a denser range for plotting the interpolated curve
            x_plot = np.linspace(np.min(x_known_sorted), np.max(x_known_sorted), 500)

            if method == "Linear Interpolation":
                y_plot = linear_interpolation(x_known_sorted, y_known_sorted, x_plot)
                ax.plot(x_plot, y_plot, '-', label='Linear Interpolation Curve', color='green')
            elif method == "Cubic Spline Interpolation":
                y_plot = cubic_spline_interpolation(x_known_sorted, y_known_sorted, x_plot)
                ax.plot(x_plot, y_plot, '-', label='Cubic Spline Interpolation Curve', color='red')

            # Plot the interpolated point
            if interp_y is not None:
                ax.plot(x_interp_single_input, interp_y, 'X', markersize=10, color='purple', label=f'Interpolated Point ({x_interp_single_input:.2f}, {interp_y:.2f})')

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(f"{method} Visualization")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)

        except ValueError:
            st.error("Error: Please enter valid comma-separated numbers for x and y values.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")