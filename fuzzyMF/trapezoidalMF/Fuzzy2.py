#         (b) Trapezoidal MF

import numpy as np
import matplotlib.pyplot as plt

def trapezoidal_mf(x, a, b, c, d):
    """
    Compute the trapezoidal membership function.

    :param x: Input values (numpy array)
    :param a: Left shoulder
    :param b: Start of the plateau
    :param c: End of the plateau
    :param d: Right shoulder
    :return: Membership values (numpy array)
    """
    return np.where(
        (x <= a) | (x >= d), 0,
        np.where((a <= x) & (x < b), (x - a) / (b - a),
                 np.where((b <= x) & (x <= c), 1, (d - x) / (d - c)))
    )

def plot_trapezoidal_mf(a, b, c, d, x_range=(-10, 10)):
    """
    Plot the trapezoidal membership function.

    :param a: Left shoulder
    :param b: Start of the plateau
    :param c: End of the plateau
    :param d: Right shoulder
    :param x_range: Range of x values for plotting
    """
    x = np.linspace(x_range[0], x_range[1], 500)
    y = trapezoidal_mf(x, a, b, c, d)

    plt.figure(figsize=(7, 4))
    plt.plot(x, y, label=f'Trapezoidal MF (a={a}, b={b}, c={c}, d={d})', color='b')
    plt.xlabel('x')
    plt.ylabel('Membership Degree')
    plt.title('Trapezoidal Membership Function')
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(a, color='r', linestyle='--', label="a (Left Shoulder)")
    plt.axvline(b, color='g', linestyle='--', label="b (Start of Plateau)")
    plt.axvline(c, color='m', linestyle='--', label="c (End of Plateau)")
    plt.axvline(d, color='y', linestyle='--', label="d (Right Shoulder)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage with different parameter values
plot_trapezoidal_mf(a=2, b=4, c=6, d=8)
plot_trapezoidal_mf(a=-5, b=-2, c=2, d=5)
