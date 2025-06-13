# 1. Write python functions to generate the following parameterized fuzzy membership functions and visualize them for different parameter values: (a) Triangular MF

import numpy as np
import matplotlib.pyplot as plt

def triangular_mf(x, a, b, c):
    """
    :param x: Input values (numpy array)
    :param a: Lower bound of the triangle
    :param b: Peak point of the triangle
    :param c: Upper bound of the triangle
    """
    return np.where(
        (x <= a) | (x >= c), 0, 
        np.where(x < b, (x - a) / (b - a), (c - x) / (c - b))
    )

def plot_triangular_mf(a, b, c, x_range=(-10, 10)):
    """
    Plot the triangular membership function.

    :param a: Lower bound of the triangle
    :param b: Peak point of the triangle
    :param c: Upper bound of the triangle
    :param x_range: Range of x values for plotting
    """
    x = np.linspace(x_range[0], x_range[1], 500)
    y = triangular_mf(x, a, b, c)

    plt.figure(figsize=(7, 4))
    plt.plot(x, y, label=f'Triangular MF (a={a}, b={b}, c={c})', color='b')
    plt.xlabel('x')
    plt.ylabel('Membership Degree')
    plt.title('Triangular Membership Function')
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(a, color='r', linestyle='--', label="a (Lower Bound)")
    plt.axvline(b, color='g', linestyle='--', label="b (Peak)")
    plt.axvline(c, color='m', linestyle='--', label="c (Upper Bound)")
    plt.legend()
    plt.grid(True)
    plt.show()

# To show the effect of shifting a, b, and c.
plot_triangular_mf(a=2, b=5, c=8)
plot_triangular_mf(a=-5, b=0, c=5)
