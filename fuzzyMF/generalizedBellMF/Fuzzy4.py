#         (d)    Generalized Bell MF

import numpy as np
import matplotlib.pyplot as plt

def generalized_bell_mf(x, a, b, c):
    """
    Compute the Generalized Bell membership function.

    :param x: Input values (numpy array)
    :param a: Width parameter (spread)
    :param b: Slope parameter (controls shape)
    :param c: Center of the curve
    :return: Membership values (numpy array)
    """
    return 1 / (1 + np.abs((x - c) / a) ** (2 * b))

def plot_generalized_bell_mf(a, b, c, x_range=(-10, 10)):
    """
    Plot the Generalized Bell membership function.

    :param a: Width parameter (spread)
    :param b: Slope parameter (controls shape)
    :param c: Center of the curve
    :param x_range: Range of x values for plotting
    """
    x = np.linspace(x_range[0], x_range[1], 500)
    y = generalized_bell_mf(x, a, b, c)

    plt.figure(figsize=(7, 4))
    plt.plot(x, y, label=f'Bell MF (a={a}, b={b}, c={c})', color='b')
    plt.xlabel('x')
    plt.ylabel('Membership Degree')
    plt.title('Generalized Bell Membership Function')
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(c, color='r', linestyle='--', label="c (Center)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage with different parameter values
plot_generalized_bell_mf(a=2, b=2, c=0)
plot_generalized_bell_mf(a=3, b=4, c=-2)
