#         (c)    Gaussian MF

import numpy as np
import matplotlib.pyplot as plt

def gaussian_mf(x, c, sigma):
    """
    Compute the Gaussian membership function.

    :param x: Input values (numpy array)
    :param c: Mean (center of the Gaussian curve)
    :param sigma: Standard deviation (controls the spread)
    :return: Membership values (numpy array)
    """
    return np.exp(-((x - c) ** 2) / (2 * sigma ** 2))

def plot_gaussian_mf(c, sigma, x_range=(-10, 10)):
    """
    Plot the Gaussian membership function.

    :param c: Mean (center of the Gaussian curve)
    :param sigma: Standard deviation (spread)
    :param x_range: Range of x values for plotting
    """
    x = np.linspace(x_range[0], x_range[1], 500)
    y = gaussian_mf(x, c, sigma)

    plt.figure(figsize=(7, 4))
    plt.plot(x, y, label=f'Gaussian MF (c={c}, sigma={sigma})', color='b')
    plt.xlabel('x')
    plt.ylabel('Membership Degree')
    plt.title('Gaussian Membership Function')
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(c, color='r', linestyle='--', label="c (Mean)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage with different parameter values
plot_gaussian_mf(c=0, sigma=2)
plot_gaussian_mf(c=3, sigma=1)
