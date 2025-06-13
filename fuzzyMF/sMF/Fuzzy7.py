#         (g)    S-MF
 
import numpy as np
import matplotlib.pyplot as plt

def s_mf(x, a, b):
    """
    Compute the S (S-shape) membership function.

    :param x: Input values (numpy array)
    :param a: Left spread (start of transition)
    :param b: Right spread (end of transition)
    :return: Membership values (numpy array)
    """
    y = np.zeros_like(x)
    midpoint = (a + b) / 2

    # Increasing part (smooth transition)
    left_mask = (x > a) & (x <= midpoint)
    y[left_mask] = 2 * ((x[left_mask] - a) / (b - a)) ** 2

    right_mask = (x > midpoint) & (x < b)
    y[right_mask] = 1 - 2 * ((x[right_mask] - b) / (b - a)) ** 2

    # Beyond b, membership is 1
    y[x >= b] = 1

    return y

def plot_s_mf(a, b, x_range=(-10, 10)):
    """
    Plot the S (S-shape) membership function.

    :param a: Left spread (start of transition)
    :param b: Right spread (end of transition)
    :param x_range: Range of x values for plotting
    """
    x = np.linspace(x_range[0], x_range[1], 500)
    y = s_mf(x, a, b)

    plt.figure(figsize=(7, 4))
    plt.plot(x, y, label=f'S MF (a={a}, b={b})', color='b')
    plt.xlabel('x')
    plt.ylabel('Membership Degree')
    plt.title('S-Shape Membership Function')
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(a, color='r', linestyle='--', label="a (Start)")
    plt.axvline(b, color='g', linestyle='--', label="b (End)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage with different parameter values
plot_s_mf(a=-3, b=3)
plot_s_mf(a=0, b=5)
