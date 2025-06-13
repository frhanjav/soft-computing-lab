#         (e)    PI-MF 
 
import numpy as np
import matplotlib.pyplot as plt

def pi_mf(x, a, c):
    """
    Compute the Pi (π) membership function.

    :param x: Input values (numpy array)
    :param a: Lower bound (left spread)
    :param c: Upper bound (right spread)
    :return: Membership values (numpy array)
    """
    y = np.zeros_like(x)
    midpoint = (a + c) / 2
    
    # First half (increasing curve)
    left_mask = (x > a) & (x <= midpoint)
    y[left_mask] = 2 * ((x[left_mask] - a) / (c - a)) ** 2
    
    # Second half (decreasing curve)
    right_mask = (x > midpoint) & (x <= c)
    y[right_mask] = 1 - 2 * ((x[right_mask] - c) / (c - a)) ** 2
    
    # Beyond c, membership is 1
    y[x > c] = 1
    
    return y

def plot_pi_mf(a, c, x_range=(-10, 10)):
    """
    Plot the Pi (π) membership function.

    :param a: Lower bound (left spread)
    :param c: Upper bound (right spread)
    :param x_range: Range of x values for plotting
    """
    x = np.linspace(x_range[0], x_range[1], 500)
    y = pi_mf(x, a, c)

    plt.figure(figsize=(7, 4))
    plt.plot(x, y, label=f'Pi MF (a={a}, c={c})', color='b')
    plt.xlabel('x')
    plt.ylabel('Membership Degree')
    plt.title('Pi (π) Membership Function')
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(a, color='r', linestyle='--', label="a (Left Bound)")
    plt.axvline(c, color='g', linestyle='--', label="c (Right Bound)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage with different parameter values
plot_pi_mf(a=-3, c=3)
plot_pi_mf(a=0, c=5)
