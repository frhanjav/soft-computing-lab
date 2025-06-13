#         (h)    Sigmoid MF
import numpy as np
import matplotlib.pyplot as plt

def sigmoid_mf(x, a, c):
    """
    Compute the Sigmoid membership function.

    :param x: Input values (numpy array)
    :param a: Center of the sigmoid curve (where y = 0.5)
    :param c: Steepness of the curve (positive for increasing, negative for decreasing)
    :return: Membership values (numpy array)
    """
    return 1 / (1 + np.exp(-c * (x - a)))

def plot_sigmoid_mf(a, c, x_range=(-10, 10)):
    """
    Plot the Sigmoid membership function.

    :param a: Center of the sigmoid curve
    :param c: Steepness of the curve
    :param x_range: Range of x values for plotting
    """
    x = np.linspace(x_range[0], x_range[1], 500)
    y = sigmoid_mf(x, a, c)

    plt.figure(figsize=(7, 4))
    plt.plot(x, y, label=f'Sigmoid MF (a={a}, c={c})', color='b')
    plt.xlabel('x')
    plt.ylabel('Membership Degree')
    plt.title('Sigmoid Membership Function')
    plt.axhline(0.5, color='gray', linestyle='--', lw=0.7, label="y = 0.5")
    plt.axvline(a, color='r', linestyle='--', label="a (Center)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage with different parameter values
plot_sigmoid_mf(a=0, c=1)   # Standard increasing sigmoid
plot_sigmoid_mf(a=0, c=-1)  # Decreasing sigmoid
plot_sigmoid_mf(a=-2, c=2)  # Steeper increasing sigmoid
