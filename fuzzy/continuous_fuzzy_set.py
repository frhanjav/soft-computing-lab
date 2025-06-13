import numpy as np
import matplotlib.pyplot as plt

# Define the domain (avoiding 0 to prevent division by zero)
# Include both positive and negative values
x_pos = np.linspace(0.1, 10, 250)
x_neg = np.linspace(-10, -0.1, 250)
x = np.concatenate([x_neg, x_pos])

# Define fuzzy membership functions using lambda
mu1 = lambda x: 1 / (x ** 2)
mu2 = lambda x: 1 / abs(x)  # Use abs(x) to handle negative values

# Evaluate membership values over the domain
mu1_values = mu1(x)
mu2_values = mu2(x)

# Compute union and intersection
mu_union = np.maximum(mu1_values, mu2_values)
mu_intersection = np.minimum(mu1_values, mu2_values)

# Plot all functions
plt.figure(figsize=(12, 8))
plt.plot(x, mu1_values, label=r'$\mu_1(x) = \frac{1}{x^2}$', linewidth=2, color='blue')
plt.plot(x, mu2_values, label=r'$\mu_2(x) = \frac{1}{|x|}$', linewidth=2, color='red')
plt.plot(x, mu_union, label=r'Union: $\max(\mu_1, \mu_2)$', linestyle='--', linewidth=2, color='green')
plt.plot(x, mu_intersection, label=r'Intersection: $\min(\mu_1, \mu_2)$', linestyle='--', linewidth=2, color='purple')

plt.title('Continuous Fuzzy Sets: μ₁(x), μ₂(x), Union, and Intersection')
plt.xlabel('x')
plt.ylabel('Membership Degree')
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim(0, 2)  # Adjusted to show more detail
plt.axvline(x=0, color='black', linestyle=':', alpha=0.5)  # Reference line at x=0
plt.tight_layout()
plt.show()

# Print some sample values to verify the operations
print("Sample membership values:")
print("-" * 40)
test_values = [-2, -1, -0.5, 0.5, 1, 2]

for val in test_values:
    mu1_val = mu1(val)
    mu2_val = mu2(val)
    union_val = max(mu1_val, mu2_val)
    intersection_val = min(mu1_val, mu2_val)
    
    print(f"x = {val:4.1f}: μ₁ = {mu1_val:.3f}, μ₂ = {mu2_val:.3f}, "
          f"Union = {union_val:.3f}, Intersection = {intersection_val:.3f}")