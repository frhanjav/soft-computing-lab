import numpy as np
import matplotlib.pyplot as plt

# Create a figure with subplots to compare both approaches
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Discrete vs Continuous Fuzzy Set Representations', fontsize=16, fontweight='bold')

# ===== DISCRETE APPROACH =====
# Define discrete fuzzy sets
fuzzy_set_A = {'x1': 0.2, 'x2': 0.7, 'x3': 1.0, 'x4': 0.4, 'x5': 0.0, 'x6': 0.6, 'x7': 0.3}
fuzzy_set_B = {'x1': 0.5, 'x2': 0.3, 'x3': 0.8, 'x4': 0.9, 'x5': 0.2, 'x6': 0.1, 'x7': 0.7}

elements = list(fuzzy_set_A.keys())
membership_A = [fuzzy_set_A[x] for x in elements]
membership_B = [fuzzy_set_B[x] for x in elements]
union_discrete = [max(fuzzy_set_A[x], fuzzy_set_B[x]) for x in elements]
intersection_discrete = [min(fuzzy_set_A[x], fuzzy_set_B[x]) for x in elements]

x_pos = np.arange(len(elements))
bar_width = 0.35

# Plot discrete sets
ax1.bar(x_pos - bar_width/2, membership_A, bar_width, label='Set A', color='skyblue', alpha=0.7)
ax1.bar(x_pos + bar_width/2, membership_B, bar_width, label='Set B', color='lightcoral', alpha=0.7)
ax1.set_title('Discrete Fuzzy Sets\n(Categorical Elements)', fontweight='bold')
ax1.set_xlabel('Elements')
ax1.set_ylabel('Membership Degree')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(elements)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot discrete operations
ax2.bar(x_pos - bar_width/2, union_discrete, bar_width, label='Union', color='gold', alpha=0.7)
ax2.bar(x_pos + bar_width/2, intersection_discrete, bar_width, label='Intersection', color='lightgreen', alpha=0.7)
ax2.set_title('Discrete Operations\n(Bar Charts)', fontweight='bold')
ax2.set_xlabel('Elements')
ax2.set_ylabel('Membership Degree')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(elements)
ax2.legend()
ax2.grid(True, alpha=0.3)

# ===== CONTINUOUS APPROACH =====
# Define continuous fuzzy sets
X = np.arange(0, 11)
mu_A = np.array([0.0, 0.2, 0.5, 0.8, 1.0, 0.7, 0.4, 0.1, 0.0, 0.0, 0.0])
mu_B = np.array([0.0, 0.0, 0.3, 0.6, 0.9, 1.0, 0.9, 0.5, 0.2, 0.0, 0.0])
mu_union = np.maximum(mu_A, mu_B)
mu_intersection = np.minimum(mu_A, mu_B)

# Plot continuous sets
ax3.plot(X, mu_A, 'o-', label='Set A', color='blue', linewidth=2, markersize=6)
ax3.plot(X, mu_B, 's--', label='Set B', color='red', linewidth=2, markersize=6)
ax3.set_title('Continuous Fuzzy Sets\n(Numerical Domain)', fontweight='bold')
ax3.set_xlabel('Universe of Discourse')
ax3.set_ylabel('Membership Degree')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xlim(-0.5, 10.5)
ax3.set_ylim(-0.1, 1.1)

# Plot continuous operations
ax4.plot(X, mu_union, 'x:', label='Union', color='orange', linewidth=3, markersize=8)
ax4.plot(X, mu_intersection, 'd-.', label='Intersection', color='green', linewidth=3, markersize=6)
ax4.set_title('Continuous Operations\n(Line Plots)', fontweight='bold')
ax4.set_xlabel('Universe of Discourse')
ax4.set_ylabel('Membership Degree')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xlim(-0.5, 10.5)
ax4.set_ylim(-0.1, 1.1)

plt.tight_layout()
plt.show()

# Print comparison
print("COMPARISON OF APPROACHES:")
print("=" * 50)
print("\nDISCRETE APPROACH (Your first code):")
print("- Elements: Named categories (x1, x2, x3, etc.)")
print("- Use case: Qualitative data (e.g., 'tall', 'medium', 'short')")
print("- Visualization: Bar charts")
print("- Example: Survey responses, quality ratings")

print("\nCONTINUOUS APPROACH (Second code):")
print("- Elements: Numerical values (0, 1, 2, ..., 10)")
print("- Use case: Quantitative data (e.g., temperature, height, speed)")
print("- Visualization: Line plots")
print("- Example: Temperature control, image processing")

print("\nBOTH APPROACHES:")
print("- Use max() for union operations")
print("- Use min() for intersection operations")
print("- Follow the same fuzzy logic principles")