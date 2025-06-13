import numpy as np
import matplotlib.pyplot as plt

def triangular_membership(x, a, b, c):
    """
    Triangular membership function.
    a, b, c: left, peak, right points
    """
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a)
    else:  # b < x < c
        return (c - x) / (c - b)

def find_alpha_cut(membership_func, domain, alpha):
    """
    Find the α-cut intervals where membership >= alpha
    """
    x_values = np.linspace(domain[0], domain[1], 1000)
    intervals = []
    
    in_cut = False
    start = None
    
    for x in x_values:
        membership = membership_func(x)
        
        if membership >= alpha and not in_cut:
            start = x
            in_cut = True
        elif membership < alpha and in_cut:
            intervals.append((start, x))
            in_cut = False
    
    # Handle case where cut extends to the end
    if in_cut:
        intervals.append((start, x_values[-1]))
    
    return intervals

def demonstrate_alpha_cuts():
    """Simple demonstration of α-cuts on one fuzzy set"""
    
    # Define a simple triangular fuzzy set: "around 5"
    def membership_func(x):
        return triangular_membership(x, 2, 5, 8)
    
    domain = (0, 10)
    alpha_levels = [0.3, 0.6, 0.9]
    colors = ['red', 'blue', 'green']
    
    # Create the plot
    x = np.linspace(domain[0], domain[1], 1000)
    y = [membership_func(xi) for xi in x]
    
    plt.figure(figsize=(10, 6))
    
    # Plot the membership function
    plt.plot(x, y, 'black', linewidth=3, label='Fuzzy Set: "Around 5"')
    plt.fill_between(x, y, alpha=0.2, color='gray')
    
    # Show α-cuts
    for i, alpha in enumerate(alpha_levels):
        # Find α-cut intervals
        intervals = find_alpha_cut(membership_func, domain, alpha)
        
        # Draw horizontal line at α level
        plt.axhline(y=alpha, color=colors[i], linestyle='--', 
                   label=f'α = {alpha}')
        
        # Highlight α-cut intervals
        for start, end in intervals:
            plt.axvspan(start, end, ymin=0, ymax=alpha, 
                       color=colors[i], alpha=0.4)
            
            # Print the interval
            print(f"α = {alpha}: [{start:.2f}, {end:.2f}] (length: {end-start:.2f})")
    
    plt.title('α-Cuts on a Triangular Fuzzy Set', fontsize=14, fontweight='bold')
    plt.xlabel('x')
    plt.ylabel('Membership Degree μ(x)')
    plt.ylim(0, 1.1)
    plt.xlim(domain)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    # Show what α-cuts mean
    print("\nWhat each α-cut means:")
    print("α = 0.3: 'Somewhat around 5' → crisp interval")
    print("α = 0.6: 'Moderately around 5' → crisp interval") 
    print("α = 0.9: 'Very close to 5' → crisp interval")

if __name__ == "__main__":
    demonstrate_alpha_cuts()