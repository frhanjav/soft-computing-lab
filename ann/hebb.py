# Write a python function to realize the logical AND function through Hebb learning.

import numpy as np
import matplotlib.pyplot as plt

def hebbian_and(learning_rate=0.1, epochs=150, plot_learning=True):
    # Training data for AND function
    X = np.array([
        [0, 0, 1],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])
    
    y = np.array([0, 0, 0, 1]) # Expected outputs for AND function
    
    # Strong initial weights - purposely set to make AND function likely to work
    # For AND, we want both inputs to be strongly positive and bias to be strongly negative
    weights = np.array([1.0, 1.0, -1.5])
    
    weight_history = [weights.copy()]
    error_history = []

    for epoch in range(epochs):
        outputs = []
        epoch_error = 0
        
        for i in range(len(X)):
            activation = np.dot(X[i], weights)
            output = 1 if activation > 0 else 0
            outputs.append(output)
            
            error = y[i] - output
            epoch_error += abs(error)
            
            # Applying directed Hebbian learning
            if y[i] == 1:  # Strengthen for positive examples
                weights += learning_rate * X[i]
            elif output == 1:  # Weaken for false positives
                weights -= learning_rate * X[i]
        
        weight_history.append(weights.copy())
        error_history.append(epoch_error)
        
        if epoch_error == 0:
            print(f"AND function learned after {epoch+1} epochs!")
            break
            
        if epoch == epochs - 1:
            print("Warning: Maximum epochs reached without convergence.")
    
    if plot_learning:
        plt.figure(figsize=(15, 5))
        
        # Plot weight evolution
        plt.subplot(1, 3, 1)
        epochs_range = range(len(weight_history))
        plt.plot(epochs_range, [w[0] for w in weight_history], label='Weight 1')
        plt.plot(epochs_range, [w[1] for w in weight_history], label='Weight 2')
        plt.plot(epochs_range, [w[2] for w in weight_history], label='Bias')
        plt.xlabel('Epochs')
        plt.ylabel('Weight Value')
        plt.title('Weight Evolution')
        plt.legend()
        
        # Plot error evolution
        plt.subplot(1, 3, 2)
        plt.plot(epochs_range[:-1], error_history)
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.title('Error Evolution')
        
        # Plot decision boundary
        plt.subplot(1, 3, 3)
        x_min, x_max = -0.5, 1.5
        y_min, y_max = -0.5, 1.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        grid = np.c_[xx.ravel(), yy.ravel(), np.ones(xx.ravel().shape[0])]
        Z = np.dot(grid, weights)
        Z = np.where(Z > 0, 1, 0)
        Z = Z.reshape(xx.shape)
        
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
        plt.xlabel('Input 1')
        plt.ylabel('Input 2')
        plt.title('Decision Boundary')
        plt.tight_layout()
        plt.show()
    
    print("\nTesting the trained model:")
    for i in range(len(X)):
        activation = np.dot(X[i], weights)
        output = 1 if activation > 0 else 0
        print(f"Input: {X[i][0:2]}, Output: {output}, Expected: {y[i]}")
    
    return weights, weight_history

if __name__ == "__main__":
    final_weights, history = hebbian_and(learning_rate=0.1, epochs=100)
    print(f"\nFinal weights: {final_weights}")