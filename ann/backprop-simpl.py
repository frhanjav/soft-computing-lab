import numpy as np

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# XOR dataset
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# Network architecture
input_size = 2      # Input layer: 2 neurons
hidden_size = 4     # Hidden layer: 4 neurons  
output_size = 1     # Output layer: 1 neuron

# Hyperparameters
learning_rate = 0.5
epochs = 10000

# Initialize weights and biases
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size)   # Input to hidden weights
b1 = np.zeros((1, hidden_size))                 # Hidden layer biases
W2 = np.random.randn(hidden_size, output_size)  # Hidden to output weights
b2 = np.zeros((1, output_size))                 # Output layer biases

# Training loop
for epoch in range(epochs):
    
    # FORWARD PASS
    # Input to hidden layer
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    
    # Hidden to output layer
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    
    # BACKWARD PASS
    # Calculate error
    error = y - a2
    
    # Output layer gradients
    d_output = error * sigmoid_derivative(a2)
    d_W2 = np.dot(a1.T, d_output)
    d_b2 = np.sum(d_output, axis=0, keepdims=True)
    
    # Hidden layer gradients
    d_hidden = np.dot(d_output, W2.T) * sigmoid_derivative(a1)
    d_W1 = np.dot(X.T, d_hidden)
    d_b1 = np.sum(d_hidden, axis=0, keepdims=True)
    
    # Update weights and biases
    W2 += learning_rate * d_W2
    b2 += learning_rate * d_b2
    W1 += learning_rate * d_W1
    b1 += learning_rate * d_b1
    
    # Print progress
    if epoch % 2000 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Test the trained network
print("\nFinal Results:")
print("Input -> Target | Prediction")
print("-" * 30)
for i in range(len(X)):
    pred = a2[i][0]
    target = y[i][0]
    print(f"{X[i]} -> {target}     | {pred:.4f}")

print(f"\nFinal Loss: {np.mean(np.square(error)):.6f}")