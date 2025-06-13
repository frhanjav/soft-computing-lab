# Write a Python program to implement backpropagation with:
# - Input layer
# - At least one hidden layer
# - Output layer

import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        """
        Initialize neural network with random weights and biases
        
        Args:
            input_size: Number of input neurons
            hidden_size: Number of hidden neurons
            output_size: Number of output neurons
            learning_rate: Learning rate for gradient descent
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights with small random values
        # Weights from input to hidden layer
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.5
        # Weights from hidden to output layer
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.5
        
        # Initialize biases
        self.b1 = np.zeros((1, self.hidden_size))
        self.b2 = np.zeros((1, self.output_size))
        
        # Store training history
        self.loss_history = []
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        # Clip x to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        return x * (1 - x)
    
    def forward_propagation(self, X):
        """
        Forward pass through the network
        
        Args:
            X: Input data (samples x features)
        
        Returns:
            Output of the network
        """
        # Input to hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Hidden to output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
    
    def backward_propagation(self, X, y, output):
        """
        Backward pass to compute gradients and update weights
        
        Args:
            X: Input data
            y: True labels
            output: Network output from forward pass
        """
        m = X.shape[0]  # Number of samples
        
        # Calculate output layer error
        output_error = output - y
        output_delta = output_error * self.sigmoid_derivative(output)
        
        # Calculate hidden layer error
        hidden_error = output_delta.dot(self.W2.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.a1)
        
        # Update weights and biases
        # Output layer weights and biases
        self.W2 -= self.learning_rate * self.a1.T.dot(output_delta) / m
        self.b2 -= self.learning_rate * np.sum(output_delta, axis=0, keepdims=True) / m
        
        # Hidden layer weights and biases
        self.W1 -= self.learning_rate * X.T.dot(hidden_delta) / m
        self.b1 -= self.learning_rate * np.sum(hidden_delta, axis=0, keepdims=True) / m
    
    def compute_loss(self, y_true, y_pred):
        """Compute mean squared error loss"""
        return np.mean((y_true - y_pred) ** 2)
    
    def train(self, X, y, epochs=1000, print_every=100):
        """
        Train the neural network
        
        Args:
            X: Training input data
            y: Training labels
            epochs: Number of training epochs
            print_every: Print loss every N epochs
        """
        for epoch in range(epochs):
            # Forward propagation
            output = self.forward_propagation(X)
            
            # Compute loss
            loss = self.compute_loss(y, output)
            self.loss_history.append(loss)
            
            # Backward propagation
            self.backward_propagation(X, y, output)
            
            # Print progress
            if epoch % print_every == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
    
    def predict(self, X):
        """Make predictions on new data"""
        return self.forward_propagation(X)
    
    def plot_loss(self):
        """Plot training loss over time"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history)
        plt.title('Training Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()

# Example usage and testing
def create_xor_dataset():
    """Create XOR dataset for testing"""
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([[0],
                  [1],
                  [1],
                  [0]])
    return X, y

def create_classification_dataset():
    """Create a simple 2D classification dataset"""
    np.random.seed(42)
    
    # Generate two classes of data
    class1 = np.random.randn(50, 2) + np.array([2, 2])
    class2 = np.random.randn(50, 2) + np.array([-2, -2])
    
    X = np.vstack([class1, class2])
    y = np.vstack([np.ones((50, 1)), np.zeros((50, 1))])
    
    return X, y

def main():
    print("Neural Network with Backpropagation Demo")
    print("=" * 50)
    
    # Example 1: XOR Problem
    print("\n1. Training on XOR Dataset:")
    print("-" * 30)
    
    X_xor, y_xor = create_xor_dataset()
    
    # Create and train network
    nn_xor = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, learning_rate=1.0)
    nn_xor.train(X_xor, y_xor, epochs=5000, print_every=1000)
    
    # Test predictions
    print("\nXOR Predictions:")
    predictions = nn_xor.predict(X_xor)
    for i, (input_val, target, pred) in enumerate(zip(X_xor, y_xor, predictions)):
        print(f"Input: {input_val}, Target: {target[0]:.0f}, Prediction: {pred[0]:.4f}")
    
    # Example 2: 2D Classification
    print("\n\n2. Training on 2D Classification Dataset:")
    print("-" * 40)
    
    X_class, y_class = create_classification_dataset()
    
    # Create and train network
    nn_class = NeuralNetwork(input_size=2, hidden_size=8, output_size=1, learning_rate=0.5)
    nn_class.train(X_class, y_class, epochs=2000, print_every=400)
    
    # Test accuracy
    predictions = nn_class.predict(X_class)
    predicted_classes = (predictions > 0.5).astype(int)
    accuracy = np.mean(predicted_classes == y_class)
    print(f"\nClassification Accuracy: {accuracy:.2%}")
    
    # Plot loss curves
    print("\nPlotting training loss curves...")
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(nn_xor.loss_history)
    plt.title('XOR Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(nn_class.loss_history)
    plt.title('Classification Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()