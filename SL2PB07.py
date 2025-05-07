import numpy as np

class XORNetwork:
    def __init__(self):
        # Initialize weights and biases with random values
        self.W1 = np.random.randn(2, 2)  # Weights from input layer (2 neurons) to hidden layer (2 neurons)
        self.b1 = np.random.randn(2)     # Biases for the 2 hidden neurons
        self.W2 = np.random.randn(2, 1)  # Weights from hidden layer (2 neurons) to output layer (1 neuron)
        self.b2 = np.random.randn(1)     # Bias for the output neuron
    
    def sigmoid(self, x):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        # Derivative of sigmoid (used in backpropagation)
        return x * (1 - x)
    
    def forward(self, X):
        # Forward pass through the network
        self.z1 = np.dot(X, self.W1) + self.b1  # Linear transformation for hidden layer
        self.a1 = self.sigmoid(self.z1)         # Activation of hidden layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2  # Linear transformation for output layer
        self.a2 = self.sigmoid(self.z2)         # Activation of output layer (final prediction)
        return self.a2
    
    def backward(self, X, y, output):
        # Backward pass for adjusting weights based on error
        
        # Calculate error in output layer
        self.output_error = y - output
        # Calculate delta for output layer using sigmoid derivative
        self.output_delta = self.output_error * self.sigmoid_derivative(output)

        # Calculate error in hidden layer by backpropagating output delta
        self.z1_error = self.output_delta.dot(self.W2.T)
        # Calculate delta for hidden layer using sigmoid derivative
        self.z1_delta = self.z1_error * self.sigmoid_derivative(self.a1)

        # Update weights and biases using the calculated deltas
        self.W1 += X.T.dot(self.z1_delta)              # Update hidden layer weights
        self.b1 += np.sum(self.z1_delta, axis=0)       # Update hidden layer biases
        self.W2 += self.a1.T.dot(self.output_delta)    # Update output layer weights
        self.b2 += np.sum(self.output_delta, axis=0)   # Update output layer bias
    
    def train(self, X, y, epochs):
        # Train the network for a fixed number of epochs
        for i in range(epochs):
            output = self.forward(X)  # Perform forward pass
            self.backward(X, y, output)  # Perform backward pass and update weights
    
    def predict(self, X):
        # Predict output for new input data
        return self.forward(X)

# Create an instance of XORNetwork
xor_nn = XORNetwork()

# XOR input and output
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Inputs
y = np.array([[0], [1], [1], [0]])              # Expected outputs for XOR

# Train the network
xor_nn.train(X, y, epochs=10000)

# Predict results after training
predictions = xor_nn.predict(X)

# Print predictions
print(predictions)
