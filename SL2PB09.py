import numpy as np

# Define sigmoid activation function
def sigmoid(x):
    # Sigmoid function returns values between 0 and 1, which is used in the output layer
    return 1 / (1 + np.exp(-x))

# Define derivative of sigmoid function
def sigmoid_derivative(x):
    # Derivative of sigmoid function, used for backpropagation during training
    return x * (1 - x)

# Define input dataset (XOR problem)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Define expected output dataset (XOR labels)
y = np.array([[0], [1], [1], [0]])

# Define hyperparameters
learning_rate = 0.01   # Controls the step size for weight updates
num_epochs = 500000    # Number of iterations for training

# Initialize weights randomly with mean 0
# Hidden layer weights: 2 input neurons -> 4 hidden neurons
hidden_weights = 2 * np.random.random((2, 4)) - 1
# Output layer weights: 4 hidden neurons -> 1 output neuron
output_weights = 2 * np.random.random((4, 1)) - 1

# Train the neural network
for i in range(num_epochs):
    # Forward propagation:
    # Calculate activations for the hidden layer by multiplying inputs with hidden layer weights
    hidden_layer = sigmoid(np.dot(X, hidden_weights))
    # Calculate activations for the output layer by multiplying hidden layer output with output layer weights
    output_layer = sigmoid(np.dot(hidden_layer, output_weights))
    
    # Backpropagation:
    # Calculate the error in the output layer (difference between expected and predicted output)
    output_error = y - output_layer
    # Calculate the delta (error term) for the output layer
    output_delta = output_error * sigmoid_derivative(output_layer)
    
    # Calculate the error in the hidden layer by backpropagating the error from the output layer
    hidden_error = output_delta.dot(output_weights.T)
    # Calculate the delta (error term) for the hidden layer
    hidden_delta = hidden_error * sigmoid_derivative(hidden_layer)
    
    # Update weights using the calculated deltas (adjust weights to minimize error)
    output_weights += hidden_layer.T.dot(output_delta) * learning_rate
    hidden_weights += X.T.dot(hidden_delta) * learning_rate

# Display input and output after training
print("Input:")
print(X)
print("Output (Predictions after training):")
# Round the output to get binary predictions (0 or 1), since XOR outputs are binary
print(np.round(output_layer))
