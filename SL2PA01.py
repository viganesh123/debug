import numpy as np
import matplotlib.pyplot as plt

# Define the activation functions

# Sigmoid Activation Function: Maps input to the range (0, 1)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Hyperbolic Tangent Function (tanh): Maps input to the range (-1, 1)
def tanh(x):
    return np.tanh(x)

# ReLU (Rectified Linear Unit): Outputs the input if it's positive, else zero
def relu(x):
    return np.maximum(0, x)

# Leaky ReLU: Similar to ReLU but allows small negative slope for negative values
def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)

# PReLU (Parametric ReLU): Similar to Leaky ReLU but alpha is a parameter that can be learned
def prelu(x, alpha):
    return np.maximum(alpha * x, x)

# ELU (Exponential Linear Unit): Outputs negative values exponentially for negative inputs, and input for positive
def elu(x, alpha=1.0):
    return np.where(x < 0, alpha * (np.exp(x) - 1), x)

# Softmax Activation Function: Normalizes the input into a probability distribution
def softmax(x):
    exp_values = np.exp(x - np.max(x))  # for numerical stability
    return exp_values / np.sum(exp_values, axis=0, keepdims=True)

# Linear Activation Function: Output equals input (no transformation)
def linear(x):
    return x

# Define input values for plotting the activation functions
x_values = np.linspace(-5, 5, 100)

# Create a figure for plotting
plt.figure(figsize=(15, 10))

# Plot each activation function in its own subplot

# Sigmoid Function
plt.subplot(2, 4, 1)
plt.title('Sigmoid Function')
plt.plot(x_values, sigmoid(x_values))
plt.grid(True)

# Hyperbolic Tangent Function (tanh)
plt.subplot(2, 4, 2)
plt.title('Hyperbolic Tangent Function (tanh)')
plt.plot(x_values, tanh(x_values))
plt.grid(True)

# ReLU (Rectified Linear Unit)
plt.subplot(2, 4, 3)
plt.title('ReLU (Rectified Linear Unit)')
plt.plot(x_values, relu(x_values))
plt.grid(True)

# Leaky ReLU
plt.subplot(2, 4, 4)
plt.title('Leaky ReLU')
plt.plot(x_values, leaky_relu(x_values))
plt.grid(True)

# PReLU (Parametric ReLU) with alpha = 0.01
plt.subplot(2, 4, 5)
plt.title('PReLU (Parametric ReLU)')
alpha_prelu = 0.01
plt.plot(x_values, prelu(x_values, alpha_prelu))
plt.grid(True)

# ELU (Exponential Linear Unit)
plt.subplot(2, 4, 6)
plt.title('ELU (Exponential Linear Unit)')
plt.plot(x_values, elu(x_values))
plt.grid(True)

# Softmax: For simplicity, we show a bar chart as softmax produces probabilities for a vector
plt.subplot(2, 4, 7)
plt.title('Softmax')
input_softmax = np.array([2.0, 1.0, 0.1])  # Example input for softmax
plt.bar(range(len(input_softmax)), softmax(input_softmax))
plt.grid(True)

# Linear Activation Function
plt.subplot(2, 4, 8)
plt.title('Linear')
plt.plot(x_values, linear(x_values))
plt.grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plots
plt.show()
