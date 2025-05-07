import numpy as np
import matplotlib.pyplot as plt

# Define a simple perceptron class
class SimplePerceptron:
    def __init__(self, lr=0.1, epochs=10):
        self.lr = lr  # Learning rate
        self.epochs = epochs  # Number of training iterations

    # Training the perceptron
    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1] + 1)  # Initialize weights including bias

        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                xi = np.insert(xi, 0, 1)  # Insert bias term at position 0
                prediction = np.dot(self.weights, xi)  # Compute weighted sum
                update = self.lr * (target - np.where(prediction >= 0, 1, -1))  # Calculate update
                self.weights += update * xi  # Update weights
        return self

    # Predict class labels
    def predict(self, X):
        # Apply net input and return class label
        return np.where(np.dot(X, self.weights[1:]) + self.weights[0] >= 0, 1, -1)

# Generate synthetic 2D dataset
np.random.seed(42)
X = np.random.randn(100, 2)  # 100 samples, 2 features
y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)  # Linearly separable decision boundary

# Initialize and train perceptron
ppn = SimplePerceptron(lr=0.1, epochs=10)
ppn.fit(X, y)

# Function to visualize decision regions
def plot_regions(X, y, model):
    # Define bounds of the plot
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Predict class for each point in the mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')  # Plot data points
    plt.title('Perceptron Decision Regions')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Call the plotting function
plot_regions(X, y, ppn)
