import numpy as np

class HopfieldNetwork:
    def __init__(self, n_neurons):
        # Initialize the number of neurons and the weight matrix
        self.n_neurons = n_neurons
        # Weight matrix initialized to zeros (n_neurons x n_neurons)
        self.weights = np.zeros((n_neurons, n_neurons))

    def train(self, patterns):
        # Train the network using a set of patterns
        for pattern in patterns:
            # Update the weight matrix using the outer product of the pattern with itself
            self.weights += np.outer(pattern, pattern)
        # Normalize the weight matrix by dividing by the number of neurons
        self.weights /= self.n_neurons
        # Set the diagonal elements to 0 (no self-connections)
        np.fill_diagonal(self.weights, 0)

    def predict(self, pattern, max_iterations=10):
        # Predict the pattern based on the trained weights
        pattern = pattern.copy()  # Make a copy of the input pattern to avoid modifying it
        for _ in range(max_iterations):
            # Calculate the new pattern by applying the sign function to the dot product
            new_pattern = np.sign(np.dot(self.weights, pattern))
            # If the value is zero, set it to 1 (because Hopfield networks use bipolar values: 1 or -1)
            new_pattern[new_pattern == 0] = 1  
            # If the new pattern is equal to the old pattern, convergence is reached, break the loop
            if np.array_equal(new_pattern, pattern):
                break
            pattern = new_pattern  # Update the pattern for the next iteration
        return pattern

# Main execution
if __name__ == '__main__':
    # Define training patterns (each pattern is a 1D array of -1 and 1 values)
    patterns = np.array([
        [1, 1, -1, -1],    # Pattern 1
        [-1, -1, 1, 1],    # Pattern 2
        [1, -1, 1, -1],    # Pattern 3
        [-1, 1, -1, 1]     # Pattern 4
    ])

    # Number of neurons is the length of each pattern
    n_neurons = patterns.shape[1]
    # Create a Hopfield Network with the specified number of neurons
    network = HopfieldNetwork(n_neurons)
    # Train the network with the given patterns
    network.train(patterns)

    # Display the trained patterns and their predictions
    print("=== Trained Patterns ===")
    for pattern in patterns:
        # For each input pattern, predict the output using the trained network
        prediction = network.predict(pattern)
        # Print the input and predicted patterns
        print("Input pattern:     ", pattern)
        print("Predicted pattern: ", prediction)
        print()

    # Test with a noisy version of a pattern (one bit is flipped)
    noisy_input = np.array([1, 1, -1, 1])  # Flip the last bit
    # Recover the pattern from the noisy input
    recovered = network.predict(noisy_input)

    # Display the noisy input and the recovered pattern
    print("=== Noisy Pattern Test ===")
    print("Noisy input:        ", noisy_input)
    print("Recovered pattern:  ", recovered)
