""" 
Group A: Assignment No. 5
Assignment Title: Write a python Program for Bidirectional Associative
Memory with two pairs of vectors.

Sample input : 
Enter the input vector size: 4
Enter the output vector size: 2

Menu:
1. Train BAM
2. Recall BAM
3. Exit
Enter your choice (1/2/3): 1
Enter the number of input-output pairs: 2
Enter input pattern (space-separated values): 1 1 1 -1
Enter corresponding output pattern (space-separated values): 1 -1
Enter input pattern (space-separated values): -1 -1 1 1
Enter corresponding output pattern (space-separated values): -1 1
BAM trained successfully.

Menu:
1. Train BAM
2. Recall BAM
3. Exit
Enter your choice (1/2/3): 2
Enter input pattern to recall (space-separated values): 1 -1 -1 1
Recalled Output: [-1.  1.]

Menu:
1. Train BAM
2. Recall BAM
3. Exit
Enter your choice (1/2/3): 3
Exiting program.
"""

import numpy as np

# BAM class for training and recalling patterns
class BAM:
    def __init__(self, input_size, output_size):
        self.input_size = input_size  # Number of elements in input vector
        self.output_size = output_size  # Number of elements in output vector
        self.weights = np.zeros((input_size, output_size))  # Initialize weight matrix with zeros

    # Train the BAM with input-output pattern pairs
    def train(self, input_patterns, output_patterns):
        for x, y in zip(input_patterns, output_patterns):
            self.weights += np.outer(x, y)  # Update weights using outer product

    # Recall output pattern from input pattern
    def recall(self, input_pattern):
        output_pattern = np.dot(input_pattern, self.weights)  # Multiply input by weight matrix
        # Apply sign function (bipolar threshold)
        output_pattern[output_pattern >= 0] = 1
        output_pattern[output_pattern < 0] = -1
        return output_pattern

# Function to handle BAM training from user input
def train_bam():
    input_patterns = []
    output_patterns = []
    num_pairs = int(input("Enter the number of input-output pairs: "))
    for _ in range(num_pairs):
        # Accept input pattern and convert to NumPy array
        input_pattern = np.array([int(x) for x in input("Enter input pattern (space-separated values): ").split()])
        # Accept corresponding output pattern
        output_pattern = np.array([int(x) for x in input("Enter corresponding output pattern (space-separated values): ").split()])
        input_patterns.append(input_pattern)
        output_patterns.append(output_pattern)
    bam.train(input_patterns, output_patterns)  # Train BAM
    print("BAM trained successfully.")

# Function to handle BAM recall from user input
def recall_bam():
    # Accept input pattern for recall
    input_pattern = np.array([int(x) for x in input("Enter input pattern to recall (space-separated values): ").split()])
    recalled_output = bam.recall(input_pattern)  # Get recalled output
    print("Recalled Output:", recalled_output)

# Main execution block
if __name__ == "__main__":
    # Get sizes for input and output vectors
    input_size = int(input("Enter the input vector size: "))
    output_size = int(input("Enter the output vector size: "))

    # Create BAM instance
    bam = BAM(input_size, output_size)

    # Menu-driven loop
    while True:
        print("\nMenu:")
        print("1. Train BAM")
        print("2. Recall BAM")
        print("3. Exit")
        choice = input("Enter your choice (1/2/3): ")
        
        if choice == '1':
            train_bam()  # Train BAM
        elif choice == '2':
            recall_bam()  # Recall from BAM
        elif choice == '3':
            print("Exiting program.")
            break  # Exit the loop
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")  # Handle invalid input
