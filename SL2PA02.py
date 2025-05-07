import numpy as np

# Class representing McCulloch-Pitts Neural Network
class McCullochPittsNN:
    def __init__(self, num_inputs):
        # Initialize weights as zero for each input and set threshold to 0
        self.weights = np.zeros(num_inputs)
        self.threshold = 0

    def set_weights(self, weights):
        # Set the custom weights for the inputs
        if len(weights) != len(self.weights):
            raise ValueError("Number of weights must match number of inputs")
        self.weights = np.array(weights)

    def set_threshold(self, threshold):
        # Set the threshold value for the activation function
        self.threshold = threshold

    def activation_function(self, net_input):
        # Binary step activation function
        # Returns 1 if net input >= threshold, else returns 0
        return 1 if net_input >= self.threshold else 0

    def forward_pass(self, inputs):
        # Compute the net input by dot product of weights and inputs
        net_input = np.dot(inputs, self.weights)
        # Pass the result through the activation function
        return self.activation_function(net_input)

# Function to generate the ANDNOT truth table using the McCulloch-Pitts NN
def generate_ANDNOT():
    # Create a McCulloch-Pitts network with 2 inputs
    mp_nn = McCullochPittsNN(2)
    
    # Set weights for inputs: A=1, B=-1 to implement A AND NOT B
    mp_nn.set_weights([1, -1])
    
    # Set the activation threshold to 1
    mp_nn.set_threshold(1)

    # Define all input combinations for 2 inputs (truth table)
    truth_table = [(0, 0), (0, 1), (1, 0), (1, 1)]

    # Print table header
    print("Truth Table for ANDNOT Function:")
    print("Input1\tInput2\tOutput")
    
    # Compute and print output for each input pair
    for inputs in truth_table:
        output = mp_nn.forward_pass(inputs)
        print(f"{inputs[0]}\t{inputs[1]}\t{output}")

# Main function providing a menu-based interface
def main():
    while True:
        # Display menu options
        print("\nMenu:")
        print("1. Generate ANDNOT Function")
        print("2. Exit")

        # Read user choice
        choice = input("Enter your choice: ")

        # Execute based on choice
        if choice == "1":
            generate_ANDNOT()
        elif choice == "2":
            print("Exiting program...")
            break
        else:
            print("Invalid choice. Please enter a valid option.")

# Run main function if the script is executed directly
if __name__ == "__main__":
    main()
