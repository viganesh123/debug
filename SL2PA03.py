import numpy as np

# Perceptron class definition
class Perceptron:
    def __init__(self, input_size, lr=1, epochs=100):
        # Initialize weights (including bias) and set learning rate and number of epochs
        self.W = np.zeros(input_size + 1)  # Adding 1 for the bias weight
        self.epochs = epochs  # Set the number of training epochs
        self.lr = lr  # Set the learning rate
    
    # Activation function: returns 1 for positive inputs, 0 for non-positive
    def activation_fn(self, x):
        return 1 if x >= 0 else 0
    
    # Prediction method: adds bias to input and computes the weighted sum
    def predict(self, x):
        x = np.insert(x, 0, 1)  # Insert the bias input (1) at the start of the input
        z = self.W.dot(x)  # Compute the weighted sum (dot product)
        a = self.activation_fn(z)  # Apply the activation function
        return a
    
    # Training method: adjusts weights based on error for each input-output pair
    def train(self, X, labels):
        for _ in range(self.epochs):  # Loop through epochs
            for i in range(len(labels)):  # Loop through all training examples
                x = np.insert(X[i], 0, 1)  # Add bias to the input
                y_pred = self.predict(X[i])  # Get the model's prediction
                error = labels[i] - y_pred  # Calculate error
                self.W = self.W + self.lr * error * x  # Update weights

# Function to train the perceptron on data representing digits 0-9
def train_perceptron():
    # Training data: representing digits 0-9 in binary form (each digit is a 10-bit vector)
    X_train = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 0
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 3
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 4
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 5
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 6
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 7
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 8
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]   # 9
    ]
    # Labels: 0 for even, 1 for odd
    y_train = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

    # Create and train the perceptron
    perceptron = Perceptron(input_size=10)  # Initialize perceptron with 10 inputs (one for each bit)
    perceptron.train(X_train, y_train)  # Train the perceptron with the data
    print("Perceptron trained successfully!")  # Indicate training success
    return perceptron  # Return the trained perceptron model

# Function to test the trained perceptron on test data
def test_perceptron(perceptron):
    # Test cases: same 10-bit binary inputs for digits 0-9
    test_cases = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 0
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 3
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 4
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 5
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 6
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 7
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 8
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]   # 9
    ]
    
    correct_predictions = 0  # Counter for correct predictions

    # Check if predictions are correct (even or odd)
    for i, test_case in enumerate(test_cases):
        prediction = perceptron.predict(test_case)  # Get model's prediction
        if prediction == 0 and i % 2 == 0:  # Even numbers should be predicted as 0
            correct_predictions += 1
        elif prediction == 1 and i % 2 != 0:  # Odd numbers should be predicted as 1
            correct_predictions += 1
    
    # Output predictions for each number
    for i, test_case in enumerate(test_cases):
        prediction = perceptron.predict(test_case)
        if prediction == 0:
            print(f"{test_cases[i]} which is number {i} is even")  # Even number
        else:
            print(f"Number {i} is odd")  # Odd number
    
    # Calculate and print the accuracy
    accuracy = (correct_predictions / len(test_cases)) * 100
    print(f"Accuracy: {accuracy:.2f}%")  # Print accuracy of predictions

# Main function to run the program
def main():
    perceptron = None  # Initialize perceptron variable as None (not yet trained)
    
    while True:
        # Display menu options
        print("\nMENU:")
        print("1. Train Perceptron")
        print("2. Test Perceptron")
        print("3. Exit")
        choice = input("Enter your choice: ")  # Get user input
        
        if choice == '1':
            if perceptron is None:  # Train the perceptron if not already trained
                perceptron = train_perceptron()
            else:
                print("Perceptron already trained!")  # Notify if already trained
        elif choice == '2':
            if perceptron is None:
                print("Please train the perceptron first!")  # Prompt user to train if not done yet
            else:
                print("Testing the perceptron:")
                test_perceptron(perceptron)  # Test the perceptron on test data
        elif choice == '3':
            print("Exiting...")  # Exit the program
            break
        else:
            print("Invalid choice. Please enter a valid option.")  # Handle invalid input

# Check if the script is being run directly (not imported) and start the program
if __name__ == "__main__":
    main()
