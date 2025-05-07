# Import necessary TensorFlow and Keras modules
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

# Load the MNIST dataset
# MNIST contains 70,000 grayscale images of handwritten digits (0 to 9)
# Split into training and testing sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize the image pixel values from [0, 255] to [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Build a sequential neural network model
model = Sequential([
    # Flatten layer to convert each 28x28 image into a 1D array of 784 pixels
    Flatten(input_shape=(28, 28)), 

    # First Dense (fully connected) layer with 128 neurons and ReLU activation
    Dense(128, activation='relu'), 

    # Output Dense layer with 10 neurons (one for each digit) and softmax activation
    Dense(10, activation='softmax') 
])

# Compile the model
# - Optimizer: Adam with a learning rate of 0.001
# - Loss function: sparse_categorical_crossentropy (used for integer labels)
# - Metric: Accuracy
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model on the training data
# - Batch size: 64 images per training step
# - Epochs: 10 complete passes through the training data
# - Verbose: 1 to display training progress
model.fit(X_train, y_train, batch_size=64, epochs=10, verbose=1)

# Evaluate the trained model on the test set
# Returns the loss value and accuracy
loss, accuracy = model.evaluate(X_test, y_test)

# Print the final test loss and accuracy
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
