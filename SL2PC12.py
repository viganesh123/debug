import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess the data: reshape to 4D array (samples, height, width, channels) and normalize
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0  # Reshape and normalize the training images
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0  # Reshape and normalize the test images

# Convert labels to one-hot encoding
y_train = to_categorical(y_train)  # One-hot encoding for training labels
y_test = to_categorical(y_test)  # One-hot encoding for test labels

# Define the Convolutional Neural Network (CNN) model
model = Sequential([
    # First Convolutional layer with 32 filters and 3x3 kernel size
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),  # MaxPooling layer with 2x2 pool size
    
    # Second Convolutional layer with 64 filters and 3x3 kernel size
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),  # MaxPooling layer with 2x2 pool size
    
    # Third Convolutional layer with 64 filters and 3x3 kernel size
    Conv2D(64, (3, 3), activation='relu'),
    
    Flatten(),  # Flatten the 3D tensor to 1D to feed into Dense layer
    
    # Fully connected layer with 64 units and ReLU activation
    Dense(64, activation='relu'),
    
    # Output layer with 10 units for classification (10 classes) and softmax activation
    Dense(10, activation='softmax')
])

# Compile the model with Adam optimizer and categorical crossentropy loss
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model on the training data for 10 epochs with a batch size of 64
model.fit(X_train, y_train, batch_size=64, epochs=10, verbose=1)

# Evaluate the model on the test data to get the final test loss and accuracy
loss, accuracy = model.evaluate(X_test, y_test)

# Print the final loss and accuracy of the model on the test data
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
