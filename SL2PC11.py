import numpy as np
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import torch
import torch.nn as nn
import torch.optim as optim

# Load the breast cancer dataset
def load_data():
    data = load_breast_cancer()  # Load the breast cancer dataset from sklearn
    X = data.data  # Features
    y = data.target  # Labels
    return X, y

# Preprocess the data: scaling and splitting into train-test sets
def preprocess_data(X, y):
    scaler = StandardScaler()  # Standardize the data
    X_scaled = scaler.fit_transform(X)  # Fit and transform the features
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Train a logistic regression model using TensorFlow
def train_logistic_regression(X_train, y_train):
    logistic_model = Sequential([  # Define the model
        Dense(1, input_dim=X_train.shape[1], activation='sigmoid')  # Single layer with sigmoid activation
    ])
    logistic_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Compile the model
    # Fit the model on training data
    logistic_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    return logistic_model

# Train a simple neural network model using TensorFlow
def train_neural_network(X_train, y_train):
    nn_model = Sequential([  # Define the neural network model
        Dense(64, input_dim=X_train.shape[1], activation='relu'),  # Hidden layer with ReLU activation
        Dense(1, activation='sigmoid')  # Output layer with sigmoid activation
    ])
    nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Compile the model
    # Fit the model on training data
    nn_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    return nn_model

# Evaluate the model performance on test data
def evaluate_model(model, X_test, y_test):
    # Predict class probabilities and convert them into binary predictions
    y_pred = (model.predict(X_test) > 0.5).astype("int32").flatten()
    # Calculate accuracy, classification report, and ROC AUC score
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    return acc, report, auc

# Display the evaluation results
def display_results(acc, report, auc):
    print("Accuracy:", acc)  # Print accuracy
    print("Classification Report:\n", report)  # Print detailed classification report
    print("ROC AUC Score:", auc)  # Print ROC AUC score

# Main menu function
def main():
    X, y = load_data()  # Load data
    X_train, X_test, y_train, y_test = preprocess_data(X, y)  # Preprocess data
    
    while True:
        print("\nMenu:")  # Display menu options
        print("1. Train Logistic Regression Model")
        print("2. Train Neural Network Model")
        print("3. Exit")

        choice = input("Enter your choice: ")  # Get user input

        if choice == '1':  # If user chooses logistic regression
            logistic_model = train_logistic_regression(X_train, y_train)  # Train logistic regression model
            acc, report, auc = evaluate_model(logistic_model, X_test, y_test)  # Evaluate the model
            print("\nLogistic Regression Model:")  # Display results
            display_results(acc, report, auc)
        elif choice == '2':  # If user chooses neural network
            nn_model = train_neural_network(X_train, y_train)  # Train neural network model
            acc, report, auc = evaluate_model(nn_model, X_test, y_test)  # Evaluate the model
            print("\nNeural Network Model:")  # Display results
            display_results(acc, report, auc)
        elif choice == '3':  # Exit the program
            print("Exiting...")
            break
        else:  # If user enters an invalid option
            print("Invalid choice. Please enter a valid option.")

# Start the program execution
if __name__ == "__main__":
    main()
