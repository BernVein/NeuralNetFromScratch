import os
import numpy as np
import pandas as pd
from keras.datasets import mnist

# Disable oneDNN optimizations (optional)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the MNIST dataset
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# Preprocess and store training data
train_data = []
print("Processing training data...")
for i in range(train_X.shape[0]):
    # Scale pixel values to [0, 1]
    flattened_pixels = train_X[i].flatten() / 255.0  # 784 columns for pixel values
    
    # Append pixel values to the data list (784 values)
    train_data.append(flattened_pixels.tolist())
    
    # Second row: one-hot encoded label (10 values)
    label = np.zeros(10)
    label[train_y[i]] = 1  # One-hot encoding for the label
    train_data.append(label.tolist())
    
    # Print progress
    if (i + 1) % 1000 == 0:
        print(f"Processed {i + 1}/{train_X.shape[0]} training images.")

# Convert to DataFrame and save as CSV
train_df = pd.DataFrame(train_data)
train_df.to_csv('mnist_train.csv', index=False, header=False)

print("Training data saved to mnist_train.csv.")

# Preprocess and store test data
test_data = []
print("Processing test data...")
for i in range(test_X.shape[0]):
    # Scale pixel values to [0, 1]
    flattened_pixels = test_X[i].flatten() / 255.0  # 784 columns for pixel values
    
    # Append pixel values to the data list (784 values)
    test_data.append(flattened_pixels.tolist())
    
    # Second row: one-hot encoded label (10 values)
    label = np.zeros(10)
    label[test_y[i]] = 1  # One-hot encoding for the label
    test_data.append(label.tolist())
    
    # Print progress
    if (i + 1) % 1000 == 0:
        print(f"Processed {i + 1}/{test_X.shape[0]} test images.")

# Convert to DataFrame and save as CSV
test_df = pd.DataFrame(test_data)
test_df.to_csv('mnist_test.csv', index=False, header=False)

print("Test data saved to mnist_test.csv.")
