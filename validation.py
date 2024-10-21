import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist

# Function to load the CSV data
def load_csv_data(csv_file):
    data = pd.read_csv(csv_file, header=None)
    images = data.values[::2]  # Extract pixel rows
    labels = data.values[1::2]  # Extract label rows
    return images, labels

# Load the MNIST dataset
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# Load the training and testing data from CSV
train_images, train_labels = load_csv_data('mnist_train.csv')
test_images, test_labels = load_csv_data('mnist_test.csv')

# Function to display an image and its corresponding values
def show_image(index, images, labels):
    plt.figure(figsize=(6, 6))
    
    # Reshape and display the image
    image = images[index].reshape(28, 28)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
    # Extract expected label
    expected_label = np.argmax(labels[index])

    # Display values
    plt.title(f"Index: {(index + 1) * 2}")
    plt.show()

# Main loop
while True:
    # Ask user for data type selection
    data_type = input("Type 'train' to view training images or 'test' for testing images (or 'exit' to quit): ").strip().lower()

    if data_type == 'exit':
        break
    elif data_type not in ['train', 'test']:
        print("Invalid input. Please type 'train', 'test', or 'exit'.")
        continue
    
    # Set images and labels based on selection
    if data_type == 'train':
        images, labels = train_images, train_labels
        max_index = len(train_images) - 1
    else:
        images, labels = test_images, test_labels
        max_index = len(test_images) - 1

    # Ask user for the index of the image
    while True:
        index_input = input(f"Enter an index (1 to {max_index + 1}) to see the image (or 'back' to change data type): ").strip().lower()
        
        if index_input == 'back':
            break
        
        try:
            index = int(index_input) - 1  # Adjusting for 0-based indexing
            if index < 0 or index > max_index:
                print(f"Please enter a valid index between 1 and {max_index + 1}.")
                continue
            show_image(index, images, labels)
        except ValueError:
            print("Invalid input. Please enter a valid integer index.")

print("Exiting the viewer.")
