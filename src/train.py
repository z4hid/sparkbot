# This script trains a neural network model using PyTorch

# Import necessary libraries
import json  # Library for parsing JSON files
from utils import tokenize, bag_of_words, stem  # Functions for tokenizing and stemming words
import numpy as np  # Library for numerical computing

import torch  # Library for deep learning
import torch.nn as nn  # Library for neural networks
from torch.utils.data import Dataset, DataLoader  # Library for creating datasets and dataloaders

from model import NeuralNet  # Custom neural network class

# Load the data from the JSON file
with open("./data/intents.json") as file:
    data = json.load(file)  # Load the JSON data into a Python dictionary

# Prepare the data for training
# Extract all words and tags from the data
all_words = []  # List to store all words
tags = []  # List to store all tags
xy = []  # List to store both patterns and texts

# Iterate over the intents in the data
for intent in data['intents']:
    tag = intent["tag"]  # Get the tag for the intent
    tags.append(tag)  # Add the tag to the list of tags
    
    # Iterate over the patterns in the intent
    for pattern in intent["patterns"]:
        w = tokenize(pattern)  # Tokenize the pattern into individual words
        all_words.extend(w)  # Add the words to the list of all words
        xy.append((w, tag))  # Add the pattern and tag to the list of patterns and texts

# Remove common punctuation and convert words to their stem form
ignore_words = ["?", "!", ".", ","]  # List of punctuation to ignore
all_words = [stem(w) for w in all_words if w not in ignore_words]  # Stem the words and remove ignored words
all_words = sorted(set(all_words))  # Remove duplicates and sort the words
tags = sorted(set(tags))  # Remove duplicates and sort the tags

# Convert the patterns and tags into numerical representations
X_train = []  # List to store the bag-of-words representations of the patterns
y_train = []  # List to store the numerical labels for the patterns

# Iterate over the patterns and tags
for (pattern_sentence, tag) in xy:
    # Convert the pattern sentence into a bag-of-words representation
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)  # Add the bag-of-words representation to the list of patterns
    
    # Get the index of the tag in the tags list
    label = tags.index(tag)
    y_train.append(label)  # Add the label to the list of labels

# Convert the lists into numpy arrays
X_train = np.array(X_train)  # Convert the list of patterns into a numpy array
y_train = np.array(y_train)  # Convert the list of labels into a numpy array

# Create a custom dataset class for the training data
class ChatDataset(Dataset):
    def __init__(self):
        """
        Initializes the ChatDataset object by setting the number of samples, X data, and y data.
        """
        self.n_samples = len(X_train)  # Number of samples in the dataset
        self.x_data = X_train  # X data, representing the bag-of-words representations of the patterns
        self.y_data = y_train  # y data, representing the labels for the patterns

    def __getitem__(self, index):
        """
        A method to get an item from the dataset based on the given index.
        
        Parameters:
            index (int): The index of the item to retrieve.
        
        Returns:
            tuple: A tuple containing the X data and the y data corresponding to the index.
        """
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        """
        A method to return the length of the dataset.
        """
        return self.n_samples

# Set hyperparameters for training
batch_size = 8  # Number of samples per batch
hideden_size = 64  # Size of the hidden layer in the neural network
input_size = len(X_train[0])  # Size of the input layer in the neural network
output_size = len(tags)  # Number of output classes in the neural network
learning_rate = 0.0001  # Learning rate for the optimizer
num_epochs = 1000  # Number of epochs to train the model for

# Create a DataLoader object for training
dataset = ChatDataset()  # Create an instance of the ChatDataset class
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)  # Create a DataLoader object for the training data

# Set the device to 'cuda' if available, otherwise use the CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create an instance of the NeuralNet model
model = NeuralNet(input_size=input_size, hidden_size=hideden_size, num_classes=output_size).to(device)  # Create an instance of the NeuralNet model and move it to the device

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Define the cross-entropy loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Define the Adam optimizer with the specified learning rate

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)  # Move the words tensor to the device
        labels = labels.to(dtype=torch.long).to(device)  # Move the labels tensor to the device
        
        # Forward pass
        outputs = model(words)  # Pass the words through the model to get the outputs
        loss = criterion(outputs, labels)  # Calculate the loss between the outputs and the labels
        
        # Backward and optimizer step
        optimizer.zero_grad()  # Zero the gradients of all model parameters
        loss.backward()  # Calculate the gradients of the loss with respect to the model parameters
        optimizer.step()  # Update the model parameters using the gradients and the optimizer
        
    if (epoch + 1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')  # Print the loss for every 100th epoch
        
# Save the trained model to a file
data = {
    'model_state': model.state_dict(),  # Save the state of the model
    'input_size': input_size,  # Save the input size of the model
    'output_size': output_size,  # Save the output size of the model
    'hidden_size': hideden_size,  # Save the hidden size of the model
    'all_words': all_words,  # Save the list of all words
    'tags': tags  # Save the list of tags
}

FILE = './models/model.pt'  # File to save the trained model to
torch.save(data, FILE)  # Save the trained model to the file

print(f'training complete. file saved to {FILE}')  # Print a message indicating that training is complete and the model has been saved

