# This script trains a neural network model using PyTorch

# Import necessary libraries
import json
from utils import tokenize, stem, bag_of_words

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet

# Load the data from the JSON file
with open("./data/intents.json") as file:
    data = json.load(file)

# Prepare the data for training
# Extract all words and tags from the data
all_words = []
tags = []
xy = []  # both patterns and texts

for intent in data['intents']:
    tag = intent["tag"]
    tags.append(tag)
    for pattern in intent["patterns"]:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

# Remove common punctuation and convert words to their stem form
ignore_words = ["?", "!", ".", ","]
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Convert the patterns and tags into numerical representations
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # Convert the pattern sentence into a bag-of-words representation
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # Get the index of the tag in the tags list
    label = tags.index(tag)

    # y_train should contain a single value for each label
    y_train.append(label)  

# Convert the lists into numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# Create a custom dataset class for the training data
class ChatDataset(Dataset):
    def __init__(self):
        """
        Initializes the ChatDataset object by setting the number of samples, X data, and y data.
        """
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

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
batch_size = 8
hideden_size = 64
input_size = len(X_train[0])
output_size = len(tags)
learning_rate = 0.0001
num_epochs = 1000

# Create a DataLoader object for training
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# Set the device to 'cuda' if available, otherwise use the CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create an instance of the NeuralNet model
model = NeuralNet(input_size=input_size, hidden_size=hideden_size, num_classes=output_size).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        # Backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch + 1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')
        
# Save the trained model to a file
data = {
    'model_state': model.state_dict(),
    'input_size': input_size,
    'output_size': output_size,
    'hidden_size': hideden_size,
    'all_words': all_words,
    'tags': tags
}

FILE = './models/model.pt'
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')

