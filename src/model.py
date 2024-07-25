import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        """
        Initializes the NeuralNet class.

        Parameters:
            input_size (int): The size of the input.
            hidden_size (int): The size of the hidden layer.
            num_classes (int): The number of classes in the output layer.
        """
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Performs the forward pass of the neural network.

        Parameters:
            x (tensor): The input tensor to the neural network.

        Returns:
            tensor: The output tensor of the neural network.
        """
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out