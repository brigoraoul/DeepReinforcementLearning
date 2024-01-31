import torch.nn as nn
import torch.nn.functional as F  # The 'functional' module contains loss-functions for neural networks.

from staticParameters import StaticParameters


# Class 'NeuralNet2Layer':
#   Reference: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
class NeuralNet1Layer(nn.Module):

    def __init__(self, device):
        super(NeuralNet1Layer, self).__init__()
        self.input_dim = StaticParameters.INPUT
        self.fc1_dim = StaticParameters.HIDDEN_1
        self.output_dim = StaticParameters.OUTPUT

        self.fc1 = nn.Linear(self.input_dim, self.fc1_dim, bias=True)
        self.fc2 = nn.Linear(self.fc1_dim, self.output_dim, bias=True)
        self.device = device

    # Method 'forward':
    #   Purpose: Passing samples through the neural net, to calculate an output.
    #   Parameters:
    #       'input_values': Input values for the first layer of the neural net.
    #   Return:
    #       'out': Output of the neural net of length 'self.output_dim'.
    def forward(self, input_values):
        input_values = input_values.to(self.device)
        x = F.relu(self.fc1(input_values))
        out = self.fc2(x)
        return out


# Class 'NeuralNet2Layer':
#   Reference: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
class NeuralNet2Layer(nn.Module):

    def __init__(self, device):
        super(NeuralNet2Layer, self).__init__()
        self.input_dim = StaticParameters.INPUT
        self.fc1_dim = StaticParameters.HIDDEN_1
        self.fc2_dim = StaticParameters.HIDDEN_2
        self.output_dim = StaticParameters.OUTPUT

        self.fc1 = nn.Linear(self.input_dim, self.fc1_dim, bias=True)
        self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim, bias=True)
        self.fc3 = nn.Linear(self.fc2_dim, self.output_dim, bias=True)
        self.device = device

    # Method 'forward':
    #   Purpose: Passing samples through the neural net, to calculate an output.
    #   Parameters:
    #       'input_values': Input values for the first layer of the neural net.
    #   Return:
    #       'out': Output of the neural net of length 'self.output_dim'.
    def forward(self, input_values):
        input_values = input_values.to(self.device)
        x = F.relu(self.fc1(input_values))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out
