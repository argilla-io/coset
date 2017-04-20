# Now we can start defining our predictive model. The first step is to define the 'architecture' of the model
# and its main operations with the data that goes through the network.

# The core neural network components of Pytorch belong to the nn module
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

# Let's start with a very simple baseline model
class BaseClassifier(nn.Module):

    def __init__(self, input_dim, hidden_size, output_dim, batch_size=32, debug=None):
        super(BaseClassifier, self).__init__()
        self.embed = nn.Embedding(input_dim, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size*2)
        self.fc2 = nn.Linear(hidden_size*2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_dim)
        self.debug = debug

    def forward(self, input):
        # The forward pass defines how the input data is processed by the network
        # to make a prediction
        embed = self.embed(input)
        # This operation summarizes a 3D tensor 200x32x200 into a 32x200 matrix
        out = F.max_pool1d(embed.transpose(0,2), input.size()[0]).squeeze().transpose(0,1)
        out = F.relu(self.fc1(out))
        out = F.dropout(out, training=self.training, p=0.5)
        out = F.relu(self.fc2(out))
        out = F.dropout(out, training=self.training, p=0.5)
        out = self.fc3(out)
        return out
