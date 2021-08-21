import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, dropout=0.2, use_l_relu=False):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        self.drop_rate = dropout
        self.use_l_relu = use_l_relu
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_sizes = [self.state_size,
                            128, 64, 64,
                            self.action_size]
        self.fc1 = nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1])
        self.fc2 = nn.Linear(self.hidden_sizes[1], self.hidden_sizes[2])
        self.fc3 = nn.Linear(self.hidden_sizes[2], self.hidden_sizes[3])
        self.action_layer = nn.Linear(self.hidden_sizes[-2], self.hidden_sizes[-1])

#         self.layers = nn.ModuleList([self.fc1, self.fc2, self.action_layer])

        self.droput = nn.Dropout(self.drop_rate)

        if self.use_l_relu:
            # default negative slope == 0.01
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.ReLU()        

    def forward(self, state):
        """Build a network that maps state -> action values."""
        # for layer in self.layers:
        #     state = self.activation(layer(state))
        #     state = self.droput(state)
        
        x = F.relu(self.fc1(state))
        x = self.droput(x)
        x = F.relu(self.fc2(x))
        x = self.droput(x)
        x = F.relu(self.fc3(x))
        x = self.droput(x)
        x = self.action_layer(x)
        
        return x
