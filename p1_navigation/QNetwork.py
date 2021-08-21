import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, layers, seed, dropout=0.2, use_l_relu=False):
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

        self.tmp_layers = [nn.Linear(state_size, layers[0])]
        self._add_activation_function(use_l_relu, self.tmp_layers)

        for i, _ in enumerate(layers):
            if i < len(layers) -1:
                self.tmp_layers.append(nn.Linear(layers[i], layers[i+1]))
                self._add_activation_function(use_l_relu, self.tmp_layers)
            else:
                self.tmp_layers.append(nn.Linear(layers[i], action_size))

        self.layers = nn.Sequential(*self.tmp_layers)
        self.droput = nn.Dropout(self.drop_rate)


    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = torch.sigmoid(self.layers(state))       
        return x

    def _add_activation_function(self, use_l_relu, layers):
        if use_l_relu:
            self.tmp_layers.append(nn.LeakyReLU())
        else:
            self.tmp_layers.append(nn.ReLU())

    def print_parameters(self, episode):
        with open(str(f'model_weights_{episode}.txt'), 'a') as fp:
            for name, param in self.named_parameters():
                print(f'Name: {name}\tParam: {param}', file=fp)
            print('{}'.format('='*40), file=fp)