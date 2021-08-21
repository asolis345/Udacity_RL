import numpy as np
import os
import random
from collections import namedtuple, deque

from QNetwork import QNetwork
from ReplayBuffer import ReplayBuffer
from Constants import (BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR, UPDATE_EVERY,
                       EXPLORATION, EXPLORATION_DEC, EXPLORATION_MIN, EPISODES)

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, layers, seed, dropout=0.2, use_l_relu=False):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state, used as input to ANN
            action_size (int): dimension of each action, used as output to ANN
            layers (list): units for each hidden layer
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.layers = layers
        self.seed = seed
        self.drop_out = dropout
        self.use_l_relu = use_l_relu

        # Q-Network
        self.qnetwork_local = QNetwork(self.state_size, self.action_size, self.layers,
                                       self.seed, self.drop_out, self.use_l_relu).to(device)
        self.qnetwork_target = QNetwork(self.state_size, self.action_size, self.layers,
                                       self.seed, self.drop_out, self.use_l_relu).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # self.epsilon_list = self._calculate_epsilon_exp_decay_table(EXPLORATION, EXPLORATION_MIN)
        self.epsilon_list = self._calculate_epsilon_linear_decay_table(EXPLORATION, EXPLORATION_DEC, EXPLORATION_MIN)
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def act(self, state, episode_i):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        eps = self.epsilon_list[episode_i]
        ran = random.random()
        if ran > eps:
            return int(np.argmax(action_values.cpu().data.numpy()))
        else:
            return int(random.choice(np.arange(self.action_size)))

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        Q_target_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_target_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def _calculate_epsilon_linear_decay_table(self, epsilon, decay_rate, min_epsilon):
        """Calculates epsilon in a linear decay manner, sets min value for epsilon once reached

        Args:
            epsilon (float): starting epsilon
            decay_rate (float): linear decay rate
            min_epsilon (float): minimum allowable epsilon value

        Returns:
            np.array: array of epsilons at given episode
        """
        epsilon_list = np.ndarray((EPISODES))
        for i, _ in enumerate(epsilon_list):
            epsilon_list[i] = max(epsilon + (-decay_rate * i), min_epsilon)
     
        # with open('epsilon_list.txt', 'w') as fp:
        #     for e in epsilon_list:
        #         print(e, file=fp)

        return epsilon_list

    def save_q_networks(self):
        torch.save({'qnetwork_local_state_dict': 
                    self.qnetwork_local.state_dict(),
                    'qnetwork_target_state_dict': 
                    self.qnetwork_target.state_dict(),
                    'optimizer_dict': 
                    self.optimizer.state_dict()},                    
                    os.path.join(os.getcwd(), 'Monkey_Q_Networks.pth'))
    