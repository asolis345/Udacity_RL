import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.n_episodes = 20000
        self.episode_n = 0
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = 0.7
        self.gamma = 0.2
        self.epsilon_start =  0.9
        self.epsilon_decay = self.epsilon_start / self.n_episodes
        self.epsilon_min = 1e-06
        self.epsilon_list = self.calculate_epsilon_linear_decay_table(self.epsilon_start,
                                                                self.epsilon_decay,
                                                                self.epsilon_min)
        # self.epsilon_list = self.calculate_epsilon_exp_decay_table(self.epsilon_start,
        #                                                         self.epsilon_min)

    def calculate_epsilon_linear_decay_table(self, epsilon, decay_rate, min_epsilon):
        """Calculates epsilon in a linear decay manner, sets min value for epsilon once reached

        Args:
            epsilon (float): starting epsilon
            decay_rate (float): linear decay rate
            min_epsilon (float): minimum allowable epsilon value

        Returns:
            np.array: array of epsilons at given episode
        """
        epsilon_list = np.ndarray((self.n_episodes))
        for i in range(len(epsilon_list)):
            epsilon_list[i] = min(epsilon + (-decay_rate * i), min_epsilon)
            
        return epsilon_list
    
    def calculate_epsilon_exp_decay_table(self, epsilon, min_epsilon):
        """Calculates epsilon in a exponential decay manner, sets min value for epsilon once reached

        Args:
            epsilon (float): starting epsilon
            min_epsilon (float): minimum allowable epsilon value

        Returns:
            np.array: array of epsilons at given episode
        """
        epsilon_list = np.ndarray((self.n_episodes))
        for i in range(1, len(epsilon_list) + 1):
            epsilon_list[i - 1] = min(epsilon / (epsilon * i), min_epsilon)
            
        return epsilon_list
        
    
    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment
        - episode: the current episode number, need to subtract for list indexing

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        p = np.random.sample()
        action = np.argmax(self.Q[state])
        if p < self.epsilon_list[self.episode_n]:
            action = np.random.randint(0, self.nA)
        return action 

    def step(self, state, action, reward, next_state, done):
        """ Using Q-Learning (SARSA Max), update the agent's knowledge, using the 
        most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        next_action = np.argmax(self.Q[next_state])
        self.Q[state][action] = (self.Q[state][action]) + (self.alpha * reward + \
            (self.gamma * self.Q[next_state][next_action] - self.Q[state][action]))
        if done:
            self.episode_n += 1
        
