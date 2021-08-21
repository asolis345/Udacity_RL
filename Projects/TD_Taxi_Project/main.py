from agent import Agent
from monitor import interact
import gym
import numpy as np

env = gym.make('Taxi-v3')
agent = Agent()
with open('test.txt', 'a') as fp:
    print("Gamma: {0.gamma}\nAlpha: {0.alpha}\nEpsilon: {0.epsilon_start}\nEpsilon Min: {0.epsilon_min}".format(agent), file=fp)
    mean_best_avg_reward = 0.0
    trials = 5
    for i in range(trials):
        print("Run {}".format(i+1))
        avg_rewards, best_avg_reward = interact(env, agent)
        mean_best_avg_reward += best_avg_reward
        agent.episode_n = 0
        
    print("Best Avg. Reward: {}".format(mean_best_avg_reward / trials), file=fp)
