import numpy as np
from collections import defaultdict
import sys
import random
from tqdm import trange

import gym
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle
from tqdm import trange
import seaborn as sns
import pandas as pd
from random import random
import abc
import warnings
warnings.filterwarnings("ignore")

class ChessAgent:
    def __init__(self):
        self.example = 0 
        
    #mcts_pred
    def select_action(self, state, epsilon):
        return
        # if epsilon != None and np.random.rand() < epsilon:
        #     action = self.env.action_space.sample()
        # else:
        #     if state in self.q:
        #         action = self.policy[state]
        #     else:
        #         action = self.env.action_space.sample()
        # return action

    #bot_vs_bot
    def run_episode(self, eps=None):
        return
        # result = []
        # state = self.env.reset()
        # while True:
        #     action = self.select_action(state, eps)
        #     next_state, reward, done, info = self.env.step(action)
        #     result.append((state, action, reward, next_state, done))
        #     state = next_state
        #     if done:
        #         break
        # return result

    #backpropagation
    def update_policy_q(self, eps=None):
        return
        # for state, values in self.q.items():
        #     if eps != None: # e-Greedy policy updates ?
        #         if np.random.rand() < eps:
        #             self.policy[state] = self.env.action_space.sample() # sample a random action
        #         else:
        #             self.policy[state] = np.argmax(values)

        #     else: # Greedy policy updates
        #         self.policy[state] = np.argmax(values)
    
    #train
    def mc_control_glie(self, n_episode=500000, firstVisit=True, update_policy_every=1):
        return
        # for t in trange(n_episode):
        #     traversed = []
        #     # Get an epsilon for this episode - used in e-greedy policy update
        #     eps = self.get_epsilon(t)

        #     # Generate an episode following current policy
        #     transitions = self.run_episode(eps=None)

        #     # zip operation will convert transitions to list of states, list of actions, rewards etc.
        #     states, actions, rewards, next_states, dones = zip(*transitions)

        #     # If firstVisit version is used, create a table stateAction_firstVisitTime that stores
        #     # when the pair (State, action) was seen first in this episode
        #     if firstVisit == True:
        #         stateAction_firstVisitTime = {}
        #         for t, state in enumerate(states):
        #             stateAction_t = (state, actions[t])
        #             if stateAction_t not in stateAction_firstVisitTime:
        #                 stateAction_firstVisitTime[stateAction_t] = t

        #     # Iterate over episode steps in reversed order, T-1, T-2, ....0
        #     G = 0 # return output
        #     for t in range(len(transitions)-1, -1, -1):
        #         St = states[t]
        #         At = actions[t]

        #         if firstVisit == True:
        #             # Is t first time when we see the (state, action) pair ?
        #             if stateAction_firstVisitTime[(St, At)] < t:
        #                 continue

        #         G = self.gamma * G + rewards[t]


        #         self.n_q[St][At] += 1
        #         alpha = (1.0 / self.n_q[St][At]) # Remember that with this formula all episodes experiences have equal importance, if you want to forget older episode put a bigger probability
        #         self.q[St][At] = self.q[St][At] + alpha*(G - self.q[St][At])

        #     if t % update_policy_every == 0:
        #         self.update_policy_q(eps)

        # # final policy update at the end
        # self.update_policy_q(eps)
