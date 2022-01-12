import gym
import gym_chess
import random
import time
import numpy as np
import chess.engine
import chess
from treelib import Node, Tree
from math import log,sqrt,e,inf
from paprika import *

from ChessAgent import ChessAgent

env = gym.make('Chess-v0')


#init



# intelege cand se opreste?

#alb e cu litere mici

#metoda player vs bot

#player is black



agent = ChessAgent(env,exploration_constant=2)
print(agent.run_episode())




# environment : Chess Table
# agent : Players
# states - a lot
# actions - legal_actions, a lot, not constant
# rewards - not defined
# episodes - not defined


# docs
# https://www.geeksforgeeks.org/ml-monte-carlo-tree-search-mcts/
# https://medium.com/@ishaan.gupta0401/monte-carlo-tree-search-application-on-chess-5573fc0efb75

# implementare
# https://github.com/Ish2K/Chess-Bot-AI-Algorithms/blob/main/Git_chess/monte_carlo_implementation.py


# Other

# miscari de sah rate-uite : https://www.kaggle.com/ethanmai/chess-moves