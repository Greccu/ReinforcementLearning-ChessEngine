import gym
import gym_chess
import random
import time
import numpy as np
import chess.engine
from treelib import Node, Tree
from math import log,sqrt,e,inf

env = gym.make('Chess-v0')

env.reset()

# chess-v0 usage
print(env.legal_moves)
board,reward,done,info = env.step(env.legal_moves[0])
print(env.render())
board2,reward2,done2,info2 = env.step(env.legal_moves[0])
print(env.render())



# nod din arbore
class nodeInfo:
    def __init__(self):
        self.state = []
        self.action = ''
        self.N = 0
        self.n = 0
        self.v = 0

print(env)


#example of treelib
tree = Tree()
tree.create_node("Harry", "harry")  # root node
info = nodeInfo()
print(info.state)
info2 = nodeInfo()
info2.state = env.observation_space
print(info2.state)
tree.create_node("Jane", 'jane', parent="harry", data=info)
tree.create_node("Bill", "bill", parent="harry")
tree.create_node("Diane", "diane", parent="jane")
tree.create_node("Mary", "mary", parent="diane")

node = tree.create_node("Mark", "mark", parent="jane")
print(tree.get_node('jane').data.state)
tree.show()

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