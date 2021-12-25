import gym
import gym_chess
import random
import time
import numpy as np
import chess.engine
import chess
from treelib import Node, Tree
from math import log,sqrt,e,inf

env = gym.make('Chess-v0')

# chess-v0 usage
# board,a,b,info = env.step(env.legal_moves[0])
# print(board.turn)

# print(env.step(env.legal_moves[0]))
# print(env.step(env.legal_moves[0]))
# print(env.step(env.legal_moves[0]))

# board,a,b,info = env.step(env.legal_moves[0])

#example of treelib
# tree = Tree()
# tree.create_node("Harry", "harry")  # root node
# info = nodeInfo()
# print(info.state)
# info2 = nodeInfo()
# info2.state = env.observation_space
# print(info2.state)
# tree.create_node("Jane", 'jane', parent="harry", data=info)
# tree.create_node("Bill", "bill", parent="harry")
# tree.create_node("Diane", "diane", parent="jane")
# tree.create_node("Mary", "mary", parent="diane")

# node = tree.create_node("Mark", "mark", parent="jane")
# print(tree.get_node('jane').data.state)
# tree.show()

def ucb1(curr_node):
    ans = curr_node.v+2*(sqrt(log(curr_node.N+e+(10**-6))/(curr_node.n+(10**-10))))
    return ans

def rollout(curr_node):
    
    if(curr_node.state.is_game_over()):
        board = curr_node.state
        if(board.result()=='1-0'):
            #print("h1")
            return (1,curr_node)
        elif(board.result()=='0-1'):
            #print("h2")
            return (-1,curr_node)
        else:
            return (0.5,curr_node)
    
    all_moves = [curr_node.state.san(i) for i in list(curr_node.state.legal_moves)]
    
    for i in all_moves:
        tmp_state = chess.Board(curr_node.state.fen())
        tmp_state.push_san(i)
        child = nodeInfo()
        child.state = tmp_state
        child.parent = curr_node
        curr_node.children.add(child)
    rnd_state = random.choice(list(curr_node.children))

    return rollout(rnd_state)

def expand(curr_node,white):
    if(len(curr_node.children)==0):
        return curr_node
    max_ucb = -inf
    if(white):
        idx = -1
        max_ucb = -inf
        sel_child = None
        for i in curr_node.children:
            tmp = ucb1(i)
            if(tmp>max_ucb):
                idx = i
                max_ucb = tmp
                sel_child = i

        return(expand(sel_child,0))

    else:
        idx = -1
        min_ucb = inf
        sel_child = None
        for i in curr_node.children:
            tmp = ucb1(i)
            if(tmp<min_ucb):
                idx = i
                min_ucb = tmp
                sel_child = i

        return expand(sel_child,1)

def rollback(curr_node,reward):
    curr_node.n+=1
    curr_node.v+=reward
    while(curr_node.parent!=None):
        curr_node.N+=1
        curr_node = curr_node.parent
    return curr_node

def mcts_pred(curr_node,over,white,iterations=10):
    if(over):
        return -1
    all_moves = [curr_node.state.san(i) for i in list(curr_node.state.legal_moves)]
    map_state_move = dict()
    
    for i in all_moves:
        tmp_state = chess.Board(curr_node.state.fen())
        tmp_state.push_san(i)
        child = nodeInfo()
        child.state = tmp_state
        child.parent = curr_node
        curr_node.children.add(child)
        map_state_move[child] = i
    while(iterations>0):
        if(white):
            idx = -1
            max_ucb = -inf
            sel_child = None
            for i in curr_node.children:
                tmp = ucb1(i)
                if(tmp>max_ucb):
                    idx = i
                    max_ucb = tmp
                    sel_child = i
            ex_child = expand(sel_child,0)
            reward,state = rollout(ex_child)
            curr_node = rollback(state,reward)
            iterations-=1
        else:
            idx = -1
            min_ucb = inf
            sel_child = None
            for i in curr_node.children:
                tmp = ucb1(i)
                if(tmp<min_ucb):
                    idx = i
                    min_ucb = tmp
                    sel_child = i

            ex_child = expand(sel_child,1)

            reward,state = rollout(ex_child)

            curr_node = rollback(state,reward)
            iterations-=1
    if(white):
        
        mx = -inf
        idx = -1
        selected_move = ''
        for i in (curr_node.children):
            tmp = ucb1(i)
            if(tmp>mx):
                mx = tmp
                selected_move = map_state_move[i]
        return selected_move
    else:
        mn = inf
        idx = -1
        selected_move = ''
        for i in (curr_node.children):
            tmp = ucb1(i)
            if(tmp<mn):
                mn = tmp
                selected_move = map_state_move[i]
        return selected_move

#final cod copiat


# nod din arbore
class nodeInfo:
    def __init__(self):
        self.state = []
        self.action = ''
        self.children = set()
        self.parent = None
        self.N = 0
        self.n = 0
        self.v = 0


#init
env.reset()
whites_turn = True
moves = 0
pgn = []
evaluations = []
sm = 0
cnt = 0
board = chess.Board()
terminal = False

#cod scris de mine
while not terminal:
    all_moves = env.legal_moves
    root = nodeInfo()
    root.state = board

    result = mcts_pred(root,board.is_game_over(),whites_turn)
    print(env.render())
    print("\n"*10)
    
    board,reward,terminal,info = env.step(board.parse_san(result))
    whites_turn = 1-whites_turn
    moves+=1
    






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