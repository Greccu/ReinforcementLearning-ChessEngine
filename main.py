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

env = gym.make('Chess-v0')

# chess-v0 usage
# board,a,b,info = env.step(env.legal_moves[0])
# print(board.turn)

# print(env.step(env.legal_moves[0]))
# print(env.step(env.legal_moves[0]))
# print(env.step(env.legal_moves[0]))

# board,a,b,info = env.step(env.legal_moves[0])

# VERIFICA SA AIBA SENS CE GENEREZI TU AICI IN MONTE CARLO SEARCH TREE

# PAS CU PAS

exploration_constant = 2

def ucb1(curr_node):
    return curr_node.exploitation+exploration_constant*(sqrt(log(curr_node.parentVisits+e+(10**-6))/(curr_node.visits+(10**-10))))

#choose best child
def selection(curr_node):
    children_ucbs = [ucb1(child) for child in curr_node.children]
    sel_node = None
    sel_node = list(curr_node.children)[np.argmax(children_ucbs)]
    return sel_node

#use to teach
def rollout(curr_node):
    
    if(curr_node.state.is_game_over()):
        # print(curr_node.white)
        # print(curr_node.state)
        # print(curr_node.state.result())

        board = curr_node.state
        if (curr_node.white):
            if(board.result()=='1-0'):
                return (-1,curr_node)
            elif(board.result()=='0-1'):
                return (1,curr_node)
            else:
                return (0.5,curr_node)
        else:
            if(board.result()=='1-0'):
                return (1,curr_node)
            elif(board.result()=='0-1'):
                return (-1,curr_node)
            else:
                return (0.5,curr_node)

    # verifica ca se genereaza ok si ca are sens
    all_moves = [curr_node.state.san(i) for i in list(curr_node.state.legal_moves)]
    
    # generaza toate miscarile posibile
    for i in all_moves:
        tmp_state = chess.Board(curr_node.state.fen())
        tmp_state.push_san(i)
        child = node()
        child.white = curr_node.white
        child.state = tmp_state
        child.parent = curr_node
        curr_node.children.add(child)

    # alege miscare random
    rnd_state = random.choice(list(curr_node.children))

    return rollout(rnd_state)


def expand(curr_node,white):
    if(len(curr_node.children)==0):
        return curr_node
    return expand(selection(curr_node),white)


def rollback(curr_node,reward):
    curr_node.visits+=1
    curr_node.exploitation+=reward
    while(curr_node.parent!=None):
        curr_node.parentVisits+=1
        curr_node = curr_node.parent
    return curr_node

def mcts_pred(curr_node,over,white,iterations=10):
    if(over):
        return -1
    all_moves = [curr_node.state.san(i) for i in list(curr_node.state.legal_moves)]
    map_state_move = dict()
    
    for move in all_moves:
        tmp_state = chess.Board(curr_node.state.fen())
        tmp_state.push_san(move)
        child = node()
        child.state = tmp_state
        child.parent = curr_node
        curr_node.children.add(child)
        map_state_move[child] = move

    #extending the best node by ucb
    while(iterations>0):
            #selecting node to expand
            sel_child = selection(curr_node)

            ex_child = expand(sel_child,white)
            reward,state = rollout(ex_child)
            curr_node = rollback(state,reward)
            iterations-=1
            
    #choosing next state
    child_list = list(curr_node.children)
    sel_move = map_state_move[child_list[np.argmax([ucb1(child) for child in child_list])]]
    return sel_move



# nod din arbore
@to_string
class node:
    def __init__(self):
        self.state = [] # board
        self.action = '' #action
        self.white = True
        self.children = set()
        self.parent = None
        self.parentVisits = 0
        self.visits = 0
        self.exploitation = 0


#init



# intelege cand se opreste?

#alb e cu litere mici

#metoda player vs bot

#player is black
def bot_vs_player():
    env.reset()
    whites_turn = True
    moves = 0
    board = chess.Board()
    terminal = False
    nodes = []
    
    while not terminal:
        #bot move
        root = node()
        root.state = board
        root.white = whites_turn

        result = mcts_pred(root,board.is_game_over(),whites_turn)
        root.action = str(board.parse_san(result))
        
        print("\n"*10)
        nodes.append(root)
        
        board,reward,terminal,info = env.step(board.parse_san(result))
        whites_turn = True
        moves+=1
        
        print(env.render())
        
        if (terminal):
            break
        #player move
        # E NEGRU
        while (True):
            move = input("Enter your move: (ex: g2g4 moves the piece from g2 to g4) \n")
            print(env.legal_moves)
            if (not chess.Move.from_uci(move) in env.legal_moves):
                move = input("Incorrect move: \n")
            else:
                break
        board,reward,terminal,info = env.step(board.parse_uci(move))
        env.render()
        moves+=1
    
        if (terminal):
            print(reward)
    

#metoda bot vs bot 
def bot_vs_bot():
    
    env.reset()
    whites_turn = True
    moves = 0
    board = chess.Board()
    terminal = False
    nodes = []
    while not terminal:
        root = node()
        root.state = board
        root.white = whites_turn

        result = mcts_pred(root,board.is_game_over(),whites_turn,iterations=1000)
        root.action = str(board.parse_san(result))
        
        print("\n"*10)
        nodes.append(root)

        # print(board.is_stalemate())
        # print(board.can_claim_fifty_moves())
        # print(board.can_claim_draw())
        
        board,reward,terminal,info = env.step(board.parse_san(result))
        whites_turn = bool(1-whites_turn)
        moves+=1
        print(env.render())
        if (terminal):
            print(reward)
    
bot_vs_bot()


#Agent

agent = MonteCarlo_BJAgent(env)
agent.mc_control_glie(n_episode=50, firstVisit=False)





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