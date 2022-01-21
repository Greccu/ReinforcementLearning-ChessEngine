from time import time
import numpy as np
from collections import defaultdict
import sys
import random
import gym
import numpy as np
from collections import defaultdict
# import matplotlib.pyplot as plt
import pickle
from tqdm import trange
# import seaborn as sns
# import pandas as pd
import abc
import warnings
from math import log, sqrt, e, inf
import chess.engine
import chess


class node:
    def __init__(self):
        self.state = []  # board
        self.action = ''  # action
        self.white = True
        self.children = set()
        self.parent = None
        self.parentVisits = 0
        self.visits = 0
        self.exploitation = 0

    def __repr__(self) -> str:
        return str(self.state)+"\n"+str(self.white)


class ChessAgent:
    def __init__(self, env, exploration_constant):
        self.env = env
        self.exploration_constant = exploration_constant

    def ucb1(self, curr_node, parentVisits):
        if (curr_node.visits == 0):
            return float('inf')
        return curr_node.exploitation + self.exploration_constant * (
            sqrt(log(curr_node.visits + e + (10 ** -6)) / (parentVisits + (10 ** -10))))
        # return (curr_node.exploitation / curr_node.visits + 1.41 * sqrt(log(parentVisits) / curr_node.visits))

    def load_agent(self):
        pass

    # choose best child
    def selection(self, curr_node):
        children_ucbs = [self.ucb1(child, curr_node.visits)
                         for child in curr_node.children]
        sel_node = None
        sel_node = list(curr_node.children)[np.argmax(children_ucbs)]
        return sel_node

    def rollout(self, curr_node):
        # print(curr_node.white)

        if (curr_node.state.is_game_over()):
            # print(curr_node.white)
            # print(curr_node.state)
            # print(curr_node.state.result())

            board = curr_node.state
            if (curr_node.white == True):

                if (board.result() == '1-0'):
                    return (-1, curr_node)
                elif (board.result() == '0-1'):
                    return (1, curr_node)
                else:
                    return (0.5, curr_node)
            else:
                if (board.result() == '1-0'):
                    return (1, curr_node)
                elif (board.result() == '0-1'):
                    return (-1, curr_node)
                else:
                    return (0.5, curr_node)

        #  de verificat : verifica ca se genereaza ok si ca are sens
        all_moves = [curr_node.state.san(i)
                     for i in list(curr_node.state.legal_moves)]

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

        # incearca sa alegi cea mai buna miscare
        next_state = self.selection(curr_node)

        return self.rollout(rnd_state)

    def expand(self, curr_node, white):
        if (len(curr_node.children) == 0):
            return curr_node
        # daca nodul ales a mai fost parcurs si are copii mergem pana la ultimul nivel alegand copii cei mai buni dupa ucb1
        return self.expand(self.selection(curr_node), white)

    # this has to be changed
    def rollback(self, curr_node, reward):
        while (curr_node.parent != None):
            curr_node.visits += 1
            curr_node.exploitation += reward
            curr_node = curr_node.parent
        return curr_node

    # mcts_pred
    def mcts_pred(self, curr_node, over, white, iterations=15):
        print(curr_node)
        if (over):
            return -1
        all_moves = [curr_node.state.san(i)
                     for i in list(curr_node.state.legal_moves)]
        map_state_move = dict()
        # creem array cu noduri cu board-ul dupa fiecare miscare

        for move in all_moves:
            tmp_state = chess.Board(curr_node.state.fen())
            tmp_state.push_san(move)
            child = node()
            child.state = tmp_state
            child.parent = curr_node
            child.white = curr_node.white
            curr_node.children.add(child)
            map_state_move[child] = move

        # iterate over all moves + once
        # (so that if every node has not been explored, they will be, and choose out of them)
        neparcursi = 0
        for k in curr_node.children:
            if (k.visits == 0):
                neparcursi += 1
        iterations = neparcursi+1
        #iterations = len(all_moves)+1
        # extending the best node by ucb
        while (iterations > 0):
            # selecting node to expand
            sel_child = self.selection(curr_node)
            # coboram pana la ultimul copil pe care l ar putea avea
            ex_child = self.expand(sel_child, white)
            #
            reward, state = self.rollout(ex_child)
            curr_node = self.rollback(state, reward)
            iterations -= 1

        # choosing next state
        child_list = list(curr_node.children)
        print("Lungime lista:")
        print(len(child_list))
        sel_move = map_state_move[child_list[np.argmax(
            [self.ucb1(child, curr_node.visits) for child in child_list])]]

        print(np.argmax(
            [self.ucb1(child, curr_node.visits) for child in child_list]))

        # primul din lista e mereu vizitat de 20 de ori, numarul de iteratii
        return sel_move

    # bot_vs_bot
    def run_episode(self):
        print("Running episode")
        self.env.reset()
        whites_turn = True
        moves = 0
        pgn = []
        f = open("game.txt", "w")
        board = chess.Board()
        terminal = False
        nodes = []
        while not terminal:
            root = node()
            root.state = board
            root.white = whites_turn

            result = self.mcts_pred(root, board.is_game_over(), whites_turn)
            root.action = str(board.parse_san(result))

            print("\n" * 7)
            # poate iti trebuie pentru run_episode sa returnezi
            # nodes.append(root)

            # print(board.is_stalemate())
            # print(board.can_claim_fifty_moves())
            # print(board.can_claim_draw())

            board, reward, terminal, info = self.env.step(
                board.parse_san(result))
            # print("OUT")
            whites_turn = bool(1 - whites_turn)
            moves += 1
            print(f'Move: {(moves + 1)//2}\n\n')
            if moves/2 != moves//2:
                print('White to move\n')
            else:
                print('Black to move\n')

            print(f"Move made: {result}\n")
            f.write(f'{(moves + 1)//2}. {result} ')
            print(self.env.render())

            if board.is_stalemate():
                print("Draw by stalemate")
                terminal = True
            elif board.can_claim_threefold_repetition():
                print("Draw by 3F")
                terminal = True
            elif board.is_insufficient_material():
                print("Draw by insufficient material")
                terminal = True
            elif board.can_claim_fifty_moves():
                print('Draw by 50 Move Rule')
                terminal = True
            if (terminal):
                f.close()
                return reward

    def bot_vs_player(self):
        print("Game started")
        self.env.reset()
        whites_turn = True
        moves = 0
        board = chess.Board()
        terminal = False
        nodes = []

        while not terminal:
            # bot move
            root = node()
            root.state = board
            root.white = whites_turn

            result = self.mcts_pred(root, board.is_game_over(), whites_turn)
            root.action = str(board.parse_san(result))

            print("\n"*10)
            nodes.append(root)

            board, reward, terminal, info = self.env.step(
                board.parse_san(result))
            whites_turn = True
            moves += 1

            print(self.env.render())

            if (terminal):
                break
            # player move
            # E NEGRU
            while (True):
                print('Legal moves:', self.env.legal_moves)
                move = input(
                    "Enter your move: (ex: g2g4 moves the piece from g2 to g4) \n")
                if (not chess.Move.from_uci(move) in self.env.legal_moves):
                    move = input("Incorrect/illegal move: \n")
                else:
                    break
            board, reward, terminal, info = self.env.step(
                board.parse_uci(move))
            self.env.render()
            moves += 1
            print(f'Move: {moves//2}\n\n')
            print(self.env.render())

            if board.is_stalemate():
                print("Draw by stalemate")
                terminal = True
            elif board.can_claim_fifty_moves():
                print('Draw by 50 Move Rule')
                terminal = True
            if (terminal):
                print(reward)

    # def update_policy_q(self, eps=None):
    #     return
    #     for state, values in self.q.items():
    #         if eps != None: # e-Greedy policy updates ?
    #             if np.random.rand() < eps:
    #                 self.policy[state] = self.env.action_space.sample() # sample a random action
    #             else:
    #                 self.policy[state] = np.argmax(values)

    #         else: # Greedy policy updates
    #             self.policy[state] = np.argmax(values)

    # train
    # def mc_control_glie(self, n_episode=500000, firstVisit=True, update_policy_every=1):

    #     for t in trange(n_episode):
    #         traversed = []
    #         # Get an epsilon for this episode - used in e-greedy policy update
    #         eps = self.get_epsilon(t)

    #         # Generate an episode following current policy
    #         transitions = self.run_episode(eps=None)

    #         # zip operation will convert transitions to list of states, list of actions, rewards etc.
    #         states, actions, rewards, next_states, dones = zip(*transitions)

    #         # If firstVisit version is used, create a table stateAction_firstVisitTime that stores
    #         # when the pair (State, action) was seen first in this episode
    #         if firstVisit == True:
    #             stateAction_firstVisitTime = {}
    #             for t, state in enumerate(states):
    #                 stateAction_t = (state, actions[t])
    #                 if stateAction_t not in stateAction_firstVisitTime:
    #                     stateAction_firstVisitTime[stateAction_t] = t

    #         # Iterate over episode steps in reversed order, T-1, T-2, ....0
    #         G = 0 # return output
    #         for t in range(len(transitions)-1, -1, -1):
    #             St = states[t]
    #             At = actions[t]

    #             if firstVisit == True:
    #                 # Is t first time when we see the (state, action) pair ?
    #                 if stateAction_firstVisitTime[(St, At)] < t:
    #                     continue
    #             G = self.gamma * G + rewards[t]

    #             self.n_q[St][At] += 1
    #             alpha = (1.0 / self.n_q[St][At]) # Remember that with this formula all episodes experiences have equal importance, if you want to forget older episode put a bigger probability
    #             self.q[St][At] = self.q[St][At] + alpha*(G - self.q[St][At])

    #         if t % update_policy_every == 0:
    #             self.update_policy_q(eps)

    #     # final policy update at the end
    #     self.update_policy_q(eps)
