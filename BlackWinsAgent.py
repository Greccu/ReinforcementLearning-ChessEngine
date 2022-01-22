
import chess
import chess.pgn
import chess.engine
import random
import time
from math import log, sqrt, e, inf
import gym
import gym_chess
import random
import time
import numpy as np
import chess.engine
import chess
from treelib import Node, Tree
from math import log, sqrt, e, inf
from paprika import *


class node():
    def __init__(self):
        self.state = chess.Board()
        self.action = ''
        self.children = set()
        self.parent = None
        self.parentVisits = 0
        self.visits = 0
        self.exploitation = 0


def ucb1(curr_node):
    ans = curr_node.exploitation+exploration_constant * \
        (sqrt(log(curr_node.parentVisits+e+(10**-6))/(curr_node.visits+(10**-10))))
    return ans


class ChessAgent:
    def __init__(self, env, exploration_constant):
        self.env = env
        self.exploration_constant = exploration_constant

    def rollout(self, curr_node):

        if(curr_node.state.is_game_over()):
            board = curr_node.state
            if(board.result() == '1-0'):
                return (100, curr_node)
            elif(board.result() == '0-1'):
                return (-1, curr_node)
            else:
                return (0.5, curr_node)

        all_moves = [curr_node.state.san(i)
                     for i in list(curr_node.state.legal_moves)]

        for i in all_moves:
            tmp_state = chess.Board(curr_node.state.fen())
            tmp_state.push_san(i)
            child = node()
            child.state = tmp_state
            child.parent = curr_node
            curr_node.children.add(child)
        rnd_state = random.choice(list(curr_node.children))

        return self.rollout(rnd_state)

    def expand(self, curr_node, white):
        if(len(curr_node.children) == 0):
            return curr_node
        if(white):
            child_list = list(curr_node.children)
            sel_child = child_list[np.argmax(
                [ucb1(child) for child in child_list])]

            return(self.expand(sel_child, 0))

        else:
            child_list = list(curr_node.children)
            sel_child = child_list[np.argmin(
                [ucb1(child) for child in child_list])]

            return self.expand(sel_child, 1)

    def rollback(self, curr_node, reward):
        curr_node.visits += 1
        curr_node.exploitation += reward
        while(curr_node.parent != None):
            curr_node.parentVisits += 1
            curr_node = curr_node.parent
        return curr_node

    def mcts_pred(self, curr_node, over, white, iterations=1):
        if(over):
            return -1
        all_moves = [curr_node.state.san(i)
                     for i in list(curr_node.state.legal_moves)]
        map_state_move = dict()

        for i in all_moves:
            tmp_state = chess.Board(curr_node.state.fen())
            tmp_state.push_san(i)
            child = node()
            child.state = tmp_state
            child.parent = curr_node
            curr_node.children.add(child)
            map_state_move[child] = i

        while(iterations > 0):
            if(white):
                child_list = list(curr_node.children)
                sel_child = child_list[np.argmax(
                    [ucb1(child) for child in child_list])]
                ex_child = self.expand(sel_child, 0)
                reward, state = self.rollout(ex_child)
                curr_node = self.rollback(state, reward)
                iterations -= 1
            else:
                child_list = list(curr_node.children)
                sel_child = child_list[np.argmin(
                    [ucb1(child) for child in child_list])]

                ex_child = self.expand(sel_child, 1)

                reward, state = self.rollout(ex_child)

                curr_node = self.rollback(state, reward)
                iterations -= 1
        if(white):

            child_list = list(curr_node.children)
            sel_move = map_state_move[child_list[np.argmax(
                [ucb1(child) for child in child_list])]]
            return sel_move
        else:
            child_list = list(curr_node.children)
            sel_move = map_state_move[child_list[np.argmin(
                [ucb1(child) for child in child_list])]]
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
                if (reward == -1.0):
                    print("Black won")
                return reward

    #player is white
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


exploration_constant = 2
env = gym.make('Chess-v0')

# Black should win

agent = ChessAgent(env, exploration_constant=2)
# print(agent.run_episode())
print(agent.bot_vs_player())
