import gym
import gym_chess
import random

env = gym.make('ChessAlphaZero-v0')
print(env.render())

env.reset()
done = False

while not done:
    action = random.choice(env.legal_moves)
    env.step(action)
    print(env.render(mode='unicode'))
    print()

env.close()