import gym
import gym_chess
import random
import time
import numpy as np

env = gym.make('Chess-v0')
print(env.render())

env.reset()
done = False


# environment : Chess Table
# agent : Players
# states - a lot
# actions - legal_actions, a lot, not constant
# rewards - not defined
# episodes - not defined

# varianta 1 facem cu monte carlo tree search

# https://medium.com/@ishaan.gupta0401/monte-carlo-tree-search-application-on-chess-5573fc0efb75
# https://github.com/Ish2K/Chess-Bot-AI-Algorithms/blob/main/Git_chess/monte_carlo_implementation.py


# Other

# miscari de sah rate-uite : https://www.kaggle.com/ethanmai/chess-moves



#COD DIN LABURI:

#
# def runEpisode(env, policy, maxSteps=100):
#     # We count here the total
#     total_reward = 0
#
#     # THis is how we reset the environment to an initial state, it returns the observation.
#     # As documented, in this case the observation is the state where the agent currently is positionaed,
#     # , which is a number in [0, nS-1]. We can use local function stateToRC to get the row and column of the agent
#     # The action give is in range [0, nA-1], check the enum defined above to understand what each number means
#     obs = env.reset()
#     for t in range(maxSteps):
#         # Draw the environment on screen
#         env.render()
#         # Sleep a bit between decisions
#         time.sleep(0.25)
#
#         # Here we sample an action from our policy, we consider it deterministically at this point
#         action = policy[obs]
#
#         # Hwere we interact with the enviornment. We give it an action to do and it returns back:
#         # - the new observation (observable state by the agent),
#         # - the reward of the action just made
#         # - if the simulation is done (terminal state)
#         # - last parameters is an "info" output, we are not interested in this one that's why we ignore the parameter
#         newObs, reward, done, _ = env.step(action)
#         print(f"Agent was in state {obs}, took action {action}, now in state {newObs}")
#         obs = newObs
#
#         total_reward += reward
#         # Close the loop before maxSteps  if we are in a terminal state
#         if done:
#             break
#
#     if not done:
#         print(f"The agent didn't reach a terminal state in {maxSteps} steps.")
#     else:
#         print(f"Episode reward: {total_reward}")
#     env.render()  # One last  rendering of the episode.
#
#
# def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
#     """Evaluate the value function from a given policy.
#     Parameters
#     ----------
#     P, nS, nA, gamma:
#         defined at beginning of file
#     policy: np.array[nS]
#         The policy to evaluate. Maps states to actions, deterministic !
#     tol: float
#         Terminate policy evaluation when
#             max |value_function(s) - prev_value_function(s)| < tol
#     Returns
#     -------
#     value_function: np.ndarray[nS]
#         The value function of the given policy, where value_function[s] is
#         the value of state s
#     """
#     # Init with 0 for all states,
#     # Remember that terminal states MUST have 0 always whatever you initialize them with here
#     value_function = np.zeros(nS)
#
#     ############################
#     # YOUR IMPLEMENTATION HERE #
#     maxChange = np.inf
#     numIters = 0
#     while maxChange > tol:
#         numIters += 1
#         maxChange = -np.inf
#         for s in range(nS):
#             a = policy[s]  # We have a deterministic policy, no need to iterate over actions in this case
#
#             # Let's check the next moves we get from starting in state s and applying action a
#             new_value_func = 0.0
#             for nextMove in P[s][a]:
#                 probability, nextstate, reward, terminal = nextMove
#                 new_value_func += probability * (reward + gamma * value_function[
#                     nextstate])  # if policy wouldn't be deterministic  we would have to multiply all this with probability given by each a, pi(a|s)
#
#             maxChange = max(maxChange, abs(new_value_func - value_function[s]))
#             value_function[s] = new_value_func
#
#     print(f"Policy evaluation converged after {numIters} iterations")
#
#     ############################
#
#     return value_function
