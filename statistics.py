from shutil import move
from turtle import width
import matplotlib.pyplot as plt
import numpy as np
game_file = open("game.txt", "r")
nr_moves_file = open("nmoves.txt","r")
move_duration_file = open("tmoves.txt", "r")
results_file = open("results.txt", "r")
move_dfg = open("gametime.txt","r")


# plot for move duration

# move_duration_list = move_dfg.readlines()[0].split(" ")
# move_duration_list = [float(i) for i in move_duration_list if i != '']
# print(move_duration_list)
# print(len(move_duration_list))

# move_nr = list(range(1, len(move_duration_list)+1))

# print(len(move_nr))

# plt.plot(move_nr,move_duration_list)
# plt.title('Time to make moves')
# plt.xlabel('Move')
# plt.ylabel('Move duration')
# plt.show()

#plot for move average compared to nr of moves

game_duration_list = move_duration_file.readlines()
game_duration_list = [float(i) for i in game_duration_list]

nr_moves_list = nr_moves_file.readlines()
nr_moves_list = [float(i) for i in nr_moves_list]

index = move_nr = list(range(1, len(game_duration_list)+1))

result_list = results_file.readlines()
result_list = [float(i) for i in result_list]


fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Game')
ax1.set_ylabel('Average time per move', color=color)
ax1.plot(index, game_duration_list, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Nr. of moves', color=color)  # we already handled the x-label with ax1
ax2.plot(index, nr_moves_list, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()



game_file.close()
move_duration_file.close()
results_file.close()
nr_moves_file.close()
move_dfg.close()