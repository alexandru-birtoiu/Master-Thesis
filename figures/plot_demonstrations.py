import matplotlib.pyplot as plt

# Constants
task_number = 3
max_episodes = 1000
episode_increment = 250

# Number of episodes
episodes = [100] + list(range(episode_increment, max_episodes + episode_increment, episode_increment))

accuracy_B = [ 8.3, 6,  5.9, 5.8 , 5.85]
accuracy_E = [ 7.4, 5, 4.8, 4.3, 4.1]

accuracy_EB = [ 3.5, 2.3, 2.1, 2, 1.75]
accuracy_SB = [ 3.4, 2.1, 1.9, 1.8, 1.65]

if len(accuracy_E) != len(episodes):
    raise ValueError("Length of accuracy values must match the number of episodes")

# Plotting the lines
plt.figure(figsize=(10, 6))

plt.plot(episodes, accuracy_E, marker='o', label='E')
plt.plot(episodes, accuracy_B, marker='o', label='B')
plt.plot(episodes, accuracy_EB, marker='o', label='E+B')
plt.plot(episodes, accuracy_SB, marker='o', label='SB')

# Adding titles and labels
plt.title(f'Number of Episodes vs Accuracy - Task {task_number}')
plt.xlabel('Number of Episodes')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 10)
plt.xticks(episodes)
plt.grid(True)

# Adding legend
plt.legend()

# Show plot
plt.show()


## Task 1
accuracy_E = [0, 40, 60, 76, 88]
accuracy_B = [0, 42, 64, 78, 90]

accuracy_EB = [0, 70, 86, 92, 98]
accuracy_SB = [0, 62, 82, 88, 94]

## Task 2
accuracy_E = [0, 16, 24, 28, 32]
accuracy_B = [0, 32, 44, 46, 48]

accuracy_EB = [0, 34, 50, 68, 76]
accuracy_SB = [0, 32, 48, 62, 68]

## Task 3

accuracy_E = [0, 12, 24, 30, 34, 36]
accuracy_B = [0, 8, 16, 18, 19, 20]

accuracy_EB = [0, 14, 28, 44, 58, 64]
accuracy_SB = [0, 16, 32, 48, 54, 58]
