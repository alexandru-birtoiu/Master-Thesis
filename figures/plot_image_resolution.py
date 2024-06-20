import matplotlib.pyplot as plt

# Constant for the task number
task_number = 4

# Image resolutions and corresponding x positions for even spacing
resolutions = [32, 64, 84, 128]
x_positions = range(len(resolutions)) 

accuracy_B = [8.23, 7.4, 7.0, 5.85]
accuracy_E = [6.2, 4.4, 4.2, 4.1]
accuracy_EB = [5.9, 4.1, 2.4, 1.75]
accuracy_SB = [2.5, 2.5, 1.8, 1.65]

plt.figure(figsize=(10, 6))

plt.plot(x_positions, accuracy_E, marker='o', label='E')
plt.plot(x_positions, accuracy_B, marker='o', label='B')
plt.plot(x_positions, accuracy_EB, marker='o', label='E+B')
plt.plot(x_positions, accuracy_SB, marker='o', label='SB')

# Adding titles and labels
plt.title(f'Image Resolution vs Distance to ear - Task {task_number}')
plt.xlabel('Image Resolution')
plt.ylabel('Distance to ear (%)')
plt.ylim(0, 10)
plt.xticks(x_positions, resolutions)
plt.grid(True)

# Adding legend
plt.legend()

# Show plot
plt.show()


## Task 1
accuracy_B = [56, 72, 80, 88]
accuracy_E = [70, 82, 84, 90]
accuracy_SB = [60, 78, 88, 94]
accuracy_EB = [70, 86, 90, 98]


## Task 2
accuracy_B = [6, 12, 28, 48]
accuracy_E = [8, 16, 26, 32]
accuracy_SB = [32, 46, 62, 68]
accuracy_EB = [42, 52, 70, 76]

## Task 3
accuracy_B = [6, 12, 18, 20]
accuracy_E = [10, 22, 32, 36]
accuracy_EB = [46, 54, 60, 64]
accuracy_SB = [40, 48, 54, 58]