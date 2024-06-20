import pybullet as p
import pybullet_data as pd
import time
import simulation as panda_sim
from config import *
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

os.makedirs(f'evaluation', exist_ok=True)

# Connect to PyBullet
p.connect(p.GUI)

# Configure the visualizer and set physics parameters
p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
p.resetDebugVisualizerCamera(cameraDistance=CAMERA_DISTANCE, cameraYaw=CAMERA_YAW, cameraPitch=CAMERA_PITCH,
                             cameraTargetPosition=CAMERA_TARGET_POSITION)
p.setAdditionalSearchPath(pd.getDataPath())

# Set time step and gravity
p.setTimeStep(TIME_STEP)
p.setGravity(0, -9.8, 0)

# Initialize the Panda simulation and task
panda = panda_sim.PandaSim(p, True, TASK_TYPE)
panda.control_dt = TIME_STEP

# Initialize variables
all_epoch_results = {}
min_full_distance = float('inf')
best_epoch = 0

print(MODEL_PATH)

# Iterate through all the epochs from 1 to EPOCH
for epoch in range(1, EPOCH + 1):
    # Load model for the current epoch
    panda.load_model(epoch)
    print(f'Loading new model {epoch}..')

    min_distances = []
    episode_durations = []
    start_time = time.time()
    time_limit = 12  
    episode_count = 0
    max_episodes = 50
    current_min_distance = float('inf')
    episode_duration = 0

    # Main simulation loop
    while episode_count < max_episodes:
        panda.step()
        p.stepSimulation()

        # Check distance to teddy bear ear
        state, _ = panda.get_state()
        current_pos = np.array(state[-3:])
        distance_to_target = panda.task.check_near_object(current_pos)
        current_min_distance = min(current_min_distance, distance_to_target)

        if distance_to_target < 0.01:
            elapsed_time = time.time() - start_time
            print(f"Reached target within {elapsed_time:.2f} seconds.")
            episode_duration = elapsed_time

            panda.next_episode()
            min_distances.append(current_min_distance)
            episode_durations.append(episode_duration)
            current_min_distance = float('inf')
            episode_duration = 0
    

            start_time = time.time()
            episode_count += 1

        elapsed_time = time.time() - start_time
        if elapsed_time > time_limit:
            print("Time limit exceeded. Resetting simulation.")
            episode_duration = elapsed_time

            panda.next_episode()
            min_distances.append(current_min_distance)
            episode_durations.append(episode_duration)
            current_min_distance = float('inf')
            episode_duration = 0
    

            start_time = time.time()
            episode_count += 1

    average_min_distance = np.mean(min_distances)
    average_duration = np.mean(episode_durations)

    all_epoch_results[epoch] = {
        'average_min_distance': average_min_distance,
        'average_duration': average_duration,
        'total_episodes': episode_count
    }

    if average_min_distance < min_full_distance:
        min_full_distance = average_min_distance
        best_epoch = epoch

# Plot average minimum distance per epoch
epochs = list(all_epoch_results.keys())
average_min_distances = [all_epoch_results[epoch]['average_min_distance'] for epoch in epochs]

task_name = TASK_TYPE.name
model_name = task_name + ' _' + MODEL_PATH.split('/')[-1] + '_' + str(best_epoch)

plt.plot(epochs, average_min_distances, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Average Minimum Distance to Target')
plt.title('Average Minimum Distance to Target per Epoch')
plt.savefig(f'evaluation/average_min_distance_per_epoch_{model_name}.png')
plt.show(block=True)

# Plot average duration per epoch
average_durations = [all_epoch_results[epoch]['average_duration'] for epoch in epochs]

plt.plot(epochs, average_durations, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Average Duration (seconds)')
plt.title('Average Duration per Epoch')
plt.savefig(f'evaluation/average_duration_per_epoch_{model_name}.png')
plt.show(block=True)

# Save all data 
torch.save(all_epoch_results, f'evaluation/all_epoch_data_{model_name}')

print(f'The best epoch is {best_epoch} with an average minimum distance of {min_full_distance:.2f}.')
