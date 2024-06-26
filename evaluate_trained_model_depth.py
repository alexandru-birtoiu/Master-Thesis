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

def reset_simulation(panda, checkpoints, checkpoints_count):
    for checkpoint, reached in checkpoints.items():
        if reached:
            checkpoints_count[checkpoint] += 1
    panda.task.next_episode()
    checkpoints.update({
        "near_object": False,
        "grasped_object": False,
        "above_table": False,
        "on_table": False
    })
    return time.time()

def update_checkpoints(checkpoints, checkpoint_name):
    checkpoint_order = ["near_object", "grasped_object", "above_table", "on_table"]
    for cp in checkpoint_order:
        checkpoints[cp] = True
        if cp == checkpoint_name:
            break

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

# Initialize the Panda simulation
panda = panda_sim.PandaSim(p, True, TASK_TYPE)
panda.control_dt = TIME_STEP

# Initialize variables
all_epoch_results = {}
max_full_checkpoints = -1
best_epoch = 0

print(MODEL_PATH)

# Iterate through all the epochs from 1 to EPOCH
for epoch in range(6, EPOCH + 1):
    # Load model for the current epoch
    panda.load_model(epoch)
    print(f'Loading new model {epoch}..')
    
    checkpoints = {
        "near_object": False,
        "grasped_object": False,
        "above_table": False,
        "on_table": False
    }
    checkpoints_count = {
        "near_object": 0,
        "grasped_object": 0,
        "above_table": 0,
        "on_table": 0
    }

    start_time = time.time()
    time_limit = 10
    episode_count = 0
    max_episodes = 50
    full_checkpoints_count = 0

    # Main simulation loop
    while episode_count < max_episodes:
        panda.step()
        p.stepSimulation()

        # Check checkpoints
        state, positions = panda.get_state()
        current_pos = np.array(state[-3:])
        ee_pos = state[-3:]  # End effector position

        if not checkpoints["near_object"] and panda.task.check_near_object(current_pos):
            update_checkpoints(checkpoints, "near_object")
            print("Checkpoint 1: Robot is near the object.")
        
        if not checkpoints["grasped_object"] and panda.task.check_grasped_object(ee_pos, panda.finger_target):
            update_checkpoints(checkpoints, "grasped_object")
            print("Checkpoint 2: Robot has grasped the object.")

        if not checkpoints["above_table"] and panda.task.above_table():
            update_checkpoints(checkpoints, "above_table")
            print("Checkpoint 3: Object is above the table.")
        
        if not checkpoints["on_table"] and panda.task.on_table():
            update_checkpoints(checkpoints, "on_table")
            print("Checkpoint 4: Object is on the table.")

        # Check if the final checkpoint is reached within the time limit
        elapsed_time = time.time() - start_time
        if checkpoints["on_table"]:
            full_checkpoints_count += 1
            print(f"All checkpoints reached in {elapsed_time:.2f} seconds.")
            start_time = reset_simulation(panda, checkpoints, checkpoints_count)
            episode_count += 1

        if elapsed_time > time_limit:
            print("Time limit exceeded. Resetting simulation.")
            start_time = reset_simulation(panda, checkpoints, checkpoints_count)
            episode_count += 1

    all_epoch_results[epoch] = {
        'checkpoints_count': checkpoints_count,
        'full_checkpoints_count': full_checkpoints_count,
        'total_episodes': episode_count
    }

    if full_checkpoints_count > max_full_checkpoints:
        max_full_checkpoints = full_checkpoints_count
        best_epoch = epoch

# Plot the histogram of checkpoint achievements for the best epoch
best_epoch_results = all_epoch_results[best_epoch]
checkpoint_labels = list(best_epoch_results['checkpoints_count'].keys())
checkpoint_values = list(best_epoch_results['checkpoints_count'].values())

task_name = TASK_TYPE.name
model_name = task_name + ' _' + MODEL_PATH.split('/')[-1] + '_' + str(best_epoch)


plt.bar(checkpoint_labels, checkpoint_values, edgecolor='black')
plt.xlabel('Checkpoints')
plt.ylabel('Number of Episodes')
plt.title(f'Histogram of Checkpoint Achievements for Epoch {best_epoch}')
plt.savefig(f'evaluation/checkpoint_{model_name}.png')
plt.show(block=True)

# Plot number of full checkpoints per epoch
epochs = list(all_epoch_results.keys())
full_checkpoints_counts = [all_epoch_results[epoch]['full_checkpoints_count'] for epoch in epochs]

plt.plot(epochs, full_checkpoints_counts, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Number of Full Checkpoints')
plt.title('Number of Full Checkpoints per Epoch')
plt.savefig(f'evaluation/full_checkpoints_per_epoch_{model_name}.png')
plt.show(block=True)

# Save all data
torch.save(all_epoch_results, f'evaluation/all_epoch_checkpoints_data_{model_name}')

print(f'The best epoch is {best_epoch} with {max_full_checkpoints} full checkpoints.')
