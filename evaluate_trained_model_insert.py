import pybullet as p
import pybullet_data as pd
import time
import simulation as panda_sim
from config import *
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def reset_simulation(panda, checkpoints, checkpoints_count):
    for checkpoint, reached in checkpoints.items():
        if reached:
            checkpoints_count[checkpoint] += 1
    panda.task.next_episode()
    checkpoints.update({
        "near_object": False,
        "grasped_object": False,
        "on_tray": False,
        "in_position": False
    })
    return time.time()

def update_checkpoints(checkpoints, checkpoint_name):
    checkpoint_order = ["near_object", "grasped_object", "on_tray", "in_position"]
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

# Start logging
logId = panda.bullet_client.startStateLogging(panda.bullet_client.STATE_LOGGING_PROFILE_TIMINGS, "log.json")

# Initialize variables
images = []
i = 0
checkpoints = {
    "near_object": False,
    "grasped_object": False,
    "on_tray": False,
    "in_position": False
}
checkpoints_count = {
    "near_object": 0,
    "grasped_object": 0,
    "on_tray": 0,
    "in_position": 0
}
start_time = time.time()
time_limit = 30  # 30 seconds
episode_count = 0
max_episodes = 30

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

    if not checkpoints["on_tray"] and panda.task.check_on_tray():
        update_checkpoints(checkpoints, "on_tray")
        print("Checkpoint 3: Object is on the tray.")
    
    if not checkpoints["in_position"] and panda.task.check_in_position():
        update_checkpoints(checkpoints, "in_position")
        print("Checkpoint 4: Object is in the correct position.")
    
    # Check if the final checkpoint is reached within the time limit
    elapsed_time = time.time() - start_time
    if checkpoints["in_position"]:
        print(f"All checkpoints reached in {elapsed_time:.2f} seconds.")
        start_time = reset_simulation(panda, checkpoints, checkpoints_count)
        episode_count += 1
        i = 0  # Reset step counter

    if elapsed_time > time_limit:
        print("Time limit exceeded. Resetting simulation.")
        start_time = reset_simulation(panda, checkpoints, checkpoints_count)
        episode_count += 1
        i = 0  # Reset step counter

    i += 1

# Stop logging
panda.bullet_client.stopStateLogging(logId)

# Plot the histogram of checkpoint achievements
checkpoint_labels = list(checkpoints_count.keys())
checkpoint_values = list(checkpoints_count.values())

model_name = MODEL_PATH.split('/')[-1] + '_' + str(EPOCH)

plt.bar(checkpoint_labels, checkpoint_values, edgecolor='black')
plt.xlabel('Checkpoints')
plt.ylabel('Number of Episodes')
plt.title('Histogram of Checkpoint Achievements')
plt.savefig(f'evaluation/checkpoint_{model_name}.png')
plt.show(block=True)


data_to_save = {
    'checkpoints_count': checkpoints_count,
    'total_episodes': episode_count
}

torch.save(data_to_save, f'evaluation/checkpoints_data_{model_name}')


