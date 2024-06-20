import pybullet as p
import pybullet_data as pd

import simulation as panda_sim
from PIL import Image
import torch
from tqdm import tqdm
import time
from config import *
import os


os.makedirs(f'labels/{TASK_TYPE.name.lower()}', exist_ok=True)
os.makedirs(f'images/{TASK_TYPE.name.lower()}', exist_ok=True)


p.connect(p.GUI)

p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
p.resetDebugVisualizerCamera(cameraDistance=CAMERA_DISTANCE, cameraYaw=CAMERA_YAW, cameraPitch=CAMERA_PITCH,
                             cameraTargetPosition=CAMERA_TARGET_POSITION)
p.setAdditionalSearchPath(pd.getDataPath())


p.setTimeStep(TIME_STEP)
p.setGravity(0, -9.8, 0)

panda = panda_sim.PandaSim(p, False, TASK_TYPE)
panda.control_dt = TIME_STEP

logId = panda.bullet_client.startStateLogging(panda.bullet_client.STATE_LOGGING_PROFILE_TIMINGS, "log.json")

images = []

episode = 0
sample_dict = {}
no_episodes = NO_EPISODES

if GATHER_DATA_MORE:
    #Continue gathering demonstrations
    sample_dict = torch.load(GATHER_MORE_LABEL_PATH)

    panda.episode = STARTING_EPISODES
    episode = STARTING_EPISODES

    no_episodes += STARTING_EPISODES
    
    print('Previous number of episodes: ' + str(len(sample_dict.keys())))

sample_dict[episode] = {}

pbar = tqdm(total=NO_EPISODES)
while episode < no_episodes:
    panda.step()
    p.stepSimulation()
    if GATHER_DATA:
        if panda.task.is_moving():
            sample_per_episode = panda.capture_images()

            current_episode = panda.get_episode()

            if current_episode not in sample_dict:
                if current_episode % 50 == 0:
                    torch.save(sample_dict, get_label_path(current_episode))
                pbar.update(1)
                sample_dict[current_episode] = {}

            episode = current_episode
           
            if sample_per_episode == -1:
                continue

            labels, positions = panda.get_state()
            
            panda.show_cube_pos(labels[-6:-3], labels[-3:])

            task_data = panda.task.get_task_type()
        
            sample_dict[current_episode][sample_per_episode] = {
                "labels": labels,
                "positions": positions,
                "task": task_data
            }
        else:
            time.sleep(TIME_STEP)
    else:
        state, positions = panda.get_state()
        print(panda.task.state.name + ' ' + str(panda.task.is_moving()))
        print(panda.task.is_gripper_closed())
        if panda.task.is_moving():
            panda.show_cube_pos(state[-3:6], state[-3:])
        time.sleep(0.05)
        
    panda.bullet_client.stopStateLogging(logId)


# Save the labels
torch.save(sample_dict,  get_label_path(episode))
p.disconnect()

