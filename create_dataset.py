import pybullet as p
import pybullet_data as pd

import panda_sim_grasp as panda_sim
from PIL import Image
import torch
from tqdm import tqdm
import time

# video requires ffmpeg available in path
from config import TIME_STEP, NO_SAMPLES, GATHER_DATA, IMAGE_SIZE, CAMERA_DISTANCE, CAMERA_YAW, CAMERA_PITCH, \
    CAMERA_TARGET_POSITION, IMAGES_PATH

p.connect(p.GUI)

p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
p.resetDebugVisualizerCamera(cameraDistance=CAMERA_DISTANCE, cameraYaw=CAMERA_YAW, cameraPitch=CAMERA_PITCH,
                             cameraTargetPosition=CAMERA_TARGET_POSITION)
p.setAdditionalSearchPath(pd.getDataPath())

p.setTimeStep(TIME_STEP)
p.setGravity(0, -9.8, 0)

panda = panda_sim.PandaSimAuto(p, False)
panda.control_dt = TIME_STEP

logId = panda.bullet_client.startStateLogging(panda.bullet_client.STATE_LOGGING_PROFILE_TIMINGS, "log.json")

images = []
sample = 0
sample_dict = {}
i = 0

pbar = tqdm(total=NO_SAMPLES)
while sample < NO_SAMPLES:
    panda.step()
    p.stepSimulation()
    if GATHER_DATA:
        if panda.is_moving():
            if len(images) < 4:
                img = p.getCameraImage(IMAGE_SIZE, IMAGE_SIZE, renderer=p.ER_BULLET_HARDWARE_OPENGL)
                rgbBuffer = img[2]
                rgbim = Image.fromarray(rgbBuffer)
                images.append(rgbim)
            else:
                for idx, img in enumerate(images):
                    img.save(IMAGES_PATH + str(sample) + '_' + str(idx) + '.png')
                images.clear()
                i = 0
                labels, positions = panda.get_state()
                sample_dict[sample] = {}
                sample_dict[sample]["labels"] = labels
                sample_dict[sample]["positions"] = positions

                sample += 1
                pbar.update(1)
        else:
            time.sleep(TIME_STEP)
            images.clear()
        if sample % 1000 == 0:
            torch.save(sample_dict, 'sample_labels')
    else:
        state, positions = panda.get_state()
        print(state)
    i += 1
    panda.bullet_client.stopStateLogging(logId)

torch.save(sample_dict, 'sample_labels')

