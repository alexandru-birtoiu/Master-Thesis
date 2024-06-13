import pybullet as p
import pybullet_data as pd
import time
import simulation as panda_sim
from config import *

p.connect(p.GUI)

p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
p.resetDebugVisualizerCamera(cameraDistance=CAMERA_DISTANCE, cameraYaw=CAMERA_YAW, cameraPitch=CAMERA_PITCH,
                             cameraTargetPosition=CAMERA_TARGET_POSITION)
p.setAdditionalSearchPath(pd.getDataPath())

p.setTimeStep(TIME_STEP)
p.setGravity(0, -9.8, 0)

panda = panda_sim.PandaSim(p, True, TASK_TYPE)
panda.control_dt = TIME_STEP

logId = panda.bullet_client.startStateLogging(panda.bullet_client.STATE_LOGGING_PROFILE_TIMINGS, "log.json")

images = []
i = 0

print(MODEL_PATH)

while i < 100000:
    panda.step()
    p.stepSimulation()
    i += 1
    panda.bullet_client.stopStateLogging(logId)
