from enum import Enum
from Task import Task
from config import *
import numpy as np
import math

class TaskStages(Enum):
    IDLE = 1
    GO_ABOVE_CUBE = 2
    GO_ON_CUBE = 3
    GRASP_CUBE = 4
    GO_INSERT_POSITION_1 = 5
    GO_INSERT_POSITION_2 = 6
    DROP_CUBE = 7
    DONE = 8

INSERT_TRAY_OFFSET = np.array([-0.09, 0, 0.13])

INSERT_TRAY_POSITIONS = {
    0: np.array([-0.4, -0.05, -0.5]),
    1: np.array([0.5, -0.05, -0.5])               
}

class InsertTask(Task):
    def __init__(self, bullet_client, flags, next_episode_callback=None):
        super().__init__(bullet_client)
        self.flags = flags
        self.bullet_client = bullet_client
        self.start_position = PLANE_START_POSITION
        self.bodies = []
        self.state = TaskStages.IDLE
        self.next_episode = next_episode_callback
        self.tray_offset = INSERT_TRAY_OFFSET

    def next_state(self):
        current_stage_index = self.state.value - 1
        next_stage_index = (current_stage_index + 1) % len(TaskStages)
        self.state = TaskStages(next_stage_index + 1)

        if self.state == TaskStages.GO_ABOVE_CUBE:
            pos, _ = self.bullet_client.getBasePositionAndOrientation(self.cube)
            diff = 0.1
            self.target_pos = np.array([pos[0], pos[1] + diff, pos[2]])
        elif self.state == TaskStages.GO_ON_CUBE:
            pos, _ = self.bullet_client.getBasePositionAndOrientation(self.cube)
            pos = [pos[0], pos[1] - 0.01, pos[2]]
            self.target_pos = np.array(pos)
        elif self.state == TaskStages.GO_INSERT_POSITION_1:
            pos, _ = self.bullet_client.getBasePositionAndOrientation(self.insertTray)
            self.target_pos = np.array([pos[0], pos[1] + 0.3, pos[2]]) + INSERT_TRAY_OFFSET
        elif self.state == TaskStages.GO_INSERT_POSITION_2:
            pos, _ = self.bullet_client.getBasePositionAndOrientation(self.insertTray)
            self.target_pos = np.array([pos[0], pos[1] + 0.2, pos[2]]) + INSERT_TRAY_OFFSET
        elif self.state == TaskStages.DONE:
            self.next_episode()
            self.next_state()

    def is_drop_state(self):
        return self.state == TaskStages.DROP_CUBE

    def is_grab_state(self):
        return self.state == TaskStages.GRASP_CUBE

    def is_idle_state(self):
        return self.state == TaskStages.IDLE

    def is_moving(self):
        return self.state.value > 1 and self.state.value <= 7
    
    def is_gripper_closed(self):
        return self.state.value >= 4 and self.state.value <= 6

    def randomize_task(self):
        self.randx = np.random.uniform(-MAX_CUBE_RANDOM, MAX_CUBE_RANDOM)
        self.randz = np.random.uniform(-MAX_CUBE_RANDOM, MAX_CUBE_RANDOM)
        self.insertObjectPosition = INSERT_TRAY_POSITIONS[np.random.randint(0, 2)]

    def randomize_environment(self):
        self.randomize_task()

        cubeOrientation = self.bullet_client.getQuaternionFromEuler([-1, 0, 0])
        self.cube = self.bullet_client.loadURDF(
                                                "lego/lego.urdf", 
                                                np.array([
                                                    self.start_position[0] + self.randx, 
                                                    self.start_position[1], 
                                                    self.start_position[2] + self.randz]), 
                                                cubeOrientation, 
                                                globalScaling=1.5, 
                                                flags=self.flags)
        self.bullet_client.changeVisualShape(self.cube, -1, rgbaColor=[1, 0, 0, 1])

        shape_id = self.bullet_client.createCollisionShape(
            shapeType=self.bullet_client.GEOM_MESH,
            fileName="franka_panda/tray/tray.obj",
            meshScale=[0.013, 0.013, 0.013],
            flags = self.bullet_client.GEOM_FORCE_CONCAVE_TRIMESH,
        )

        insertTrayOrientation =  self.bullet_client.getQuaternionFromEuler([-math.pi / 2, math.pi / 2, 0])
        self.insertTray = self.bullet_client.createMultiBody(
            baseCollisionShapeIndex=shape_id,
            basePosition=self.insertObjectPosition,
            baseOrientation=insertTrayOrientation,
            flags=self.flags,
            
        )

        # insertTrayOrientation = self.bullet_client.getQuaternionFromEuler([-math.pi / 2, math.pi / 2, 0])
        # self.insertTray = self.bullet_client.loadURDF(
        #                                         "franka_panda/tray.urdf", 
        #                                         self.insertObjectPosition, 
        #                                         insertTrayOrientation,
        #                                         globalScaling=0.02,
        #                                         flags=self.flags,
        #                                         useFixedBase=True)
        self.bullet_client.changeVisualShape(self.insertTray, -1, rgbaColor=[0.01, 0.35, 0.5, 1])
        self.bodies = [self.cube, self.insertTray]