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
    GO_TABLE = 5
    DROP_CUBE = 6
    DONE = 7

MAX_CUBE_RANDOM_Y = 0.23
CUBE_SCALING = 1.3
MAX_RANDOM = 1.6

class CubeDepthTask(Task):
    def __init__(self, bullet_client, next_episode_callback=None):
        super().__init__(bullet_client)
        self.bullet_client = bullet_client
        self.start_position = PLANE_START_POSITION
        self.bodies = []
        self.state = TaskStages.IDLE
        self.next_episode = next_episode_callback
        self.bullet_client.configureDebugVisualizer(self.bullet_client.COV_ENABLE_SHADOWS, 0)
        self.randSize = 0

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
        elif self.state == TaskStages.GO_TABLE:
            pos, _ = self.bullet_client.getBasePositionAndOrientation(self.table)
            self.target_pos = np.array([pos[0], pos[1] + 0.3, pos[2]])
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
        return self.state.value > 1 and self.state.value < 7
    
    def is_gripper_closed(self):
        return self.state.value >= 4 and self.state.value < 6

    def randomize_task(self):
        self.randx = np.random.uniform(-MAX_CUBE_RANDOM, MAX_CUBE_RANDOM)
        self.randz = np.random.uniform(-MAX_CUBE_RANDOM, MAX_CUBE_RANDOM)

        
        # if self.randSize == 1:
        #     self.randSize = MAX_RANDOM
        #     self.randz = 0.06
        #     self.randx = 0
        # else:
        #     self.randSize = 1
        #     self.randz = -0.06
        #     self.randx = 0
        
        self.randSize = np.random.uniform(1, MAX_RANDOM)
        self.cubeScale = CUBE_SCALING * self.randSize

        self.randy = (2 - self.randSize) * MAX_CUBE_RANDOM_Y

    def randomize_environment(self):
        self.randomize_task()

        cubePosition = np.array([self.start_position[0] + self.randx, 
                        self.start_position[1] + self.randy, 
                        self.start_position[2] + self.randz])
    
        # Calculate the height of the cuboid from the ground to the cube
        cuboid_height = self.randy

        # Create a collision shape for the rectangular cuboid
        # halfExtents=[0.1, 0.1, cuboid_height / 2] to match the cube height
        cuboid_shape = self.bullet_client.createCollisionShape(self.bullet_client.GEOM_BOX, 
                                                            halfExtents=[0.1, cuboid_height, 0.1])

        # Create a multi-body for the rectangular cuboid
        self.transparent_cuboid = self.bullet_client.createMultiBody(baseMass=0,
                                                                    baseCollisionShapeIndex=cuboid_shape,
                                                                    basePosition=np.array(
                                                                        [cubePosition[0],
                                                                         self.start_position[1],
                                                                         cubePosition[2]]
                                                                    ),
                                                                    baseOrientation=[0, 0, 0, 1])

        # Change the visual shape to be semi-transparent
        self.bullet_client.changeVisualShape(self.transparent_cuboid, -1, rgbaColor=[1, 1, 1, 0])  # Semi-transparent
        # Remove shadow from the rectangular cuboid
        # self.bullet_client.setShadow(self.transparent_cuboid, 0)
    

        cubeOrientation = self.bullet_client.getQuaternionFromEuler([-1, 0, 0])
        self.cube = self.bullet_client.loadURDF("lego/lego.urdf", 
                                                cubePosition, 
                                                cubeOrientation, 
                                                globalScaling=self.cubeScale)
        self.bullet_client.changeVisualShape(self.cube, -1, rgbaColor=[1, 0, 0, 1])


        tableOrientation = self.bullet_client.getQuaternionFromEuler([-math.pi / 2, math.pi / 2, 0])
        self.table = self.bullet_client.loadURDF("table/table.urdf", np.array([-0.5, 0.0, -0.2]), tableOrientation,
                                                 globalScaling=0.25)
        


        
        self.bodies = [self.cube, self.table, self.transparent_cuboid]