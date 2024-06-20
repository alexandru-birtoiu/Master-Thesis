from enum import Enum
from tasks.Task import Task
from config import *
import numpy as np
import math
from utils import check_distance

# Enum to represent the different stages of the cube task
class TaskStages(Enum):
    IDLE = 1
    GO_ABOVE_CUBE = 2
    GO_ON_CUBE = 3
    GRASP_CUBE = 4
    GO_TABLE = 5
    DROP_CUBE = 6
    DONE = 7

# Constants for randomization and table dimensions
MAX_CUBE_RANDOM_Y = 0.15
CUBE_SCALING = 0.3
MAX_RANDOM = 1
TABLE_LENGTH = 0.4
TABLE_WIDTH = 0.25
ON_TABLE_HEIGHT = 0.15

class CubeDepthTask(Task):
    """
    Class to manage the cube depth task. Inherits from Task.
    """
    def __init__(self, bullet_client, next_episode_callback=None):
        """
        Initializes the CubeDepthTask.
        """
        super().__init__(bullet_client)
        self.bullet_client = bullet_client
        self.start_position = PLANE_START_POSITION
        self.bodies = []
        self.state = TaskStages.IDLE
        self.next_episode = next_episode_callback
        self.bullet_client.configureDebugVisualizer(self.bullet_client.COV_ENABLE_SHADOWS, 0)
        self.randSize = 0

    def next_state(self):
        """
        Transitions to the next state in the task.
        """
        current_stage_index = self.state.value - 1
        next_stage_index = (current_stage_index + 1) % len(TaskStages)
        self.state = TaskStages(next_stage_index + 1)

        # Set target positions based on the next state
        if self.state == TaskStages.GO_ABOVE_CUBE:
            pos, _ = self.bullet_client.getBasePositionAndOrientation(self.cube)
            diff = 0.02
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
        """
        Checks if the current state is DROP_CUBE.
        
        Returns:
        - bool: True if the current state is DROP_CUBE, False otherwise.
        """
        return self.state == TaskStages.DROP_CUBE

    def is_grab_state(self):
        """
        Checks if the current state is GRASP_CUBE.
        
        Returns:
        - bool: True if the current state is GRASP_CUBE, False otherwise.
        """
        return self.state == TaskStages.GRASP_CUBE

    def is_idle_state(self):
        """
        Checks if the current state is IDLE.
        
        Returns:
        - bool: True if the current state is IDLE, False otherwise.
        """
        return self.state == TaskStages.IDLE

    def is_moving(self):
        """
        Checks if the current state is between GO_ABOVE_CUBE and DROP_CUBE.
        
        Returns:
        - bool: True if the current state is between GO_ABOVE_CUBE and DROP_CUBE, False otherwise.
        """
        return self.state.value > 1 and self.state.value < 7
    
    def is_gripper_closed(self):
        """
        Checks if the gripper should be closed (between GRASP_CUBE and before DROP_CUBE).
        
        Returns:
        - bool: True if the gripper should be closed, False otherwise.
        """
        return self.state.value >= 4 and self.state.value < 6
    
    def get_task_type(self):
        """
        Gets the type of the task.
        """
        return []

    def randomize_task(self):
        """
        Randomizes the position and size of the cube within the allowed range.
        """
        self.randx = np.random.uniform(-MAX_CUBE_RANDOM, MAX_CUBE_RANDOM)
        self.randz = np.random.uniform(-MAX_CUBE_RANDOM, MAX_CUBE_RANDOM)
        
        self.randSize = np.random.uniform(0, MAX_RANDOM)
        self.cubeScale = 1.5 + CUBE_SCALING * self.randSize
        self.randy = (1 - self.randSize) * MAX_CUBE_RANDOM_Y

    def randomize_environment(self):
        """
        Randomizes the environment by placing the cube and table in random positions
        and creating a transparent cuboid under the cube.
        """
        self.randomize_task()

        cubePosition = np.array([self.start_position[0] + self.randx + 0.03, 
                                 self.start_position[1] + self.randy, 
                                 self.start_position[2] + self.randz])
    
        # Calculate the height of the cuboid from the ground to the cube
        cuboid_height = self.randy

        # Create a collision shape for the rectangular cuboid
        cuboid_shape = self.bullet_client.createCollisionShape(self.bullet_client.GEOM_BOX, 
                                                               halfExtents=[0.05, cuboid_height, 0.05])

        # Create a multi-body for the rectangular cuboid
        self.transparent_cuboid = self.bullet_client.createMultiBody(baseMass=0,
                                                                     baseCollisionShapeIndex=cuboid_shape,
                                                                     basePosition=np.array([cubePosition[0],
                                                                                            self.start_position[1],
                                                                                            cubePosition[2]]),
                                                                     baseOrientation=[0, 0, 0, 1])

        # Change the visual shape to be transparent
        self.bullet_client.changeVisualShape(self.transparent_cuboid, -1, rgbaColor=[1, 1, 1, 0]) 
    
        cubeOrientation = self.bullet_client.getQuaternionFromEuler([-1, 0, 0])
        self.cube = self.bullet_client.loadURDF("lego/lego.urdf", 
                                                cubePosition, 
                                                cubeOrientation, 
                                                globalScaling=self.cubeScale)
        self.bullet_client.changeVisualShape(self.cube, -1, rgbaColor=[1, 0, 0, 1])

        tableOrientation = self.bullet_client.getQuaternionFromEuler([-math.pi / 2, math.pi / 2, 0])
        self.table = self.bullet_client.loadURDF("table/table.urdf", np.array([-0.7, 0.0, -0.1]), tableOrientation,
                                                 globalScaling=0.25)
        
        self.bodies = [self.cube, self.table, self.transparent_cuboid]

    def check_near_object(self, position, threshold=0.05):
        """
        Checks if the given position is near the cube within a specified threshold.
        
        Parameters:
        - position: The position to check.
        - threshold: The distance threshold.
        
        Returns:
        - bool: True if the position is within the threshold distance from the cube, False otherwise.
        """
        object_pos, _ = self.bullet_client.getBasePositionAndOrientation(self.cube)
        return check_distance(np.array(object_pos), np.array(position), threshold)

    def check_grasped_object(self, ee_pos, finger_target, threshold=0.03):
        """
        Checks if the cube is grasped by the end effector.
        
        Parameters:
        - ee_pos: The position of the end effector.
        - finger_target: The target position of the fingers.
        - threshold: The distance threshold.
        
        Returns:
        - bool: True if the cube is grasped, False otherwise.
        """
        cube_pos, _ = self.bullet_client.getBasePositionAndOrientation(self.cube)
        return finger_target == GRIPPER_CLOSE and check_distance(np.array(ee_pos), np.array(cube_pos), threshold)

    def above_table(self):
        """
        Checks if the cube is above the table.
        
        Returns:
        - bool: True if the cube is above the table, False otherwise.
        """
        object_pos, _ = self.bullet_client.getBasePositionAndOrientation(self.cube)
        table_pos, _ = self.bullet_client.getBasePositionAndOrientation(self.table)
        
        within_length = (table_pos[2] - TABLE_LENGTH / 2 <= object_pos[2] <= table_pos[2] + TABLE_LENGTH / 2)
        within_width = (table_pos[0] - TABLE_WIDTH / 2 <= object_pos[0] <= table_pos[0] + TABLE_WIDTH / 2)
        
        return within_length and within_width and (object_pos[1] > table_pos[1])

    def on_table(self, threshold=0.05):
        """
        Checks if the cube is on the table within a specified height threshold.
        
        Parameters:
        - threshold: The height threshold.
        
        Returns:
        - bool: True if the cube is on the table within the height threshold, False otherwise.
        """
        object_pos, _ = self.bullet_client.getBasePositionAndOrientation(self.cube)
        table_pos, _ = self.bullet_client.getBasePositionAndOrientation(self.table)
        
        within_length = (table_pos[2] - TABLE_LENGTH / 2 <= object_pos[2] <= table_pos[2] + TABLE_LENGTH / 2)
        within_width = (table_pos[0] - TABLE_WIDTH / 2 <= object_pos[0] <= table_pos[0] + TABLE_WIDTH / 2)
        
        height_within_threshold = (abs(object_pos[1] - ON_TABLE_HEIGHT) <= threshold)
        
        return within_length and within_width and height_within_threshold
