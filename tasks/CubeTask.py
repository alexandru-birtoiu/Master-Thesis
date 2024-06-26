from enum import Enum
from config import *
import numpy as np
import math
import sys
from utils import check_distance
from tasks.Task import Task

# Enum to represent the different stages of the cube task
class TaskStages(Enum):
    IDLE = 1
    GO_ABOVE_CUBE = 2
    GO_ON_CUBE = 3
    GRASP_CUBE = 4
    GO_TABLE = 5
    DROP_CUBE = 6
    DONE = 7

# Constants for table dimensions
TABLE_LENGTH = 0.4
TABLE_WIDTH = 0.25
ON_TABLE_HEIGHT = 0.15

MAX_CUBE_RANDOM = 0.

class CubeTask(Task):
    """
    Class to manage the cube grasping task. Inherits from Task.
    """
    def __init__(self, bullet_client, next_episode_callback=None):
        """
        Initializes the CubeTask.
        """
        super().__init__(bullet_client)
        self.bullet_client = bullet_client
        self.start_position = PLANE_START_POSITION
        self.bodies = []
        self.state = TaskStages.IDLE
        self.next_episode = next_episode_callback

    def next_state(self):
        """
        Transitions to the next state in the task.
        """
        current_stage_index = self.state.value - 1
        next_stage_index = (current_stage_index + 1) % len(TaskStages)
        self.state = TaskStages(next_stage_index + 1)

        # Set target positions based on the next state
        if self.state == TaskStages.GO_ABOVE_CUBE:
            pos, _ = self.bullet_client.getBasePositionAndOrientation(self.lego)
            self.target_pos = np.array([pos[0], pos[1] + 0.1, pos[2]])
        elif self.state == TaskStages.GO_ON_CUBE:
            pos, _ = self.bullet_client.getBasePositionAndOrientation(self.lego)
            self.target_pos = np.array([pos[0], pos[1] - 0.01, pos[2]])
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

    def randomize_task(self):
        """
        Randomizes the position of the cube within the allowed range.
        """
        self.randx = np.random.uniform(-MAX_CUBE_RANDOM, MAX_CUBE_RANDOM)
        self.randz = np.random.uniform(-MAX_CUBE_RANDOM, MAX_CUBE_RANDOM)

    def get_task_type(self):
        """
        Gets the type of the task.
        
        Returns:
        - list: An empty list representing the type of the task.
        """
        return []   

    def randomize_environment(self):
        """
        Randomizes the environment by placing the cube and table in random positions.
        """
        self.randomize_task()

        legoOrientation = self.bullet_client.getQuaternionFromEuler([-1, 0, 0])
        self.lego = self.bullet_client.loadURDF("lego/lego.urdf", 
                                                np.array([
                                                    self.start_position[0] + self.randx, 
                                                    self.start_position[1], 
                                                    self.start_position[2] + self.randz]), 
                                                legoOrientation, 
                                                globalScaling=1.5)
        self.bullet_client.changeVisualShape(self.lego, -1, rgbaColor=[1, 0, 0, 1])

        tableOrientation = self.bullet_client.getQuaternionFromEuler([-math.pi / 2, math.pi / 2, 0])
        self.table = self.bullet_client.loadURDF("table/table.urdf", np.array([-0.5, 0.0, -0.2]), tableOrientation,
                                                 globalScaling=0.25)
        
        self.bodies = [self.lego, self.table]

    def check_near_object(self, position, threshold=0.05):
        """
        Checks if the given position is near the cube within a specified threshold.
        
        Parameters:
        - position: The position to check.
        - threshold: The distance threshold.
        
        Returns:
        - bool: True if the position is within the threshold distance from the cube, False otherwise.
        """
        object_pos, _ = self.bullet_client.getBasePositionAndOrientation(self.lego)
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
        cube_pos, _ = self.bullet_client.getBasePositionAndOrientation(self.lego)
        return finger_target == GRIPPER_CLOSE and check_distance(np.array(ee_pos), np.array(cube_pos), threshold)

    def above_table(self):
        """
        Checks if the cube is above the table.
        
        Returns:
        - bool: True if the cube is above the table, False otherwise.
        """
        object_pos, _ = self.bullet_client.getBasePositionAndOrientation(self.lego)
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
        object_pos, _ = self.bullet_client.getBasePositionAndOrientation(self.lego)
        table_pos, _ = self.bullet_client.getBasePositionAndOrientation(self.table)
        
        within_length = (table_pos[2] - TABLE_LENGTH / 2 <= object_pos[2] <= table_pos[2] + TABLE_LENGTH / 2)
        within_width = (table_pos[0] - TABLE_WIDTH / 2 <= object_pos[0] <= table_pos[0] + TABLE_WIDTH / 2)
        
        height_within_threshold = (abs(object_pos[1] - ON_TABLE_HEIGHT) <= threshold)
        
        return within_length and within_width and height_within_threshold
