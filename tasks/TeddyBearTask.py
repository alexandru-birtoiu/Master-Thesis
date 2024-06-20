from enum import Enum
from tasks.Task import Task
from config import *
import numpy as np
import math
from utils import check_distance

# Enum to represent the different stages of the teddy bear task
class TaskStages(Enum):
    IDLE = 1
    GRAB_STATE_1 = 2
    GO_ABOVE_TEDDY = 3
    TOUCH_TEDDY_EAR = 4
    GRAB_STATE_2 = 5
    DONE = 6

# Constants for teddy bear ear offsets and randomization
TEDDY_EAR_OFFSET_X = 0.025 
TEDDY_EAR_OFFSET_Y = 0.12
TEDDY_EAR_OFFSET_Z = 0
MAX_TEDDY_RANDOM = 0.01

class TeddyBearTask(Task):
    """
    Class to manage the teddy bear task. Inherits from Task.
    """
    def __init__(self, bullet_client, next_episode_callback=None):
        """
        Initializes the TeddyBearTask.
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
        if self.state == TaskStages.GO_ABOVE_TEDDY:
            pos, _ = self.bullet_client.getBasePositionAndOrientation(self.teddy)
            self.target_pos = np.array([pos[0], pos[1] + 0.2, pos[2]])
        elif self.state == TaskStages.TOUCH_TEDDY_EAR:
            self.target_pos = self.get_teddy_ear_position()
        elif self.state == TaskStages.DONE:
            if self.next_episode:
                self.next_episode()
            self.state = TaskStages.IDLE  # Reset to IDLE after completing the episode

    def is_drop_state(self):
        """
        Checks if the current state is a drop state.
        
        Returns:
        - bool: False, as this task does not have a drop state.
        """
        return False

    def is_grab_state(self):
        """
        Checks if the current state is a grab state (GRAB_STATE_1 or GRAB_STATE_2).
        
        Returns:
        - bool: True if the current state is GRAB_STATE_1 or GRAB_STATE_2, False otherwise.
        """
        return self.state == TaskStages.GRAB_STATE_1 or self.state == TaskStages.GRAB_STATE_2

    def is_idle_state(self):
        """
        Checks if the current state is IDLE.
        
        Returns:
        - bool: True if the current state is IDLE, False otherwise.
        """
        return self.state == TaskStages.IDLE

    def is_moving(self):
        """
        Checks if the current state is not IDLE.
        
        Returns:
        - bool: True if the current state is not IDLE, False otherwise.
        """
        return not (self.state == TaskStages.IDLE)

    def is_gripper_closed(self):
        """
        Checks if the gripper should be closed.
        
        Returns:
        - bool: True, as the gripper is always closed in this task.
        """
        return True

    def randomize_task(self):
        """
        Randomizes the position and orientation of the teddy bear within the allowed range.
        """
        self.randx = np.random.uniform(-MAX_TEDDY_RANDOM, MAX_TEDDY_RANDOM)
        self.randz = np.random.uniform(-MAX_TEDDY_RANDOM, MAX_TEDDY_RANDOM)
        
        self.rand_orientation = self.bullet_client.getQuaternionFromEuler([
            0,
            np.random.uniform(0, 2 * math.pi),
            0
        ])

    def get_task_type(self):
        return []

    def randomize_environment(self):
        """
        Randomizes the environment by placing the teddy bear in a random position and orientation.
        """
        self.randomize_task()

        self.teddy = self.bullet_client.loadURDF("teddy_vhacd.urdf", 
                                                 np.array([
                                                     self.start_position[0] + self.randx, 
                                                     self.start_position[1], 
                                                     self.start_position[2] + self.randz]), 
                                                 self.rand_orientation, 
                                                 globalScaling=2,
                                                 useFixedBase=1)

        self.bodies = [self.teddy]

    def check_near_object(self, position):
        """
        Checks if the given position is near the teddy bear's ear.
        
        Parameters:
        - position: The position to check.
        
        Returns:
        - float: The distance between the given position and the teddy bear's ear.
        """
        ear = self.get_teddy_ear_position()
        return np.linalg.norm(np.array(ear) - np.array(position))

    def get_teddy_ear_position(self):
        """
        Gets the position of the teddy bear's ear based on its orientation and offsets.
        
        Returns:
        - numpy.ndarray: The position of the teddy bear's ear.
        """
        teddy_pos, teddy_orientation = self.bullet_client.getBasePositionAndOrientation(self.teddy)
        ear_offset = np.array([TEDDY_EAR_OFFSET_X, TEDDY_EAR_OFFSET_Y, TEDDY_EAR_OFFSET_Z])
        rotation_matrix = self.bullet_client.getMatrixFromQuaternion(teddy_orientation)
        rotation_matrix = np.array(rotation_matrix).reshape(3, 3)
        rotated_offset = rotation_matrix.dot(ear_offset)
        ear_pos = teddy_pos + rotated_offset
        return ear_pos
