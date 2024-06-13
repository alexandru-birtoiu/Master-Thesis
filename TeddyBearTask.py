from enum import Enum
from Task import Task
from config import *
import numpy as np
import math
from utils import check_distance

class TaskStages(Enum):
    IDLE = 1
    GRAB_STATE_1 = 2
    GO_ABOVE_TEDDY = 3
    TOUCH_TEDDY_NOSE = 4
    GRAB_STATE_2 = 5
    DONE = 6

TEDDY_NOSE_OFFSET_X = 0.025  # Adjust based on the teddy bear model
TEDDY_NOSE_OFFSET_Y = 0.12  # Adjust based on the teddy bear model
TEDDY_NOSE_OFFSET_Z = 0  # Adjust based on the teddy bear model


class TeddyBearTask(Task):
    def __init__(self, bullet_client, next_episode_callback=None):
        super().__init__(bullet_client)
        self.bullet_client = bullet_client
        self.start_position = PLANE_START_POSITION
        self.bodies = []
        self.state = TaskStages.IDLE
        self.next_episode = next_episode_callback

    def next_state(self):
        current_stage_index = self.state.value - 1
        next_stage_index = (current_stage_index + 1) % len(TaskStages)
        self.state = TaskStages(next_stage_index + 1)

        if self.state == TaskStages.GO_ABOVE_TEDDY:
            pos, _ = self.bullet_client.getBasePositionAndOrientation(self.teddy)
            self.target_pos = np.array([pos[0], pos[1] + 0.2, pos[2]])
        elif self.state == TaskStages.TOUCH_TEDDY_NOSE:
            self.target_pos = self.get_teddy_nose_position()
        elif self.state == TaskStages.DONE:
            if self.next_episode:
                self.next_episode()
            self.state = TaskStages.IDLE  # Reset to IDLE after completing the episode

    def is_drop_state(self):
        return False

    def is_grab_state(self):
        return self.state == TaskStages.GRAB_STATE_1 or self.state == TaskStages.GRAB_STATE_2

    def is_idle_state(self):
        return self.state == TaskStages.IDLE

    def is_moving(self):
        return not (self.state == TaskStages.IDLE)

    def is_gripper_closed(self):
        return True

    def randomize_task(self):
        self.randx = np.random.uniform(-MAX_CUBE_RANDOM, MAX_CUBE_RANDOM)
        self.randz = np.random.uniform(-MAX_CUBE_RANDOM, MAX_CUBE_RANDOM)
        
        self.rand_orientation = self.bullet_client.getQuaternionFromEuler([
            0,
            np.random.uniform(0, 2 * math.pi),
            0
        ])

    def get_task_type(self):
        return []

    def randomize_environment(self):
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

    def check_near_object(self, position, threshold=0.05):
        nose_pos = self.get_teddy_nose_position()
        return check_distance(np.array(nose_pos), np.array(position), threshold)

    def get_teddy_nose_position(self):
        teddy_pos, teddy_orientation = self.bullet_client.getBasePositionAndOrientation(self.teddy)
        nose_offset = np.array([TEDDY_NOSE_OFFSET_X, TEDDY_NOSE_OFFSET_Y, TEDDY_NOSE_OFFSET_Z])
        rotation_matrix = self.bullet_client.getMatrixFromQuaternion(teddy_orientation)
        rotation_matrix = np.array(rotation_matrix).reshape(3, 3)
        rotated_offset = rotation_matrix.dot(nose_offset)
        nose_pos = teddy_pos + rotated_offset
        return nose_pos
