from enum import Enum
from tasks.Task import Task
from config import *
import numpy as np
import math
from utils import check_distance

# Enum to represent the different stages of the insert task
class TaskStages(Enum):
    IDLE = 1
    GO_ABOVE_CUBE = 2
    GO_ON_CUBE = 3
    GRASP_CUBE = 4
    GO_INSERT_POSITION_1 = 5
    GO_INSERT_POSITION_2 = 6
    DROP_CUBE = 7
    DONE = 8

# Enum to represent different objects in the task
class Objects(Enum):
    CUBE = 1
    SPHERE = 2
    POLYGON = 3

# Constants for insert tray offsets and object settings
INSERT_TRAY_OFFSET_MIDDLE = np.array([-0.09, 0, 0.13])
INSERT_TRAY_OFFSET_BOTTOM = np.array([-0.09, 0, -0.02])
INSERT_TRAY_OFFSET_TOP = np.array([-0.09, 0, 0.28])

OBJECT_SETTINGS = {
    1: {
        "insert_offset": INSERT_TRAY_OFFSET_MIDDLE,
        "object_scaling": 1.5,
        "object_orientation": [-1, 0 ,0],
        "urdf_path": "lego/lego.urdf"
    },
    2: {
        "insert_offset": INSERT_TRAY_OFFSET_TOP,
        "object_scaling": 1.0,
        "object_orientation": [-1, 0 ,0],
        "urdf_path": "objects/sphere/sphere.urdf"
    },
    3: {
        "insert_offset": INSERT_TRAY_OFFSET_BOTTOM,
        "object_scaling": 1.2,
        "object_orientation": [0, 0 ,0],
        "urdf_path": "objects/polygon/polygon.urdf"
    }
}

INSERT_TRAY_POSITIONS = {
    0: np.array([-0.4, -0.05, -0.5]),
    1: np.array([0.5, -0.05, -0.5])               
}

class InsertTask(Task):
    """
    Class to manage the insert task. Inherits from Task.
    """
    def __init__(self, bullet_client, next_episode_callback=None):
        """
        Initializes the InsertTask.
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
            pos, _ = self.bullet_client.getBasePositionAndOrientation(self.object)
            diff = 0.1
            self.target_pos = np.array([pos[0], pos[1] + diff, pos[2]])
        elif self.state == TaskStages.GO_ON_CUBE:
            pos, _ = self.bullet_client.getBasePositionAndOrientation(self.object)
            pos = [pos[0], pos[1] - 0.01, pos[2]]
            self.target_pos = np.array(pos)
        elif self.state == TaskStages.GO_INSERT_POSITION_1:
            pos, _ = self.bullet_client.getBasePositionAndOrientation(self.insertTray)
            self.target_pos = np.array([pos[0], pos[1] + 0.3, pos[2]]) + self.tray_offset
        elif self.state == TaskStages.GO_INSERT_POSITION_2:
            pos, _ = self.bullet_client.getBasePositionAndOrientation(self.insertTray)
            self.target_pos = np.array([pos[0], pos[1] + 0.2, pos[2]]) + self.tray_offset
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
        return self.state.value > 1 and self.state.value <= 7
    
    def is_gripper_closed(self):
        """
        Checks if the gripper should be closed (between GRASP_CUBE and before DROP_CUBE).
        
        Returns:
        - bool: True if the gripper should be closed, False otherwise.
        """
        return self.state.value >= 4 and self.state.value <= 6

    def get_task_type(self):
        """
        Gets the type of the task.
        
        Returns:
        - list: A list representing the type of the task.
        """
        return [self.objectType.value, int(all(self.insertTrayPosition == INSERT_TRAY_POSITIONS[0]))]

    def randomize_task(self):
        """
        Randomizes the position and type of the object within the allowed range.
        """
        self.randx = np.random.uniform(-MAX_CUBE_RANDOM, MAX_CUBE_RANDOM)
        self.randz = np.random.uniform(-MAX_CUBE_RANDOM, MAX_CUBE_RANDOM)
        self.insertTrayPosition = INSERT_TRAY_POSITIONS[np.random.randint(0, 2)]
        
        self.objectType = Objects(np.random.randint(1, 4))

        settings = OBJECT_SETTINGS[self.objectType.value]
        self.tray_offset = settings["insert_offset"]
        self.object_scaling = settings["object_scaling"]
        self.object_orientation = settings["object_orientation"]
        self.urdf_path = settings["urdf_path"]

    def randomize_environment(self):
        """
        Randomizes the environment by placing the object and insert tray in random positions and orientations.
        """
        self.randomize_task()

        objectOrientation = self.bullet_client.getQuaternionFromEuler(self.object_orientation)
        self.object = self.bullet_client.loadURDF(
            self.urdf_path, 
            np.array([
                self.start_position[0] + self.randx, 
                self.start_position[1], 
                self.start_position[2] + self.randz
            ]), 
            objectOrientation, 
            globalScaling=self.object_scaling
        )
        self.bullet_client.changeVisualShape(self.object, -1, shapeIndex=-1, rgbaColor=[1, 0, 0, 1])

        tray_id = self.bullet_client.createCollisionShape(
            shapeType=self.bullet_client.GEOM_MESH,
            fileName="objects/tray/tray.obj",
            meshScale=[0.013, 0.013, 0.013],
            flags=self.bullet_client.GEOM_FORCE_CONCAVE_TRIMESH,
        )

        insertTrayOrientation = self.bullet_client.getQuaternionFromEuler([-math.pi / 2, math.pi / 2, 0])
        self.insertTray = self.bullet_client.createMultiBody(
            baseCollisionShapeIndex=tray_id,
            basePosition=self.insertTrayPosition,
            baseOrientation=insertTrayOrientation
        )

        self.bullet_client.changeVisualShape(self.insertTray, -1, rgbaColor=[0.01, 0.35, 0.5, 1])
        self.bodies = [self.object, self.insertTray]

    def check_near_object(self, position, threshold=0.05):
        """
        Checks if the given position is near the object within a specified threshold.
        
        Parameters:
        - position: The position to check.
        - threshold: The distance threshold.
        
        Returns:
        - bool: True if the position is within the threshold distance from the object, False otherwise.
        """
        object_pos, _ = self.bullet_client.getBasePositionAndOrientation(self.object)
        return check_distance(np.array(object_pos), np.array(position), threshold)

    def check_grasped_object(self, ee_pos, finger_target, threshold=0.03):
        """
        Checks if the object is grasped by the end effector.
        
        Parameters:
        - ee_pos: The position of the end effector.
        - finger_target: The target position of the fingers.
        - threshold: The distance threshold.
        
        Returns:
        - bool: True if the object is grasped, False otherwise.
        """
        object_pos, _ = self.bullet_client.getBasePositionAndOrientation(self.object)
        return finger_target == GRIPPER_CLOSE and check_distance(np.array(ee_pos), np.array(object_pos), threshold)

    def check_on_tray(self, threshold=0.1):
        """
        Checks if the object is on the tray within a specified threshold.
        
        Parameters:
        - threshold: The distance threshold.
        
        Returns:
        - bool: True if the object is on the tray, False otherwise.
        """
        object_pos, _ = self.bullet_client.getBasePositionAndOrientation(self.object)
        tray_pos, _ = self.bullet_client.getBasePositionAndOrientation(self.insertTray)
        offset_top = tray_pos + INSERT_TRAY_OFFSET_TOP
        offset_bottom = tray_pos + INSERT_TRAY_OFFSET_BOTTOM
        return (object_pos[1] > tray_pos[1] and
                offset_bottom[0] - threshold <= object_pos[0] <= offset_top[0] + threshold and
                offset_bottom[2] - threshold <= object_pos[2] <= offset_top[2] + threshold)

    def check_in_position(self, threshold=0.11):
        """
        Checks if the object is in the correct position within a specified threshold.
        
        Parameters:
        - threshold: The distance threshold.
        
        Returns:
        - bool: True if the object is in the correct position, False otherwise.
        """
        object_pos, _ = self.bullet_client.getBasePositionAndOrientation(self.object)
        tray_pos, _ = self.bullet_client.getBasePositionAndOrientation(self.insertTray)
        insert_pos = tray_pos + self.tray_offset
        insert_pos[1] = 0
        return check_distance(insert_pos, object_pos, threshold)
