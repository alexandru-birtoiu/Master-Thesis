import numpy as np
from config import *

class Task:
    def __init__(self, bullet_client):
        self.bullet_client = bullet_client

    def next_state(self):
        raise NotImplementedError

    def is_drop_state(self):
        raise NotImplementedError

    def is_grab_state(self):
        raise NotImplementedError

    def is_idle_state(self):
        raise NotImplementedError

    def is_moving(self):
        raise NotImplementedError
    
    def is_gripper_closed(self):
        raise NotImplementedError

    def randomize_task(self):
        raise NotImplementedError

    def randomize_environment(self):
        raise NotImplementedError
    
    def get_task_type(self):
        raise NotImplementedError
