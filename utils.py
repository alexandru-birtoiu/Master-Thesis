
from config import IMAGES_PATH
import numpy as np

def get_image_path(camera, episode, sample_episode):
    return f"{IMAGES_PATH}{camera}/image_{episode}_{sample_episode}.png"

def get_depth_path(camera, episode, sample_episode):
    return f"{IMAGES_PATH}{camera}/image_{episode}_{sample_episode}_depth"

def check_distance(point1, point2, threshold):
    distance = np.linalg.norm(point1 - point2)
    return distance < threshold