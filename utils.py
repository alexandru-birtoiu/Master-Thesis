
from config import IMAGES_PATH

def get_image_path(camera, episode, sample_episode):
    return f"{IMAGES_PATH}{camera}/image_{episode}_{sample_episode}.png"

def get_depth_path(camera, episode, sample_episode):
    return f"{IMAGES_PATH}{camera}/image_{episode}_{sample_episode}_depth"