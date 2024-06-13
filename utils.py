
from config import IMAGES_PATH
import numpy as np
from config import IMAGE_SIZE
import torch.nn.functional as F

def get_image_path(camera, episode, sample_episode):
    return f"{IMAGES_PATH}{camera}/image_{episode}_{sample_episode}.png"

def get_depth_path(camera, episode, sample_episode):
    return f"{IMAGES_PATH}{camera}/image_{episode}_{sample_episode}_depth"

def check_distance(point1, point2, threshold):
    distance = np.linalg.norm(point1 - point2)
    return distance < threshold

def normalize_depth_data(depth_data):
    depth_data = depth_data.view(1, IMAGE_SIZE, IMAGE_SIZE)
    min_val = depth_data.min()
    max_val = depth_data.max()
    return ((depth_data - min_val) / (max_val - min_val)) * 255

def resize_tensor(image_tensor, target_size):
    return F.interpolate(image_tensor.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
