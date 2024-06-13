import torch
from collections import deque, defaultdict
from config import *
from torch.utils.data import Dataset
from utils import *
from torchvision.io import read_image 

class ImageCache:
    def __init__(self, max_size):
        self.cache = {}
        self.order = deque()
        self.max_size = max_size

    def get(self, path):
        return self.cache.get(path)

    def put(self, path, tensor):
        if path not in self.cache:
            if len(self.order) >= self.max_size:
                oldest = self.order.pop()
                del self.cache[oldest]
            self.order.appendleft(path)
        self.cache[path] = tensor

class CustomImageDataset(Dataset):
    def __init__(self, labels, device):
        self.labels = labels
        self.device = device
        self.cameras = ACTIVE_CAMERAS

        self.episode_sample_pairs = [
            (episode, sample) for episode, samples in self.labels.items() \
                for sample in samples.keys()
        ]

        self.image_cache = ImageCache(max_size=len(self.cameras) * MAX_SEQUENCE * BATCH_SIZE * (2 if IMAGE_TYPE == ImageType.RGBD else 1))

    def __len__(self):
        return len(self.episode_sample_pairs)

    def __getitem__(self, idx):
        episode, sample = self.episode_sample_pairs[idx]
        inputs = []

        for camera in self.cameras:
            sample_images = []
            for i in range(SEQUENCE_LENGTH - 1, -1, -1):
                    if IMAGE_TYPE == ImageType.D:
                        depth_path = get_depth_path(camera, episode, sample - i)
                        depth_data_normalized = self.image_cache.get(depth_path)

                        if depth_data_normalized is None:
                            depth_data = torch.tensor(torch.load(depth_path))
                            depth_data_normalized = normalize_depth_data(depth_data).to(self.device)
                            self.image_cache.put(depth_path, depth_data_normalized)

                        sample_images.append(depth_data_normalized)

                    elif IMAGE_TYPE == ImageType.RGB:
                        image_path = get_image_path(camera, episode, sample - i)
                        img = self.image_cache.get(image_path)

                        if img is None:
                            img = read_image(image_path).to(self.device, dtype=torch.float)
                            self.image_cache.put(image_path, img)

                        sample_images.append(img)

                    elif IMAGE_TYPE == ImageType.RGBD:
                        image_path = get_image_path(camera, episode, sample - i)
                        depth_path = get_depth_path(camera, episode, sample - i)

                        img = self.image_cache.get(image_path)
                        depth_data_normalized = self.image_cache.get(depth_path)

                        if img is None:
                            img = read_image(image_path).to(self.device, dtype=torch.float)
                            self.image_cache.put(image_path, img)

                        if depth_data_normalized is None:
                            depth_data = torch.tensor(torch.load(depth_path))
                            depth_data_normalized = normalize_depth_data(depth_data).to(self.device)
                            self.image_cache.put(depth_path, depth_data_normalized)

                        # Concatenate RGB and depth images
                        rgbd_data = torch.cat((img, depth_data_normalized), dim=0)
                        sample_images.append(rgbd_data)

            resized_images = [resize_tensor(image, (IMAGE_SIZE_TRAIN, IMAGE_SIZE_TRAIN)) for image in sample_images]
            inputs.append(torch.cat(resized_images, dim=0))
        
        label = torch.tensor(self.labels[episode][sample]["labels"]).to(self.device)
        positions = torch.tensor(self.labels[episode][sample]["positions"]).to(self.device)

        #TODO: ADD TASK LABEL FOR TRAY POSITION
        if USE_TASK_LOSS:
            task = self.labels[episode][sample]["task"][0]
            task_one_hot = torch.nn.functional.one_hot(torch.tensor(task - 1), num_classes=3).to(self.device)  # Adjust task to zero-based
            label = torch.cat((label[:-8], task_one_hot.float(), label[-8:]), dim=0)

        return inputs, positions, label
    
def calculate_loss(criterion, outputs, labels, mask=None):
    if mask == None:
        mask = torch.zeros(outputs.shape[0]) == 0
    
    if PREDICTION_TYPE == PredictionType.VELOCITY:
        loss_1 = criterion(outputs[mask, :7], labels[mask, :7])
    elif PREDICTION_TYPE == PredictionType.POSITION:
        loss_1 = criterion(outputs[mask, :3], labels[mask, 7:10])
    elif PREDICTION_TYPE == PredictionType.TARGET_POSITION:
        loss_1 = criterion(outputs[mask, :3], labels[mask, 10:13])
    
    task_loss = 0
    if USE_TASK_LOSS:
        task_loss = criterion(outputs[mask, -11:-8], labels[mask, -11:-8])

    loss_2 = criterion(outputs[mask, -8:-6], labels[mask, -8:-6])
    loss_3 = criterion(outputs[mask, -6:-3], labels[mask, -6:-3])
    loss_4 = criterion(outputs[mask, -3:], labels[mask, -3:])

    return loss_1 + loss_2 + loss_3 + loss_4 + task_loss

def prepare_labels(labels):
    """
    Prepare labels for the CustomImageDataset with skipping steps.

    Parameters:
    labels (dict): A dictionary containing episode and sample data.
                   Structure:
                   {
                       episode_number: {
                           sample_number: sample_data,
                           ...
                       },
                       ...
                   }

    Returns:
    list: A list of tuples (episode_number, sample_number, sample_data) with the appropriate samples skipped.
    """
    prepared_labels = {}

    for episode_number, samples in labels.items():
        sample_keys = list(samples.keys()) 

        adjusted_skip = max(1, STEPS_SKIPED // 2)
        if adjusted_skip == 1:
            return labels

        prepared_labels[episode_number] = {}
        
        for i in range(0, len(sample_keys), adjusted_skip):
            sample_number = sample_keys[i]
            sample_data = samples[sample_number]
            prepared_labels[episode_number][sample_number] = sample_data

    return prepared_labels
