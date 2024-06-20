import torch
from collections import deque, defaultdict
from config import *
from torch.utils.data import Dataset
from utils import *
from torchvision.io import read_image 
from torch.utils.data import Sampler
import math
import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms


class ImageLayerBase(nn.Module):
    def __init__(self, input_channels):
        super(ImageLayerBase, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 12, kernel_size=(5, 5), stride=2, padding=2)  # Common layer 1
        self.conv2 = nn.Conv2d(12, 24, kernel_size=(3, 3), stride=2, padding=1)  # Common layer 2
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)  # Common max-pooling layer

    def forward_base(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.maxpool1(x)
        return x

class ImageLayer32(ImageLayerBase):
    def __init__(self, input_channels):
        super(ImageLayer32, self).__init__(input_channels)
        self.conv3 = nn.Conv2d(24, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 256, kernel_size=(4, 4), stride=1, padding=0)

    def forward(self, x):
        x = self.forward_base(x)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        return x

class ImageLayer64(ImageLayerBase):
    def __init__(self, input_channels):
        super(ImageLayer64, self).__init__(input_channels)
        self.conv3 = nn.Conv2d(24, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=(4, 4), stride=1, padding=0)

    def forward(self, x):
        x = self.forward_base(x)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        return x

class ImageLayer84(ImageLayerBase):
    def __init__(self, input_channels):
        super(ImageLayer84, self).__init__(input_channels)
        self.conv3 = nn.Conv2d(24, 36, kernel_size=(3, 3), stride=1, padding=1)
        self.conv4 = nn.Conv2d(36, 48, kernel_size=(3, 3), stride=2, padding=1)
        self.conv5 = nn.Conv2d(48, 128, kernel_size=(3, 3), stride=2, padding=1)
        self.conv6 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=0)

    def forward(self, x):
        x = self.forward_base(x)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = torch.relu(self.conv6(x))
        return x

class ImageLayer128(ImageLayerBase):
    def __init__(self, input_channels):
        super(ImageLayer128, self).__init__(input_channels)
        self.conv3 = nn.Conv2d(24, 36, kernel_size=(3, 3), stride=1, padding=1)
        self.conv4 = nn.Conv2d(36, 48, kernel_size=(3, 3), stride=2, padding=1)
        self.conv5 = nn.Conv2d(48, 64, kernel_size=(3, 3), stride=2, padding=1)
        self.conv6 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding=1)
        self.conv7 = nn.Conv2d(128, 256, kernel_size=(2, 2), stride=1, padding=0)

    def forward(self, x):
        x = self.forward_base(x)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = torch.relu(self.conv6(x))
        x = torch.relu(self.conv7(x))
        return x
    
class CrossViewAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_hidden_dim):
        super(CrossViewAttention, self).__init__()
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        self.q_linear1 = nn.Linear(embed_dim, embed_dim)
        self.k_linear2 = nn.Linear(embed_dim, embed_dim)
        self.v_linear2 = nn.Linear(embed_dim, embed_dim)
        
        self.q_linear2 = nn.Linear(embed_dim, embed_dim)
        self.k_linear1 = nn.Linear(embed_dim, embed_dim)
        self.v_linear1 = nn.Linear(embed_dim, embed_dim)
        
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1)
        
        self.mlp1 = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, embed_dim)
        )
        
        self.mlp2 = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, embed_dim)
        )

    def forward(self, z1, z2):
        ln_z1 = self.layer_norm(z1)
        ln_z2 = self.layer_norm(z2)

        q1 = self.q_linear1(ln_z1)
        k2 = self.k_linear2(ln_z2)
        v2 = self.v_linear2(ln_z2)

        q2 = self.q_linear2(ln_z2)
        k1 = self.k_linear1(ln_z1)
        v1 = self.v_linear1(ln_z1)

        attn_output1, _ = self.attention(q1, k2, v2)
        attn_output2, _ = self.attention(q2, k1, v1)

        ln_attn_output1 = self.layer_norm(z1 + attn_output1)
        ln_attn_output2 = self.layer_norm(z2 + attn_output2)

        h1 = self.mlp1(ln_attn_output1)
        h2 = self.mlp2(ln_attn_output2)

        combined_output = h1 + h2
        
        return combined_output
    

class ImageLayer(nn.Module):
    def __init__(self, input_channels):
        super(ImageLayer, self).__init__()

        self.input_channels = input_channels
        if IMAGE_SIZE_TRAIN == 32:
            self.model = ImageLayer32(input_channels)
        elif IMAGE_SIZE_TRAIN == 64:
            self.model = ImageLayer64(input_channels)
        elif IMAGE_SIZE_TRAIN == 84:
            self.model = ImageLayer84(input_channels)
        elif IMAGE_SIZE_TRAIN == 128:
            self.model = ImageLayer128(input_channels)
        else:
            raise ValueError("Unsupported IMAGE_SIZE_TRAIN value")

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        number_of_images = channels // self.input_channels
        x = x.view(batch_size * number_of_images, self.input_channels, height, width)
        x = self.model(x)
        x = x.view(batch_size, number_of_images, 256)
        return x


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
    def __init__(self, labels, augment=False):
        self.labels = labels
        self.cameras = ACTIVE_CAMERAS
        self.augment = augment

        self.episode_sample_pairs = [
            (episode, sample) for episode, samples in self.labels.items() 
                for sample in samples.keys()
        ]

        self.image_cache = ImageCache(max_size=len(self.cameras) * MAX_SEQUENCE * BATCH_SIZE * (2 if IMAGE_TYPE == ImageType.RGBD else 1))

        # Define the augmentation sequence
        max_translate = 4 / IMAGE_SIZE_TRAIN  # Calculate translation as a fraction of the image size
        self.transform = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(max_translate, max_translate))
            ], p=0.5)
        ])

    def augment_image(self, image):
        image = self.transform(image)
        return image
    
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
                        depth_data_normalized = normalize_depth_data(depth_data)
                        if self.augment:
                            depth_data_normalized = self.augment_image(depth_data_normalized)
                        self.image_cache.put(depth_path, depth_data_normalized)

                    sample_images.append(depth_data_normalized)

                elif IMAGE_TYPE == ImageType.RGB:
                    image_path = get_image_path(camera, episode, sample - i)
                    img = self.image_cache.get(image_path)

                    if img is None:
                        img = read_image(image_path).to(dtype=torch.float)
                        img = normalize_image(img)  # Normalize the RGB image
                        if self.augment:
                            img = self.augment_image(img)
                        self.image_cache.put(image_path, img)

                    sample_images.append(img)

                elif IMAGE_TYPE == ImageType.RGBD:
                    image_path = get_image_path(camera, episode, sample - i)
                    depth_path = get_depth_path(camera, episode, sample - i)

                    img = self.image_cache.get(image_path)
                    depth_data_normalized = self.image_cache.get(depth_path)

                    if img is None:
                        img = read_image(image_path).to(dtype=torch.float)
                        img = normalize_image(img)  # Normalize the RGB image
                        if self.augment:
                            img = self.augment_image(img)
                        self.image_cache.put(image_path, img)

                    if depth_data_normalized is None:
                        depth_data = torch.tensor(torch.load(depth_path))
                        depth_data_normalized = normalize_depth_data(depth_data)
                        if self.augment:
                            depth_data_normalized = self.augment_image(depth_data_normalized)
                        self.image_cache.put(depth_path, depth_data_normalized)

                    # Add the depth data as the 4th channel
                    rgbd_data = torch.cat((img, depth_data_normalized), dim=0)
                    sample_images.append(rgbd_data)

            if IMAGE_SIZE_TRAIN != IMAGE_SIZE:
                resized_images = [resize_tensor(image, (IMAGE_SIZE_TRAIN, IMAGE_SIZE_TRAIN)) for image in sample_images]
                inputs.append(torch.cat(resized_images, dim=0))
            else:
                inputs.append(torch.cat(sample_images, dim=0))
        
        label = torch.tensor(self.labels[episode][sample]["labels"])
        positions = torch.tensor(self.labels[episode][sample]["positions"])

        if USE_TASK_LOSS:
            task = self.labels[episode][sample]["task"][0]
            task_one_hot = torch.nn.functional.one_hot(torch.tensor(task - 1), num_classes=3)  # Adjust task to zero-based
            label = torch.cat((label[:-8], task_one_hot.float(), label[-8:]), dim=0)

        return inputs, positions, label

class EpisodeBatchSampler(Sampler):
    def __init__(self, labels, batch_size, split=0.85, train=True, random=False):
        self.labels = labels
        self.batch_size = batch_size
        self.train = train
        self.random = random
        self.dataset_indices = self._create_dataset_indices()

        episodes = list(self.labels.keys())
        split_idx = int(len(episodes) * split)
        
        if self.train:
            self.episodes = episodes[:split_idx]
        else:
            self.episodes = episodes[split_idx:]

        self.shuffle_episodes()

    def shuffle_episodes(self):
        np.random.shuffle(self.episodes)
        self.batches = []
        self.prepare_batches()

    def _create_dataset_indices(self):
        episode_sample_pairs = [
            (episode, sample) for episode, samples in self.labels.items() \
                for sample in list(samples.keys())
        ]
        return {pair: idx for idx, pair in enumerate(episode_sample_pairs)}

    def prepare_batches(self):
        num_batches = math.ceil(len(self.episodes) / self.batch_size)

        for batch_idx in range(num_batches):
            batch_episodes = self.episodes[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
            batch = []
            if self.random == False:
                max_samples = max(len(self.labels[episode]) for episode in batch_episodes)
                
                for sample_idx in range(SEQUENCE_LENGTH - 1, max_samples):
                    sample_batch = []
                    for episode in batch_episodes:
                        samples = list(self.labels[episode].keys())
                        if sample_idx < len(samples):
                            sample = samples[sample_idx]
                            idx = self.dataset_indices[(episode, sample)]
                            sample_batch.append(idx)
                    batch.append(sample_batch)

            else:
                episode_sample_sequences = {episode: np.random.permutation(list(self.labels[episode].keys())[SEQUENCE_LENGTH - 1:]) for episode in batch_episodes}
                max_samples = max(len(episode_sample_sequences[episode]) for episode in batch_episodes)

                for sample_idx in range(max_samples):
                    sample_batch = []
                    for episode in batch_episodes:
                        samples = episode_sample_sequences[episode]
                        if sample_idx < len(samples):
                            sample = samples[sample_idx]
                            idx = self.dataset_indices[(episode, sample)]
                            sample_batch.append(idx)
                    batch.append(sample_batch)
                
            self.batches.extend(batch)

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)
    

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

def run_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
              pbar: tqdm, epoch: int, device: torch.device, is_training: bool = True, 
              optimizer: optim.Optimizer = None) -> float:
    """
    Run a single epoch of training or validation.

    Parameters:
    - model: The neural network model.
    - dataloader: DataLoader for the dataset.
    - criterion: Loss function.
    - pbar: TQDM progress bar.
    - epoch: Current epoch number.
    - device: Device to run the training.
    - is_training: Boolean flag to indicate if the epoch is for training.
    - optimizer: Optimizer for training.

    Returns:
    - Mean loss for the epoch.
    """
    model.train() if is_training else model.eval()
    epoch_losses = []
    sample = 0

    with torch.set_grad_enabled(is_training):
        for i, data in tqdm(enumerate(dataloader, 0), leave=False):
            inputs, positions, labels = data
            inputs = [input.to(device) for input in inputs]
            positions = positions.to(device)
            labels = labels.to(device)
            current_batch_size = labels.shape[0]

            if is_training:
                optimizer.zero_grad()

            outputs = model(inputs, positions)
            loss = calculate_loss(criterion, outputs, labels)
            epoch_losses.append(loss.item())

            if is_training:
                loss.backward()
                optimizer.step()

            sample += current_batch_size
            pbar.set_description('[%d, %5d] %s loss: %.8f batches: %d' %
                                 (epoch, sample, 'train' if is_training else 'validation', np.mean(epoch_losses[-1]), current_batch_size))
            pbar.update(1)

    return np.mean(epoch_losses)
